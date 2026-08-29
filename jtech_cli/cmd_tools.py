"""Guarded tool protocol, shell command policy, and executor.

The AI requests work with standalone ``jtech_cmd(...)`` and ``jtech_agent(...)``
calls. One scanner owns both: they differ only in name, arity, and argument
validation, so fence handling, HTML-wrapper handling, quoted-literal parsing,
standalone-line detection, span masking, and diagnostics are written once.

Shell calls are then decided in order: absolute blacklist (immutable, all
modes) -> mode (off/yolo) -> allowlist -> prompt. Pure logic lives here so it
is unit-testable without the TUI; the TUI owns prompting, rendering, dispatch,
and re-stream.
"""

from __future__ import annotations

import ast
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import bashlex
from bashlex.errors import ParsingError

VALID_CMD_MODES = ("ask", "auto", "yolo", "off")
DEFAULT_MAX_OUTPUT = 12000

# Read-only seed for a fresh config: nothing that writes or talks to a network.
DEFAULT_ALLOW = (
    "git status:*",
    "git log:*",
    "git diff:*",
    "git show:*",
    "git branch:*",
    "git remote:*",
    "git tag:*",
    "git describe:*",
    "git config --list:*",
    "git config --get:*",
    "ls:*",
    "cat:*",
    "head:*",
    "tail:*",
    "grep:*",
    "rg:*",
    "find:*",
    "wc:*",
    "file:*",
    "stat:*",
    "tree:*",
    "du:*",
    "df:*",
    "pwd:*",
    "echo:*",
    "which:*",
    "type:*",
    "uname:*",
    "whoami:*",
    "hostname:*",
    "date:*",
    "curl:*",
    "wget:*",
    "less:*",
    "more:*",
)


@dataclass
class CmdPolicy:
    """User-configurable command policy (the ``[cmd]`` table of the config file)."""

    mode: str = "ask"
    allow: list[str] = field(default_factory=lambda: list(DEFAULT_ALLOW))
    max_output: int = DEFAULT_MAX_OUTPUT


#: Key rule for a dispatched agent; ``primary`` is the app's own conversation.
_AGENT_KEY_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")
RESERVED_AGENT_KEY = "primary"

ToolName = Literal["jtech_cmd", "jtech_agent"]


def _single_line(field_name: str, value: str) -> str:
    """Return ``value`` stripped, rejecting an empty or multi-line label.

    Raises:
        ValueError: if the stripped value is empty or spans more than one line.
    """
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    if text.splitlines() != [text]:
        raise ValueError(f"{field_name} must be a single line")
    return text


@dataclass(frozen=True, slots=True)
class AgentDispatch:
    """One validated ``jtech_agent(...)`` call.

    The boundary value between the tool protocol and orchestration. Every field
    is normalized (outer whitespace stripped) and validated here, so no caller
    truncates, synthesizes, or silently replaces a label or a task.

    Args:
        agent_key: Stable lowercase key identifying one reusable private
            conversation. Reusing it continues that agent.
        agent_label: Single-line sidebar label; immutable for the key.
        profile_name: Name of an available API profile; immutable for the key.
        task_label: Single-line task label shown beneath the agent.
        task: The complete instruction sent to the agent. May be multiline.

    Raises:
        ValueError: if any field violates those rules.
    """

    agent_key: str
    agent_label: str
    profile_name: str
    task_label: str
    task: str

    def __post_init__(self) -> None:
        # ``object.__setattr__`` because the boundary value is frozen: this is
        # normalization at construction, not mutation afterwards.
        key = self.agent_key.strip()
        if not _AGENT_KEY_PATTERN.fullmatch(key):
            raise ValueError(
                f"agent_key {key!r} is invalid: use lowercase letters, digits, "
                "'-' or '_', starting with a letter or digit"
            )
        if key == RESERVED_AGENT_KEY:
            raise ValueError(
                f"agent_key {RESERVED_AGENT_KEY!r} is reserved for the coordinator"
            )
        object.__setattr__(self, "agent_key", key)
        object.__setattr__(
            self, "agent_label", _single_line("agent_label", self.agent_label)
        )
        object.__setattr__(
            self, "profile_name", _single_line("profile_name", self.profile_name)
        )
        object.__setattr__(
            self, "task_label", _single_line("task_label", self.task_label)
        )
        task = self.task.strip()
        if not task:
            raise ValueError("task must not be empty")
        object.__setattr__(self, "task", task)


@dataclass(frozen=True, slots=True)
class ToolProtocolError:
    """One recognized tool call the runtime refuses to execute.

    Args:
        tool_name: The tool whose call line failed.
        line: One-based line of the candidate call in the model's own reply.
        message: What the model must correct.
    """

    tool_name: ToolName
    line: int
    message: str


@dataclass
class ParsedReply:
    """The executable tool calls, diagnostics, and surrounding commentary."""

    commands: list[str]
    commentary: str = ""
    dispatches: list[AgentDispatch] = field(default_factory=list)
    errors: list[ToolProtocolError] = field(default_factory=list)


class ShellParseError(ValueError):
    """Raised when a shell command cannot be analyzed safely."""


@dataclass(frozen=True)
class _ShellCommand:
    """One executable Bash command extracted from the parsed syntax tree."""

    source: str
    words: tuple[str, ...]
    redirect_types: tuple[str, ...]
    dynamic_program: bool

    @property
    def program(self) -> str | None:
        """Static program basename, or ``None`` when Bash computes the name."""
        if not self.words or self.dynamic_program:
            return None
        return self.words[0].rsplit("/", 1)[-1]


@dataclass(frozen=True)
class _ShellAnalysis:
    """Policy-relevant facts from one non-executing Bash parse."""

    commands: tuple[_ShellCommand, ...]
    pipeline_targets: tuple[str, ...]


def _command_word_nodes(parts: list[object]) -> list[object]:
    return [part for part in parts if getattr(part, "kind", None) == "word"]


def _static_program_from_parts(parts: list[object]) -> str | None:
    words = _command_word_nodes(parts)
    if not words or getattr(words[0], "parts", None):
        return None
    return words[0].word.rsplit("/", 1)[-1]


class _ShellCollector(bashlex.ast.nodevisitor):
    """Collect commands and pipeline targets while walking a Bash AST."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.commands: list[_ShellCommand] = []
        self.pipeline_targets: list[str] = []

    def visitcommand(self, node: object, parts: list[object]) -> None:
        word_nodes = _command_word_nodes(parts)
        words = tuple(word.word for word in word_nodes)
        redirects = tuple(
            part.type for part in parts if getattr(part, "kind", None) == "redirect"
        )
        self.commands.append(
            _ShellCommand(
                source=self.source[node.pos[0] : node.pos[1]].strip(),
                words=words,
                redirect_types=redirects,
                dynamic_program=bool(
                    word_nodes and getattr(word_nodes[0], "parts", None)
                ),
            )
        )

    def visitpipeline(self, _node: object, parts: list[object]) -> None:
        for index, part in enumerate(parts[:-1]):
            if getattr(part, "kind", None) != "pipe":
                continue
            target = parts[index + 1]
            if getattr(target, "kind", None) != "command":
                continue
            program = _static_program_from_parts(target.parts)
            if program is not None:
                self.pipeline_targets.append(program)


def _analyze_shell(command: str) -> _ShellAnalysis:
    """Parse Bash without execution and return the facts used by command policy."""
    if not command.strip():
        return _ShellAnalysis((), ())
    try:
        roots = bashlex.parse(command)
    except (ParsingError, NotImplementedError) as error:
        raise ShellParseError(f"shell syntax could not be analyzed: {error}") from error

    collector = _ShellCollector(command)
    for root in roots:
        collector.visit(root)
    if not collector.commands:
        raise ShellParseError("shell syntax contains no executable command")
    return _ShellAnalysis(
        commands=tuple(collector.commands),
        pipeline_targets=tuple(collector.pipeline_targets),
    )


# ---------------------------------------------------------------- parsing

_JTECH_CMD = "jtech_cmd"
_JTECH_AGENT = "jtech_agent"
#: The complete tool vocabulary, and how many string arguments each call takes.
_TOOL_ARITY: dict[str, int] = {_JTECH_CMD: 1, _JTECH_AGENT: 5}


class _CallSyntaxError(ValueError):
    """One recognized tool call that cannot be parsed or validated."""

    def __init__(self, tool_name: str, message: str) -> None:
        super().__init__(message)
        self.tool_name = tool_name


@dataclass(frozen=True, slots=True)
class _ToolCall:
    """One syntactically valid call and the span it occupies in the reply."""

    name: str
    args: tuple[str, ...]
    start: int
    end: int


def _skip_ws(text: str, position: int) -> int:
    while position < len(text) and text[position].isspace():
        position += 1
    return position


def _quoted_literal_end(text: str, start: int) -> int | None:
    """Return the end of the quoted literal beginning at ``start``."""
    quote = text[start]
    delimiter = quote * 3 if text.startswith(quote * 3, start) else quote
    position = start + len(delimiter)
    while position < len(text):
        if text[position] == "\\":
            position += 2
        elif text.startswith(delimiter, position):
            return position + len(delimiter)
        else:
            position += 1
    return None


def _tool_name_at(text: str, position: int) -> str | None:
    """The tool name beginning at ``position`` as a whole token, else ``None``.

    The token boundary matters: ``jtech_cmdline`` names no tool, so a line that
    starts with it is prose rather than a malformed call.
    """
    for name in _TOOL_ARITY:
        if not text.startswith(name, position):
            continue
        after = position + len(name)
        if after < len(text) and not (text[after].isspace() or text[after] == "("):
            continue
        return name
    return None


#: A leading GFM task-list marker, up to and including the checkbox, on either
#: list kind: a bullet or an ordered ``1.``/``1)``. Removed as a whole token
#: because a checked box carries a letter, and the rule below — decoration is
#: anything that is not a letter — has to stay letter-free to keep prose out.
#: Enumerating decoration instead was the earlier approach and it leaked: every
#: Markdown flavour spells a wrapper with different punctuation (task boxes,
#: strikethrough, table cells), so the set was never finished. Naming what
#: decoration is *not* has one edge, and the checkbox is all of it.
_TASK_LIST_MARKER = re.compile(r"^[\s>]*(?:[-*+]|\d{1,9}[.)])\s+\[[ xX]\]")
_HTML_CODE_TAG = re.compile(r"</?code\b[^>]*>", re.IGNORECASE)
#: An opening code fence: three or more backticks or tildes. Both the marker
#: character and its length are state, because a longer fence quotes a shorter
#: one — a three-backtick line inside a four-backtick block is content, and
#: closing on it hands the rest of the block to the scanner as executable text.
_FENCE_OPEN = re.compile(r"^(`{3,}|~{3,})")
#: Why a recognized call did not run, as the model is told it.
_FENCE_CONTEXT = "was written inside a code fence"
_HTML_CODE_CONTEXT = "was written inside an HTML code block"
_DECORATION_CONTEXT = "was indented or wrapped in Markdown formatting"


def _wrapped_call_name(line: str) -> str | None:
    """The tool this line opens a call to behind decoration, else ``None``.

    Recognition stops at the opening ``(``: it deliberately does not parse the
    call. A wrapped call may be multiline, may share its line with a second
    call, or may be malformed, and every one of those is still the model
    asking for a tool it will not get. Requiring a complete parse here returns
    ``None`` for exactly those shapes and leaves them silent, which is the
    failure this whole path exists to prevent.

    What discriminates is the *prefix*: HTML code tags, a task-list marker, and
    Markdown underscore escaping are removed, and then no letter may stand
    between the start of the line and the tool name. Punctuation is how every
    wrapper is spelled — bullets, quote markers, emphasis, strikethrough, code
    spans, table cells — so it is all decoration by construction, while a
    mention inside a sentence (``Run this next: jtech_cmd("ls")``) keeps its
    letters and stays commentary.
    """
    text = _HTML_CODE_TAG.sub("", line.replace("\\_", "_"))
    text = _TASK_LIST_MARKER.sub("", text, count=1)
    for start, char in enumerate(text):
        if not char.isalpha():
            continue
        name = _tool_name_at(text, start)
        if name is None:
            return None
        # A bare word is prose ("jtech_cmd runs shell commands"); the opening
        # paren is what makes the line a call the model expected to run.
        after = _skip_ws(text, start + len(name))
        return name if after < len(text) and text[after] == "(" else None
    return None


def _near_miss_message(tool_name: str, context: str) -> str:
    """What the model must change to turn a wrapped call into a running one."""
    return (
        f"{tool_name} {context}, so it did not run. A call executes only as "
        "raw text starting at the very first column of its own line, with no "
        "indent, fence, code span, list marker, quote marker, or emphasis "
        "around it. If you meant to show the syntax rather than run it, "
        "describe it in a sentence instead."
    )


def _near_miss_errors(
    text: str, skipped: Sequence[tuple[int, str]], base_line: int
) -> list[ToolProtocolError]:
    """Diagnostics for calls that were skipped because something wrapped them.

    Only for a reply that ran nothing: silence is the failure being prevented,
    and a reply that executed a call is never silent.
    """
    errors: list[ToolProtocolError] = []
    for position, context in skipped:
        line_end = text.find("\n", position)
        line = text[position:] if line_end < 0 else text[position:line_end]
        tool_name = _wrapped_call_name(line)
        if tool_name is None:
            continue
        errors.append(
            ToolProtocolError(
                tool_name,
                base_line + text.count("\n", 0, position) + 1,
                _near_miss_message(tool_name, context),
            )
        )
    return errors


def _unwrap_html_code(text: str) -> tuple[str, int]:
    """Remove one whole-response HTML code wrapper, and count the lines it hid.

    Returns the text to scan and the number of reply lines consumed by the
    wrapper, so a diagnostic still names the line the model actually wrote.
    """
    opening = "<code>"
    closing = "</code>"
    if not (text.startswith(opening) and text.endswith(closing)):
        return text, 0
    inner_raw = text[len(opening) : -len(closing)]
    inner = inner_raw.strip()
    if not inner or "<code>" in inner or "</code>" in inner:
        return text, 0
    if _tool_name_at(inner, 0) is None:
        return text, 0
    skipped = len(inner_raw) - len(inner_raw.lstrip())
    return inner, text.count("\n", 0, len(opening) + skipped)


def _scan_source(reply: str) -> tuple[str, int]:
    """The text to scan, and how many reply lines precede its first line.

    Only whole blank lines come off the front. A call's column is part of the
    contract, so trimming the first line's own indentation would quietly move
    it to column zero and run an indented code block that happened to be all
    the model said.
    """
    trailing = reply.rstrip()
    lead = 0
    while True:
        line_end = trailing.find("\n", lead)
        if line_end < 0 or trailing[lead:line_end].strip():
            break
        lead = line_end + 1
    text, wrapper_lines = _unwrap_html_code(trailing[lead:])
    return text, trailing.count("\n", 0, lead) + wrapper_lines


def _parse_call_at(text: str, start: int) -> _ToolCall:
    """Parse the recognized tool call beginning at ``start``.

    ``ast.literal_eval`` parses each argument without evaluating Python code.

    Raises:
        _CallSyntaxError: if the call is malformed, carries a non-string
            argument, or has the wrong number of arguments.
    """
    name = _tool_name_at(text, start)
    if name is None:  # pragma: no cover - callers check first
        raise ValueError(f"no tool call at position {start}")
    position = _skip_ws(text, start + len(name))
    if position >= len(text) or text[position] != "(":
        raise _CallSyntaxError(name, f"{name} must be called as {name}(...)")
    position = _skip_ws(text, position + 1)
    args: list[str] = []
    if position < len(text) and text[position] == ")":
        position += 1
    else:
        while True:
            position = _skip_ws(text, position)
            ordinal = len(args) + 1
            if position >= len(text) or text[position] not in "'\"":
                raise _CallSyntaxError(
                    name, f"{name} argument {ordinal} must be a quoted string"
                )
            literal_end = _quoted_literal_end(text, position)
            if literal_end is None:
                raise _CallSyntaxError(
                    name,
                    f"{name} argument {ordinal} has an unterminated string literal",
                )
            try:
                value = ast.literal_eval(text[position:literal_end])
            except (SyntaxError, ValueError) as error:
                raise _CallSyntaxError(
                    name, f"{name} argument {ordinal} is not a valid string literal"
                ) from error
            if not isinstance(value, str):
                raise _CallSyntaxError(
                    name, f"{name} argument {ordinal} must be a string"
                )
            args.append(value)
            position = _skip_ws(text, literal_end)
            if position < len(text) and text[position] == ",":
                position += 1
                continue
            if position < len(text) and text[position] == ")":
                position += 1
                break
            raise _CallSyntaxError(
                name, f"{name} argument {ordinal} must be followed by ',' or ')'"
            )
    arity = _TOOL_ARITY[name]
    if len(args) != arity:
        plural = "" if arity == 1 else "s"
        raise _CallSyntaxError(
            name,
            f"{name} takes exactly {arity} string argument{plural}, got {len(args)}",
        )
    return _ToolCall(name, tuple(args), start, position)


def _parse_call_line(text: str, start: int) -> tuple[list[_ToolCall], int] | None:
    """Parse the standalone tool calls on the line beginning at ``start``.

    Returns the calls and the next scan position, or ``None`` when the line is
    not a tool-call candidate at all — that is ordinary commentary.

    The call must start at column zero. Indentation is not decoration the
    scanner can see past: four spaces *is* a Markdown code block, and no
    amount of fence tracking catches a code block that has no fence. Making
    the first column the whole rule keeps one contract for the model to follow
    and leaves an indented call to the near-miss diagnostic.

    Raises:
        _CallSyntaxError: if a candidate line is malformed or carries trailing
            text that is not another call.
    """
    if _tool_name_at(text, start) is None:
        return None
    position = start

    calls: list[_ToolCall] = []
    while True:
        call = _parse_call_at(text, position)
        calls.append(call)
        tail_end = text.find("\n", call.end)
        if tail_end < 0:
            tail_end = len(text)
        tail_start = call.end
        while tail_start < tail_end and text[tail_start] in " \t":
            tail_start += 1
        if tail_start == tail_end:
            return calls, tail_end + (tail_end < len(text))
        if _tool_name_at(text, tail_start) is None:
            raise _CallSyntaxError(
                call.name,
                f"a {call.name} call line may not carry other text: "
                f"{text[tail_start:tail_end].strip()!r}",
            )
        position = tail_start


def parse_jtech_reply(reply: str) -> ParsedReply:
    """Parse standalone tool-call lines anywhere in a reply.

    A call must start at column zero and occupy the rest of that line,
    although several calls may share one line. This permits commentary before,
    between, and after calls without executing inline examples. Code blocks
    are inert in every form Markdown gives them — backtick and tilde fences of
    any marker length, and indentation, which needs no fence at all — as are
    HTML ``<code>`` blocks; a whole-response ``<code>`` wrapper remains
    supported for either tool.

    A line that opens with a tool name but is malformed, mis-typed, wrongly
    sized, or followed by other text becomes a :class:`ToolProtocolError`
    rather than silently reverting to commentary — the model asked for a tool
    and is told exactly why it did not run.

    A reply that runs nothing gets the same treatment for a call some wrapper
    hid: an indented, fenced, bulleted, quoted, or emphasized call is still
    refused, but it is reported rather than dropped, because a reply with no
    call and no diagnostic reads as a final answer and ends the turn in
    silence.
    """
    text, base_line = _scan_source(reply)
    if not text:
        return ParsedReply([])

    commands: list[str] = []
    dispatches: list[AgentDispatch] = []
    errors: list[ToolProtocolError] = []
    spans: list[tuple[int, int]] = []
    # Position and wrapper kind of every line the scan passes over. Recorded
    # unconditionally because the scan owns the fence/HTML state a second pass
    # would have to duplicate; nothing is parsed unless the reply ran nothing.
    skipped: list[tuple[int, str]] = []

    def line_of(position: int) -> int:
        return base_line + text.count("\n", 0, position) + 1

    position = 0
    # The open fence's marker character and length, or None outside a fence.
    fence: tuple[str, int] | None = None
    in_html_code = False
    while position < len(text):
        line_end = text.find("\n", position)
        if line_end < 0:
            line_end = len(text)
        line = text[position:line_end]
        stripped = line.strip()
        lower = stripped.lower()

        if fence is not None:
            marker, size = fence
            # A closing fence is the same character, at least as long as the
            # opening one, and carries nothing else on its line.
            if len(stripped) >= size and set(stripped) == {marker}:
                fence = None
            skipped.append((position, _FENCE_CONTEXT))
            position = line_end + (line_end < len(text))
            continue
        if in_html_code:
            if "</code>" in lower:
                in_html_code = False
            skipped.append((position, _HTML_CODE_CONTEXT))
            position = line_end + (line_end < len(text))
            continue
        opening = _FENCE_OPEN.match(stripped)
        if opening is not None:
            fence = (stripped[0], len(opening.group()))
            skipped.append((position, _FENCE_CONTEXT))
            position = line_end + (line_end < len(text))
            continue
        if "<code" in lower:
            in_html_code = "</code>" not in lower
            skipped.append((position, _HTML_CODE_CONTEXT))
            position = line_end + (line_end < len(text))
            continue

        try:
            parsed_line = _parse_call_line(text, position)
        except _CallSyntaxError as error:
            errors.append(
                ToolProtocolError(error.tool_name, line_of(position), str(error))
            )
            position = line_end + (line_end < len(text))
            continue
        if parsed_line is None:
            skipped.append((position, _DECORATION_CONTEXT))
            position = line_end + (line_end < len(text))
            continue
        line_calls, next_position = parsed_line
        for call in line_calls:
            spans.append((call.start, call.end))
            if call.name == _JTECH_CMD:
                commands.append(call.args[0])
                continue
            try:
                dispatches.append(AgentDispatch(*call.args))
            except ValueError as error:
                errors.append(
                    ToolProtocolError(_JTECH_AGENT, line_of(call.start), str(error))
                )
        position = next_position

    if not spans:
        # Nothing executed, so nothing keeps the turn alive. An existing
        # diagnostic already says why; otherwise a wrapped call is the one
        # remaining explanation worth giving.
        return ParsedReply(
            commands,
            text,
            dispatches,
            errors or _near_miss_errors(text, skipped, base_line),
        )

    masked = list(text)
    for start, end in spans:
        for index in range(start, end):
            if masked[index] != "\n":
                masked[index] = " "
    commentary = "".join(masked).strip()
    return ParsedReply(commands, commentary, dispatches, errors)


def duplicate_agent_keys(dispatches: Sequence[AgentDispatch]) -> tuple[str, ...]:
    """Agent keys a single reply dispatches more than once, in sorted order.

    One agent key is one conversation, so a batch cannot write to it twice
    concurrently. The caller rejects the whole batch rather than serializing it.
    """
    keys = [dispatch.agent_key for dispatch in dispatches]
    return tuple(sorted({key for key in keys if keys.count(key) > 1}))


def split_segments(command: str) -> list[str]:
    """Return executable Bash commands, including nested substitutions, in order."""
    return [segment.source for segment in _analyze_shell(command).commands]


def program_names(command: str) -> list[str]:
    """Return every statically known executable basename in a Bash command."""
    return [
        segment.program
        for segment in _analyze_shell(command).commands
        if segment.program is not None
    ]


# ---------------------------------------------------------------- blacklist

# Programs that are never allowed, in any mode, as the leading word of any segment.
FORBIDDEN_PROGRAMS = {
    "sudo", "su",
    "mkfs", "fdisk", "parted", "shred",
    "shutdown", "reboot", "halt", "poweroff",
    "mount", "umount", "mkswap", "swapon", "swapoff",
    "iptables", "nft",
    "insmod", "rmmod", "modprobe",
    "pass", "keyring", "secrets",
}

# Punctuation that can cling to an operand after Bash word parsing.
_ARG_STRIP = "();,'\"`"

# rm targets that are absolute no-go (exact match).
RM_ROOT_TARGETS = {"/", "/*", "~", "$HOME", "/etc", "/usr", "/var", "/boot", "/home", "/System"}
# rm targets whose subpaths are still system files (prefix match).
RM_PREFIX_TARGETS = ("/etc/", "/usr/", "/var/", "/boot/", "/System")

_CREDENTIAL_RE = re.compile(
    r"(~|\$HOME|/home/[^/\s]+|/root)/\.ssh|\.aws/credentials|\.netrc|id_rsa(?!\.pub)|id_ed25519|\.pgpass"
)
_PIPE_SHELL_PROGRAMS = frozenset(
    {"sh", "bash", "zsh", "dash", "ksh", "python", "python3", "perl", "ruby"}
)


def _dangerous_rm_target(token: str) -> bool:
    t = token.strip(_ARG_STRIP)
    return t in RM_ROOT_TARGETS or t.startswith(RM_PREFIX_TARGETS)


def _find_exec_commands(command: str) -> list[str]:
    """Commands find would execute via -exec/-execdir/-ok, in order.

    Each is the tokens between the flag and its ``\\;``/``+`` terminator,
    with the ``{}`` placeholder dropped. An unterminated -exec is ignored.
    """
    return _find_exec_commands_from_analysis(_analyze_shell(command))


def _find_exec_commands_from_analysis(analysis: _ShellAnalysis) -> list[str]:
    commands: list[str] = []
    for segment in analysis.commands:
        if segment.program != "find":
            continue
        current: list[str] | None = None
        for token in segment.words[1:]:
            if current is None:
                if token in ("-exec", "-execdir", "-ok"):
                    current = []
            elif token in (";", "+"):
                commands.append(" ".join(item for item in current if item != "{}"))
                current = None
            elif token != "{}":
                current.append(token)
    return commands


def _check_blacklist(command: str, analysis: _ShellAnalysis) -> str | None:
    for segment in analysis.commands:
        if segment.dynamic_program:
            return "a dynamically computed command name cannot be safety-checked"
        program = segment.program
        if program is None:
            continue
        if program in FORBIDDEN_PROGRAMS or program.startswith("mkfs"):
            return f"'{program}' is on the absolute blacklist"
        if program == "rm":
            for argument in segment.words[1:]:
                if not argument.startswith("-") and _dangerous_rm_target(argument):
                    target = argument.strip(_ARG_STRIP)
                    return f"rm targeting '{target}' is on the absolute blacklist"
        if program == "dd" and any(
            argument.startswith("of=/dev/") for argument in segment.words[1:]
        ):
            return "dd writing to a raw device is on the absolute blacklist"
        if program == "init" and any(
            argument in {"0", "6"} for argument in segment.words[1:]
        ):
            return "power-off via init is on the absolute blacklist"

    for inner in _find_exec_commands_from_analysis(analysis):
        reason = check_blacklist(inner)
        if reason is not None:
            return f"{reason} (inside find -exec)"

    if any(target in _PIPE_SHELL_PROGRAMS for target in analysis.pipeline_targets):
        return "piping data into a shell is on the absolute blacklist"
    if _CREDENTIAL_RE.search(command):
        return "accessing credential/key material is on the absolute blacklist"
    return None


def check_blacklist(command: str) -> str | None:
    """Return a reason if the command touches the absolute blacklist, else None.

    Checked against every segment of the chain plus whole-line patterns, so
    ``git status && rm -rf /`` and ``curl x | sh`` are caught, not just their
    first word. Commands embedded in ``find -exec`` are vetted too —
    ``find . -exec rm / \\;`` is an rm of ``/``, whatever wraps it.
    """
    return _check_blacklist(command, _analyze_shell(command))


# ---------------------------------------------------------------- allowlist


# CLIs whose first bare argument is a subcommand worth pinning (git status:*).
# Everything else gets the bare prog:* rule — interpreters (python:*, one
# approval covers every script) and search/file tools (grep:*, not
# grep <pattern>:*) have operands first, not subcommands.
_SUBCOMMAND_PROGRAMS = frozenset(
    {
        "git", "npm", "npx", "pnpm", "yarn", "bun", "docker", "podman",
        "pip", "pip3", "cargo", "go", "make", "kubectl", "gh", "brew",
        "composer", "gradle", "mvn",
    }
)


def _segment_matches(segment: _ShellCommand, allow: list[str]) -> bool:
    """True when one segment matches some allow rule.

    Rules: ``name`` matches the bare program with no args; ``name:*`` matches
    the program with any args; ``name sub:*`` matches the program with first
    arg ``sub`` and any further args.
    """
    program = segment.program
    if program is None:
        return False
    for rule in allow:
        rule = rule.strip()
        if not rule:
            continue
        if rule.endswith(":*"):
            parts = rule[:-2].split()
            arguments = segment.words[1 : 1 + len(parts) - 1]
            if program == parts[0] and tuple(parts[1:]) == arguments:
                return True
        elif rule == program and len(segment.words) == 1:
            return True
    return False


def _matches_allow(analysis: _ShellAnalysis, allow: list[str]) -> bool:
    return bool(analysis.commands) and all(
        _segment_matches(segment, allow) for segment in analysis.commands
    )


def matches_allow(command: str, allow: list[str]) -> bool:
    """True when every segment of the command matches an allow rule.

    A pipeline or chain auto-runs only if each part is individually
    allowlisted — ``grep x | head`` needs both ``grep:*`` and ``head:*``.
    Any segment without a rule forces a prompt, so no part runs unvetted.
    """
    return _matches_allow(_analyze_shell(command), allow)


def _rule_for_segment(segment: _ShellCommand) -> str | None:
    """The allow rule for one segment: program (+ pinned subcommand) + ``:*``.

    The subcommand is pinned only for known subcommand-style CLIs; for
    interpreters and operand-first tools the rule is the bare program so a
    single approval covers all of them.
    """
    program = segment.program
    if program is None:
        return None
    if (
        program in _SUBCOMMAND_PROGRAMS
        and len(segment.words) >= 2
        and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", segment.words[1])
    ):
        return f"{program} {segment.words[1]}:*"
    return f"{program}:*"


def _allow_rule_for(
    analysis: _ShellAnalysis, allow: list[str] | None = None
) -> str | None:
    for segment in analysis.commands:
        if allow is None or not _segment_matches(segment, allow):
            return _rule_for_segment(segment)
    return None


def allow_rule_for(command: str, allow: list[str] | None = None) -> str | None:
    """The rule to persist for 'always allow' on this command, or None.

    The rule for the first segment not already covered by ``allow`` — the
    piece that made the command prompt. Chains earn their rules one segment
    at a time, each explicitly approved; every segment covered -> None.
    """
    return _allow_rule_for(_analyze_shell(command), allow)


# ---------------------------------------------------------------- scope


def escape_project(command: str, root: Path) -> bool:
    """True when any token of the command reaches outside the project root.

    Flags ``~``/``$HOME``, ``..`` components, and absolute paths that are not
    under ``root``. Used to force a prompt in ask/auto modes; yolo ignores it.
    """
    return _escapes_project(_analyze_shell(command), root)


def _escapes_project(analysis: _ShellAnalysis, root: Path) -> bool:
    for segment in analysis.commands:
        for word in segment.words:
            token = word.strip(_ARG_STRIP)
            if not token:
                continue
            if token in ("~", "$HOME") or token.startswith(("~/", "$HOME/")):
                return True
            if ".." in token.split("/"):
                return True
            if token.startswith("/"):
                path = Path(token)
                if not (path == root or root in path.parents):
                    return True
    return False


# ---------------------------------------------------------------- decision


@dataclass(frozen=True)
class Decision:
    action: str  # "run" | "ask" | "blocked"
    reason: str = ""


# Shell-level actions an allowlist grant must never silently cover. An
# allowlist match is a read-only grant; anything acting on disk forces a prompt.
_WRITE_REDIRECT_TYPES = frozenset({">", ">>", ">|", "&>", "&>>"})
_FIND_ACTIONS = frozenset({"-exec", "-execdir", "-ok", "-delete"})


def _acting_reason(analysis: _ShellAnalysis) -> str | None:
    for segment in analysis.commands:
        if segment.program == "find" and any(
            word in _FIND_ACTIONS for word in segment.words[1:]
        ):
            return "find -exec/-ok/-delete executes or deletes"
        if any(kind in _WRITE_REDIRECT_TYPES for kind in segment.redirect_types):
            return "output redirection writes to a file"
    return None


def acting_reason(command: str) -> str | None:
    """Why an allowlisted command still must prompt, or None if it is inert."""
    return _acting_reason(_analyze_shell(command))


def decide(command: str, policy: CmdPolicy, root: Path) -> Decision:
    """The full gate: blacklist (absolute) -> off -> yolo -> allowlist -> ask.

    An allowlist match only auto-runs commands that stay read-only; a command
    that writes files or embeds execution prompts even when fully allowlisted.
    yolo is exempt (explicit max-trust mode; the blacklist still applies).
    """
    try:
        analysis = _analyze_shell(command)
    except ShellParseError as error:
        return Decision("blocked", str(error))

    reason = _check_blacklist(command, analysis)
    if reason is not None:
        return Decision("blocked", reason)
    if policy.mode == "off":
        return Decision("blocked", "command execution is disabled (cmd_mode = off)")
    if policy.mode == "yolo":
        return Decision("run")
    if _matches_allow(analysis, policy.allow):
        acting = _acting_reason(analysis)
        if acting is None:
            return Decision("run")
        return Decision("ask", f"allowlisted, but {acting}")
    why = "not in the allowlist"
    if _escapes_project(analysis, root):
        why += "; references paths outside the project"
    return Decision("ask", why)


# ---------------------------------------------------------------- execution


@dataclass
class ExecResult:
    exit_code: int
    output: str
    interrupted: bool = False
    truncated: bool = False


def _truncation_marker(dropped: int) -> str:
    """The separator standing in for the characters a cap removed.

    One definition, because two producers emit it: whole-string truncation and
    the streaming collector below. They must be indistinguishable to a reader.
    """
    return f"\n…[{dropped} chars truncated]…\n"


def truncate_output(text: str, max_output: int) -> tuple[str, bool]:
    """Cap output at ``max_output`` chars, keeping head and tail."""
    text = text.strip("\n")
    if len(text) <= max_output:
        return text, False
    half = max_output // 2
    dropped = len(text) - max_output
    return text[:half] + _truncation_marker(dropped) + text[-half:], True


class BoundedOutput:
    """Accumulate a command's output within a fixed memory budget.

    `truncate_output` needs the whole string, so capturing output for it means
    holding all of it: with no execution deadline, a verbose or endless command
    would grow the process until it died. This keeps the same head, the same
    tail, and the same dropped count while retaining ``O(limit)`` characters
    however much the command prints.

    For every ``limit >= 2`` the result is byte-for-byte what
    ``truncate_output(whole, limit)`` returns for the same bytes, so the two are
    interchangeable at the boundary. That includes ``strip("\n")``: leading
    newlines are discarded on arrival and trailing ones are held back until
    something proves them interior, which is what makes the character counts
    agree.

    At ``limit == 1`` they differ, and deliberately. ``truncate_output`` slices
    its tail as ``text[-0:]``, which is the whole string, so a one-character cap
    returns everything — the one input where it does not cap at all. Reproducing
    that here would reintroduce the unbounded retention this class exists to
    prevent, so the tail is empty instead. No configuration path produces that
    value on purpose.

    Args:
        limit: The same ``max_output`` cap ``truncate_output`` takes.
    """

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._head: list[str] = []
        self._head_len = 0
        self._tail = ""
        self._length = 0
        # Newlines seen at the end so far. They are content only if a later
        # chunk turns out to follow them; otherwise they are what strip removes.
        self._pending_newlines = 0
        self._started = False

    def add(self, chunk: str) -> None:
        """Take one chunk of output, keeping at most ``limit`` head and tail."""
        if not self._started:
            chunk = chunk.lstrip("\n")
            if not chunk:
                return
            self._started = True
        body = chunk.rstrip("\n")
        trailing = len(chunk) - len(body)
        if not body:
            self._pending_newlines += trailing
            return
        run = self._pending_newlines
        self._pending_newlines = trailing
        if run:
            # Newlines held back earlier turn out to be interior after all.
            # Past the cap only their count survives, so only that many are
            # built: rendering a million-newline run in full would spike memory
            # exactly the way whole-stream buffering did.
            self._keep("\n" * min(run, self._limit), total=run)
        self._keep(body)

    def _keep(self, text: str, *, total: int | None = None) -> None:
        """Record ``total`` characters of content, of which ``text`` is retained.

        ``total`` differs from ``len(text)`` only for a newline run longer than
        the cap, where the count is the whole of what survives.
        """
        self._length += len(text) if total is None else total
        room = self._limit - self._head_len
        if room > 0:
            kept = text[:room]
            self._head.append(kept)
            self._head_len += len(kept)
        # Slicing per chunk rather than per character: the work is bounded by
        # the cap, and chunks arrive in fixed sizes.
        self._tail = (self._tail + text)[-self._limit :] if self._limit else ""

    def result(self) -> tuple[str, bool]:
        """The capped text and whether anything was dropped."""
        if self._length <= self._limit:
            # Everything that survived stripping is still in the head window.
            return "".join(self._head), False
        half = self._limit // 2
        dropped = self._length - self._limit
        head = "".join(self._head)[:half]
        tail = self._tail[len(self._tail) - half :] if half else ""
        return head + _truncation_marker(dropped) + tail, True


def execute(
    command: str,
    root: Path,
    max_output: int = DEFAULT_MAX_OUTPUT,
) -> ExecResult:
    """Run one command with ``bash -c`` in ``root``, capturing combined output.

    The command has no elapsed-time deadline: it runs until it exits. A build,
    a test suite, or a migration takes as long as it needs, and killing one at
    an arbitrary age turned a legitimate operation into a truncated failure the
    model then had to reason from.
    """
    try:
        proc = subprocess.run(
            ["bash", "-c", command],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            # A non-zero exit is this function's result, reported to the model
            # as such. Raising on it would turn every failing build into an
            # exception the caller has to translate back into the same value.
            check=False,
        )
    except OSError as e:
        return ExecResult(127, str(e))
    out, truncated = truncate_output(proc.stdout or "", max_output)
    return ExecResult(proc.returncode, out, truncated=truncated)


def format_result(command: str, *, result: ExecResult | None = None, note: str | None = None) -> str:
    """The text appended to the session as this command's result message."""
    if note is not None:
        return f"$ {command}\n→ {note}"
    if result is None:
        raise ValueError("provide either result or note")
    if result.interrupted:
        body = f"\n{result.output}" if result.output else ""
        return f"$ {command}\n→ interrupted by user{body}"
    body = result.output if result.output else "(no output)"
    return f"$ {command}\nexit {result.exit_code}\n{body}"
