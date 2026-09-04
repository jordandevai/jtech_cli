"""Guarded tool protocol, shell command policy, and executor.

The AI requests work with ``[[[jtech_cmd]]]`` and ``[[[jtech_agent]]]`` blocks,
and a dispatched subagent ends its turn with a ``[[[jtech_result]]]`` block.
That third block executes nothing: it is terminal, carrying the status and the
report the coordinator receives. One regex owns all three, because they differ
only in name and payload shape, so matching, span masking, and diagnostics are
written once.

Protocol structure is expressed only by exact marker substrings. An opening
marker starts a payload wherever it appears, its matching closing marker ends
that payload, and everything about how the response is laid out around them —
line position, indentation, prose on the same line, Markdown, HTML — is
presentation the parser never reads. Framing depends on protocol tokens alone,
so a model that spells a complete, unambiguous block on one line gets the tool
it asked for.

The payload is never quoted, unescaped, or evaluated. Only the whitespace
framing it comes off, and even that is envelope rather than content: the
leading run for every tool, and the trailing run only for a command, whose
payload has no field of its own to normalize it. That is the whole point of the
format: a command containing quotes, triple quotes, backslashes, or heredocs
cannot damage the envelope carrying it, because the envelope does not share a
lexer with its contents.

Shell blocks are then decided in order: absolute blacklist (immutable, all
modes) -> mode (off/yolo) -> allowlist -> prompt. Pure logic lives here so it
is unit-testable without the TUI; the TUI owns prompting, rendering, dispatch,
and re-stream.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, get_args

import bashlex
from bashlex.errors import ParsingError

from jtech_cli.bash_parser import BashParserCompatibilityError, parse_bash

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

ToolName = Literal["jtech_cmd", "jtech_agent", "jtech_result"]
#: Who a diagnostic is about. A complete block that failed conversion keeps its
#: own tool name; a marker nested inside another block's payload belongs to no
#: complete block of its own, and inventing a tool for it would name one the
#: model never successfully asked for.
ProtocolErrorSource = Literal[
    "jtech_cmd", "jtech_agent", "jtech_result", "jtech_protocol"
]
AgentResultStatus = Literal["completed", "failed"]


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
    """One validated ``[[[jtech_agent]]]`` block.

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
class AgentResult:
    """One validated ``[[[jtech_result]]]`` block: a subagent's terminal status.

    The boundary value between the tool protocol and the coordinator, and the
    only thing that may end a subagent's turn. Prose is not a status, so a run
    can be reported completed only once this value exists — which is what stops
    a model saying the work failed from being recorded as a success.

    ``content`` is normalized (outer whitespace stripped) and validated here,
    and its internal newlines are preserved exactly: it reaches the coordinator
    verbatim as the whole of what the subagent has to report.

    Args:
        status: ``completed`` only when the assignment was actually achieved,
            ``failed`` when an unresolved blocker prevented it.
        content: The self-contained report. May be multiline.

    Raises:
        ValueError: on any other status, or on an empty report.
    """

    status: AgentResultStatus
    content: str

    def __post_init__(self) -> None:
        valid = get_args(AgentResultStatus)
        if self.status not in valid:
            raise ValueError(
                f"status {self.status!r} is invalid: use "
                + " or ".join(repr(name) for name in valid)
            )
        content = self.content.strip()
        if not content:
            raise ValueError("content must not be empty")
        # ``object.__setattr__`` because the boundary value is frozen: this is
        # normalization at construction, not mutation afterwards.
        object.__setattr__(self, "content", content)


@dataclass(frozen=True, slots=True)
class ToolProtocolError:
    """One protocol block the runtime refuses to execute.

    Args:
        tool_name: The tool whose block failed, or ``jtech_protocol`` for a
            marker nested inside another block's payload.
        line: One-based line the offending marker or line falls on in the
            model's own reply.
        message: What the model must correct.
    """

    tool_name: ProtocolErrorSource
    line: int
    message: str


@dataclass
class ParsedReply:
    """The executable tool blocks, the terminal result, diagnostics, and commentary.

    ``result`` and ``errors`` are never both populated for the same result
    block: a boundary carrying a diagnostic *and* an apparently usable terminal
    status is internally contradictory, so an invalid or repeated
    ``[[[jtech_result]]]`` yields diagnostics and no result at all.
    """

    commands: list[str]
    commentary: str = ""
    dispatches: list[AgentDispatch] = field(default_factory=list)
    errors: list[ToolProtocolError] = field(default_factory=list)
    result: AgentResult | None = None


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
        roots = parse_bash(command)
    except (
        ParsingError,
        NotImplementedError,
        BashParserCompatibilityError,
    ) as error:
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
_JTECH_RESULT = "jtech_result"
_JTECH_PROTOCOL = "jtech_protocol"
#: The complete tool vocabulary. Membership here is what makes a marker part of
#: the protocol at all, so the terminal result block is framed, matched, and
#: diagnosed exactly like the executable ones.
_TOOL_NAMES: tuple[ToolName, ...] = (_JTECH_CMD, _JTECH_AGENT, _JTECH_RESULT)

_OPEN_PREFIX = "[[["
_CLOSE_PREFIX = "[[[/"
_MARKER_SUFFIX = "]]]"


def _opener(tool_name: str) -> str:
    """The exact opening marker for one tool."""
    return f"{_OPEN_PREFIX}{tool_name}{_MARKER_SUFFIX}"


def _closer(tool_name: str) -> str:
    """The exact closing marker for one tool."""
    return f"{_CLOSE_PREFIX}{tool_name}{_MARKER_SUFFIX}"


_MARKER_VOCABULARY = ", ".join(_opener(name) for name in _TOOL_NAMES)

#: One complete block: an exact opening marker, its payload, and its exact
#: matching closing marker. Non-greedy so the first matching closer ends the
#: block, DOTALL so a payload spans lines, and back-referenced so an opener for
#: one tool can never be closed by another tool's marker. Position is not part
#: of it: a marker is a substring, so indentation, prose on the same line, a
#: fence, or a list marker around it are presentation, not protocol.
_BLOCK_RE = re.compile(
    r"\[\[\[(" + "|".join(re.escape(name) for name in _TOOL_NAMES) + r")\]\]\]"
    r"(.*?)"
    r"\[\[\[/\1\]\]\]",
    re.DOTALL,
)
#: Any token claiming the reserved JTECH namespace, whether or not it names a
#: real tool. The underscore group repeats so a multi-word name claims the
#: namespace too: ``[[[jtech_bad_tool]]]`` is as much a tool the model reached
#: for as ``[[[jtech_bad]]]``, and reading one as ordinary text inside a
#: payload is the concealment this token exists to prevent.
_MARKER_RE = re.compile(r"\[\[\[/?jtech(?:_[A-Za-z0-9]+)*\]\]\]")
#: The characters normalization removes from a payload's edges: the leading run
#: for every tool, and the trailing run only for a command, whose payload has
#: no field of its own to normalize it. Everything between those edges —
#: indentation, blank lines, line endings — is the command, task, or report
#: exactly as the model wrote it.
_PAYLOAD_PADDING = " \t\r\n"

#: The ordered header lines each structured block carries, and what its payload
#: is called in a diagnostic. A command block has neither: its body is the
#: command, whole and unread.
_BLOCK_HEADERS: dict[ToolName, tuple[str, ...]] = {
    _JTECH_AGENT: ("agent_key", "agent_label", "profile_name", "task_label"),
    _JTECH_RESULT: ("status",),
}
_PAYLOAD_NAMES: dict[ToolName, str] = {_JTECH_AGENT: "task", _JTECH_RESULT: "report"}


@dataclass(frozen=True, slots=True)
class _ProtocolBlock:
    """One complete block, built from one regex match.

    Args:
        tool_name: The tool both markers name.
        body: The payload with its framing whitespace removed: the leading
            run always, and the trailing run only for a command.
        line: One-based line of the opening marker in the reply.
        body_line: One-based line of ``body``'s first character in the reply.
            Tracked separately because a compact block puts the payload's first
            line on the opening marker's own line, so a header diagnostic
            cannot be counted from the marker.
    """

    tool_name: ToolName
    body: str
    line: int
    body_line: int


class _BlockSyntaxError(ValueError):
    """One protocol block the parser refuses, with where and why."""

    def __init__(
        self, tool_name: ProtocolErrorSource, line: int, message: str
    ) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.line = line


def _line_at(reply: str, offset: int) -> int:
    """The one-based line ``offset`` falls on in ``reply``."""
    return reply.count("\n", 0, offset) + 1


def _shown(line: str) -> str:
    """One offending line, quoted for a diagnostic and capped in length."""
    return repr(line if len(line) <= 80 else line[:79] + "…")


def _block_from_match(reply: str, match: re.Match[str]) -> _ProtocolBlock:
    """Convert one complete match to the block value the converters read.

    Leading padding is envelope for every tool. It is what a model writes to
    put a readable body on its own line, or the space it leaves after the
    opening marker, and removing it is what makes the compact, spaced, and
    multiline spellings of one block the same block.

    Trailing padding is envelope only for a command, whose payload is the whole
    body and has nothing else to normalize it. A structured body ends in its
    task or report, and that field is normalized and validated by its own
    boundary type. Stripping the body's tail here would take the empty
    separator line with it, so a whitespace-only report would be reported as a
    block of the wrong shape — naming a defect the model did not commit, and
    hiding the one it did.
    """
    payload = match.group(2)
    tool_name: ToolName = match.group(1)
    lead = len(payload) - len(payload.lstrip(_PAYLOAD_PADDING))
    body = payload[lead:]
    if tool_name == _JTECH_CMD:
        body = body.rstrip(_PAYLOAD_PADDING)
    return _ProtocolBlock(
        tool_name=tool_name,
        body=body,
        line=_line_at(reply, match.start()),
        body_line=_line_at(reply, match.start(2) + lead),
    )


def _opens_its_line(reply: str, start: int) -> bool:
    """Whether only whitespace precedes the marker at ``start`` on its line.

    The one position that carries meaning, and it is not a layout rule: a model
    that begins a line with an opening marker is starting a block, so a stream
    cut off before the closer leaves exactly this shape. Prose naming a marker
    reaches it through a sentence, and the words before it are what tell the
    two apart.
    """
    return not reply[reply.rfind("\n", 0, start) + 1 : start].strip()


def _residual_marker_message(marker: str, *, with_block: bool) -> str:
    """Why one marker left outside every complete block is not a block."""
    tool_name = marker.strip("[]/")
    if tool_name not in _TOOL_NAMES:
        return (
            f"{marker} names no JTECH tool, so nothing from this response ran. "
            f"The tools are {_MARKER_VOCABULARY}, each written as an exact "
            "opening marker, the payload, then its exact matching closing "
            "marker."
        )
    if with_block:
        return (
            f"{marker} belongs to no complete block, and this response carries "
            "a block, so none of it ran. A marker left over is usually a "
            "payload that was cut short at an earlier closing marker: check "
            "that each block ends where you meant it to, and remove any marker "
            "that was not opening or closing one."
        )
    return (
        f"{marker} was never closed by {_closer(tool_name)}, so nothing from "
        "this response ran. Write the payload between the two markers; they "
        "may sit anywhere in the response, including together on one line."
    )


def _residual_marker_errors(
    reply: str,
    matched_spans: Sequence[tuple[int, int]],
) -> list[ToolProtocolError]:
    """Report markers no complete block consumed, where one is a tool attempt.

    A marker on its own is not enough to refuse a response: prose is how the
    protocol gets discussed at all, and a sentence naming a marker must not
    cost a corrective round. Two residues are not prose, though.

    The first is a marker left over in a response that also carries a complete
    block. That is what a truncated payload looks like — the block ended at an
    earlier closing marker and the rest of the intended command became
    commentary — so the leftover marker is the only evidence the command that
    survived is a fragment. Refusing the whole response keeps it out of the
    shell.

    The second is an opening marker that begins its own line with no closer
    anywhere after it, which is what a stream cut off mid-block leaves. Read as
    prose it would end a primary turn as the final answer, with the work
    unstarted and nothing said about it.
    """
    errors: list[ToolProtocolError] = []
    with_block = bool(matched_spans)
    for match in _MARKER_RE.finditer(reply):
        start = match.start()
        if any(span[0] <= start < span[1] for span in matched_spans):
            continue
        marker = match.group()
        truncated_opener = not marker.startswith(_CLOSE_PREFIX) and _opens_its_line(
            reply, start
        )
        if not with_block and not truncated_opener:
            continue
        errors.append(
            ToolProtocolError(
                _JTECH_PROTOCOL,
                _line_at(reply, start),
                _residual_marker_message(marker, with_block=with_block),
            )
        )
    return errors


def _nested_marker_errors(
    reply: str,
    matches: Sequence[re.Match[str]],
) -> list[ToolProtocolError]:
    """Reject marker tokens captured inside another block's payload.

    The one marker that is not ordinary text, because a lazy match hides what
    it swallowed. Outside a block an unpaired marker is prose the model can see
    for itself and nothing acts on it; inside one it would be handed to the
    shell as payload, so a nested tool attempt would run as command text rather
    than be refused. The block's own closing marker is the only reserved
    substring in a payload; any other marker there is a tool the model tried to
    call from within one.
    """
    errors: list[ToolProtocolError] = []
    for match in matches:
        opener = _opener(match.group(1))
        for nested in _MARKER_RE.finditer(match.group(2)):
            errors.append(
                ToolProtocolError(
                    _JTECH_PROTOCOL,
                    _line_at(reply, match.start(2) + nested.start()),
                    f"{nested.group()} appears inside a {opener} block's "
                    "payload, so nothing from this response ran. "
                    "Protocol blocks cannot nest, and the only reserved "
                    f"substring inside a payload is {_closer(match.group(1))}.",
                )
            )
    return errors


def _split_ordered_headers(
    block: _ProtocolBlock, expected: tuple[str, ...]
) -> tuple[dict[str, str], str]:
    """Parse exact ordered headers, one blank separator, and raw payload.

    Position is the whole rule, so unknown, duplicated, missing, reordered, and
    multiline headers are all the same failure: the line at this position does
    not open with the header this position requires. One rule has one edge; a
    lookup keyed by name would accept any order and then need four more rules
    to reject the shapes it just admitted.

    Diagnostics count from ``body_line`` rather than from the opening marker,
    because a compact block puts the first header on the marker's own line and
    a multiline one puts it on the next.

    Raises:
        _BlockSyntaxError: on any deviation from that shape.
    """
    payload_name = _PAYLOAD_NAMES[block.tool_name]
    opener = _opener(block.tool_name)
    # Split on the line feed alone: a carriage return belongs to the line
    # ending of a header, and to the payload everywhere else.
    lines = block.body.split("\n")
    if len(lines) < len(expected) + 1:
        raise _BlockSyntaxError(
            block.tool_name,
            block.line,
            f"a {opener} block carries "
            + ", ".join(f"{name}:" for name in expected)
            + f", then one empty line, then the {payload_name}",
        )
    values: dict[str, str] = {}
    for index, name in enumerate(expected):
        line = lines[index].removesuffix("\r")
        key, separator, value = line.partition(":")
        if not separator or key != name:
            raise _BlockSyntaxError(
                block.tool_name,
                block.body_line + index,
                f"line {index + 1} of a {opener} block must be the header "
                f"'{name}:', not {_shown(line)}",
            )
        # Only the spaces or tabs framing the value come off. Everything after
        # them, colons included, is the value the boundary type validates.
        values[name] = value.lstrip(" \t")
    separator_line = lines[len(expected)].removesuffix("\r")
    if separator_line:
        raise _BlockSyntaxError(
            block.tool_name,
            block.body_line + len(expected),
            f"exactly one empty line must separate a {opener} block's headers "
            f"from its {payload_name}, not {_shown(separator_line)}",
        )
    return values, "\n".join(lines[len(expected) + 1 :])


def _dispatch_from_block(block: _ProtocolBlock) -> AgentDispatch:
    """Convert one syntactically valid agent block to its boundary value.

    Raises:
        _BlockSyntaxError: if the header shape is wrong.
        ValueError: if a field fails :class:`AgentDispatch` validation.
    """
    headers, task = _split_ordered_headers(block, _BLOCK_HEADERS[_JTECH_AGENT])
    return AgentDispatch(
        agent_key=headers["agent_key"],
        agent_label=headers["agent_label"],
        profile_name=headers["profile_name"],
        task_label=headers["task_label"],
        task=task,
    )


def _result_from_block(block: _ProtocolBlock) -> AgentResult:
    """Convert one syntactically valid result block to its boundary value.

    Raises:
        _BlockSyntaxError: if the header shape is wrong.
        ValueError: if the status or report fails :class:`AgentResult`
            validation.
    """
    headers, report = _split_ordered_headers(block, _BLOCK_HEADERS[_JTECH_RESULT])
    return AgentResult(status=headers["status"], content=report)


def _block_error(block: _ProtocolBlock, error: ValueError) -> ToolProtocolError:
    """Locate one block's failure: at the offending line, or at its opener.

    A header failure knows exactly which line it is about; a boundary-type
    failure knows only that the block's values were wrong, so it lands on the
    opening marker that names the block.
    """
    if isinstance(error, _BlockSyntaxError):
        return ToolProtocolError(error.tool_name, error.line, str(error))
    return ToolProtocolError(block.tool_name, block.line, str(error))


def parse_jtech_reply(reply: str) -> ParsedReply:
    """Parse JTECH protocol blocks wherever their markers appear in a reply.

    A block is an exact opening marker, a payload, and the exact matching
    closing marker. Nothing about where the markers sit is protocol: they may
    share a line with each other and with prose, or frame a payload across as
    many lines as it needs, and indentation, list markers, quote markers, code
    fences, code spans, and HTML around them are presentation the parser never
    reads. Framing depends only on the two uncommon tokens, so how a model
    lays out Markdown cannot decide whether its tool call runs.

    The payload's leading spaces, tabs, and line endings are removed, and a
    command's trailing ones too — a structured body's tail belongs to the task
    or report its own boundary type normalizes. What is left is raw: never
    trimmed further, unquoted, unescaped, dedented, or evaluated, so a command
    holding quotes, triple quotes, backslashes, or a heredoc reaches the shell
    exactly as written. Commentary before, between, and after blocks is
    preserved by masking their spans.

    A ``[[[jtech_result]]]`` block is matched by the same regex but executes
    nothing: it lands in ``result`` as the subagent's terminal status, and only
    one may appear in a reply, because two would leave the status ambiguous.

    A complete pair is the whole of what makes a block, so a marker that is not
    part of one is ordinary text by default: a sentence naming a marker costs
    no corrective round, and neither does a stray closer in prose.

    Three residues are not prose, and each becomes a
    :class:`ToolProtocolError` that keeps the whole response out of the
    runtime. A marker left over beside a complete block is what a payload cut
    short at an earlier closer leaves behind, so the block that survived is a
    fragment. An opening marker that begins its own line and is never closed is
    what a truncated stream leaves, and read as prose it would end a primary
    turn as the final answer with the work unstarted. A marker captured
    *inside* another block's payload is a nested tool call a lazy match would
    otherwise hand to the shell as payload text.
    """
    matches = list(_BLOCK_RE.finditer(reply))
    spans = [match.span() for match in matches]
    errors = _residual_marker_errors(reply, spans)
    errors.extend(_nested_marker_errors(reply, matches))

    commands: list[str] = []
    dispatches: list[AgentDispatch] = []
    result: AgentResult | None = None
    # Counted rather than inferred from ``result``: the second block clears it,
    # and a third must not be allowed to fill it back in.
    result_blocks = 0

    for match in matches:
        block = _block_from_match(reply, match)
        if block.tool_name == _JTECH_CMD:
            # An empty or whitespace-only payload stays parseable and reaches
            # the runtime's existing empty-command path; the parser owns
            # syntax, not policy.
            commands.append(block.body)
            continue
        if block.tool_name == _JTECH_RESULT:
            result_blocks += 1
            if result_blocks > 1:
                # Two terminal statuses cannot both be the answer, and picking
                # one would invent an intent the model never expressed, so the
                # reply carries none.
                result = None
                errors.append(
                    ToolProtocolError(
                        _JTECH_RESULT,
                        block.line,
                        f"{_opener(_JTECH_RESULT)} may appear only once in a "
                        "response, so no terminal result was recorded",
                    )
                )
                continue
            try:
                result = _result_from_block(block)
            except ValueError as error:
                errors.append(_block_error(block, error))
            continue
        try:
            dispatches.append(_dispatch_from_block(block))
        except ValueError as error:
            errors.append(_block_error(block, error))

    # Nested-marker errors and block errors are each already in source order,
    # but they arrive in two passes; merging by line keeps one reply's
    # diagnostics readable top to bottom.
    errors.sort(key=lambda error: error.line)

    masked = list(reply)
    for start, end in spans:
        for index in range(start, end):
            if masked[index] != "\n":
                masked[index] = " "
    return ParsedReply(
        commands=commands,
        commentary="".join(masked).strip(),
        dispatches=dispatches,
        errors=errors,
        result=result,
    )


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
