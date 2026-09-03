"""Guarded tool protocol, shell command policy, and executor.

The AI requests work with ``[[[jtech_cmd]]]`` and ``[[[jtech_agent]]]`` blocks,
and a dispatched subagent ends its turn with a ``[[[jtech_result]]]`` block.
That third block executes nothing: it is terminal, carrying the status and the
report the coordinator receives. One scanner owns all three, because they
differ only in name and body shape, so fence handling, HTML-wrapper handling,
delimiter detection, span masking, and diagnostics are written once.

Protocol structure is expressed only by exact, standalone delimiter lines, and
the payload between them is never quoted, unescaped, or evaluated. That is the
whole point of the format: a command containing quotes, triple quotes,
backslashes, or heredocs cannot damage the envelope carrying it, because the
envelope does not share a lexer with its contents.

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
#: Who a diagnostic is about. A recognized tool keeps its own name; a line that
#: claims the reserved ``[[[jtech_...]]]`` namespace without spelling any
#: delimiter belongs to no tool, and inventing one for it would name a tool the
#: model never asked for.
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
            line that claims the reserved namespace without naming a tool.
        line: One-based line of the offending line in the model's own reply.
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
_JTECH_RESULT = "jtech_result"
_JTECH_PROTOCOL = "jtech_protocol"
#: The complete tool vocabulary. Membership here is what makes a delimiter
#: executable at all, so the terminal result block gets the same near-miss
#: diagnostics as the executable ones.
_TOOL_NAMES: tuple[ToolName, ...] = (_JTECH_CMD, _JTECH_AGENT, _JTECH_RESULT)

_OPEN_PREFIX = "[[["
_CLOSE_PREFIX = "[[[/"
_DELIMITER_SUFFIX = "]]]"
#: Every executable delimiter line, mapped to the tool it frames. Exact strings
#: rather than a pattern: the whole line is the token, so there is nothing to
#: match loosely and no way for a payload's own punctuation to be read as one.
_OPEN_DELIMITERS: dict[str, ToolName] = {
    f"{_OPEN_PREFIX}{name}{_DELIMITER_SUFFIX}": name for name in _TOOL_NAMES
}
_CLOSE_DELIMITERS: dict[str, ToolName] = {
    f"{_CLOSE_PREFIX}{name}{_DELIMITER_SUFFIX}": name for name in _TOOL_NAMES
}
#: The namespace a delimiter line claims. A column-zero line that opens with one
#: of these without spelling a delimiter exactly is a malformed protocol line,
#: never prose: the model was reaching for a tool, and demoting the attempt to
#: final prose is the silence every diagnostic here exists to prevent.
_RESERVED_PREFIXES = (f"{_OPEN_PREFIX}jtech_", f"{_CLOSE_PREFIX}jtech_")
_DELIMITER_VOCABULARY = ", ".join(_OPEN_DELIMITERS)

#: The ordered header lines each structured block carries, and what its payload
#: is called in a diagnostic. A command block has neither: its body is the
#: command, whole and unread.
_BLOCK_HEADERS: dict[ToolName, tuple[str, ...]] = {
    _JTECH_AGENT: ("agent_key", "agent_label", "profile_name", "task_label"),
    _JTECH_RESULT: ("status",),
}
_PAYLOAD_NAMES: dict[ToolName, str] = {_JTECH_AGENT: "task", _JTECH_RESULT: "report"}

DelimiterKind = Literal["open", "close"]


@dataclass(frozen=True, slots=True)
class _ProtocolDelimiter:
    """One exact delimiter line: which side of a block it is, and whose."""

    kind: DelimiterKind
    tool_name: ToolName


@dataclass(frozen=True, slots=True)
class _ProtocolBlock:
    """One complete block and the span it occupies in the reply.

    Args:
        tool_name: The tool the delimiters name.
        body: The raw payload between the framing newlines, byte for byte.
        start: Offset of the opening delimiter line's first character.
        end: Offset just past the closing delimiter line's last character.
            The span covers the delimiters too, so masking it leaves the
            commentary around the block lossless.
        line: One-based line of the opening delimiter in the reply.
    """

    tool_name: ToolName
    body: str
    start: int
    end: int
    line: int


class _BlockSyntaxError(ValueError):
    """One protocol block the parser refuses, with where and why."""

    def __init__(
        self, tool_name: ProtocolErrorSource, line: int, message: str
    ) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.line = line


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
#: Why a recognized block did not run, as the model is told it.
_FENCE_CONTEXT = "was written inside a code fence"
_HTML_CODE_CONTEXT = "was written inside an HTML code block"
_DECORATION_CONTEXT = "was indented or wrapped in Markdown formatting"


def _exact_delimiter(line: str) -> _ProtocolDelimiter | None:
    """Return an exact column-zero delimiter, otherwise ``None``.

    A lone trailing ``\\r`` is a line ending, not content, so a CRLF reply
    spells its delimiters the same way an LF one does. Nothing else is
    forgiven: any other character on the line makes it payload or prose.
    """
    line = line.removesuffix("\r")
    tool_name = _OPEN_DELIMITERS.get(line)
    if tool_name is not None:
        return _ProtocolDelimiter("open", tool_name)
    tool_name = _CLOSE_DELIMITERS.get(line)
    if tool_name is not None:
        return _ProtocolDelimiter("close", tool_name)
    return None


def _reserved_delimiter_candidate(line: str) -> bool:
    """Whether a column-zero line claims the reserved JTECH namespace.

    Only asked of a line that is not already an exact delimiter, so a true
    answer means the model spelled one wrong — a misspelled tool, an inline
    body, or text appended to the marker — rather than writing prose.
    """
    return line.startswith(_RESERVED_PREFIXES)


def _wrapped_block_name(line: str) -> ToolName | None:
    """The tool this line opens a block for behind decoration, else ``None``.

    Recognition is deliberately shallow: it looks at one line and asks only
    whether an opening delimiter is sitting inside a wrapper. A wrapped block
    may be malformed or may never be closed, and every one of those is still
    the model asking for a tool it will not get. Requiring a complete block
    here returns ``None`` for exactly those shapes and leaves them silent,
    which is the failure this whole path exists to prevent.

    What discriminates is the *prefix*: HTML code tags, a task-list marker, and
    Markdown underscore escaping are removed, and then no letter may stand
    between the start of the line and the tool name. Punctuation is how every
    wrapper is spelled — bullets, quote markers, emphasis, strikethrough, code
    spans, table cells — so it is all decoration by construction, while a
    mention inside a sentence (``The marker is [[[jtech_cmd]]].``) keeps its
    letters and stays commentary.

    A closing delimiter is never reported: its own opener already carries the
    diagnostic, and a wrapped block would otherwise be refused twice.
    """
    text = _HTML_CODE_TAG.sub("", line.replace("\\_", "_"))
    text = _TASK_LIST_MARKER.sub("", text, count=1)
    for start, char in enumerate(text):
        if not char.isalpha():
            continue
        if start < len(_OPEN_PREFIX) or not text.startswith(
            _OPEN_PREFIX, start - len(_OPEN_PREFIX)
        ):
            # The name is not sitting in a marker at all, so this is prose that
            # happens to begin with a word — or a closing delimiter, whose
            # slash keeps it out of the marker prefix.
            return None
        for name in _TOOL_NAMES:
            if text.startswith(name, start) and text.startswith(
                _DELIMITER_SUFFIX, start + len(name)
            ):
                return name
        return None
    return None


def _near_miss_message(tool_name: ToolName, context: str) -> str:
    """What the model must change to turn a wrapped block into a running one."""
    return (
        f"{_OPEN_PREFIX}{tool_name}{_DELIMITER_SUFFIX} {context}, so it did not "
        "run. A block executes only as raw text: its opening and closing "
        "delimiter lines must each start at the very first column of their own "
        "line and carry nothing else, with no indent, fence, code span, list "
        "marker, quote marker, or emphasis around them. If you meant to show "
        "the syntax rather than run it, describe it in a sentence instead."
    )


def _near_miss_errors(
    text: str, skipped: Sequence[tuple[int, str]]
) -> list[ToolProtocolError]:
    """Diagnostics for blocks that were skipped because something wrapped them.

    Only for a reply that ran nothing: silence is the failure being prevented,
    and a reply that executed a block is never silent.
    """
    errors: list[ToolProtocolError] = []
    for position, context in skipped:
        line_end = text.find("\n", position)
        line = text[position:] if line_end < 0 else text[position:line_end]
        tool_name = _wrapped_block_name(line)
        if tool_name is None:
            continue
        errors.append(
            ToolProtocolError(
                tool_name,
                text.count("\n", 0, position) + 1,
                _near_miss_message(tool_name, context),
            )
        )
    return errors


def _shown(line: str) -> str:
    """One offending line, quoted for a diagnostic and capped in length.

    Quoted rather than stripped: the whitespace around a delimiter is usually
    the whole defect, and a trimmed echo of ``[[[jtech_cmd]]] `` reads as the
    valid delimiter the model thought it wrote.
    """
    return repr(line if len(line) <= 80 else line[:79] + "…")


def _read_block(
    reply: str, start: int, body_start: int, open_line: int, tool_name: ToolName
) -> tuple[_ProtocolBlock | None, _BlockSyntaxError | None, int, int]:
    """Read one opened block through its closing delimiter.

    The body is taken verbatim: only exact delimiter lines are looked for
    inside it, so fences, indentation, quotes, and Markdown in the payload mean
    nothing here. That is the invariant the whole format exists for.

    Returns the block (or ``None`` when it failed), the failure (or ``None``),
    and the position and one-based line the outer scan resumes at. A nested
    opener resumes *on* that opener, so the block the model actually started is
    still read; every other outcome resumes past the line that ended this one.
    """
    length = len(reply)
    position = body_start
    line_number = open_line + 1
    body_lines = 0
    while position < length:
        line_end = reply.find("\n", position)
        if line_end < 0:
            line_end = length
        after = line_end + (line_end < length)
        delimiter = _exact_delimiter(reply[position:line_end])
        if delimiter is None:
            body_lines += 1
            position = after
            line_number += 1
            continue
        opener = f"{_OPEN_PREFIX}{tool_name}{_DELIMITER_SUFFIX}"
        closer = f"{_CLOSE_PREFIX}{tool_name}{_DELIMITER_SUFFIX}"
        if delimiter.kind == "open":
            return (
                None,
                _BlockSyntaxError(
                    tool_name,
                    line_number,
                    f"{opener} must be closed by {closer} before another block "
                    "opens; protocol blocks cannot nest",
                ),
                position,
                line_number,
            )
        if delimiter.tool_name != tool_name:
            wrong = f"{_CLOSE_PREFIX}{delimiter.tool_name}{_DELIMITER_SUFFIX}"
            return (
                None,
                _BlockSyntaxError(
                    tool_name,
                    line_number,
                    f"{opener} must be closed by {closer}, not {wrong}",
                ),
                after,
                line_number + 1,
            )
        if body_lines == 0:
            return (
                None,
                _BlockSyntaxError(
                    tool_name,
                    open_line,
                    f"{opener} must be followed by a body line before {closer}; "
                    "an empty body is written as one empty line",
                ),
                after,
                line_number + 1,
            )
        # The newline introducing the closing line is framing, not payload, and
        # so is the carriage return that belongs to it. Nothing else about the
        # body's line endings is touched.
        body_end = position - 1
        if body_end > body_start and reply[body_end - 1] == "\r":
            body_end -= 1
        return (
            _ProtocolBlock(
                tool_name, reply[body_start:body_end], start, line_end, open_line
            ),
            None,
            after,
            line_number + 1,
        )
    return (
        None,
        _BlockSyntaxError(
            tool_name,
            open_line,
            f"{_OPEN_PREFIX}{tool_name}{_DELIMITER_SUFFIX} was never closed: the "
            f"block must end with a line containing exactly "
            f"{_CLOSE_PREFIX}{tool_name}{_DELIMITER_SUFFIX} and nothing else, at "
            "the first column",
        ),
        length,
        line_number,
    )


def _scan_protocol_blocks(
    reply: str,
) -> tuple[list[_ProtocolBlock], list[_BlockSyntaxError], list[tuple[int, str]]]:
    """Scan blocks, syntax errors, and skipped decorated candidates once.

    One pass owns the fence and HTML-code state, so the near-miss diagnostics
    read it rather than rebuilding it: a second scan would have to duplicate
    that state machine and could only disagree with this one.

    A block's body is consumed whole, so a fence, an HTML tag, or an indented
    line inside a payload never enters that state machine at all.
    """
    blocks: list[_ProtocolBlock] = []
    errors: list[_BlockSyntaxError] = []
    # Position and wrapper kind of every line the scan passes over. Recorded
    # unconditionally because the scan owns the state a second pass would have
    # to duplicate; nothing is examined unless the reply ran nothing.
    skipped: list[tuple[int, str]] = []
    length = len(reply)
    position = 0
    line_number = 1
    # The open fence's marker character and length, or None outside a fence.
    fence: tuple[str, int] | None = None
    in_html_code = False
    while position < length:
        line_end = reply.find("\n", position)
        if line_end < 0:
            line_end = length
        after = line_end + (line_end < length)
        line = reply[position:line_end]
        stripped = line.strip()
        lower = stripped.lower()

        if fence is not None:
            marker, size = fence
            # A closing fence is the same character, at least as long as the
            # opening one, and carries nothing else on its line.
            if len(stripped) >= size and set(stripped) == {marker}:
                fence = None
            skipped.append((position, _FENCE_CONTEXT))
        elif in_html_code:
            if "</code>" in lower:
                in_html_code = False
            skipped.append((position, _HTML_CODE_CONTEXT))
        elif (opening := _FENCE_OPEN.match(stripped)) is not None:
            fence = (stripped[0], len(opening.group()))
            skipped.append((position, _FENCE_CONTEXT))
        elif "<code" in lower:
            in_html_code = "</code>" not in lower
            skipped.append((position, _HTML_CODE_CONTEXT))
        else:
            delimiter = _exact_delimiter(line)
            if delimiter is None:
                if _reserved_delimiter_candidate(line):
                    errors.append(
                        _BlockSyntaxError(
                            _JTECH_PROTOCOL,
                            line_number,
                            f"{_shown(line)} is not a JTECH protocol delimiter. A "
                            f"delimiter line carries exactly one of "
                            f"{_DELIMITER_VOCABULARY} or its matching closing "
                            "form, alone on its own line at the first column",
                        )
                    )
                else:
                    skipped.append((position, _DECORATION_CONTEXT))
            elif delimiter.kind == "close":
                closer = f"{_CLOSE_PREFIX}{delimiter.tool_name}{_DELIMITER_SUFFIX}"
                errors.append(
                    _BlockSyntaxError(
                        delimiter.tool_name,
                        line_number,
                        f"{closer} closes a block that was never opened",
                    )
                )
            else:
                block, error, position, line_number = _read_block(
                    reply, position, after, line_number, delimiter.tool_name
                )
                if block is not None:
                    blocks.append(block)
                if error is not None:
                    errors.append(error)
                continue
        position = after
        line_number += 1
    return blocks, errors, skipped


def _split_ordered_headers(
    block: _ProtocolBlock, expected: tuple[str, ...]
) -> tuple[dict[str, str], str]:
    """Parse exact ordered headers, one blank separator, and raw payload.

    Position is the whole rule, so unknown, duplicated, missing, reordered, and
    multiline headers are all the same failure: the line at this position does
    not open with the header this position requires. One rule has one edge; a
    lookup keyed by name would accept any order and then need four more rules
    to reject the shapes it just admitted.

    Raises:
        _BlockSyntaxError: on any deviation from that shape.
    """
    payload_name = _PAYLOAD_NAMES[block.tool_name]
    opener = f"{_OPEN_PREFIX}{block.tool_name}{_DELIMITER_SUFFIX}"
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
                block.line + 1 + index,
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
            block.line + 1 + len(expected),
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
    delimiter that names the block.
    """
    if isinstance(error, _BlockSyntaxError):
        return ToolProtocolError(error.tool_name, error.line, str(error))
    return ToolProtocolError(block.tool_name, block.line, str(error))


def parse_jtech_reply(reply: str) -> ParsedReply:
    """Parse standalone JTECH protocol blocks anywhere in a reply.

    A block is an exact opening delimiter line, a body, and the matching exact
    closing delimiter line, each delimiter starting at column zero and carrying
    nothing else. The body between them is raw: it is never trimmed, unquoted,
    unescaped, or evaluated, so a command holding quotes, triple quotes,
    backslashes, or a heredoc reaches the shell exactly as written. This
    permits commentary before, between, and after blocks without executing
    examples. Code blocks are inert in every form Markdown gives them —
    backtick and tilde fences of any marker length, and indentation, which
    needs no fence at all — as are HTML ``<code>`` blocks.

    A ``[[[jtech_result]]]`` block is recognized by the same scan but executes
    nothing: it lands in ``result`` as the subagent's terminal status, and only
    one may appear in a reply, because two would leave the status ambiguous.

    A line in the reserved ``[[[jtech_...]]]`` namespace that spells no
    delimiter, a block that never closes, a nested or mismatched delimiter, and
    a stray closer all become a :class:`ToolProtocolError` rather than silently
    reverting to commentary — the model asked for a tool and is told exactly
    why it did not run.

    A reply that runs nothing gets the same treatment for a block some wrapper
    hid: an indented, fenced, bulleted, quoted, or emphasized opening delimiter
    is still refused, but it is reported rather than dropped, because a reply
    with no block and no diagnostic reads as a final answer and ends the turn
    in silence.
    """
    blocks, syntax_errors, skipped = _scan_protocol_blocks(reply)
    errors: list[ToolProtocolError] = [
        ToolProtocolError(error.tool_name, error.line, str(error))
        for error in syntax_errors
    ]
    if not blocks:
        # Nothing parsed, so nothing executed and nothing keeps the turn alive,
        # and the whole reply is commentary because no span was consumed.
        #
        # Both kinds of diagnostic are reported, never one instead of the other.
        # A block spans lines, so a single wrapped construct can raise a syntax
        # error on one delimiter and a near miss on its decorated partner: an
        # indented opener with a bare closer reports only that the closer opened
        # nothing, which names the wrong line and hides the indent that actually
        # caused it. The two cannot collide, because a line that raised a syntax
        # error was never skipped and a skipped line never raised one.
        errors.extend(_near_miss_errors(reply, skipped))
        errors.sort(key=lambda error: error.line)
        return ParsedReply(commands=[], commentary=reply.strip(), errors=errors)

    commands: list[str] = []
    dispatches: list[AgentDispatch] = []
    result: AgentResult | None = None
    # Counted rather than inferred from ``result``: the second block clears it,
    # and a third must not be allowed to fill it back in.
    result_blocks = 0

    for block in blocks:
        if block.tool_name == _JTECH_CMD:
            # Whitespace-only bodies stay parseable and reach the runtime's
            # existing empty-command path; the parser owns syntax, not policy.
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
                        f"{_OPEN_PREFIX}{_JTECH_RESULT}{_DELIMITER_SUFFIX} may "
                        "appear only once in a response, so no terminal result "
                        "was recorded",
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

    # Scan errors and block errors are each already in source order, but they
    # arrive in two passes; merging by line keeps one reply's diagnostics
    # readable top to bottom.
    errors.sort(key=lambda error: error.line)

    masked = list(reply)
    for block in blocks:
        for index in range(block.start, block.end):
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
