"""Guarded shell command execution: parse, policy, and executor.

The AI requests commands with standalone ``jtech_cmd(...)`` calls. Each call is
decided in order: absolute blacklist (immutable, all modes) -> mode (off/yolo)
-> allowlist -> prompt. Pure logic lives here so it is unit-testable without
the TUI; the TUI owns prompting, rendering, and re-stream.
"""

from __future__ import annotations

import ast
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import bashlex
from bashlex.errors import ParsingError

VALID_CMD_MODES = ("ask", "auto", "yolo", "off")
DEFAULT_TIMEOUT = 60
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
    timeout: int = DEFAULT_TIMEOUT
    max_output: int = DEFAULT_MAX_OUTPUT


@dataclass
class ParsedReply:
    """The executable command region and optional surrounding commentary."""

    commands: list[str]
    commentary: str = ""


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


def _unwrap_html_code(text: str) -> str:
    """Remove one whole-response HTML code wrapper, if present."""
    opening = "<code>"
    closing = "</code>"
    if text.startswith(opening) and text.endswith(closing):
        inner = text[len(opening) : -len(closing)].strip()
        if (
            inner
            and inner.startswith(_JTECH_CMD)
            and "<code>" not in inner
            and "</code>" not in inner
        ):
            return inner
    return text


def _parse_command_at(text: str, start: int) -> tuple[str, int] | None:
    """Parse one command call at ``start`` and return its end position."""
    position = start
    if not text.startswith(_JTECH_CMD, position):
        return None
    position += len(_JTECH_CMD)
    if position < len(text) and not (text[position].isspace() or text[position] == "("):
        return None
    while position < len(text) and text[position].isspace():
        position += 1
    if position >= len(text) or text[position] != "(":
        return None
    position += 1
    while position < len(text) and text[position].isspace():
        position += 1
    if position >= len(text) or text[position] not in "'\"":
        return None

    literal_start = position
    literal_end = _quoted_literal_end(text, literal_start)
    if literal_end is None:
        return None
    try:
        command = ast.literal_eval(text[literal_start:literal_end])
    except (SyntaxError, ValueError):
        return None
    if not isinstance(command, str):
        return None
    position = literal_end
    while position < len(text) and text[position].isspace():
        position += 1
    if position >= len(text) or text[position] != ")":
        return None
    return command, position + 1


def _parse_command_line(
    text: str, start: int
) -> tuple[list[str], list[tuple[int, int]], int] | None:
    """Parse standalone calls beginning at one line and return its next line."""
    line_end = text.find("\n", start)
    if line_end < 0:
        line_end = len(text)
    line = text[start:line_end]
    position = start + len(line) - len(line.lstrip(" \t"))
    if not text.startswith(_JTECH_CMD, position):
        return None

    commands: list[str] = []
    spans: list[tuple[int, int]] = []
    while True:
        parsed = _parse_command_at(text, position)
        if parsed is None:
            return None
        command, command_end = parsed
        tail_end = text.find("\n", command_end)
        if tail_end < 0:
            tail_end = len(text)
        tail_start = command_end
        while tail_start < tail_end and text[tail_start] in " \t":
            tail_start += 1
        commands.append(command)
        spans.append((position, command_end))
        if tail_start == tail_end:
            return commands, spans, tail_end + (tail_end < len(text))
        if not text.startswith(_JTECH_CMD, tail_start):
            return None
        position = tail_start


def parse_jtech_reply(reply: str) -> ParsedReply:
    """Parse standalone command-call lines anywhere in a reply.

    A call must begin a line (apart from indentation) and occupy the command
    line, although multiple calls may share one line. This permits commentary
    before, between, and after commands without executing inline examples.
    Markdown fences and HTML ``<code>`` blocks are inert; a whole-response
    ``<code>`` wrapper remains supported by the compatibility unwrapping above.
    ``ast.literal_eval`` parses string arguments without evaluating Python code.
    """
    text = _unwrap_html_code(reply.strip())
    if not text:
        return ParsedReply([])

    commands: list[str] = []
    spans: list[tuple[int, int]] = []
    position = 0
    in_fence = False
    in_html_code = False
    while position < len(text):
        line_end = text.find("\n", position)
        if line_end < 0:
            line_end = len(text)
        line = text[position:line_end]
        stripped = line.strip()
        lower = stripped.lower()

        if in_fence:
            if stripped.startswith("```"):
                in_fence = False
            position = line_end + (line_end < len(text))
            continue
        if in_html_code:
            if "</code>" in lower:
                in_html_code = False
            position = line_end + (line_end < len(text))
            continue
        if stripped.startswith("```"):
            in_fence = True
            position = line_end + (line_end < len(text))
            continue
        if "<code" in lower:
            in_html_code = "</code>" not in lower
            position = line_end + (line_end < len(text))
            continue

        parsed_line = _parse_command_line(text, position)
        if parsed_line is None:
            position = line_end + (line_end < len(text))
            continue
        line_commands, line_spans, next_position = parsed_line
        commands.extend(line_commands)
        spans.extend(line_spans)
        position = next_position

    if not commands:
        return ParsedReply([], text)

    masked = list(text)
    for start, end in spans:
        for index in range(start, end):
            if masked[index] != "\n":
                masked[index] = " "
    commentary = "".join(masked).strip()
    return ParsedReply(commands, commentary)


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
    timed_out: bool = False
    interrupted: bool = False
    truncated: bool = False


def timeout_partial_output(exc: subprocess.TimeoutExpired) -> str:
    """Partial stdout captured before a TimeoutExpired, decoded to text.

    ``exc.output`` is bytes on POSIX even with ``text=True`` (subprocess joins
    the partial chunks with ``b''``), so decode defensively.
    """
    out = exc.output
    if out is None:
        return ""
    if isinstance(out, bytes):
        out = out.decode(errors="replace")
    return out


def truncate_output(text: str, max_output: int) -> tuple[str, bool]:
    """Cap output at ``max_output`` chars, keeping head and tail."""
    text = text.strip("\n")
    if len(text) <= max_output:
        return text, False
    half = max_output // 2
    dropped = len(text) - max_output
    return text[:half] + f"\n…[{dropped} chars truncated]…\n" + text[-half:], True


def execute(
    command: str,
    root: Path,
    timeout: int = DEFAULT_TIMEOUT,
    max_output: int = DEFAULT_MAX_OUTPUT,
) -> ExecResult:
    """Run one command with ``bash -c`` in ``root``, capturing combined output."""
    try:
        proc = subprocess.run(
            ["bash", "-c", command],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        # Keep partial output like the TUI's _exec_command: it tells the model
        # how far the command got before being killed.
        out, truncated = truncate_output(timeout_partial_output(e), max_output)
        return ExecResult(124, out, timed_out=True, truncated=truncated)
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
    if result.timed_out:
        body = f"\n{result.output}" if result.output else ""
        return f"$ {command}\n→ timed out (killed){body}"
    if result.interrupted:
        body = f"\n{result.output}" if result.output else ""
        return f"$ {command}\n→ interrupted by user{body}"
    body = result.output if result.output else "(no output)"
    return f"$ {command}\nexit {result.exit_code}\n{body}"
