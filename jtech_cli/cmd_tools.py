"""Guarded shell command execution: parse, policy, and executor.

The AI requests commands by emitting fenced blocks with language ``cmd`` in its
reply. Each block is decided in order: absolute blacklist (immutable, all modes)
-> mode (off/yolo) -> allowlist -> prompt. Pure logic lives here so it is
unit-testable without the TUI; the TUI owns prompting, rendering, and re-stream.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

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


# ---------------------------------------------------------------- parsing

def extract_cmd_blocks(reply: str) -> list[str]:
    """Return the bodies of all `````cmd```` fenced blocks in a reply, in order.

    Only the fence language ``cmd`` is honored; an unclosed fence is ignored.
    """
    blocks: list[str] = []
    current: list[str] | None = None
    for line in reply.splitlines():
        stripped = line.strip()
        if current is None:
            if stripped.startswith("```") and stripped[3:].strip().lower() == "cmd":
                current = []
        elif stripped.startswith("```"):
            blocks.append("\n".join(current))
            current = None
        else:
            current.append(line)
    return blocks


# Shell operators that separate one command from the next inside a chain.
_OPS_RE = re.compile(r"&&|\|\||[|;]|\$\(|`")
_ENV_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=\S*")


def split_segments(command: str) -> list[str]:
    """Split a command into segments on ``&&``, ``||``, ``|``, ``;``, ``$(``, backticks.

    A lone ``&`` (background) also separates commands, so it is normalized to
    ``;`` first — but only when it stands alone: ``2>&1``, ``&>`` and ``<&``
    are fd-redirects, not command separators. Chaining means *every* segment
    must be vetted, not just the first one.
    """
    cmd = re.sub(r"(?<![<>&])&(?![<>&])", ";", command)
    return [s.strip() for s in _OPS_RE.split(cmd) if s.strip()]


def _tokens(segment: str) -> list[str]:
    """Tokens of a segment with leading env assignments (``FOO=bar``) skipped."""
    tokens = segment.split()
    i = 0
    while i < len(tokens) and _ENV_RE.fullmatch(tokens[i]):
        i += 1
    return tokens[i:]


def program_names(command: str) -> list[str]:
    """The leading program of each segment (basename, env assignments skipped)."""
    names = []
    for seg in split_segments(command):
        tokens = _tokens(seg)
        if tokens:
            names.append(tokens[0].rsplit("/", 1)[-1])
    return names


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

# rm targets that are absolute no-go (exact match).
RM_ROOT_TARGETS = {"/", "/*", "~", "$HOME", "/etc", "/usr", "/var", "/boot", "/home", "/System"}
# rm targets whose subpaths are still system files (prefix match).
RM_PREFIX_TARGETS = ("/etc/", "/usr/", "/var/", "/boot/", "/System")

_PIPE_TO_SHELL_RE = re.compile(r"\|\s*(ba|z|da|k)?sh\b|\|\s*python3?\b|\|\s*perl\b|\|\s*ruby\b")
_CREDENTIAL_RE = re.compile(
    r"(~|\$HOME|/home/[^/\s]+|/root)/\.ssh|\.aws/credentials|\.netrc|id_rsa(?!\.pub)|id_ed25519|\.pgpass"
)


def _dangerous_rm_target(token: str) -> bool:
    t = token.strip(",'\"")
    return t in RM_ROOT_TARGETS or t.startswith(RM_PREFIX_TARGETS)


def _find_exec_commands(command: str) -> list[str]:
    """Commands find would execute via -exec/-execdir/-ok, in order.

    Each is the tokens between the flag and its ``\\;``/``+`` terminator,
    with the ``{}`` placeholder dropped. An unterminated -exec is ignored.
    """
    commands: list[str] = []
    current: list[str] | None = None
    for tok in command.split():
        if current is None:
            if tok in ("-exec", "-execdir", "-ok"):
                current = []
        elif tok in ("\\;", ";", "+"):
            commands.append(" ".join(t for t in current if t != "{}"))
            current = None
        elif tok != "{}":
            current.append(tok)
    return commands


def check_blacklist(command: str) -> str | None:
    """Return a reason if the command touches the absolute blacklist, else None.

    Checked against every segment of the chain plus whole-line patterns, so
    ``git status && rm -rf /`` and ``curl x | sh`` are caught, not just their
    first word. Commands embedded in ``find -exec`` are vetted too —
    ``find . -exec rm / \\;`` is an rm of ``/``, whatever wraps it.
    """
    for prog in program_names(command):
        # mkfs is a family (mkfs, mkfs.ext4, mkfs.xfs, ...): prefix-match it.
        if prog in FORBIDDEN_PROGRAMS or prog.startswith("mkfs"):
            return f"'{prog}' is on the absolute blacklist"

    for seg in split_segments(command):
        tokens = _tokens(seg)
        if tokens and tokens[0].rsplit("/", 1)[-1] == "rm":
            for arg in tokens[1:]:
                if not arg.startswith("-") and _dangerous_rm_target(arg):
                    return f"rm targeting '{arg.strip(chr(39))}' is on the absolute blacklist"

    for inner in _find_exec_commands(command):
        reason = check_blacklist(inner)
        if reason is not None:
            return f"{reason} (inside find -exec)"

    if re.search(r"\bdd\b[^|;&]*\bof=/dev/", command):
        return "dd writing to a raw device is on the absolute blacklist"
    if _PIPE_TO_SHELL_RE.search(command):
        return "piping data into a shell is on the absolute blacklist"
    if re.search(r"\binit\s+[06]\b", command):
        return "power-off via init is on the absolute blacklist"
    if _CREDENTIAL_RE.search(command):
        return "accessing credential/key material is on the absolute blacklist"
    return None


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


def _segment_matches(segment: str, allow: list[str]) -> bool:
    """True when one segment matches some allow rule.

    Rules: ``name`` matches the bare program with no args; ``name:*`` matches
    the program with any args; ``name sub:*`` matches the program with first
    arg ``sub`` and any further args.
    """
    tokens = _tokens(segment)
    if not tokens:
        return False
    prog = tokens[0].rsplit("/", 1)[-1]
    for rule in allow:
        rule = rule.strip()
        if not rule:
            continue
        if rule.endswith(":*"):
            parts = rule[:-2].split()
            if prog == parts[0] and tokens[1 : 1 + len(parts) - 1] == parts[1:]:
                return True
        elif rule == prog and len(tokens) == 1:
            return True
    return False


def matches_allow(command: str, allow: list[str]) -> bool:
    """True when every segment of the command matches an allow rule.

    A pipeline or chain auto-runs only if each part is individually
    allowlisted — ``grep x | head`` needs both ``grep:*`` and ``head:*``.
    Any segment without a rule forces a prompt, so no part runs unvetted.
    """
    segs = split_segments(command)
    return bool(segs) and all(_segment_matches(seg, allow) for seg in segs)


def _rule_for_segment(segment: str) -> str | None:
    """The allow rule for one segment: program (+ pinned subcommand) + ``:*``.

    The subcommand is pinned only for known subcommand-style CLIs; for
    interpreters and operand-first tools the rule is the bare program so a
    single approval covers all of them.
    """
    tokens = _tokens(segment)
    if not tokens:
        return None
    prog = tokens[0].rsplit("/", 1)[-1]
    if (
        prog in _SUBCOMMAND_PROGRAMS
        and len(tokens) >= 2
        and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", tokens[1])
    ):
        return f"{prog} {tokens[1]}:*"
    return f"{prog}:*"


def allow_rule_for(command: str, allow: list[str] | None = None) -> str | None:
    """The rule to persist for 'always allow' on this command, or None.

    The rule for the first segment not already covered by ``allow`` — the
    piece that made the command prompt. Chains earn their rules one segment
    at a time, each explicitly approved; every segment covered -> None.
    """
    for seg in split_segments(command):
        if allow is None or not _segment_matches(seg, allow):
            return _rule_for_segment(seg)
    return None


# ---------------------------------------------------------------- scope


def escape_project(command: str, root: Path) -> bool:
    """True when any token of the command reaches outside the project root.

    Flags ``~``/``$HOME``, ``..`` components, and absolute paths that are not
    under ``root``. Used to force a prompt in ask/auto modes; yolo ignores it.
    """
    for tok in command.split():
        t = tok.strip("();,'\"`")
        if not t:
            continue
        if t in ("~", "$HOME") or t.startswith(("~/", "$HOME/")):
            return True
        if ".." in t.split("/"):
            return True
        if t.startswith("/"):
            p = Path(t)
            if not (p == root or root in p.parents):
                return True
    return False


# ---------------------------------------------------------------- decision


@dataclass(frozen=True)
class Decision:
    action: str  # "run" | "ask" | "blocked"
    reason: str = ""


# Shell-level actions an allowlist grant must never silently cover: writing to
# a file (output redirection) or executing other programs from inside an
# allowlisted one (find -exec/-ok/-delete). An allowlist match is a read-only
# grant; anything acting on disk forces a prompt.
_REDIRECT_RE = re.compile(r"(?<![-=])>{1,2}(?!&)")  # > or >>, not -> => or >&1
_FIND_ACT_RE = re.compile(r"(?<!\S)(-exec|-execdir|-ok|-delete)(?!\S)")


def acting_reason(command: str) -> str | None:
    """Why an allowlisted command still must prompt, or None if it is inert."""
    if _FIND_ACT_RE.search(command):
        return "find -exec/-ok/-delete executes or deletes"
    if _REDIRECT_RE.search(command):
        return "output redirection writes to a file"
    return None


def decide(command: str, policy: CmdPolicy, root: Path) -> Decision:
    """The full gate: blacklist (absolute) -> off -> yolo -> allowlist -> ask.

    An allowlist match only auto-runs commands that stay read-only; a command
    that writes files or embeds execution prompts even when fully allowlisted.
    yolo is exempt (explicit max-trust mode; the blacklist still applies).
    """
    reason = check_blacklist(command)
    if reason is not None:
        return Decision("blocked", reason)
    if policy.mode == "off":
        return Decision("blocked", "command execution is disabled (cmd_mode = off)")
    if policy.mode == "yolo":
        return Decision("run")
    if matches_allow(command, policy.allow):
        acting = acting_reason(command)
        if acting is None:
            return Decision("run")
        return Decision("ask", f"allowlisted, but {acting}")
    why = "not in the allowlist"
    if escape_project(command, root):
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
