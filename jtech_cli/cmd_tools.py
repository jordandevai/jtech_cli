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
    ``;`` first. Chaining means *every* segment must be vetted, not just the
    first one.
    """
    cmd = re.sub(r"(?<!&)&(?!&)", ";", command)
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


def check_blacklist(command: str) -> str | None:
    """Return a reason if the command touches the absolute blacklist, else None.

    Checked against every segment of the chain plus whole-line patterns, so
    ``git status && rm -rf /`` and ``curl x | sh`` are caught, not just their
    first word.
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


def matches_allow(command: str, allow: list[str]) -> bool:
    """True when a single-segment command matches an allow rule.

    Rules: ``name`` matches the bare program with no args; ``name:*`` matches
    the program with any args; ``name sub:*`` matches the program with first
    arg ``sub`` and any further args. Chained commands never match — an
    allowlist entry must cover the whole thing being run.
    """
    segs = split_segments(command)
    if len(segs) != 1:
        return False
    tokens = _tokens(segs[0])
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


def allow_rule_for(command: str) -> str | None:
    """The rule to persist for 'always allow' on this command, or None.

    Single segment only: program + first bare subcommand (if present) + ``:*``.
    """
    segs = split_segments(command)
    if len(segs) != 1:
        return None
    tokens = _tokens(segs[0])
    if not tokens:
        return None
    prog = tokens[0].rsplit("/", 1)[-1]
    if len(tokens) >= 2 and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", tokens[1]):
        return f"{prog} {tokens[1]}:*"
    return f"{prog}:*"


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


def decide(command: str, policy: CmdPolicy, root: Path) -> Decision:
    """The full gate: blacklist (absolute) -> off -> yolo -> allowlist -> ask."""
    reason = check_blacklist(command)
    if reason is not None:
        return Decision("blocked", reason)
    if policy.mode == "off":
        return Decision("blocked", "command execution is disabled (cmd_mode = off)")
    if policy.mode == "yolo":
        return Decision("run")
    if matches_allow(command, policy.allow):
        return Decision("run")
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
    except subprocess.TimeoutExpired:
        return ExecResult(124, "", timed_out=True)
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
        return f"$ {command}\n→ timed out (killed)"
    if result.interrupted:
        body = f"\n{result.output}" if result.output else ""
        return f"$ {command}\n→ interrupted by user{body}"
    body = result.output if result.output else "(no output)"
    return f"$ {command}\nexit {result.exit_code}\n{body}"
