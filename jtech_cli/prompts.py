"""Markdown-backed prompts used by configuration, commands, and the TUI."""

from collections.abc import Sequence
from pathlib import Path

from jtech_cli.resource_loader import ResourceError, load_text_resource


class PromptResourceError(RuntimeError):
    """Raised when a required Markdown prompt is missing or empty."""


class PromptSourceError(RuntimeError):
    """Raised when an explicitly selected prompt source cannot be loaded."""


def load_prompt(filename: str) -> str:
    """Load a non-empty Markdown prompt from the bundled prompt resources."""
    try:
        text = load_text_resource(f"prompts/{filename}")
    except ResourceError as error:
        raise PromptResourceError(
            f"Prompt resource {filename!r} could not be loaded"
        ) from error
    if not text.strip():
        raise PromptResourceError(f"Prompt resource {filename!r} is empty")
    return text


def format_prompt(filename: str, **values: object) -> str:
    """Load and format a prompt, surfacing missing template values."""
    return load_prompt(filename).format(**values)


def load_prompt_file(path: str | Path) -> str:
    """Load a non-empty user-selected prompt file with actionable context."""
    prompt_path = Path(path).expanduser()
    try:
        text = prompt_path.read_text(encoding="utf-8")
    except OSError as error:
        raise PromptSourceError(
            f"Prompt file {str(prompt_path)!r} could not be read: {error}"
        ) from error
    if not text.strip():
        raise PromptSourceError(f"Prompt file {str(prompt_path)!r} is empty")
    return text


def migrate_legacy_prompt(text: str) -> tuple[str, bool]:
    """Remove the retired fenced-command section from an untyped legacy prompt."""
    shell_section = text.find("\nShell commands:")
    if shell_section < 0:
        return text, False
    legacy_shell = text[shell_section:]
    if "```cmd" not in legacy_shell or "fenced code block" not in legacy_shell:
        return text, False
    return text[:shell_section].rstrip(), True


def compose_system_prompt(user_prompt: str) -> str:
    """Append optional user instructions before the current runtime contract."""
    custom = user_prompt.strip()
    if not custom:
        return DEFAULT_SYSTEM_PROMPT
    return f"{custom}\n\n{DEFAULT_SYSTEM_PROMPT}"


def _profile_availability(
    profile_names: Sequence[str], active_profile_name: str | None
) -> str:
    """The coordinator's profile section: what it may dispatch to, or nothing.

    With no available profile the section says dispatch is unavailable rather
    than leaving the model to invent a name it would only be refused for.
    """
    if not profile_names:
        return (
            "### Available profiles\n\n"
            "No API profile is available in this session, so agent dispatch is "
            "unavailable. Do not emit a [[[jtech_agent]]] block; continue the "
            "work yourself."
        )
    lines = ["### Available profiles", ""]
    for name in profile_names:
        active = name == active_profile_name
        suffix = " — the profile this conversation runs on" if active else ""
        lines.append(f"- `{name}`{suffix}")
    lines.append("")
    lines.append(
        "Use one of these names exactly. Any other name fails and runs nothing."
    )
    return "\n".join(lines)


def compose_coordinator_prompt(
    base_prompt: str,
    *,
    profile_names: Sequence[str],
    active_profile_name: str | None,
) -> str:
    """Append the coordinator dispatch contract to an effective system prompt.

    ``base_prompt`` is the caller's ``Settings.effective_system_prompt()``, so
    the user's own instructions and the shared runtime contract stay the single
    source of truth for coding and shell rules and are never restated here.

    Args:
        base_prompt: Custom instructions plus the runtime contract.
        profile_names: Dispatchable profile names, in the order to advertise.
        active_profile_name: The name this conversation itself runs on, marked
            in the list when it appears there.
    """
    fragment = format_prompt(
        "coordinator.md",
        availability=_profile_availability(profile_names, active_profile_name),
    )
    return f"{base_prompt}\n\n{fragment}"


def compose_worker_prompt(base_prompt: str) -> str:
    """Append the subagent role contract to an effective system prompt.

    Composed with the same base as the coordinator's, so one runtime contract
    governs both; this fragment only adds what being a worker changes.
    """
    return f"{base_prompt}\n\n{WORKER_PROMPT}"


DEFAULT_SYSTEM_PROMPT = load_prompt("system.md")
INSTRUCTIONS_HELP = load_prompt("instructions.md")
NUDGE_PROMPT = load_prompt("nudge.md")
COMMAND_DECLINED_PROMPT = load_prompt("command-declined.md")
WORKER_PROMPT = load_prompt("worker.md")

__all__ = [
    "COMMAND_DECLINED_PROMPT",
    "DEFAULT_SYSTEM_PROMPT",
    "INSTRUCTIONS_HELP",
    "NUDGE_PROMPT",
    "WORKER_PROMPT",
    "PromptResourceError",
    "PromptSourceError",
    "compose_coordinator_prompt",
    "compose_system_prompt",
    "compose_worker_prompt",
    "format_prompt",
    "load_prompt",
    "load_prompt_file",
    "migrate_legacy_prompt",
]
