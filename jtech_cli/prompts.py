"""Markdown-backed prompts used by configuration, commands, and the TUI."""

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


DEFAULT_SYSTEM_PROMPT = load_prompt("system.md")
INSTRUCTIONS_HELP = load_prompt("instructions.md")
NUDGE_PROMPT = load_prompt("nudge.md")
COMMAND_DECLINED_PROMPT = load_prompt("command-declined.md")

__all__ = [
    "COMMAND_DECLINED_PROMPT",
    "DEFAULT_SYSTEM_PROMPT",
    "INSTRUCTIONS_HELP",
    "NUDGE_PROMPT",
    "PromptResourceError",
    "PromptSourceError",
    "compose_system_prompt",
    "format_prompt",
    "load_prompt",
    "load_prompt_file",
    "migrate_legacy_prompt",
]
