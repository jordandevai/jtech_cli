"""Markdown-backed prompts used by configuration, commands, and the TUI."""

from jtech_cli.resource_loader import ResourceError, load_text_resource


class PromptResourceError(RuntimeError):
    """Raised when a required Markdown prompt is missing or empty."""


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


DEFAULT_SYSTEM_PROMPT = load_prompt("system.md")
INSTRUCTIONS_HELP = load_prompt("instructions.md")
NUDGE_PROMPT = load_prompt("nudge.md")
COMMAND_DECLINED_PROMPT = load_prompt("command-declined.md")
BLOCKS_DROPPED_PROMPT = load_prompt("blocks-dropped.md")
ROUNDS_EXHAUSTED_PROMPT = load_prompt("rounds-exhausted.md")

__all__ = [
    "BLOCKS_DROPPED_PROMPT",
    "COMMAND_DECLINED_PROMPT",
    "DEFAULT_SYSTEM_PROMPT",
    "INSTRUCTIONS_HELP",
    "NUDGE_PROMPT",
    "ROUNDS_EXHAUSTED_PROMPT",
    "PromptResourceError",
    "format_prompt",
    "load_prompt",
]
