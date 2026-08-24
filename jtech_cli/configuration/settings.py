"""Validated in-memory settings and their single source-of-truth schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

from jtech_cli.cmd_tools import VALID_CMD_MODES
from jtech_cli.prompts import PromptSourceError, compose_system_prompt, load_prompt_file
from jtech_cli.resource_loader import load_toml_resource
from jtech_cli.theme import VALID_THEMES, resolve_theme

_DEFAULTS = load_toml_resource("config/defaults.toml")
_SETTING_DEFAULTS = _DEFAULTS.get("settings")
if not isinstance(_SETTING_DEFAULTS, dict):
    raise TypeError("Bundled config/defaults.toml must define a [settings] table")


def _string_default(name: str) -> str:
    value = _SETTING_DEFAULTS.get(name)
    if not isinstance(value, str):
        raise TypeError(f"Bundled setting default {name!r} must be a string")
    return value


def _number_default(name: str) -> float:
    value = _SETTING_DEFAULTS.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Bundled setting default {name!r} must be numeric")
    return float(value)


DEFAULT_TEMPERATURE = _number_default("temperature")
DEFAULT_REASONING = _string_default("reasoning")
DEFAULT_THEME = _string_default("theme")
DEFAULT_CMD_MODE = _string_default("cmd_mode")
VALID_REASONING_MODES = ("hide", "transient", "tail", "always")
DEFAULT_DEBUG_LEVEL = _string_default("debug_level")
VALID_DEBUG_LEVELS = ("none", "system")
DEFAULT_PROMPT_SOURCE = "default"
VALID_PROMPT_SOURCES = ("default", "file", "inline")


@dataclass(frozen=True)
class SettingSpec:
    """One user-facing setting and its editing metadata."""

    name: str
    label: str
    description: str
    settable: bool = True


SETTINGS: tuple[SettingSpec, ...] = (
    SettingSpec(
        "model", "Model",
        "Name of the model to chat with (see /models for what the server offers)",
    ),
    SettingSpec(
        "base_url", "Base URL",
        "OpenAI-compatible endpoint, e.g. http://127.0.0.1:8080/v1 (llama-server)",
    ),
    SettingSpec(
        "temperature", "Temperature",
        "Sampling randomness, 0.0-2.0. Lower = focused, higher = varied",
    ),
    SettingSpec(
        "theme", "Theme",
        "auto follows your terminal's colors; light/dark force a Jtech scheme",
    ),
    SettingSpec(
        "reasoning", "Reasoning",
        "How thinking tokens display: hide | transient (default) | tail (last 500) | always",
    ),
    SettingSpec(
        "system_prompt", "Additional instructions",
        "Optional extra instructions. The current CLI runtime protocol is always added.",
        settable=False,
    ),
    SettingSpec(
        "cmd_mode", "Shell mode",
        "AI shell execution: ask (prompt each command) | auto (allowlist runs silently) "
        " | yolo (all run except the hard blacklist) | off",
    ),
    SettingSpec(
        "debug_level", "Debug level",
        "System messages in chat: none (session only) | system (also render in chat)",
    ),
)

SETTING_DESCRIPTIONS: dict[str, str] = {setting.name: setting.description for setting in SETTINGS}
SETTABLE_KEYS: tuple[str, ...] = tuple(setting.name for setting in SETTINGS if setting.settable)
SETTING_KEYS: tuple[str, ...] = tuple(setting.name for setting in SETTINGS) + (
    "prompt_source",
    "prompt_file",
)


@dataclass
class Settings:
    base_url: str = field(default="")
    model: str = field(default="")
    temperature: float = DEFAULT_TEMPERATURE
    system_prompt: str = field(default="")
    prompt_source: str = field(default=DEFAULT_PROMPT_SOURCE)
    prompt_file: str = field(default="")
    prompt_notice: str = field(default="")
    theme: str = field(default=DEFAULT_THEME)
    reasoning: str = field(default=DEFAULT_REASONING)
    cmd_mode: str = field(default=DEFAULT_CMD_MODE)
    debug_level: str = field(default=DEFAULT_DEBUG_LEVEL)

    def __post_init__(self) -> None:
        if self.prompt_source not in VALID_PROMPT_SOURCES:
            raise ValueError(
                f"prompt_source must be one of: {', '.join(VALID_PROMPT_SOURCES)}"
            )
        # Direct construction with the historical system_prompt argument means
        # inline instructions; config loading supplies the source explicitly.
        if self.system_prompt and self.prompt_source == DEFAULT_PROMPT_SOURCE:
            self.prompt_source = "inline"

    def set_prompt_inline(self, text: str) -> None:
        """Use ``text`` as extra inline instructions, or reset when empty."""
        if not text.strip():
            self.reset_prompt()
            return
        self.system_prompt = text
        self.prompt_source = "inline"
        self.prompt_file = ""
        self.prompt_notice = ""

    def set_prompt_file(self, path: str | Path) -> None:
        """Select and immediately load a prompt file, surfacing read errors."""
        prompt_path = Path(path).expanduser().resolve()
        text = load_prompt_file(prompt_path)
        self.system_prompt = text
        self.prompt_source = "file"
        self.prompt_file = str(prompt_path)
        self.prompt_notice = ""

    def reload_prompt(self) -> None:
        """Reload the selected file source; reject reload for non-file sources."""
        if self.prompt_source != "file":
            raise PromptSourceError(
                f"Prompt source is {self.prompt_source!r}; only file prompts can reload"
            )
        self.system_prompt = load_prompt_file(self.prompt_file)

    def reset_prompt(self) -> None:
        """Return to the bundled runtime prompt without user instructions."""
        self.system_prompt = ""
        self.prompt_source = DEFAULT_PROMPT_SOURCE
        self.prompt_file = ""
        self.prompt_notice = ""

    def effective_system_prompt(self) -> str:
        """Return current user instructions followed by the runtime contract."""
        if self.prompt_source == "file" and not self.system_prompt:
            raise PromptSourceError("The selected prompt file has not been loaded")
        return compose_system_prompt(self.system_prompt)

    def make_client(self) -> OpenAI:
        return OpenAI(base_url=self.base_url, api_key="none", timeout=30, max_retries=0)

    def set(self, key: str, value: str) -> None:
        """Update a setting by name, validating the value."""
        if key == "model":
            self.model = value
        elif key == "base_url":
            self.base_url = value
        elif key == "temperature":
            try:
                self.temperature = float(value)
            except ValueError:
                raise ValueError("temperature must be a number") from None
        elif key == "theme":
            choice = value.strip().lower()
            if choice not in VALID_THEMES:
                raise ValueError(f"theme must be one of: {', '.join(VALID_THEMES)}")
            resolve_theme(choice)
            self.theme = choice
        elif key == "reasoning":
            choice = value.strip().lower()
            if choice not in VALID_REASONING_MODES:
                raise ValueError(f"reasoning must be one of: {', '.join(VALID_REASONING_MODES)}")
            self.reasoning = choice
        elif key == "cmd_mode":
            choice = value.strip().lower()
            if choice not in VALID_CMD_MODES:
                raise ValueError(f"cmd_mode must be one of: {', '.join(VALID_CMD_MODES)}")
            self.cmd_mode = choice
        elif key == "debug_level":
            choice = value.strip().lower()
            if choice not in VALID_DEBUG_LEVELS:
                raise ValueError(f"debug_level must be one of: {', '.join(VALID_DEBUG_LEVELS)}")
            self.debug_level = choice
        else:
            raise ValueError(f"Unknown setting: {key}")
