"""Validated in-memory settings and their single source-of-truth schema."""

from __future__ import annotations

from dataclasses import dataclass, field

from openai import OpenAI

from jtech_cli.cmd_tools import VALID_CMD_MODES
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
        "system_prompt", "System prompt",
        "Instructions prepended to every message. Load from file: /prompt FILE",
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
SETTING_KEYS: tuple[str, ...] = tuple(setting.name for setting in SETTINGS)


@dataclass
class Settings:
    base_url: str = field(default="")
    model: str = field(default="")
    temperature: float = DEFAULT_TEMPERATURE
    system_prompt: str = field(default="")
    theme: str = field(default=DEFAULT_THEME)
    reasoning: str = field(default=DEFAULT_REASONING)
    cmd_mode: str = field(default=DEFAULT_CMD_MODE)
    debug_level: str = field(default=DEFAULT_DEBUG_LEVEL)

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
