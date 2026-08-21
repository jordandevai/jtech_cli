"""Configuration: the Settings object, a client factory, and TOML config file."""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

from jtech_cli.cmd_tools import (
    DEFAULT_ALLOW,
    DEFAULT_MAX_OUTPUT,
    DEFAULT_TIMEOUT,
    CmdPolicy,
    VALID_CMD_MODES,
)
from jtech_cli.prompts import DEFAULT_SYSTEM_PROMPT
from jtech_cli.theme import VALID_THEMES, resolve_theme

DEFAULT_TEMPERATURE = 0.7
DEFAULT_REASONING = "transient"
VALID_REASONING_MODES = ("hide", "transient", "tail", "always")


def home_dir() -> Path:
    """Base directory for user data (~/.mycli, overridable via $MYCLI_HOME)."""
    return Path(os.environ.get("MYCLI_HOME", "~/.mycli")).expanduser()


CONFIG_PATH = home_dir() / "config.toml"


@dataclass(frozen=True)
class SettingSpec:
    """One user-facing setting.

    Single source of truth for the setting list: the settings-dialog rows,
    /set's usage and persistence logic, and the description help all derive
    from SETTINGS. Add a setting here (plus its dataclass field/validator)
    and every surface picks it up.
    """

    name: str  # Settings field name
    label: str  # settings-dialog row label
    description: str  # one-line help under the highlighted row
    settable: bool = True  # False -> editable only via dialog/file; /set rejects


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
)

# Derived: one-line help per setting (shown by the TUI settings menu).
SETTING_DESCRIPTIONS: dict[str, str] = {s.name: s.description for s in SETTINGS}

# Derived: /set-able keys, in order (system_prompt is dialog/file only).
SETTABLE_KEYS: tuple[str, ...] = tuple(s.name for s in SETTINGS if s.settable)

# Derived: all valid config-file keys.
_SETTING_KEYS: tuple[str, ...] = tuple(s.name for s in SETTINGS)


@dataclass
class Settings:
    base_url: str = field(default="")
    model: str = field(default="")
    temperature: float = DEFAULT_TEMPERATURE
    system_prompt: str = field(default="")
    theme: str = field(default="auto")
    reasoning: str = field(default=DEFAULT_REASONING)
    cmd_mode: str = field(default="ask")

    def make_client(self) -> OpenAI:
        return OpenAI(base_url=self.base_url, api_key="none", timeout=30, max_retries=0)

    def set(self, key: str, value: str) -> None:
        """Update a setting by name, validating the value. Raises ValueError."""
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
            # validate the choice resolves before storing
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
        else:
            raise ValueError(f"Unknown setting: {key}")


def load_config_overrides(path: Path = CONFIG_PATH) -> dict:
    """Read settings from a TOML file (a [server] table). Missing/absent -> {}."""
    if not path.exists():
        return {}
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    table = data.get("server", data)
    return {k: v for k, v in table.items() if k in _SETTING_KEYS}


def build_settings(
    base_url: str | None = None,
    model: str | None = None,
    *,
    config_path: Path = CONFIG_PATH,
) -> Settings:
    """Build Settings from config file, with explicit values taking precedence."""
    overrides = load_config_overrides(config_path)
    reasoning = overrides.get("reasoning", DEFAULT_REASONING)
    if reasoning not in VALID_REASONING_MODES:  # stale/typo'd value in an old config
        reasoning = DEFAULT_REASONING
    settings = Settings(
        base_url=overrides.get("base_url", ""),
        model=overrides.get("model", ""),
        temperature=overrides.get("temperature", DEFAULT_TEMPERATURE),
        system_prompt=overrides.get("system_prompt", ""),
        theme=overrides.get("theme", "auto"),
        reasoning=reasoning,
    )
    if base_url:
        settings.base_url = base_url
    if model:
        settings.model = model
    return settings


def apply_default_prompt(settings: Settings) -> Settings:
    """Use the built-in default prompt when no prompt is configured.

    Startup-only: an empty prompt means nothing was set up, so the default
    applies. The result lives on the in-memory Settings only — it is never
    written to the config file. A prompt cleared mid-session stays cleared
    (the live setting is the source of truth, filled or empty).
    """
    if not settings.system_prompt:
        settings.system_prompt = DEFAULT_SYSTEM_PROMPT
    return settings


def _toml_str(value: str) -> str:
    """Escape a string for a TOML basic string (JSON escaping is TOML-compatible)."""
    return json.dumps(value, ensure_ascii=False)


def load_cmd_policy(path: Path = CONFIG_PATH) -> CmdPolicy:
    """Read the [cmd] table (mode, allow, timeout, max_output). Missing/absent -> defaults.

    ``allow`` defaults to DEFAULT_ALLOW (the read-only seed) when the key is
    absent; an explicit empty list is respected. Invalid values fall back to
    defaults per key.
    """
    table: dict = {}
    if path.exists():
        try:
            with path.open("rb") as fh:
                table = tomllib.load(fh).get("cmd", {})
        except (OSError, tomllib.TOMLDecodeError):
            table = {}
    mode = table.get("mode", "ask")
    if not isinstance(mode, str) or mode not in VALID_CMD_MODES:
        mode = "ask"
    allow = table.get("allow", DEFAULT_ALLOW)
    if not isinstance(allow, list) or not all(isinstance(a, str) for a in allow):
        allow = list(DEFAULT_ALLOW)
    timeout = table.get("timeout", DEFAULT_TIMEOUT)
    if not isinstance(timeout, int) or timeout <= 0:
        timeout = DEFAULT_TIMEOUT
    max_output = table.get("max_output", DEFAULT_MAX_OUTPUT)
    if not isinstance(max_output, int) or max_output <= 0:
        max_output = DEFAULT_MAX_OUTPUT
    return CmdPolicy(mode=mode, allow=list(allow), timeout=timeout, max_output=max_output)


def save_settings(settings: Settings, path: Path = CONFIG_PATH, cmd: CmdPolicy | None = None) -> None:
    """Persist settings to a TOML [server] table. Round-trips via load_config_overrides.

    With ``cmd`` given, a [cmd] table (policy for AI shell execution) is written
    too. Without it, an existing file's [cmd] table is simply not touched here —
    callers that hold a CmdPolicy should pass it so the table stays in sync.
    """
    lines = ["[server]"]
    lines.append(f"base_url = {_toml_str(settings.base_url)}")
    if settings.model:
        lines.append(f"model = {_toml_str(settings.model)}")
    lines.append(f"temperature = {settings.temperature}")
    if settings.theme and settings.theme != "auto":
        lines.append(f"theme = {_toml_str(settings.theme)}")
    if settings.reasoning and settings.reasoning != DEFAULT_REASONING:
        lines.append(f"reasoning = {_toml_str(settings.reasoning)}")
    if settings.system_prompt:
        lines.append(f"system_prompt = {_toml_str(settings.system_prompt)}")
    if cmd is not None:
        lines.append("")
        lines.append("[cmd]")
        lines.append(f"mode = {_toml_str(cmd.mode)}")
        lines.append(f"allow = {json.dumps(cmd.allow, ensure_ascii=False)}")
        lines.append(f"timeout = {cmd.timeout}")
        lines.append(f"max_output = {cmd.max_output}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
