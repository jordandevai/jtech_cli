"""TOML loading and persistence for settings and shell policy."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from jtech_cli.cmd_tools import (
    DEFAULT_ALLOW,
    DEFAULT_MAX_OUTPUT,
    DEFAULT_TIMEOUT,
    VALID_CMD_MODES,
    CmdPolicy,
)
from jtech_cli.configuration.paths import CONFIG_PATH
from jtech_cli.configuration.settings import (
    DEFAULT_CMD_MODE,
    DEFAULT_DEBUG_LEVEL,
    DEFAULT_PROMPT_SOURCE,
    DEFAULT_REASONING,
    DEFAULT_TEMPERATURE,
    DEFAULT_THEME,
    SETTING_KEYS,
    VALID_DEBUG_LEVELS,
    VALID_PROMPT_SOURCES,
    VALID_REASONING_MODES,
    Settings,
)
from jtech_cli.prompts import migrate_legacy_prompt
from jtech_cli.theme import VALID_THEMES


def load_config_overrides(path: Path = CONFIG_PATH) -> dict:
    """Read valid settings from a TOML file."""
    if not path.exists():
        return {}
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    table = data.get("server", data)
    return {key: value for key, value in table.items() if key in SETTING_KEYS}


def build_settings(
    base_url: str | None = None,
    model: str | None = None,
    *,
    config_path: Path = CONFIG_PATH,
) -> Settings:
    """Build settings from disk, with explicit values taking precedence."""
    overrides = load_config_overrides(config_path)
    reasoning = overrides.get("reasoning", DEFAULT_REASONING)
    if reasoning not in VALID_REASONING_MODES:
        reasoning = DEFAULT_REASONING
    debug_level = overrides.get("debug_level", DEFAULT_DEBUG_LEVEL)
    if debug_level not in VALID_DEBUG_LEVELS:
        debug_level = DEFAULT_DEBUG_LEVEL
    theme = overrides.get("theme", DEFAULT_THEME)
    if theme not in VALID_THEMES:
        theme = DEFAULT_THEME
    temperature = overrides.get("temperature", DEFAULT_TEMPERATURE)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        temperature = DEFAULT_TEMPERATURE
    prompt_text = overrides.get("system_prompt", "")
    if not isinstance(prompt_text, str):
        raise TypeError("system_prompt must be a string")
    configured_source = overrides.get("prompt_source")
    if configured_source is None:
        # Legacy configs stored prompt text without source metadata. Preserve
        # the useful instructions, but remove the retired command transport.
        prompt_text, migrated = migrate_legacy_prompt(prompt_text)
        prompt_source = "inline" if prompt_text else DEFAULT_PROMPT_SOURCE
    elif configured_source not in VALID_PROMPT_SOURCES:
        raise ValueError(
            f"prompt_source must be one of: {', '.join(VALID_PROMPT_SOURCES)}"
        )
    else:
        prompt_source = configured_source
    prompt_file = overrides.get("prompt_file", "")
    if not isinstance(prompt_file, str):
        raise TypeError("prompt_file must be a string")
    if prompt_source == "default" and prompt_text:
        raise ValueError("system_prompt must be empty when prompt_source is default")
    if prompt_source == "file" and not prompt_file:
        raise ValueError("prompt_file is required when prompt_source is file")
    if prompt_source == "inline" and not prompt_text:
        raise ValueError("system_prompt is required when prompt_source is inline")

    settings = Settings(
        base_url=overrides.get("base_url", ""),
        model=overrides.get("model", ""),
        temperature=float(temperature),
        prompt_source=prompt_source,
        prompt_file=prompt_file,
        theme=theme,
        reasoning=reasoning,
        debug_level=debug_level,
    )
    if prompt_source == "file":
        settings.set_prompt_file(prompt_file)
    elif prompt_source == "inline":
        settings.set_prompt_inline(prompt_text)
    if configured_source is None and migrated:
        settings.prompt_notice = (
            "Migrated a legacy prompt: its retired fenced-command section was "
            "removed; the current runtime command protocol is active."
        )
    if base_url:
        settings.base_url = base_url
    if model:
        settings.model = model
    return settings


def resolve_prompt_source(settings: Settings) -> Settings:
    """Resolve the selected prompt source before the first model request."""
    if settings.prompt_source == "file" and not settings.system_prompt:
        settings.reload_prompt()
    elif settings.prompt_source == "default":
        settings.reset_prompt()
    return settings


def _toml_str(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def load_cmd_policy(path: Path = CONFIG_PATH) -> CmdPolicy:
    """Read the [cmd] table, applying validated defaults per field."""
    table: dict = {}
    if path.exists():
        try:
            with path.open("rb") as fh:
                table = tomllib.load(fh).get("cmd", {})
        except (OSError, tomllib.TOMLDecodeError):
            table = {}
    mode = table.get("mode", DEFAULT_CMD_MODE)
    if not isinstance(mode, str) or mode not in VALID_CMD_MODES:
        mode = DEFAULT_CMD_MODE
    allow = table.get("allow", DEFAULT_ALLOW)
    if not isinstance(allow, list) or not all(isinstance(item, str) for item in allow):
        allow = list(DEFAULT_ALLOW)
    timeout = table.get("timeout", DEFAULT_TIMEOUT)
    if not isinstance(timeout, int) or timeout <= 0:
        timeout = DEFAULT_TIMEOUT
    max_output = table.get("max_output", DEFAULT_MAX_OUTPUT)
    if not isinstance(max_output, int) or max_output <= 0:
        max_output = DEFAULT_MAX_OUTPUT
    return CmdPolicy(mode=mode, allow=list(allow), timeout=timeout, max_output=max_output)


def save_settings(settings: Settings, path: Path = CONFIG_PATH, *, cmd: CmdPolicy) -> None:
    """Persist [server] settings and the complete [cmd] policy."""
    if settings.prompt_source not in VALID_PROMPT_SOURCES:
        raise ValueError(
            f"prompt_source must be one of: {', '.join(VALID_PROMPT_SOURCES)}"
        )
    lines = ["[server]"]
    lines.append(f"base_url = {_toml_str(settings.base_url)}")
    if settings.model:
        lines.append(f"model = {_toml_str(settings.model)}")
    lines.append(f"temperature = {settings.temperature}")
    if settings.theme and settings.theme != "auto":
        lines.append(f"theme = {_toml_str(settings.theme)}")
    if settings.reasoning and settings.reasoning != DEFAULT_REASONING:
        lines.append(f"reasoning = {_toml_str(settings.reasoning)}")
    if settings.debug_level and settings.debug_level != DEFAULT_DEBUG_LEVEL:
        lines.append(f"debug_level = {_toml_str(settings.debug_level)}")
    if settings.prompt_source != DEFAULT_PROMPT_SOURCE:
        lines.append(f"prompt_source = {_toml_str(settings.prompt_source)}")
        if settings.prompt_source == "file":
            if not settings.prompt_file:
                raise ValueError("prompt_file is required when prompt_source is file")
            lines.append(f"prompt_file = {_toml_str(settings.prompt_file)}")
        elif settings.prompt_source == "inline":
            if not settings.system_prompt:
                raise ValueError("system_prompt is required when prompt_source is inline")
            lines.append(f"system_prompt = {_toml_str(settings.system_prompt)}")
    lines.extend(
        [
            "",
            "[cmd]",
            f"mode = {_toml_str(cmd.mode)}",
            f"allow = {json.dumps(cmd.allow, ensure_ascii=False)}",
            f"timeout = {cmd.timeout}",
            f"max_output = {cmd.max_output}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
