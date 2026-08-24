"""Configuration implementation: paths, validation, and TOML persistence."""

from jtech_cli.configuration.paths import CONFIG_PATH, home_dir
from jtech_cli.configuration.settings import (
    DEFAULT_DEBUG_LEVEL,
    DEFAULT_REASONING,
    DEFAULT_TEMPERATURE,
    SETTABLE_KEYS,
    SETTING_DESCRIPTIONS,
    SETTINGS,
    VALID_DEBUG_LEVELS,
    VALID_REASONING_MODES,
    Settings,
    SettingSpec,
)
from jtech_cli.configuration.storage import (
    build_settings,
    load_cmd_policy,
    load_config_overrides,
    resolve_prompt_source,
    save_settings,
)

__all__ = [
    "CONFIG_PATH",
    "DEFAULT_DEBUG_LEVEL",
    "DEFAULT_REASONING",
    "DEFAULT_TEMPERATURE",
    "SETTABLE_KEYS",
    "SETTINGS",
    "SETTING_DESCRIPTIONS",
    "VALID_DEBUG_LEVELS",
    "VALID_REASONING_MODES",
    "SettingSpec",
    "Settings",
    "build_settings",
    "home_dir",
    "load_cmd_policy",
    "load_config_overrides",
    "resolve_prompt_source",
    "save_settings",
]
