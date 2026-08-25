"""Configuration implementation: paths, profiles, validation, and TOML persistence."""

from jtech_cli.configuration.paths import CONFIG_PATH, home_dir
from jtech_cli.configuration.profiles import (
    CLI_PROFILE_NAME,
    DEFAULT_PROFILE_NAME,
    LOCAL_API_KEY,
    Profile,
    ProfileError,
    Profiles,
    ResolvedProfile,
    resolve_api_key,
    resolve_profile,
)
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
    ConfigurationError,
    build_settings,
    load_cmd_policy,
    load_config_overrides,
    resolve_prompt_source,
    save_settings,
)

__all__ = [
    "CLI_PROFILE_NAME",
    "CONFIG_PATH",
    "DEFAULT_DEBUG_LEVEL",
    "DEFAULT_PROFILE_NAME",
    "DEFAULT_REASONING",
    "DEFAULT_TEMPERATURE",
    "LOCAL_API_KEY",
    "SETTABLE_KEYS",
    "SETTINGS",
    "SETTING_DESCRIPTIONS",
    "VALID_DEBUG_LEVELS",
    "VALID_REASONING_MODES",
    "ConfigurationError",
    "Profile",
    "ProfileError",
    "Profiles",
    "ResolvedProfile",
    "SettingSpec",
    "Settings",
    "build_settings",
    "home_dir",
    "load_cmd_policy",
    "load_config_overrides",
    "resolve_api_key",
    "resolve_profile",
    "resolve_prompt_source",
    "save_settings",
]
