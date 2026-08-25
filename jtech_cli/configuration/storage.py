"""TOML loading and persistence for settings, profiles, and shell policy."""

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
from jtech_cli.configuration.profiles import (
    CLI_PROFILE_NAME,
    DEFAULT_PROFILE_NAME,
    Profile,
    ProfileError,
    Profiles,
)
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

#: The only keys a [profiles.NAME] table may carry.
PROFILE_FIELDS = ("base_url", "model", "api_key_env")


class ConfigurationError(ValueError):
    """The configuration file cannot be read or does not describe a usable setup.

    Always carries the offending path and, where known, the field it came from,
    so the startup boundary can print one actionable line instead of a
    traceback. Raised rather than swallowed: a broken config must never be
    mistaken for an empty one.
    """


def _read_document(path: Path) -> dict:
    """Parse the whole TOML document, or return ``{}`` when the file is absent.

    Raises:
        ConfigurationError: on a read failure or malformed TOML.
    """
    if not path.exists():
        return {}
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except OSError as error:
        raise ConfigurationError(f"{path}: could not be read: {error}") from error
    except tomllib.TOMLDecodeError as error:
        raise ConfigurationError(f"{path}: is not valid TOML: {error}") from error


def _server_table(document: dict, path: Path) -> dict:
    """The ``[server]`` table, tolerating the historical flat top-level layout."""
    table = document.get("server", document)
    if not isinstance(table, dict):
        raise ConfigurationError(f"{path}: [server] must be a table")
    return table


def _settings_overrides(document: dict, path: Path) -> dict:
    """Global settings read from ``[server]``; endpoint keys are not among them."""
    table = _server_table(document, path)
    return {key: value for key, value in table.items() if key in SETTING_KEYS}


def load_config_overrides(path: Path = CONFIG_PATH) -> dict:
    """Read valid global settings from a TOML file.

    Raises:
        ConfigurationError: on a read failure or malformed TOML.
    """
    return _settings_overrides(_read_document(path), path)


def _migrate_legacy_profile(base_url: object, model: object, path: Path) -> Profiles:
    """Represent a pre-profile ``[server]`` endpoint as one in-memory profile.

    The file is not rewritten here; the next intentional save emits the new
    format.

    Raises:
        ConfigurationError: if the legacy endpoint is not a usable profile.
    """
    if not base_url:
        return Profiles()
    try:
        profile = Profile(
            name=DEFAULT_PROFILE_NAME, base_url=base_url, model=model or ""
        )
    except ProfileError as error:
        raise ConfigurationError(f"{path}: [server]: {error}") from error
    return Profiles(items=(profile,), active_name=DEFAULT_PROFILE_NAME)


def _load_profiles(document: dict, path: Path) -> Profiles:
    """Build the profile catalog from a config document.

    Legacy single-endpoint files migrate in memory. New-format files must name
    their active profile, and mixing the two layouts is refused rather than
    resolved by guessing which endpoint was meant.

    Raises:
        ConfigurationError: on a mixed, unknown-field, or inconsistent catalog.
    """
    server = _server_table(document, path)
    legacy_url = server.get("base_url", "")
    legacy_model = server.get("model", "")
    table = document.get("profiles")
    if table is None:
        return _migrate_legacy_profile(legacy_url, legacy_model, path)
    if not isinstance(table, dict):
        raise ConfigurationError(f"{path}: [profiles] must be a table of profiles")
    if legacy_url or legacy_model:
        raise ConfigurationError(
            f"{path}: [server].base_url/model cannot be combined with [profiles.*]. "
            "Keep the endpoint in its profile table and delete the [server] keys."
        )
    if not table:
        raise ConfigurationError(f"{path}: [profiles] defines no profiles")

    items: list[Profile] = []
    for name, entry in table.items():
        if not isinstance(entry, dict):
            raise ConfigurationError(f"{path}: [profiles.{name}] must be a table")
        unknown = sorted(set(entry) - set(PROFILE_FIELDS))
        if unknown:
            raise ConfigurationError(
                f"{path}: [profiles.{name}] has unknown field(s): {', '.join(unknown)}. "
                f"Allowed: {', '.join(PROFILE_FIELDS)}"
            )
        try:
            items.append(
                Profile(
                    name=name,
                    base_url=entry.get("base_url", ""),
                    model=entry.get("model", ""),
                    api_key_env=entry.get("api_key_env", ""),
                )
            )
        except ProfileError as error:
            raise ConfigurationError(f"{path}: [profiles.{name}]: {error}") from error

    active = server.get("active_profile")
    if not isinstance(active, str) or not active:
        raise ConfigurationError(
            f"{path}: [server].active_profile is required when [profiles.*] are configured"
        )
    try:
        return Profiles(items=tuple(items), active_name=active)
    except ProfileError as error:
        raise ConfigurationError(f"{path}: {error}") from error


def _cli_override(
    profiles: Profiles, base_url: str | None, model: str | None
) -> Profile | None:
    """Build the session-only profile for ``--base-url``/``--model``.

    Overriding an existing selection keeps that profile's name and credential
    source, so a URL-only override still authenticates the way the profile says.

    Raises:
        ProfileError: if the override values do not form a valid profile.
    """
    if not base_url and not model:
        return None
    active = profiles.active
    if active is None:
        if not base_url:
            # A model with no endpoint addresses nothing. Startup runs setup,
            # which asks for both, rather than pointing it at a guessed URL.
            return None
        return Profile(
            name=CLI_PROFILE_NAME,
            base_url=base_url.strip(),
            model=(model or "").strip(),
        )
    return Profile(
        name=active.name,
        base_url=(base_url or active.base_url).strip(),
        model=(model or active.model).strip(),
        api_key_env=active.api_key_env,
    )


def build_settings(
    base_url: str | None = None,
    model: str | None = None,
    *,
    config_path: Path = CONFIG_PATH,
) -> Settings:
    """Build settings and the profile catalog from disk.

    ``base_url`` and ``model`` are CLI overrides: they produce a session-only
    profile that is never written back to the config file.

    Raises:
        ConfigurationError: if the file cannot be read or is inconsistent.
        ProfileError: if the CLI override values are not a valid profile.
    """
    document = _read_document(config_path)
    overrides = _settings_overrides(document, config_path)
    profiles = _load_profiles(document, config_path)
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
        profiles=profiles,
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
    settings.profile_override = _cli_override(profiles, base_url, model)
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
    """Persist [server] globals, every stored profile, and the complete [cmd] policy.

    Only ``settings.profiles`` is written: a ``--base-url``/``--model`` override
    is session-only and must not leak into the file. The full ``CmdPolicy`` is
    required so a profile change can never erase always-allow rules.
    """
    if settings.prompt_source not in VALID_PROMPT_SOURCES:
        raise ValueError(
            f"prompt_source must be one of: {', '.join(VALID_PROMPT_SOURCES)}"
        )
    profiles = settings.profiles
    lines = ["[server]"]
    if profiles.active_name is not None:
        lines.append(f"active_profile = {_toml_str(profiles.active_name)}")
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
    for profile in profiles.items:
        # Profile names are restricted to a subset of TOML bare-key characters,
        # so the header needs no quoting.
        lines.extend(["", f"[profiles.{profile.name}]"])
        lines.append(f"base_url = {_toml_str(profile.base_url)}")
        if profile.model:
            lines.append(f"model = {_toml_str(profile.model)}")
        if profile.api_key_env:
            lines.append(f"api_key_env = {_toml_str(profile.api_key_env)}")
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
