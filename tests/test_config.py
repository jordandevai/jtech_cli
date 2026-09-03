"""Unit tests for the settings dataclass and config file loading."""

import pytest

from jtech_cli.cmd_tools import DEFAULT_ALLOW, CmdPolicy
from jtech_cli.config import (
    DEFAULT_TEMPERATURE,
    ConfigurationError,
    Profile,
    Profiles,
    Settings,
    build_settings,
    load_cmd_policy,
    load_config_overrides,
    resolve_prompt_source,
    save_settings,
)
from jtech_cli.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    WORKER_PROMPT,
    PromptSourceError,
    compose_coordinator_prompt,
    compose_worker_prompt,
)

LOCAL = Profile(name="local", base_url="http://x:1/v1", model="m")


def settings_with(**kwargs) -> Settings:
    """Settings carrying one active profile, so saves have an endpoint to write."""
    return Settings(profiles=Profiles(items=(LOCAL,), active_name="local"), **kwargs)


def test_defaults():
    s = Settings()
    assert s.profiles == Profiles()
    assert s.active_profile is None
    assert s.profile_is_overridden is False
    assert s.base_url == ""  # read-only projection of "no profile"
    assert s.model == ""
    assert s.temperature == 0.7
    assert s.system_prompt == ""
    assert s.prompt_source == "default"
    assert s.prompt_file == ""
    assert s.theme == "auto"
    assert s.reasoning == "transient"
    assert s.cmd_mode == "ask"


def test_set_valid_keys():
    s = Settings()
    s.set("temperature", "0.1")
    s.set("theme", "light")
    s.set("reasoning", "always")
    s.set("cmd_mode", "yolo")
    assert s.temperature == 0.1
    assert s.theme == "light"
    assert s.reasoning == "always"
    assert s.cmd_mode == "yolo"


def test_set_cmd_mode_invalid_raises():
    s = Settings()
    with pytest.raises(ValueError):
        s.set("cmd_mode", "maybe")
    assert s.cmd_mode == "ask"  # default unchanged


@pytest.mark.parametrize(
    "key,value",
    [
        ("temperature", "abc"),
        ("bogus", "x"),
        ("theme", "blue"),
        ("reasoning", "sometimes"),
        # Endpoint and model belong to a profile: /set is not a second route.
        ("model", "my-model"),
        ("base_url", "http://host:9000/v1"),
    ],
)
def test_set_invalid_raises(key, value):
    s = Settings()
    with pytest.raises(ValueError):
        s.set(key, value)


def test_endpoint_projections_are_read_only():
    """One source of truth: nothing can assign an endpoint onto Settings."""
    s = settings_with()
    assert s.base_url == "http://x:1/v1"
    with pytest.raises(AttributeError):
        s.base_url = "http://elsewhere/v1"
    with pytest.raises(AttributeError):
        s.model = "other"


def test_load_config_overrides_missing(tmp_path):
    assert load_config_overrides(tmp_path / "nope.toml") == {}


def test_load_config_overrides_valid(tmp_path):
    """Only global settings live in [server]; the endpoint is not one of them."""
    path = tmp_path / "config.toml"
    path.write_text('[server]\nbase_url = "http://x:1/v1"\ntemperature = 0.2\ntheme = "light"\n')
    assert load_config_overrides(path) == {"temperature": 0.2, "theme": "light"}


def test_load_config_overrides_rejects_bad_toml(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("not = toml [[")
    with pytest.raises(ConfigurationError):
        load_config_overrides(path)


def test_build_settings_precedence(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[server]\nbase_url = "http://cfg:1/v1"\nmodel = "cfg-model"\n')
    # config file wins over defaults
    s = build_settings(config_path=path)
    assert s.base_url == "http://cfg:1/v1"
    assert s.model == "cfg-model"
    # explicit args win over config file, without touching the stored profile
    s2 = build_settings(base_url="http://cli:2/v1", model="cli-model", config_path=path)
    assert s2.base_url == "http://cli:2/v1"
    assert s2.model == "cli-model"
    assert s2.profiles.get("default").base_url == "http://cfg:1/v1"


def test_resolve_default_prompt_keeps_runtime_prompt_out_of_saved_settings():
    s = Settings()
    result = resolve_prompt_source(s)
    assert result is s  # mutates and returns the same object
    assert s.system_prompt == ""
    assert s.effective_system_prompt() == DEFAULT_SYSTEM_PROMPT


def test_resolve_inline_prompt_keeps_custom_instructions():
    s = Settings(system_prompt="my custom prompt")
    resolve_prompt_source(s)
    assert s.system_prompt == "my custom prompt"
    assert s.prompt_source == "inline"
    assert "my custom prompt" in s.effective_system_prompt()
    assert s.effective_system_prompt().endswith(DEFAULT_SYSTEM_PROMPT)


# ------------------------------------------------- orchestration prompts


def test_coordinator_prompt_keeps_the_base_contract_and_adds_dispatch():
    base = Settings(system_prompt="my custom prompt").effective_system_prompt()
    prompt = compose_coordinator_prompt(
        base, profile_names=("local", "cloud"), active_profile_name="local"
    )
    assert prompt.startswith(base)
    assert "my custom prompt" in prompt
    assert DEFAULT_SYSTEM_PROMPT in prompt
    assert "[[[jtech_agent]]]" in prompt
    # The shell/coding contract is composed, never restated by the fragment.
    assert prompt.count(DEFAULT_SYSTEM_PROMPT) == 1


def test_coordinator_prompt_lists_profiles_in_order_and_marks_the_current_one():
    prompt = compose_coordinator_prompt(
        "BASE", profile_names=("cli", "local", "cloud"), active_profile_name="cli"
    )
    listing = prompt[prompt.index("### Available profiles") :]
    assert listing.index("`cli`") < listing.index("`local`") < listing.index("`cloud`")
    assert "`cli` — the profile this conversation runs on" in listing
    assert "`local` —" not in listing


def test_coordinator_prompt_without_profiles_forbids_dispatch():
    prompt = compose_coordinator_prompt(
        "BASE", profile_names=(), active_profile_name=None
    )
    assert "dispatch is unavailable" in prompt
    assert "Do not emit a [[[jtech_agent]]] block" in prompt


def test_coordinator_prompt_states_the_batch_and_continuation_rules():
    prompt = compose_coordinator_prompt(
        "BASE", profile_names=("local",), active_profile_name="local"
    )
    assert "Reuse a key to continue that agent" in prompt
    assert "genuinely independent" in prompt
    assert "may not dispatch the same key twice" in prompt
    assert "shell blocks or agent blocks, never both" in prompt
    assert "[JTECH agent result]" in prompt
    assert "no tool block at all" in prompt


def test_worker_prompt_keeps_the_base_contract_and_restricts_the_worker():
    base = Settings(system_prompt="my custom prompt").effective_system_prompt()
    prompt = compose_worker_prompt(base)
    assert prompt.startswith(base)
    assert "my custom prompt" in prompt
    assert prompt.endswith(WORKER_PROMPT)
    assert "You cannot dispatch agents" in prompt
    assert "### Available profiles" not in prompt
    assert prompt.count(DEFAULT_SYSTEM_PROMPT) == 1


def test_messages_with_system_empty_prompt_sends_none():
    """Runtime semantic: what is set is what is used — empty sends no system message."""
    from jtech_cli.session import Session

    s = Session(persist=False)
    s.add("user", "hi")
    assert s.messages_with_system("") == [{"role": "user", "content": "hi"}]
    assert s.messages_with_system("prompt")[0] == {
        "role": "system",
        "content": "prompt",
    }


def test_save_settings_roundtrip(tmp_path):
    path = tmp_path / "config.toml"
    s = settings_with(
        temperature=0.5,
        system_prompt='line1\nline2 "quoted"',
    )
    save_settings(s, path, cmd=CmdPolicy())
    assert path.exists()
    loaded = build_settings(config_path=path)
    assert loaded.base_url == s.base_url
    assert loaded.model == s.model
    assert loaded.temperature == s.temperature
    assert loaded.system_prompt == s.system_prompt
    assert loaded.prompt_source == "inline"
    assert loaded.theme == "auto"


def test_legacy_prompt_without_source_is_preserved_as_inline(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[server]\nsystem_prompt = "legacy instructions"\n')
    loaded = build_settings(config_path=path)
    assert loaded.prompt_source == "inline"
    assert loaded.system_prompt == "legacy instructions"
    assert DEFAULT_SYSTEM_PROMPT in loaded.effective_system_prompt()


def test_legacy_command_section_is_migrated_without_discarding_other_text(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        '[server]\nsystem_prompt = """custom rules\n\nShell commands:\n'
        '- emit a fenced code block with language `cmd`\n```cmd\npwd\n```\n"""\n'
    )
    loaded = build_settings(config_path=path)
    assert loaded.prompt_source == "inline"
    assert loaded.system_prompt == "custom rules"
    assert loaded.prompt_notice.startswith("Migrated a legacy prompt")
    assert "```cmd" not in loaded.effective_system_prompt()
    assert DEFAULT_SYSTEM_PROMPT in loaded.effective_system_prompt()


def test_custom_prompt_cannot_replace_runtime_contract():
    settings = Settings(system_prompt="legacy custom instructions")
    effective = settings.effective_system_prompt()
    assert effective.startswith("legacy custom instructions")
    assert effective.endswith(DEFAULT_SYSTEM_PROMPT)


def test_prompt_file_source_reloads_from_path(tmp_path):
    config = tmp_path / "config.toml"
    prompt = tmp_path / "instructions.md"
    prompt.write_text("first version")
    settings = Settings()
    settings.set_prompt_file(prompt)
    save_settings(settings, config, cmd=CmdPolicy())

    prompt.write_text("second version")
    loaded = build_settings(config_path=config)
    assert loaded.prompt_source == "file"
    assert loaded.prompt_file == str(prompt)
    assert loaded.system_prompt == "second version"


def test_prompt_file_selection_surfaces_missing_file(tmp_path):
    settings = Settings()
    with pytest.raises(PromptSourceError, match="could not be read"):
        settings.set_prompt_file(tmp_path / "missing.md")


def test_reset_prompt_returns_to_bundled_runtime(tmp_path):
    settings = Settings(system_prompt="custom")
    settings.reset_prompt()
    save_settings(settings, tmp_path / "config.toml", cmd=CmdPolicy())
    loaded = build_settings(config_path=tmp_path / "config.toml")
    assert loaded.prompt_source == "default"
    assert loaded.system_prompt == ""
    assert loaded.effective_system_prompt() == DEFAULT_SYSTEM_PROMPT


def test_save_settings_theme_roundtrip(tmp_path):
    path = tmp_path / "config.toml"
    s = settings_with(theme="light")
    save_settings(s, path, cmd=CmdPolicy())
    loaded = build_settings(config_path=path)
    assert loaded.theme == "light"


def test_save_settings_reasoning_roundtrip(tmp_path):
    path = tmp_path / "config.toml"
    s = settings_with(reasoning="always")
    save_settings(s, path, cmd=CmdPolicy())
    loaded = build_settings(config_path=path)
    assert loaded.reasoning == "always"


def test_save_settings_omits_default_reasoning(tmp_path):
    path = tmp_path / "config.toml"
    s = settings_with()
    save_settings(s, path, cmd=CmdPolicy())
    assert "reasoning" not in path.read_text()


def test_build_settings_bad_reasoning_falls_back_to_default(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[server]\nreasoning = "sometimes"\n')
    s = build_settings(config_path=path)
    assert s.reasoning == "transient"


@pytest.mark.parametrize(
    "prompt",
    [
        "carriage\rreturn\n",
        "tab\there\n",
        "form\ffeed\0nul",
        "all\r\n\t\f\\ \"escapes\"",
    ],
)
def test_save_settings_control_chars_roundtrip(tmp_path, prompt):
    path = tmp_path / "config.toml"
    s = settings_with(system_prompt=prompt)
    save_settings(s, path, cmd=CmdPolicy())
    loaded = build_settings(config_path=path)
    assert loaded.system_prompt == prompt


def test_load_cmd_policy_missing_file(tmp_path):
    p = load_cmd_policy(tmp_path / "nope.toml")
    assert p.mode == "ask"
    assert p.allow == list(DEFAULT_ALLOW)
    assert p.max_output == 12000


def test_load_cmd_policy_valid(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        '[server]\nbase_url = "http://x:1/v1"\n\n[cmd]\n'
        'mode = "yolo"\nallow = ["git status:*", "ls:*"]\nmax_output = 100\n'
    )
    p = load_cmd_policy(path)
    assert p.mode == "yolo"
    assert p.allow == ["git status:*", "ls:*"]
    assert p.max_output == 100


def test_load_cmd_policy_explicit_empty_allow(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[cmd]\nallow = []\n')
    assert load_cmd_policy(path).allow == []


def test_load_cmd_policy_invalid_falls_back(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[cmd]\nmode = "bogus"\nallow = "nope"\nmax_output = "big"\n')
    p = load_cmd_policy(path)
    assert p.mode == "ask"
    assert p.allow == list(DEFAULT_ALLOW)
    assert p.max_output == 12000


def test_load_cmd_policy_ignores_a_legacy_timeout_key(tmp_path):
    """Every generated config carries `[cmd].timeout`; upgrading must still start.

    The key is non-operative rather than migrated in place: loading is read-only,
    so the next intentional save is what drops it.
    """
    path = tmp_path / "config.toml"
    path.write_text(
        '[server]\nbase_url = "http://x:1/v1"\n\n[cmd]\n'
        'mode = "yolo"\nallow = ["ls:*"]\ntimeout = 60\nmax_output = 100\n'
    )
    p = load_cmd_policy(path)
    assert not hasattr(p, "timeout")
    assert (p.mode, p.allow, p.max_output) == ("yolo", ["ls:*"], 100)
    assert "timeout" in path.read_text()  # loading rewrote nothing

    save_settings(settings_with(), path, cmd=p)
    text = path.read_text()
    assert "timeout" not in text
    assert 'mode = "yolo"' in text
    assert 'allow = ["ls:*"]' in text
    assert "max_output = 100" in text


def test_save_settings_with_cmd_roundtrip(tmp_path):
    path = tmp_path / "config.toml"
    s = settings_with(cmd_mode="auto")
    cmd = CmdPolicy(mode="auto", allow=["ls:*", "git status:*"], max_output=2000)
    save_settings(s, path, cmd=cmd)
    text = path.read_text()
    assert "[cmd]" in text
    assert 'mode = "auto"' in text
    assert 'allow = ["ls:*", "git status:*"]' in text
    loaded = load_cmd_policy(path)
    assert loaded.mode == "auto"
    assert loaded.allow == ["ls:*", "git status:*"]
    assert loaded.max_output == 2000


def test_save_settings_requires_a_cmd_policy(tmp_path):
    """Omitting the policy is a TypeError, never a silent wipe of [cmd]."""
    path = tmp_path / "config.toml"
    with pytest.raises(TypeError):
        save_settings(settings_with(), path)


@pytest.mark.parametrize(
    "line, name, expected",
    [
        ('theme = "solarized"', "theme", "auto"),
        ('temperature = "hot"', "temperature", DEFAULT_TEMPERATURE),
        ("temperature = true", "temperature", DEFAULT_TEMPERATURE),
    ],
)
def test_build_settings_rejects_unusable_values(tmp_path, line, name, expected):
    """A stale or mistyped value falls back instead of reaching the app."""
    path = tmp_path / "config.toml"
    path.write_text(f"[server]\n{line}\n")
    assert getattr(build_settings(config_path=path), name) == expected


def test_bad_theme_in_config_resolves_instead_of_raising(tmp_path):
    """The TUI resolves the theme at mount, before the dialog that would fix it."""
    from jtech_cli.theme import textual_theme_name

    path = tmp_path / "config.toml"
    path.write_text('[server]\ntheme = "solarized"\n')
    name = textual_theme_name(build_settings(config_path=path).theme)
    assert name in ("jtech-dark", "jtech-light")


def test_persist_settings_syncs_cmd_mode_without_a_policy_in_hand(tmp_path):
    """/set cmd_mode must reach the file even when the context holds no policy."""
    from rich.console import Console

    from jtech_cli.commands import CommandContext
    from jtech_cli.session import Session

    path = tmp_path / "config.toml"
    save_settings(
        settings_with(), path,
        cmd=CmdPolicy(mode="ask", allow=["cargo build:*"], max_output=7000),
    )
    ctx = CommandContext(
        session=Session(tmp_path / "s.jsonl", persist=False),
        settings=settings_with(cmd_mode="yolo"),
        console=Console(record=True, width=100),
        cmd=None,
        config_path=path,
    )
    ctx.persist_settings()

    policy = load_cmd_policy(path)
    assert policy.mode == "yolo"            # the live setting reached the file
    assert policy.allow == ["cargo build:*"]  # the rest was carried through
    assert policy.max_output == 7000
