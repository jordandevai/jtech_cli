"""Unit tests for the command registry and handlers."""

import pytest
from rich.console import Console

from jtech_cli.commands import CommandContext, _no_multiline, build_registry
from jtech_cli.config import Profile, ProfileError, Profiles, Settings
from jtech_cli.prompts import DEFAULT_SYSTEM_PROMPT
from jtech_cli.session import Session

LOCAL = Profile(name="local", base_url="http://x:1/v1", model="m")
CLOUD = Profile(
    name="cloud",
    base_url="https://api.example.com/v1",
    model="cloud-model",
    api_key_env="CLOUD_API_KEY",
)


def local_settings(**kwargs) -> Settings:
    """Settings with one active local profile."""
    return Settings(profiles=Profiles(items=(LOCAL,), active_name="local"), **kwargs)


async def _multiline_stub() -> str:
    return "pasted"


def make_ctx(tmp_path, multiline=None, settings=None, session=None):
    console = Console(record=True, width=100)
    ctx = CommandContext(
        session=session or Session(tmp_path / "s.jsonl"),
        settings=settings if settings is not None else local_settings(),
        console=console,
        enter_multiline=multiline or _multiline_stub,
        config_path=tmp_path / "config.toml",
    )
    return ctx, console


def output(console: Console) -> str:
    return console.export_text()


def test_dispatch_unknown(tmp_path):
    ctx, console = make_ctx(tmp_path)
    reg = build_registry(ctx)
    reg.handle("/bogus")
    assert "Unknown command" in output(console)


def test_help_lists_commands(tmp_path):
    ctx, console = make_ctx(tmp_path)
    reg = build_registry(ctx)
    reg.handle("/help")
    out = output(console)
    assert "/exit" in out
    assert "/read" in out


def test_clear_empties_session(tmp_path):
    session = Session(tmp_path / "s.jsonl")
    session.add("user", "hi")
    ctx, console = make_ctx(tmp_path, session=session)
    reg = build_registry(ctx)
    reg.handle("/clear")
    assert session.messages == []
    assert "cleared" in output(console).lower()


@pytest.mark.parametrize("key", ["model", "base_url"])
def test_set_refuses_endpoint_keys(tmp_path, key):
    """The endpoint has one route: a profile. /set is not a second one."""
    settings = local_settings()
    ctx, console = make_ctx(tmp_path, settings=settings)
    reg = build_registry(ctx)
    reg.handle(f"/set {key} something")
    assert settings.base_url == "http://x:1/v1"
    assert settings.model == "m"
    assert "Unknown setting" in output(console)


def test_set_help_no_longer_advertises_endpoint_keys(tmp_path):
    ctx, console = make_ctx(tmp_path)
    reg = build_registry(ctx)
    reg.handle("/set")
    out = output(console)
    assert "temperature" in out
    assert "model" not in out
    assert "base_url" not in out


def test_set_invalid_temperature(tmp_path):
    settings = local_settings()
    ctx, console = make_ctx(tmp_path, settings=settings)
    reg = build_registry(ctx)
    reg.handle("/set temperature nope")
    assert settings.temperature == 0.7
    assert "must be a number" in output(console)


def test_set_theme(tmp_path):
    settings = local_settings()
    ctx, console = make_ctx(tmp_path, settings=settings)
    reg = build_registry(ctx)
    reg.handle("/set theme light")
    assert settings.theme == "light"
    assert "theme = light" in output(console)


def test_theme_cycles(tmp_path):
    settings = local_settings()
    ctx, console = make_ctx(tmp_path, settings=settings)
    reg = build_registry(ctx)
    reg.handle("/theme")
    assert settings.theme == "light"
    reg.handle("/theme")
    assert settings.theme == "dark"
    reg.handle("/theme dark")
    assert settings.theme == "dark"
    assert "Theme set" in output(console)


def test_settings_help_lists_entries(tmp_path):
    ctx, console = make_ctx(tmp_path)
    reg = build_registry(ctx)
    reg.handle("/help")
    out = output(console)
    assert "/settings" in out
    assert "/theme" in out


def test_settings_calls_open_settings(tmp_path):
    opened = []
    ctx, _console = make_ctx(tmp_path)
    ctx.open_settings = lambda: opened.append(True)
    reg = build_registry(ctx)
    reg.handle("/settings")
    assert opened == [True]


def test_clear_calls_clear_chat(tmp_path):
    cleared = []
    session = Session(tmp_path / "s.jsonl")
    session.add("user", "hi")
    ctx, _console = make_ctx(tmp_path, session=session)
    ctx.clear_chat = lambda: cleared.append(True)
    reg = build_registry(ctx)
    reg.handle("/clear")
    assert session.messages == []
    assert cleared == [True]


def test_clear_does_not_reset_prompt_source(tmp_path):
    settings = local_settings(system_prompt="keep these instructions")
    ctx, _console = make_ctx(tmp_path, settings=settings)
    reg = build_registry(ctx)
    reg.handle("/clear")
    assert settings.prompt_source == "inline"
    assert settings.system_prompt == "keep these instructions"


def test_prompt_file_reload_and_reset(tmp_path):
    prompt = tmp_path / "instructions.md"
    prompt.write_text("first")
    ctx, console = make_ctx(tmp_path)
    reg = build_registry(ctx)

    reg.handle(f"/prompt {prompt}")
    assert ctx.settings.prompt_source == "file"
    assert ctx.settings.system_prompt == "first"
    assert 'prompt_source = "file"' in (tmp_path / "config.toml").read_text()

    prompt.write_text("second")
    reg.handle("/prompt reload")
    assert ctx.settings.system_prompt == "second"
    assert "Reloaded prompt file" in output(console)

    reg.handle("/prompt reset")
    assert ctx.settings.prompt_source == "default"
    assert ctx.settings.system_prompt == ""
    assert "Reset to the bundled runtime prompt" in output(console)


def test_set_theme_persists(tmp_path):
    settings = local_settings()
    ctx, _console = make_ctx(tmp_path, settings=settings)
    reg = build_registry(ctx)
    reg.handle("/set theme light")
    assert settings.theme == "light"
    assert (tmp_path / "config.toml").exists()
    assert 'theme = "light"' in (tmp_path / "config.toml").read_text()


async def test_write_uses_multiline_reader(tmp_path):
    calls = []
    async def reader():
        calls.append(None)
        return "hello\nworld"
    ctx, _console = make_ctx(tmp_path, multiline=reader)
    reg = build_registry(ctx)
    await reg.handle(f"/write {tmp_path / 'out.txt'}")
    assert len(calls) == 1
    assert (tmp_path / "out.txt").read_text() == "hello\nworld"


async def test_write_cancelled_never_reaches_the_writer(tmp_path, monkeypatch):
    """``None`` is a cancel: the file writer must not be called at all."""
    written = []
    monkeypatch.setattr(
        "jtech_cli.commands.file_tools.cmd_write",
        lambda path_arg, content: written.append((path_arg, content)) or "",
    )
    async def reader():
        return None
    target = tmp_path / "out.txt"
    ctx, console = make_ctx(tmp_path, multiline=reader)
    reg = build_registry(ctx)
    await reg.handle(f"/write {target}")
    assert written == []
    assert not target.exists()
    assert "Write cancelled." in output(console)


async def test_write_empty_submission_still_writes_an_empty_file(tmp_path):
    """Submitting an empty editor is a real instruction, unlike cancelling."""
    async def reader():
        return ""
    target = tmp_path / "out.txt"
    ctx, console = make_ctx(tmp_path, multiline=reader)
    reg = build_registry(ctx)
    await reg.handle(f"/write {target}")
    assert target.read_text() == ""
    assert "Write cancelled." not in output(console)


async def test_write_without_an_editor_raises_instead_of_guessing(tmp_path):
    """A host that injected no editor gets an error, not a fabricated result."""
    target = tmp_path / "out.txt"
    ctx, _console = make_ctx(tmp_path, multiline=_no_multiline)
    reg = build_registry(ctx)
    with pytest.raises(RuntimeError, match="Multi-line input is unavailable"):
        await reg.handle(f"/write {target}")
    assert not target.exists()


def test_read_prints_file(tmp_path):
    (tmp_path / "a.txt").write_text("x\ny\n")
    ctx, console = make_ctx(tmp_path)
    reg = build_registry(ctx)
    reg.handle(f"/read {tmp_path / 'a.txt'}")
    assert "1 | x" in output(console)


def test_exit_raises_systemexit(tmp_path):
    ctx, _console = make_ctx(tmp_path)
    reg = build_registry(ctx)
    with pytest.raises(SystemExit):
        reg.handle("/exit")


async def test_stats(tmp_path, monkeypatch):
    monkeypatch.setattr("jtech_cli.server_info.fetch_token_count", lambda profile, text: None)
    session = Session(tmp_path / "s.jsonl")
    session.add("user", "hello")
    ctx, console = make_ctx(tmp_path, session=session)
    reg = build_registry(ctx)
    await reg.handle("/stats")
    assert "messages=1" in output(console)


async def test_stats_shows_tokens_and_context(tmp_path, monkeypatch):
    from jtech_cli.server_info import ServerInfo
    session = Session(tmp_path / "s.jsonl")
    session.add("user", "hello world")
    ctx, console = make_ctx(tmp_path, session=session)
    ctx.server = ServerInfo(models=["m"], context_length=100)
    monkeypatch.setattr("jtech_cli.server_info.fetch_token_count", lambda profile, text: 3)
    reg = build_registry(ctx)
    await reg.handle("/stats")
    out = output(console)
    assert "history_tokens=3" in out
    assert "context_length=100" in out
    assert "context_remaining=97" in out


def test_models_lists_server_models(tmp_path):
    from jtech_cli.server_info import ServerInfo
    ctx, console = make_ctx(tmp_path)
    ctx.server = ServerInfo(models=["model-a", "model-b"], context_length=4096)
    reg = build_registry(ctx)
    reg.handle("/models")
    out = output(console)
    assert "model-a" in out
    assert "model-b" in out
    assert "4096" in out


def test_models_no_info(tmp_path):
    ctx, console = make_ctx(tmp_path)
    reg = build_registry(ctx)
    reg.handle("/models")
    assert "No model info" in output(console)


def test_render_last_reply(tmp_path):
    ctx, console = make_ctx(tmp_path)
    ctx.last_reply = "hello `code`"
    reg = build_registry(ctx)
    reg.handle("/render")
    assert "hello" in output(console)


def test_render_without_reply(tmp_path):
    ctx, console = make_ctx(tmp_path)
    reg = build_registry(ctx)
    reg.handle("/render")
    assert "No reply" in output(console)


# --- profile commands ------------------------------------------------------


def test_profiles_calls_open_profiles(tmp_path):
    opened = []
    ctx, _console = make_ctx(tmp_path)
    ctx.open_profiles = lambda: opened.append(True)
    reg = build_registry(ctx)
    reg.handle("/profiles")
    assert opened == [True]


async def test_profile_delegates_activation_to_the_app(tmp_path):
    """Dispatch stays free of settings, storage, and widget work."""
    switched = []
    settings = Settings(
        profiles=Profiles(items=(LOCAL, CLOUD), active_name="local")
    )
    ctx, _console = make_ctx(tmp_path, settings=settings)

    async def switch(name):
        switched.append(name)

    ctx.switch_profile = switch
    reg = build_registry(ctx)
    await reg.handle("/profile cloud")

    assert switched == ["cloud"]
    # the handler itself changed nothing
    assert settings.profiles.active_name == "local"
    assert not (tmp_path / "config.toml").exists()


async def test_profile_without_a_name_prints_usage_and_the_current_profile(tmp_path):
    settings = Settings(profiles=Profiles(items=(LOCAL, CLOUD), active_name="local"))
    ctx, console = make_ctx(tmp_path, settings=settings)
    called = []

    async def switch(name):
        called.append(name)

    ctx.switch_profile = switch
    reg = build_registry(ctx)
    await reg.handle("/profile")

    out = output(console)
    assert "Usage: /profile NAME" in out
    assert "local" in out
    assert "http://x:1/v1" in out
    assert "cloud" in out
    assert called == []


async def test_profile_without_a_name_and_no_catalog_says_none(tmp_path):
    ctx, console = make_ctx(tmp_path, settings=Settings())
    reg = build_registry(ctx)
    await reg.handle("/profile")
    assert "current: none" in output(console)


async def test_profile_with_an_unknown_name_still_delegates(tmp_path):
    """The app owns the unknown-name error; dispatch must not second-guess it."""
    settings = local_settings()
    ctx, console = make_ctx(tmp_path, settings=settings)

    async def switch(name):
        ctx.console.print(f"[red]No profile named {name!r}[/red]")

    ctx.switch_profile = switch
    reg = build_registry(ctx)
    await reg.handle("/profile nope")

    assert "No profile named 'nope'" in output(console)
    assert settings.profiles.active_name == "local"


def test_profile_commands_appear_in_help(tmp_path):
    ctx, console = make_ctx(tmp_path)
    reg = build_registry(ctx)
    reg.handle("/help")
    out = output(console)
    assert "/profiles" in out
    assert "/profile" in out


# --- /stats against the selected profile -----------------------------------


async def test_stats_counts_tokens_against_the_selected_profile(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(
        "jtech_cli.server_info.fetch_token_count",
        lambda profile, text: seen.append(profile) or 5,
    )
    session = Session(tmp_path / "s.jsonl")
    session.add("user", "hello")
    settings = Settings(profiles=Profiles(items=(LOCAL, CLOUD), active_name="cloud"))
    ctx, console = make_ctx(tmp_path, session=session, settings=settings)
    reg = build_registry(ctx)
    await reg.handle("/stats")

    assert seen == [CLOUD]
    assert "history_tokens=5" in output(console)


async def test_stats_reports_a_missing_profile_instead_of_zero_tokens(tmp_path):
    session = Session(tmp_path / "s.jsonl")
    session.add("user", "hello")
    ctx, console = make_ctx(tmp_path, session=session, settings=Settings())
    reg = build_registry(ctx)
    await reg.handle("/stats")

    out = output(console)
    assert "messages=1" in out
    assert "No API profile is configured" in out
    assert "history_tokens" not in out


async def test_stats_surfaces_a_credential_error(tmp_path, monkeypatch):
    def boom(profile, text):
        raise ProfileError(
            "Profile 'cloud' reads its API key from $CLOUD_API_KEY, which is "
            "unset or empty in this environment"
        )

    monkeypatch.setattr("jtech_cli.server_info.fetch_token_count", boom)
    session = Session(tmp_path / "s.jsonl")
    session.add("user", "hello")
    settings = Settings(profiles=Profiles(items=(CLOUD,), active_name="cloud"))
    ctx, console = make_ctx(tmp_path, session=session, settings=settings)
    reg = build_registry(ctx)
    await reg.handle("/stats")

    out = output(console)
    assert "messages=1" in out
    assert "CLOUD_API_KEY" in out
    assert "history_tokens" not in out


# ---------------------------------------------------------------- /system


def test_system_prints_the_settings_prompt_without_a_host_callback(tmp_path):
    """Standalone contexts have no extra composition: what settings say is sent."""
    ctx, console = make_ctx(tmp_path, settings=local_settings(system_prompt="mine"))
    reg = build_registry(ctx)
    reg.handle("/system")
    text = output(console)
    assert "Prompt source: inline" in text
    assert "mine" in text
    assert "Agent dispatch" not in text


def test_system_prints_the_host_composed_prompt_when_one_is_injected(tmp_path):
    """The TUI adds the coordinator contract, so /system must show that, not
    the settings prompt the next request would not actually carry."""
    ctx, console = make_ctx(tmp_path)
    ctx.effective_prompt = lambda: "COMPOSED PROMPT\n\n## Agent dispatch"
    reg = build_registry(ctx)
    reg.handle("/system")
    text = output(console)
    assert "COMPOSED PROMPT" in text
    assert "Agent dispatch" in text
    assert DEFAULT_SYSTEM_PROMPT.splitlines()[0] not in text
