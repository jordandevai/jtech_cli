"""Unit tests for the command registry and handlers."""

import pytest
from rich.console import Console

from jtech_cli.commands import CommandContext, build_registry
from jtech_cli.config import Settings
from jtech_cli.session import Session


async def _multiline_stub(_terminator: str) -> str:
    return "pasted"


def make_ctx(tmp_path, multiline=None, settings=None, session=None):
    console = Console(record=True, width=100)
    ctx = CommandContext(
        session=session or Session(tmp_path / "s.jsonl"),
        settings=settings or Settings(),
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


def test_set_model(tmp_path):
    settings = Settings()
    ctx, console = make_ctx(tmp_path, settings=settings)
    reg = build_registry(ctx)
    reg.handle("/set model qwen3")
    assert settings.model == "qwen3"
    assert "model = qwen3" in output(console)


def test_set_invalid_temperature(tmp_path):
    settings = Settings()
    ctx, console = make_ctx(tmp_path, settings=settings)
    reg = build_registry(ctx)
    reg.handle("/set temperature nope")
    assert settings.temperature == 0.7
    assert "must be a number" in output(console)


def test_set_theme(tmp_path):
    settings = Settings()
    ctx, console = make_ctx(tmp_path, settings=settings)
    reg = build_registry(ctx)
    reg.handle("/set theme light")
    assert settings.theme == "light"
    assert "theme = light" in output(console)


def test_theme_cycles(tmp_path):
    settings = Settings()
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
    settings = Settings(system_prompt="keep these instructions")
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
    settings = Settings()
    ctx, _console = make_ctx(tmp_path, settings=settings)
    reg = build_registry(ctx)
    reg.handle("/set theme light")
    assert settings.theme == "light"
    assert (tmp_path / "config.toml").exists()
    assert 'theme = "light"' in (tmp_path / "config.toml").read_text()


async def test_write_uses_multiline_reader(tmp_path):
    captured = []
    async def reader(term):
        captured.append(term)
        return "hello\nworld"
    ctx, _console = make_ctx(tmp_path, multiline=reader)
    reg = build_registry(ctx)
    await reg.handle(f"/write {tmp_path / 'out.txt'}")
    assert captured == ["END"]
    assert (tmp_path / "out.txt").read_text() == "hello\nworld"


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
    monkeypatch.setattr("jtech_cli.server_info.fetch_token_count", lambda settings, text: None)
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
    monkeypatch.setattr("jtech_cli.server_info.fetch_token_count", lambda settings, text: 3)
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
