"""Unit tests for the first-run setup wizard (endpoint mocked)."""


from rich.console import Console

from jtech_cli.server_info import ServerInfo
from jtech_cli.wizard import run_setup


def _ask_script(answers: list[str]):
    it = iter(answers)
    return lambda prompt: next(it)


def _console():
    return Console(record=True, width=100)


def _fake_info(models, context=None):
    return ServerInfo(models=models, context_length=context)


def test_setup_success_single_model(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "jtech_cli.wizard.fetch_server_info",
        lambda settings: _fake_info(["real-model"], 4096),
    )
    console = _console()
    settings = run_setup(console, ask=_ask_script(["http://x:1/v1"]), config_path=tmp_path / "c.toml")
    assert settings.base_url == "http://x:1/v1"
    assert settings.model == "real-model"
    assert (tmp_path / "c.toml").exists()
    out = console.export_text()
    assert "Success!" in out
    assert "real-model" in out
    assert "4096" in out
    assert "http://127.0.0.1:8080/v1" in out  # URL shape hint shown up front


def test_setup_retries_after_failure(tmp_path, monkeypatch):
    responses = iter([
        _fake_info([]),          # first URL fails
        _fake_info(["ok-model"]),  # second URL succeeds
    ])
    monkeypatch.setattr("jtech_cli.wizard.fetch_server_info", lambda settings: next(responses))
    console = _console()
    settings = run_setup(
        console,
        ask=_ask_script(["http://bad/v1", "http://good/v1"]),
        config_path=tmp_path / "c.toml",
    )
    assert settings.base_url == "http://good/v1"
    assert settings.model == "ok-model"
    out = console.export_text()
    assert "Connection failed" in out
    assert "Success!" in out


def test_setup_multiple_models_prompts(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "jtech_cli.wizard.fetch_server_info",
        lambda settings: _fake_info(["model-a", "model-b"]),
    )
    console = _console()
    settings = run_setup(
        console,
        ask=_ask_script(["http://x:1/v1", "model-b"]),
        config_path=tmp_path / "c.toml",
    )
    assert settings.model == "model-b"


def test_setup_requires_url_when_no_default(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "jtech_cli.wizard.fetch_server_info",
        lambda settings: _fake_info(["m"]),
    )
    console = _console()
    settings = run_setup(console, ask=_ask_script(["", "http://srv:1234/v1"]), config_path=tmp_path / "c.toml")
    assert settings.base_url == "http://srv:1234/v1"
    assert "A server URL is required" in console.export_text()


def test_setup_uses_existing_config_as_default(tmp_path, monkeypatch):
    from jtech_cli.config import Settings, save_settings
    path = tmp_path / "c.toml"
    save_settings(Settings(base_url="http://srv:4321/v1", model="m"), path)

    monkeypatch.setattr(
        "jtech_cli.wizard.fetch_server_info",
        lambda settings: _fake_info(["m"]),
    )
    console = _console()
    # empty input -> falls back to the config file's base_url
    settings = run_setup(console, ask=_ask_script([""]), config_path=path)
    assert settings.base_url == "http://srv:4321/v1"
