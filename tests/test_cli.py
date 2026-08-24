"""Tests for the composition root: arg parsing and make_app wiring."""

from jtech_cli.cli import make_app, parse_args
from jtech_cli.config import Settings
from jtech_cli.prompts import DEFAULT_SYSTEM_PROMPT


def test_parse_args_defaults():
    args = parse_args([])
    assert args.no_persist is False
    assert args.no_discover is False
    assert args.base_url is None
    assert args.instructions is None


def test_parse_args_flags():
    args = parse_args(["--no-persist", "--no-discover", "--base-url", "http://x/v1"])
    assert args.no_persist
    assert args.no_discover
    assert str(args.base_url) == "http://x/v1"


def test_make_app_applies_default_prompt(tmp_path, monkeypatch):
    """No prompt configured -> the built-in default is used."""
    monkeypatch.setattr(
        "jtech_cli.cli.resolve_settings", lambda args, console: Settings(base_url="http://x/v1")
    )
    app = make_app(parse_args(["--no-persist", "--no-discover"]))
    assert app.settings.prompt_source == "default"
    assert app.settings.effective_system_prompt() == DEFAULT_SYSTEM_PROMPT


def test_make_app_keeps_configured_prompt(tmp_path, monkeypatch):
    """A configured prompt is preserved, never overwritten by the default."""
    monkeypatch.setattr(
        "jtech_cli.cli.resolve_settings",
        lambda args, console: Settings(base_url="http://x/v1", system_prompt="custom prompt"),
    )
    app = make_app(parse_args(["--no-persist", "--no-discover"]))
    assert app.settings.system_prompt == "custom prompt"
    assert "custom prompt" in app.settings.effective_system_prompt()
    assert app.settings.effective_system_prompt().endswith(DEFAULT_SYSTEM_PROMPT)


def test_make_app_loads_instruction_file_as_a_source(tmp_path, monkeypatch):
    prompt = tmp_path / "instructions.md"
    prompt.write_text("file instructions")
    def resolve(args, console):
        settings = Settings(base_url="http://x/v1")
        settings.set_prompt_file(args.instructions)
        return settings

    monkeypatch.setattr("jtech_cli.cli.resolve_settings", resolve)
    app = make_app(
        parse_args(["--instructions", str(prompt), "--no-persist", "--no-discover"])
    )
    assert app.settings.prompt_source == "file"
    assert app.settings.prompt_file == str(prompt)
    assert app.settings.system_prompt == "file instructions"


def test_make_app_wires_cmd_policy(tmp_path, monkeypatch):
    from jtech_cli.cmd_tools import CmdPolicy
    from jtech_cli.config import load_cmd_policy

    policy_path = tmp_path / "config.toml"
    policy_path.write_text('[cmd]\nmode = "yolo"\n')
    monkeypatch.setattr("jtech_cli.cli.load_cmd_policy", lambda: load_cmd_policy(policy_path))
    monkeypatch.setattr(
        "jtech_cli.cli.resolve_settings", lambda args, console: Settings(base_url="http://x/v1")
    )
    app = make_app(parse_args(["--no-persist", "--no-discover"]))
    assert isinstance(app.cmd, CmdPolicy)
    assert app.cmd.mode == "yolo"
    assert app.settings.cmd_mode == "yolo"
