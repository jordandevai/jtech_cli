"""Tests for the composition root: arg parsing, profile resolution, and wiring."""

import functools

import pytest

from jtech_cli.cli import main, make_app, make_settings, parse_args, resolve_settings
from jtech_cli.config import (
    ConfigurationError,
    Profile,
    ProfileError,
    Profiles,
    Settings,
    build_settings,
)
from jtech_cli.prompts import DEFAULT_SYSTEM_PROMPT

CONFIG = (
    '[server]\nactive_profile = "local"\n\n'
    '[profiles.local]\nbase_url = "http://cfg/v1"\nmodel = "cfg-model"\n'
)


def local_settings(**kwargs) -> Settings:
    """Settings with one active local profile, for wiring tests."""
    return Settings(
        profiles=Profiles(
            items=(Profile(name="local", base_url="http://x/v1"),), active_name="local"
        ),
        **kwargs,
    )


def use_config(monkeypatch, tmp_path, text=CONFIG):
    """Point cli.build_settings at a config file under tmp_path."""
    path = tmp_path / "config.toml"
    path.write_text(text)
    monkeypatch.setattr(
        "jtech_cli.cli.build_settings", functools.partial(build_settings, config_path=path)
    )
    return path


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
        "jtech_cli.cli.resolve_settings", lambda args, console: local_settings()
    )
    app = make_app(parse_args(["--no-persist", "--no-discover"]))
    assert app.settings.prompt_source == "default"
    assert app.settings.effective_system_prompt() == DEFAULT_SYSTEM_PROMPT


def test_make_app_keeps_configured_prompt(tmp_path, monkeypatch):
    """A configured prompt is preserved, never overwritten by the default."""
    monkeypatch.setattr(
        "jtech_cli.cli.resolve_settings",
        lambda args, console: local_settings(system_prompt="custom prompt"),
    )
    app = make_app(parse_args(["--no-persist", "--no-discover"]))
    assert app.settings.system_prompt == "custom prompt"
    assert "custom prompt" in app.settings.effective_system_prompt()
    assert app.settings.effective_system_prompt().endswith(DEFAULT_SYSTEM_PROMPT)


def test_make_app_loads_instruction_file_as_a_source(tmp_path, monkeypatch):
    prompt = tmp_path / "instructions.md"
    prompt.write_text("file instructions")

    def resolve(args, console):
        settings = local_settings()
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
        "jtech_cli.cli.resolve_settings", lambda args, console: local_settings()
    )
    app = make_app(parse_args(["--no-persist", "--no-discover"]))
    assert isinstance(app.cmd, CmdPolicy)
    assert app.cmd.mode == "yolo"
    assert app.settings.cmd_mode == "yolo"


# --- profile resolution ----------------------------------------------------


def test_the_persisted_active_profile_is_selected(tmp_path, monkeypatch):
    use_config(monkeypatch, tmp_path)
    settings = make_settings(parse_args([]))
    assert settings.active_profile.name == "local"
    assert settings.base_url == "http://cfg/v1"
    assert settings.model == "cfg-model"
    assert settings.profile_is_overridden is False


def test_cli_flags_override_the_session_only(tmp_path, monkeypatch):
    path = use_config(monkeypatch, tmp_path)
    settings = make_settings(
        parse_args(["--base-url", "http://cli/v1", "--model", "cli-model"])
    )
    assert settings.profile_is_overridden
    assert settings.base_url == "http://cli/v1"
    assert settings.model == "cli-model"
    # the stored profile is untouched, in memory and on disk
    assert settings.profiles.get("local").base_url == "http://cfg/v1"
    assert path.read_text() == CONFIG


def test_an_override_never_reaches_an_unrelated_settings_save(tmp_path, monkeypatch):
    from jtech_cli.cmd_tools import CmdPolicy
    from jtech_cli.config import save_settings

    path = use_config(monkeypatch, tmp_path)
    settings = make_settings(parse_args(["--base-url", "http://cli/v1"]))
    settings.theme = "light"
    save_settings(settings, path, cmd=CmdPolicy())

    text = path.read_text()
    assert "http://cli/v1" not in text
    assert 'base_url = "http://cfg/v1"' in text
    assert 'theme = "light"' in text


def test_setup_runs_when_no_profile_resolves(tmp_path, monkeypatch):
    use_config(monkeypatch, tmp_path, text="")
    called = {}

    def fake_setup(console, *, default_url=None, theme="auto"):
        called["default_url"] = default_url
        called["theme"] = theme
        return local_settings()

    monkeypatch.setattr("jtech_cli.cli.run_setup", fake_setup)
    resolve_settings(parse_args([]), console=None)
    assert called == {"default_url": None, "theme": "auto"}


def test_setup_is_skipped_for_a_one_shot_base_url(tmp_path, monkeypatch):
    use_config(monkeypatch, tmp_path, text="")

    def fail_setup(*args, **kwargs):
        raise AssertionError("setup must not run when --base-url resolves")

    monkeypatch.setattr("jtech_cli.cli.run_setup", fail_setup)
    settings = resolve_settings(parse_args(["--base-url", "http://oneshot/v1"]), console=None)
    assert settings.active_profile.name == "cli"
    assert settings.base_url == "http://oneshot/v1"


def test_setup_flag_forces_the_wizard_with_the_current_url(tmp_path, monkeypatch):
    use_config(monkeypatch, tmp_path)
    seen = {}

    def fake_setup(console, *, default_url=None, theme="auto"):
        seen["default_url"] = default_url
        return local_settings()

    monkeypatch.setattr("jtech_cli.cli.run_setup", fake_setup)
    resolve_settings(parse_args(["--setup"]), console=None)
    assert seen["default_url"] == "http://cfg/v1"


# --- startup failures ------------------------------------------------------


def as_tty(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)


@pytest.mark.parametrize(
    "error",
    [
        ConfigurationError("/home/u/.mycli/config.toml: is not valid TOML: line 1"),
        ProfileError("profile 'x': base_url must be an absolute http:// or https:// URL"),
    ],
)
def test_a_bad_config_exits_non_zero_with_one_concise_line(monkeypatch, capsys, error):
    as_tty(monkeypatch)

    def boom(args):
        raise error

    monkeypatch.setattr("jtech_cli.cli.make_app", boom)
    with pytest.raises(SystemExit) as excinfo:
        main([])

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == f"jtech-cli: {error}"
    assert "Traceback" not in captured.err


def test_an_invalid_base_url_flag_is_reported_not_traced(monkeypatch, capsys, tmp_path):
    as_tty(monkeypatch)
    use_config(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        main(["--base-url", "not-a-url"])

    assert excinfo.value.code == 1
    assert "base_url" in capsys.readouterr().err
