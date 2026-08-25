"""Unit tests for theme detection, resolution, and Textual mapping."""

import pytest

from jtech_cli.theme import (
    VALID_THEMES,
    detect_theme,
    resolve_theme,
    textual_theme_name,
)


def test_detect_theme_dark_default(monkeypatch):
    monkeypatch.delenv("COLORFGBG", raising=False)
    assert detect_theme() == "dark"


def test_detect_theme_dark_bg(monkeypatch):
    monkeypatch.setenv("COLORFGBG", "15;0")
    assert detect_theme() == "dark"


def test_detect_theme_light_bg(monkeypatch):
    monkeypatch.setenv("COLORFGBG", "0;15")
    assert detect_theme() == "light"


def test_detect_theme_list_bg_uses_actual(monkeypatch):
    # "15;15,0": fg=15, bg could not be determined (15) but actual is 0 (black) -> dark
    monkeypatch.setenv("COLORFGBG", "15;15,0")
    assert detect_theme() == "dark"


def test_resolve_theme_auto_delegates_to_detect(monkeypatch):
    monkeypatch.setenv("COLORFGBG", "0;15")
    assert resolve_theme("auto") == "light"


@pytest.mark.parametrize("choice", ["light", "dark"])
def test_resolve_theme_explicit(choice):
    assert resolve_theme(choice) == choice


def test_resolve_theme_invalid_raises():
    with pytest.raises(ValueError):
        resolve_theme("blue")


def test_valid_themes():
    assert VALID_THEMES == ("auto", "light", "dark")


def test_textual_theme_name_light(monkeypatch):
    monkeypatch.setenv("COLORFGBG", "0;15")
    assert textual_theme_name("auto") == "jtech-light"
    assert textual_theme_name("light") == "jtech-light"


def test_textual_theme_name_dark(monkeypatch):
    monkeypatch.setenv("COLORFGBG", "15;0")
    assert textual_theme_name("auto") == "jtech-dark"
    assert textual_theme_name("dark") == "jtech-dark"


def test_textual_theme_name_invalid_raises():
    with pytest.raises(ValueError):
        textual_theme_name("blue")


def test_custom_theme_objects():
    from jtech_cli.theme import JTECH_DARK, JTECH_LIGHT

    assert JTECH_DARK.name == "jtech-dark"
    assert JTECH_DARK.dark is True
    assert JTECH_LIGHT.name == "jtech-light"
    assert JTECH_LIGHT.dark is False


async def test_custom_themes_registered_and_available(tmp_path):
    from jtech_cli.config import Profile, Profiles, Settings
    from jtech_cli.server_info import ServerInfo
    from jtech_cli.session import Session
    from jtech_cli.tui import ChatApp

    profile = Profile(name="local", base_url="http://host:9000/v1", model="qwen3")
    app = ChatApp(
        settings=Settings(profiles=Profiles(items=(profile,), active_name="local")),
        session=Session(tmp_path / "s.jsonl", persist=False),
        server=ServerInfo(models=["qwen3"], context_length=4096),
        config_path=tmp_path / "config.toml",
    )
    async with app.run_test():
        assert "jtech-dark" in app.available_themes
        assert "jtech-light" in app.available_themes
        assert app.theme in ("jtech-dark", "jtech-light")
