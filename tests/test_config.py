"""Unit tests for the settings dataclass and config file loading."""

import pytest

from jtech_cli.config import (
    Settings,
    build_settings,
    load_config_overrides,
    save_settings,
)


def test_defaults():
    s = Settings()
    assert s.base_url == ""
    assert s.model == ""
    assert s.temperature == 0.7
    assert s.system_prompt == ""
    assert s.theme == "auto"
    assert s.reasoning == "transient"


def test_set_valid_keys():
    s = Settings()
    s.set("model", "my-model")
    s.set("base_url", "http://host:9000/v1")
    s.set("temperature", "0.1")
    s.set("theme", "light")
    s.set("reasoning", "always")
    assert s.model == "my-model"
    assert s.base_url == "http://host:9000/v1"
    assert s.temperature == 0.1
    assert s.theme == "light"
    assert s.reasoning == "always"


@pytest.mark.parametrize(
    "key,value",
    [
        ("temperature", "abc"),
        ("bogus", "x"),
        ("theme", "blue"),
        ("reasoning", "sometimes"),
    ],
)
def test_set_invalid_raises(key, value):
    s = Settings()
    with pytest.raises(ValueError):
        s.set(key, value)


def test_make_client_uses_base_url():
    s = Settings(base_url="http://example.com:1234/v1")
    client = s.make_client()
    assert str(client.base_url) == "http://example.com:1234/v1/"


def test_load_config_overrides_missing(tmp_path):
    assert load_config_overrides(tmp_path / "nope.toml") == {}


def test_load_config_overrides_valid(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[server]\nbase_url = "http://x:1/v1"\nmodel = "real-model"\ntemperature = 0.2\n')
    overrides = load_config_overrides(path)
    assert overrides == {"base_url": "http://x:1/v1", "model": "real-model", "temperature": 0.2}


def test_load_config_overrides_ignores_bad_toml(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("not = toml [[")
    assert load_config_overrides(path) == {}


def test_build_settings_precedence(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[server]\nbase_url = "http://cfg:1/v1"\nmodel = "cfg-model"\n')
    # config file wins over defaults
    s = build_settings(config_path=path)
    assert s.base_url == "http://cfg:1/v1"
    assert s.model == "cfg-model"
    # explicit args win over config file
    s2 = build_settings(base_url="http://cli:2/v1", model="cli-model", config_path=path)
    assert s2.base_url == "http://cli:2/v1"
    assert s2.model == "cli-model"


def test_save_settings_roundtrip(tmp_path):
    path = tmp_path / "config.toml"
    s = Settings(
        base_url="http://x:1/v1",
        model="m",
        temperature=0.5,
        system_prompt='line1\nline2 "quoted"',
    )
    save_settings(s, path)
    assert path.exists()
    loaded = build_settings(config_path=path)
    assert loaded.base_url == s.base_url
    assert loaded.model == s.model
    assert loaded.temperature == s.temperature
    assert loaded.system_prompt == s.system_prompt
    assert loaded.theme == "auto"


def test_save_settings_theme_roundtrip(tmp_path):
    path = tmp_path / "config.toml"
    s = Settings(base_url="http://x:1/v1", model="m", theme="light")
    save_settings(s, path)
    loaded = build_settings(config_path=path)
    assert loaded.theme == "light"


def test_save_settings_reasoning_roundtrip(tmp_path):
    path = tmp_path / "config.toml"
    s = Settings(base_url="http://x:1/v1", model="m", reasoning="always")
    save_settings(s, path)
    loaded = build_settings(config_path=path)
    assert loaded.reasoning == "always"


def test_save_settings_omits_default_reasoning(tmp_path):
    path = tmp_path / "config.toml"
    s = Settings(base_url="http://x:1/v1", model="m")
    save_settings(s, path)
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
    s = Settings(base_url="http://x:1/v1", system_prompt=prompt)
    save_settings(s, path)
    loaded = build_settings(config_path=path)
    assert loaded.system_prompt == prompt
