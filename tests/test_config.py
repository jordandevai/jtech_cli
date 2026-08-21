"""Unit tests for the settings dataclass and config file loading."""

import pytest

from jtech_cli.cmd_tools import DEFAULT_ALLOW, CmdPolicy
from jtech_cli.config import (
    Settings,
    apply_default_prompt,
    build_settings,
    load_cmd_policy,
    load_config_overrides,
    save_settings,
)
from jtech_cli.prompts import DEFAULT_SYSTEM_PROMPT


def test_defaults():
    s = Settings()
    assert s.base_url == ""
    assert s.model == ""
    assert s.temperature == 0.7
    assert s.system_prompt == ""
    assert s.theme == "auto"
    assert s.reasoning == "transient"
    assert s.cmd_mode == "ask"


def test_set_valid_keys():
    s = Settings()
    s.set("model", "my-model")
    s.set("base_url", "http://host:9000/v1")
    s.set("temperature", "0.1")
    s.set("theme", "light")
    s.set("reasoning", "always")
    s.set("cmd_mode", "yolo")
    assert s.model == "my-model"
    assert s.base_url == "http://host:9000/v1"
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


def test_apply_default_prompt_when_empty():
    s = Settings()
    result = apply_default_prompt(s)
    assert result is s  # mutates and returns the same object
    assert s.system_prompt == DEFAULT_SYSTEM_PROMPT


def test_apply_default_prompt_keeps_custom():
    s = Settings(system_prompt="my custom prompt")
    apply_default_prompt(s)
    assert s.system_prompt == "my custom prompt"


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


def test_load_cmd_policy_missing_file(tmp_path):
    p = load_cmd_policy(tmp_path / "nope.toml")
    assert p.mode == "ask"
    assert p.allow == list(DEFAULT_ALLOW)
    assert p.timeout == 60
    assert p.max_output == 12000


def test_load_cmd_policy_valid(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        '[server]\nbase_url = "http://x:1/v1"\n\n[cmd]\n'
        'mode = "yolo"\nallow = ["git status:*", "ls:*"]\ntimeout = 5\nmax_output = 100\n'
    )
    p = load_cmd_policy(path)
    assert p.mode == "yolo"
    assert p.allow == ["git status:*", "ls:*"]
    assert p.timeout == 5
    assert p.max_output == 100


def test_load_cmd_policy_explicit_empty_allow(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[cmd]\nallow = []\n')
    assert load_cmd_policy(path).allow == []


def test_load_cmd_policy_invalid_falls_back(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[cmd]\nmode = "bogus"\nallow = "nope"\ntimeout = -1\nmax_output = "big"\n')
    p = load_cmd_policy(path)
    assert p.mode == "ask"
    assert p.allow == list(DEFAULT_ALLOW)
    assert p.timeout == 60
    assert p.max_output == 12000


def test_save_settings_with_cmd_roundtrip(tmp_path):
    path = tmp_path / "config.toml"
    s = Settings(base_url="http://x:1/v1", model="m", cmd_mode="auto")
    cmd = CmdPolicy(mode="auto", allow=["ls:*", "git status:*"], timeout=30, max_output=2000)
    save_settings(s, path, cmd=cmd)
    text = path.read_text()
    assert "[cmd]" in text
    assert 'mode = "auto"' in text
    assert 'allow = ["ls:*", "git status:*"]' in text
    loaded = load_cmd_policy(path)
    assert loaded.mode == "auto"
    assert loaded.allow == ["ls:*", "git status:*"]
    assert loaded.timeout == 30
    assert loaded.max_output == 2000


def test_save_settings_without_cmd_omits_table(tmp_path):
    path = tmp_path / "config.toml"
    save_settings(Settings(base_url="http://x:1/v1"), path)
    assert "[cmd]" not in path.read_text()
