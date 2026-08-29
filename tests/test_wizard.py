"""Unit tests for the first-run setup wizard (endpoint mocked)."""

import pytest
from rich.console import Console

from jtech_cli.cmd_tools import CmdPolicy
from jtech_cli.config import (
    Profile,
    ProfileError,
    Profiles,
    Settings,
    build_settings,
    load_cmd_policy,
    save_settings,
)
from jtech_cli.server_info import ServerInfo
from jtech_cli.wizard import run_setup

# Answers are scripted in prompt order: API-key variable, then URL, then model
# when more than one is served.
NO_AUTH = ""


def _ask_script(answers: list[str]):
    it = iter(answers)
    return lambda prompt: next(it)


def _console():
    return Console(record=True, width=100)


def _fake_info(models, context=None):
    return ServerInfo(models=models, context_length=context)


def _stub_probe(monkeypatch, result):
    """Stub discovery, recording the profile and environment each probe used."""
    seen: list[tuple[Profile, object]] = []

    def probe(profile, *, environ=None):
        seen.append((profile, environ))
        outcome = result(profile) if callable(result) else result
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr("jtech_cli.wizard.fetch_server_info", probe)
    return seen


def _saved(path) -> Settings:
    return build_settings(config_path=path)


def test_setup_success_single_model(tmp_path, monkeypatch):
    _stub_probe(monkeypatch, _fake_info(["real-model"], 4096))
    console = _console()
    settings = run_setup(
        console,
        ask=_ask_script([NO_AUTH, "http://x:1/v1"]),
        config_path=tmp_path / "c.toml",
    )
    assert settings.base_url == "http://x:1/v1"
    assert settings.model == "real-model"
    assert (tmp_path / "c.toml").exists()
    out = console.export_text()
    assert "Success!" in out
    assert "real-model" in out
    assert "4096" in out
    assert "http://127.0.0.1:8080/v1" in out  # URL shape hint shown up front


def test_first_run_creates_and_activates_the_default_profile(tmp_path, monkeypatch):
    _stub_probe(monkeypatch, _fake_info(["m"]))
    path = tmp_path / "c.toml"
    settings = run_setup(_console(), ask=_ask_script([NO_AUTH, "http://x:1/v1"]), config_path=path)

    assert settings.profiles.names == ("default",)
    assert settings.profiles.active_name == "default"
    assert _saved(path).profiles.active == Profile(
        name="default", base_url="http://x:1/v1", model="m"
    )


def test_setup_edits_the_active_profile_and_leaves_the_others_alone(tmp_path, monkeypatch):
    path = tmp_path / "c.toml"
    other = Profile(name="cloud", base_url="https://api.example.com/v1", model="cloud-model")
    settings = build_settings(config_path=path)
    settings.profiles = Profiles().add(other).add(
        Profile(name="workstation", base_url="http://old:1/v1", model="old"), activate=True
    )
    save_settings(settings, path, cmd=CmdPolicy())

    _stub_probe(monkeypatch, _fake_info(["new-model"]))
    console = _console()
    run_setup(console, ask=_ask_script([NO_AUTH, "http://new:2/v1"]), config_path=path)

    loaded = _saved(path)
    assert loaded.profiles.names == ("cloud", "workstation")
    assert loaded.profiles.active_name == "workstation"
    assert loaded.profiles.get("workstation").base_url == "http://new:2/v1"
    assert loaded.profiles.get("workstation").model == "new-model"
    assert loaded.profiles.get("cloud") == other
    assert "Configuring profile: workstation" in console.export_text()


def test_setup_probes_an_authenticated_endpoint_with_its_variable(tmp_path, monkeypatch):
    seen = _stub_probe(monkeypatch, _fake_info(["cloud-model"]))
    path = tmp_path / "c.toml"
    settings = run_setup(
        _console(),
        ask=_ask_script(["CLOUD_API_KEY", "https://api.example.com/v1"]),
        config_path=path,
        environ={"CLOUD_API_KEY": "sk-secret"},
    )

    probed, environ = seen[0]
    assert probed.api_key_env == "CLOUD_API_KEY"
    assert environ == {"CLOUD_API_KEY": "sk-secret"}
    assert settings.profiles.active.api_key_env == "CLOUD_API_KEY"
    # only the variable name is persisted, never the value
    text = path.read_text()
    assert 'api_key_env = "CLOUD_API_KEY"' in text
    assert "sk-secret" not in text


def test_a_missing_credential_returns_to_the_api_key_field(tmp_path, monkeypatch):
    calls = {"n": 0}

    def result(profile):
        calls["n"] += 1
        if calls["n"] == 1:
            return ProfileError(
                "Profile 'default' reads its API key from $ABSENT, which is unset "
                "or empty in this environment"
            )
        return _fake_info(["m"])

    _stub_probe(monkeypatch, result)
    console = _console()
    # first key name resolves to nothing, URL is fine, second key name works
    settings = run_setup(
        console,
        ask=_ask_script(["ABSENT", "http://x:1/v1", "PRESENT"]),
        config_path=tmp_path / "c.toml",
        environ={"PRESENT": "sk-secret"},
    )
    out = console.export_text()
    assert "ABSENT" in out
    assert "unset or empty" in out
    assert settings.profiles.active.api_key_env == "PRESENT"


def test_an_invalid_api_key_variable_name_is_rejected_before_probing(tmp_path, monkeypatch):
    seen = _stub_probe(monkeypatch, _fake_info(["m"]))
    console = _console()
    settings = run_setup(
        console,
        ask=_ask_script(["not a var", "http://x:1/v1", "GOOD_KEY"]),
        config_path=tmp_path / "c.toml",
        environ={"GOOD_KEY": "sk"},
    )
    assert "api_key_env" in console.export_text()
    assert len(seen) == 1  # nothing was probed with the invalid name
    assert settings.profiles.active.api_key_env == "GOOD_KEY"


def test_setup_retries_after_a_connection_failure(tmp_path, monkeypatch):
    responses = iter([_fake_info([]), _fake_info(["ok-model"])])
    _stub_probe(monkeypatch, lambda profile: next(responses))
    console = _console()
    settings = run_setup(
        console,
        ask=_ask_script([NO_AUTH, "http://bad/v1", "http://good/v1"]),
        config_path=tmp_path / "c.toml",
    )
    assert settings.base_url == "http://good/v1"
    assert settings.model == "ok-model"
    out = console.export_text()
    assert "Connection failed" in out
    assert "Success!" in out


def test_a_connection_failure_reports_the_transport_error(tmp_path, monkeypatch):
    responses = iter(
        [ServerInfo(error="OSError: refused"), _fake_info(["ok-model"])]
    )
    _stub_probe(monkeypatch, lambda profile: next(responses))
    console = _console()
    run_setup(
        console,
        ask=_ask_script([NO_AUTH, "http://bad/v1", "http://good/v1"]),
        config_path=tmp_path / "c.toml",
    )
    assert "refused" in console.export_text()


def test_an_invalid_url_is_rejected_without_probing(tmp_path, monkeypatch):
    seen = _stub_probe(monkeypatch, _fake_info(["m"]))
    console = _console()
    settings = run_setup(
        console,
        ask=_ask_script([NO_AUTH, "not-a-url", "http://good/v1"]),
        config_path=tmp_path / "c.toml",
    )
    assert "base_url" in console.export_text()
    assert len(seen) == 1
    assert settings.base_url == "http://good/v1"


def test_setup_multiple_models_prompts(tmp_path, monkeypatch):
    _stub_probe(monkeypatch, _fake_info(["model-a", "model-b"]))
    settings = run_setup(
        _console(),
        ask=_ask_script([NO_AUTH, "http://x:1/v1", "model-b"]),
        config_path=tmp_path / "c.toml",
    )
    assert settings.model == "model-b"


def test_setup_requires_url_when_no_default(tmp_path, monkeypatch):
    _stub_probe(monkeypatch, _fake_info(["m"]))
    console = _console()
    settings = run_setup(
        console,
        ask=_ask_script([NO_AUTH, "", "http://srv:1234/v1"]),
        config_path=tmp_path / "c.toml",
    )
    assert settings.base_url == "http://srv:1234/v1"
    assert "A server URL is required" in console.export_text()


def test_setup_uses_the_active_profile_as_the_default(tmp_path, monkeypatch):
    path = tmp_path / "c.toml"
    settings = build_settings(config_path=path)
    settings.profiles = Profiles().add(
        Profile(name="default", base_url="http://srv:4321/v1", model="m"), activate=True
    )
    save_settings(settings, path, cmd=CmdPolicy())

    _stub_probe(monkeypatch, _fake_info(["m"]))
    # blank input -> falls back to the active profile's base_url
    result = run_setup(_console(), ask=_ask_script([NO_AUTH, ""]), config_path=path)
    assert result.base_url == "http://srv:4321/v1"


def test_a_blank_api_key_answer_keeps_the_configured_variable(tmp_path, monkeypatch):
    path = tmp_path / "c.toml"
    settings = build_settings(config_path=path)
    settings.profiles = Profiles().add(
        Profile(
            name="default",
            base_url="https://api.example.com/v1",
            model="m",
            api_key_env="CLOUD_API_KEY",
        ),
        activate=True,
    )
    save_settings(settings, path, cmd=CmdPolicy())

    _stub_probe(monkeypatch, _fake_info(["m"]))
    result = run_setup(
        _console(),
        ask=_ask_script(["", ""]),
        config_path=path,
        environ={"CLOUD_API_KEY": "sk"},
    )
    assert result.profiles.active.api_key_env == "CLOUD_API_KEY"


def test_setup_preserves_global_settings(tmp_path, monkeypatch):
    path = tmp_path / "c.toml"
    settings = build_settings(config_path=path)
    settings.profiles = Profiles().add(
        Profile(name="default", base_url="http://old:1/v1", model="m"), activate=True
    )
    settings.temperature = 0.15
    settings.reasoning = "always"
    settings.debug_level = "system"
    settings.set_prompt_inline("keep these instructions")
    save_settings(settings, path, cmd=CmdPolicy())

    _stub_probe(monkeypatch, _fake_info(["m"]))
    run_setup(
        _console(), ask=_ask_script([NO_AUTH, "http://new:2/v1"]), config_path=path
    )

    loaded = _saved(path)
    assert loaded.temperature == 0.15
    assert loaded.reasoning == "always"
    assert loaded.debug_level == "system"
    assert loaded.system_prompt == "keep these instructions"
    assert loaded.base_url == "http://new:2/v1"


def test_rerunning_setup_preserves_the_cmd_policy(tmp_path, monkeypatch):
    """Re-running setup re-points the endpoint; it must not reset shell policy."""
    path = tmp_path / "c.toml"
    settings = build_settings(config_path=path)
    settings.profiles = Profiles().add(
        Profile(name="default", base_url="http://old:1/v1", model="m"), activate=True
    )
    save_settings(
        settings,
        path,
        cmd=CmdPolicy(mode="yolo", allow=["cargo build:*"], max_output=99),
    )
    _stub_probe(monkeypatch, _fake_info(["m"]))

    run_setup(_console(), ask=_ask_script([NO_AUTH, "http://new:2/v1"]), config_path=path)

    policy = load_cmd_policy(path)
    assert policy.mode == "yolo"
    assert policy.allow == ["cargo build:*"]
    assert policy.max_output == 99


def test_nothing_is_written_until_a_profile_is_complete(tmp_path, monkeypatch):
    """A wizard abandoned mid-form leaves no half-configured config behind."""
    path = tmp_path / "c.toml"
    _stub_probe(monkeypatch, _fake_info([]))  # every probe fails

    with pytest.raises(StopIteration):
        run_setup(
            _console(),
            ask=_ask_script([NO_AUTH, "http://bad/v1"]),  # runs out on the retry
            config_path=path,
        )
    assert not path.exists()
