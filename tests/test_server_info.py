"""Unit tests for server introspection (endpoint mocked)."""

import json
from urllib.error import HTTPError

import pytest

from jtech_cli.config import Profile, ProfileError
from jtech_cli.server_info import fetch_server_info, fetch_token_count

LOCAL = Profile(name="local", base_url="http://srv:1234/v1")
CLOUD = Profile(
    name="cloud", base_url="https://api.example.com/v1", api_key_env="CLOUD_API_KEY"
)
CLOUD_ENV = {"CLOUD_API_KEY": "sk-secret"}


class _FakeOpener:
    def __init__(self, responses: dict[str, dict]):
        self.responses = responses
        self.calls: list[tuple[str, bytes | None]] = []
        self.headers: list[dict] = []

    def __call__(self, req, timeout=None):
        url = req.full_url
        self.calls.append((url, req.data))
        self.headers.append(dict(req.headers))
        for path, payload in self.responses.items():
            if url.endswith(path):
                return _FakeResp(payload)
        raise OSError(f"no stub for {url}")


class _FakeResp:
    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return json.dumps(self._payload).encode()


def authorization(headers: dict) -> str | None:
    """urllib title-cases header names, so look the value up case-insensitively."""
    for name, value in headers.items():
        if name.lower() == "authorization":
            return value
    return None


# --- discovery -------------------------------------------------------------


def test_fetch_server_info_unreachable(monkeypatch):
    def boom(req, timeout=None):
        raise OSError("refused")

    monkeypatch.setattr("jtech_cli.server_info.urlopen", boom)
    info = fetch_server_info(LOCAL)
    assert info.models == []
    assert info.context_length is None
    assert info.known is False
    assert info.error == "OSError: refused"


def test_fetch_server_info_reads_models_and_context(monkeypatch):
    opener = _FakeOpener(
        {"/models": {"data": [{"id": "qwen3", "meta": {"n_ctx": 4096}}]}}
    )
    monkeypatch.setattr("jtech_cli.server_info.urlopen", opener)
    info = fetch_server_info(LOCAL)
    assert info.models == ["qwen3"]
    assert info.context_length == 4096
    assert info.error is None


def test_a_local_profile_sends_no_authorization(monkeypatch):
    opener = _FakeOpener({"/models": {"data": [{"id": "qwen3"}]}})
    monkeypatch.setattr("jtech_cli.server_info.urlopen", opener)
    fetch_server_info(LOCAL)
    assert authorization(opener.headers[0]) is None


def test_an_authenticated_profile_sends_a_bearer_token(monkeypatch):
    opener = _FakeOpener({"/models": {"data": [{"id": "cloud-model"}]}})
    monkeypatch.setattr("jtech_cli.server_info.urlopen", opener)
    info = fetch_server_info(CLOUD, environ=CLOUD_ENV)
    assert info.models == ["cloud-model"]
    assert authorization(opener.headers[0]) == "Bearer sk-secret"


def test_a_missing_credential_is_explicit_and_makes_no_request(monkeypatch):
    opener = _FakeOpener({"/models": {"data": [{"id": "cloud-model"}]}})
    monkeypatch.setattr("jtech_cli.server_info.urlopen", opener)
    with pytest.raises(ProfileError, match="CLOUD_API_KEY"):
        fetch_server_info(CLOUD, environ={})
    assert opener.calls == []


def test_an_http_failure_is_reported_without_credential_data(monkeypatch):
    def unauthorized(req, timeout=None):
        raise HTTPError(
            req.full_url, 401, "Unauthorized", {"WWW-Authenticate": "Bearer"}, None
        )

    monkeypatch.setattr("jtech_cli.server_info.urlopen", unauthorized)
    info = fetch_server_info(CLOUD, environ=CLOUD_ENV)
    assert info.known is False
    assert "401" in info.error
    assert "sk-secret" not in info.error
    assert "Bearer" not in info.error


def test_a_malformed_response_is_reported_as_a_discovery_error(monkeypatch):
    class _BadResp(_FakeResp):
        def read(self):
            return b"not json"

    monkeypatch.setattr(
        "jtech_cli.server_info.urlopen", lambda req, timeout=None: _BadResp({})
    )
    info = fetch_server_info(LOCAL)
    assert info.known is False
    assert info.error is not None
    assert "JSONDecodeError" in info.error


def test_discovery_works_while_the_profile_model_is_empty(monkeypatch):
    """Discovery is what makes an empty Profile.model usable; it may not resolve one."""
    opener = _FakeOpener({"/models": {"data": [{"id": "served"}]}})
    monkeypatch.setattr("jtech_cli.server_info.urlopen", opener)
    assert LOCAL.model == ""
    assert fetch_server_info(LOCAL).model == "served"


# --- token count -----------------------------------------------------------


def test_fetch_token_count(monkeypatch):
    opener = _FakeOpener({"/tokenize": {"tokens": [1, 2, 3]}})
    monkeypatch.setattr("jtech_cli.server_info.urlopen", opener)

    assert fetch_token_count(LOCAL, "hello world") == 3
    url, data = opener.calls[0]
    assert url.endswith("/tokenize")
    assert data is not None
    assert authorization(opener.headers[0]) is None


def test_fetch_token_count_sends_a_bearer_token(monkeypatch):
    opener = _FakeOpener({"/tokenize": {"tokens": [1, 2]}})
    monkeypatch.setattr("jtech_cli.server_info.urlopen", opener)

    assert fetch_token_count(CLOUD, "hi", environ=CLOUD_ENV) == 2
    assert authorization(opener.headers[0]) == "Bearer sk-secret"


def test_fetch_token_count_failure(monkeypatch):
    def boom(req, timeout=None, data=None):
        raise OSError("down")

    monkeypatch.setattr("jtech_cli.server_info.urlopen", boom)
    assert fetch_token_count(LOCAL, "x") is None


def test_fetch_token_count_credential_failure_is_explicit(monkeypatch):
    opener = _FakeOpener({"/tokenize": {"tokens": [1]}})
    monkeypatch.setattr("jtech_cli.server_info.urlopen", opener)
    with pytest.raises(ProfileError, match="CLOUD_API_KEY"):
        fetch_token_count(CLOUD, "x", environ={})
    assert opener.calls == []
