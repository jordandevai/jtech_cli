"""Unit tests for server introspection (endpoint mocked)."""

from jtech_cli.config import Settings
from jtech_cli.server_info import fetch_server_info, fetch_token_count


class _FakeOpener:
    def __init__(self, responses: dict[str, dict]):
        self.responses = responses
        self.calls: list[tuple[str, bytes | None]] = []

    def __call__(self, req, timeout=None):
        url = req.full_url
        data = req.data
        self.calls.append((url, data))
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
        import json
        return json.dumps(self._payload).encode()


def test_fetch_server_info_single_model(monkeypatch):
    settings = Settings(base_url="http://srv:1234/v1")
    opener = _FakeOpener({
        "/models": {"data": [{"id": "real-model"}]},
        "/models/real-model": {"meta": {"llama.context_length": 32768}},
    })
    monkeypatch.setattr("jtech_cli.server_info.urlopen", opener)

    info = fetch_server_info(settings)
    assert info.models == ["real-model"]
    assert info.context_length == 32768
    assert info.model == "real-model"
    assert info.known is True


def test_fetch_server_info_unreachable(monkeypatch):
    settings = Settings(base_url="http://srv:1234/v1")

    def boom(req, timeout=None):
        raise OSError("refused")

    monkeypatch.setattr("jtech_cli.server_info.urlopen", boom)
    info = fetch_server_info(settings)
    assert info.models == []
    assert info.context_length is None
    assert info.known is False


def test_fetch_token_count(monkeypatch):
    settings = Settings(base_url="http://srv:1234/v1")
    opener = _FakeOpener({"/tokenize": {"tokens": [1, 2, 3]}})
    monkeypatch.setattr("jtech_cli.server_info.urlopen", opener)

    assert fetch_token_count(settings, "hello world") == 3
    url, data = opener.calls[0]
    assert url.endswith("/tokenize")
    assert data is not None


def test_fetch_token_count_failure(monkeypatch):
    settings = Settings(base_url="http://srv:1234/v1")

    def boom(req, timeout=None, data=None):
        raise OSError("down")

    monkeypatch.setattr("jtech_cli.server_info.urlopen", boom)
    assert fetch_token_count(settings, "x") is None
