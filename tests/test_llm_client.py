"""Unit tests for the streaming LLM client (mocked OpenAI endpoint)."""

from types import SimpleNamespace

from jtech_cli.config import Settings
from jtech_cli.llm_client import stream_reply


def _chunk(content):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])


def _empty_chunk():
    return SimpleNamespace(choices=[])


class _FakeClient:
    def __init__(self):
        self.sent = None

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.sent = kwargs
        return iter([
            _empty_chunk(),
            _chunk("Hello"),
            _chunk(" world"),
            _empty_chunk(),
        ])


def test_stream_reply_concatenates_deltas(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(Settings, "make_client", lambda self: fake)

    settings = Settings(model="m", temperature=0.3)
    messages = [{"role": "user", "content": "hi"}]
    assert "".join(stream_reply(settings, messages)) == "Hello world"

    assert fake.sent["model"] == "m"
    assert fake.sent["temperature"] == 0.3
    assert fake.sent["stream"] is True
    assert fake.sent["messages"] == messages


def _usage_chunk(prompt_tokens):
    """The final chunk ``include_usage`` produces: usage set, choices empty."""
    return SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=prompt_tokens))


def test_stream_reply_emits_usage_from_the_choiceless_final_chunk(monkeypatch):
    """The usage chunk has no choices, so it must be read before that guard."""

    class _Client(_FakeClient):
        def create(self, **kwargs):
            self.sent = kwargs
            return iter([_chunk("hi"), _usage_chunk(4242)])

    monkeypatch.setattr(Settings, "make_client", lambda self: _Client())
    # Distinct base_url on purpose: llm_client caches one client per base_url,
    # so sharing the default "" would hand this test the previous test's fake.
    items = list(stream_reply(Settings(base_url="http://usage-test/v1", model="m"), []))
    assert ("usage", {"prompt_tokens": 4242}) in items
    assert "".join(i for i in items if isinstance(i, str)) == "hi"
