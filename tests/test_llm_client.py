"""Unit tests for the streaming LLM client (mocked OpenAI endpoint)."""

from types import SimpleNamespace

import pytest

from jtech_cli import llm_client
from jtech_cli.config import ResolvedProfile
from jtech_cli.llm_client import make_client, stream_reply

LOCAL = ResolvedProfile(
    name="local", base_url="http://host:9000/v1", model="qwen3", api_key="none"
)
CLOUD = ResolvedProfile(
    name="cloud",
    base_url="https://api.example.com/v1",
    model="cloud-model",
    api_key="sk-secret",
)


@pytest.fixture(autouse=True)
def _isolated_client_cache():
    """Clients are cached per transport identity; tests must not share them."""
    llm_client._client_cache.clear()
    yield
    llm_client._client_cache.clear()


def _chunk(content):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])


def _empty_chunk():
    return SimpleNamespace(choices=[])


def _usage_chunk(prompt_tokens):
    """The final chunk ``include_usage`` produces: usage set, choices empty."""
    return SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=prompt_tokens))


def fake_openai(monkeypatch, chunks=(), fail_on_stream_options=False):
    """Replace the SDK constructor; return the list of clients it builds."""
    built = []

    class _Client:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.sent = None
            self.calls = []
            built.append(self)

        @property
        def chat(self):
            return self

        @property
        def completions(self):
            return self

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if fail_on_stream_options and "stream_options" in kwargs:
                raise TypeError("stream_options unsupported")
            self.sent = kwargs
            return iter(list(chunks))

    monkeypatch.setattr(llm_client, "OpenAI", _Client)
    return built


def test_stream_reply_concatenates_deltas(monkeypatch):
    built = fake_openai(
        monkeypatch,
        [_empty_chunk(), _chunk("Hello"), _chunk(" world"), _empty_chunk()],
    )
    messages = [{"role": "user", "content": "hi"}]
    assert "".join(stream_reply(LOCAL, 0.3, messages)) == "Hello world"

    sent = built[0].sent
    assert sent["model"] == "qwen3"
    assert sent["temperature"] == 0.3
    assert sent["stream"] is True
    assert sent["messages"] == messages
    assert sent["stream_options"] == {"include_usage": True}


def test_the_sdk_receives_the_resolved_url_and_credential(monkeypatch):
    built = fake_openai(monkeypatch)
    make_client(CLOUD)
    assert built[0].kwargs == {
        "base_url": "https://api.example.com/v1",
        "api_key": "sk-secret",
        "timeout": 30,
        "max_retries": 0,
    }


def test_an_identical_transport_identity_reuses_one_client(monkeypatch):
    built = fake_openai(monkeypatch)
    first = make_client(CLOUD)
    same_identity = ResolvedProfile(
        name="renamed",  # the name is not part of the transport identity
        base_url=CLOUD.base_url,
        model="another-model",
        api_key=CLOUD.api_key,
    )
    assert make_client(same_identity) is first
    assert len(built) == 1


def test_two_credentials_for_one_url_never_share_a_client(monkeypatch):
    built = fake_openai(monkeypatch)
    first = make_client(CLOUD)
    second = make_client(
        ResolvedProfile(
            name="cloud-alt",
            base_url=CLOUD.base_url,
            model=CLOUD.model,
            api_key="sk-other",
        )
    )
    assert first is not second
    assert len(built) == 2
    assert built[0].kwargs["api_key"] == "sk-secret"
    assert built[1].kwargs["api_key"] == "sk-other"


def test_the_model_comes_only_from_the_profile(monkeypatch):
    """No 'default' fallback remains: an empty model is sent as empty."""
    built = fake_openai(monkeypatch, [_chunk("x")])
    empty_model = ResolvedProfile(
        name="local", base_url="http://host:9000/v1", model="", api_key="none"
    )
    list(stream_reply(empty_model, 0.7, []))
    assert built[0].sent["model"] == ""


def test_stream_options_failure_retries_without_them(monkeypatch):
    built = fake_openai(monkeypatch, [_chunk("hi")], fail_on_stream_options=True)
    assert "".join(stream_reply(LOCAL, 0.7, [])) == "hi"

    client = built[0]
    assert "stream_options" in client.calls[0]
    assert "stream_options" not in client.calls[1]
    assert client.calls[1]["model"] == "qwen3"
    assert client.calls[1]["temperature"] == 0.7


def test_stream_reply_emits_usage_from_the_choiceless_final_chunk(monkeypatch):
    """The usage chunk has no choices, so it must be read before that guard."""
    fake_openai(monkeypatch, [_chunk("hi"), _usage_chunk(4242)])
    items = list(stream_reply(LOCAL, 0.7, []))
    assert ("usage", {"prompt_tokens": 4242}) in items
    assert "".join(i for i in items if isinstance(i, str)) == "hi"


def test_reasoning_and_timings_events_are_preserved(monkeypatch):
    reasoning = SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=None, reasoning_content="hmm"),
                finish_reason=None,
            )
        ]
    )
    final = SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content="4"), finish_reason="stop"
            )
        ],
        timings={"prompt_n": 12},
    )
    fake_openai(monkeypatch, [reasoning, final])
    items = list(stream_reply(LOCAL, 0.7, []))
    assert ("reasoning", "hmm") in items
    assert ("timings", {"prompt_n": 12}) in items
    assert "".join(i for i in items if isinstance(i, str)) == "4"
