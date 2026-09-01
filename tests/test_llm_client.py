"""Unit tests for the streaming LLM client (mocked OpenAI endpoint)."""

import threading
from types import SimpleNamespace

import httpx2
import pytest
from openai import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
    UnprocessableEntityError,
)

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


async def items_of(stream):
    """Every `StreamItem` an opened stream produces, in order."""
    return [item async for item in stream]


def _chunk(content):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])


def _empty_chunk():
    return SimpleNamespace(choices=[])


def _usage_chunk(prompt_tokens):
    """The final chunk ``include_usage`` produces: usage set, choices empty."""
    return SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=prompt_tokens))


#: The SDK parses ``status_code``, ``param``, and ``body`` out of a real
#: response, so errors here are built through it rather than asserted into
#: existence. That needs the HTTP types openai vendors — ``httpx2`` for 3.x. A
#: future SDK changing transports breaks this import loudly, which is the point:
#: a hand-built stand-in would keep passing while proving nothing.
_REQUEST = httpx2.Request("POST", "http://host:9000/v1/chat/completions")


def api_error(cls, status, *, message="refused", body=None):
    """One genuine SDK error, with the fields the production predicate reads."""
    return cls(message, response=httpx2.Response(status, request=_REQUEST), body=body)


def unsupported_param_error():
    """A 400 shaped the way a server naming the offending argument sends one."""
    return api_error(
        BadRequestError,
        400,
        message="Unrecognized request argument supplied: stream_options",
        body={
            "message": "Unrecognized request argument supplied: stream_options",
            "type": "invalid_request_error",
            "param": "stream_options",
            "code": None,
        },
    )


class _SDKStream:
    """One SDK completion stream: async-iterable, with a counted ``close()``.

    Shaped like ``openai.AsyncStream``, which is what production wraps so that
    every read is a cancellable await.
    """

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.closed = 0

    def __aiter__(self):
        return self._items()

    async def _items(self):
        for chunk in self._chunks:
            yield chunk

    async def close(self):
        self.closed += 1


def fake_openai(monkeypatch, chunks=(), fail_on_stream_options=False, raises=None):
    """Replace the SDK constructor; return the list of clients it builds.

    ``fail_on_stream_options`` refuses only the usage request, as a server or an
    old SDK does; pass an exception to choose which refusal. ``raises`` fails
    every call, standing in for a transport that is down.

    Each client records the streams it opened in ``streams`` and counts its own
    ``close()`` in ``closed``: cancelling one response must never take the
    shared, cached client down with it.
    """
    built = []

    class _Client:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.sent = None
            self.calls = []
            self.streams = []
            self.closed = 0
            built.append(self)

        @property
        def chat(self):
            return self

        @property
        def completions(self):
            return self

        async def close(self):
            self.closed += 1

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            if raises is not None:
                raise raises
            if fail_on_stream_options and "stream_options" in kwargs:
                raise (
                    TypeError("stream_options unsupported")
                    if fail_on_stream_options is True
                    else fail_on_stream_options
                )
            self.sent = kwargs
            stream = _SDKStream(chunks)
            self.streams.append(stream)
            return stream

    monkeypatch.setattr(llm_client, "AsyncOpenAI", _Client)
    return built


async def test_stream_reply_concatenates_deltas(monkeypatch):
    built = fake_openai(
        monkeypatch,
        [_empty_chunk(), _chunk("Hello"), _chunk(" world"), _empty_chunk()],
    )
    messages = [{"role": "user", "content": "hi"}]
    assert "".join(await items_of(await stream_reply(LOCAL, 0.3, messages))) == "Hello world"

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


async def test_the_model_comes_only_from_the_profile(monkeypatch):
    """No 'default' fallback remains: an empty model is sent as empty."""
    built = fake_openai(monkeypatch, [_chunk("x")])
    empty_model = ResolvedProfile(
        name="local", base_url="http://host:9000/v1", model="", api_key="none"
    )
    await items_of(await stream_reply(empty_model, 0.7, []))
    assert built[0].sent["model"] == ""


async def test_stream_options_failure_retries_without_them(monkeypatch):
    built = fake_openai(monkeypatch, [_chunk("hi")], fail_on_stream_options=True)
    assert "".join(await items_of(await stream_reply(LOCAL, 0.7, []))) == "hi"

    client = built[0]
    assert "stream_options" in client.calls[0]
    assert "stream_options" not in client.calls[1]
    assert client.calls[1]["model"] == "qwen3"
    assert client.calls[1]["temperature"] == 0.7


@pytest.mark.parametrize(
    "name, refusal",
    [
        # The structured signal: the server names the argument in `param`.
        ("param names it", unsupported_param_error()),
        # Servers that answer prose instead of a parsed error body.
        (
            "message names it",
            api_error(BadRequestError, 400, message="unknown field stream_options"),
        ),
        (
            "body names it",
            api_error(
                BadRequestError,
                400,
                message="Invalid request",
                body={"error": "stream_options is not supported by this server"},
            ),
        ),
        # 422 is the other rejected-request status in the wild.
        (
            "unprocessable names it",
            api_error(
                UnprocessableEntityError, 422, message="stream_options not allowed"
            ),
        ),
    ],
)
async def test_a_server_refusing_the_usage_option_retries_without_it(monkeypatch, name, refusal):
    """A rejected *request* that identifies the argument is the real trigger."""
    built = fake_openai(monkeypatch, [_chunk("hi")], fail_on_stream_options=refusal)
    assert "".join(await items_of(await stream_reply(LOCAL, 0.7, []))) == "hi", name

    client = built[0]
    assert "stream_options" in client.calls[0]
    assert "stream_options" not in client.calls[1]
    # The retry changes that one argument and nothing else.
    assert {k: v for k, v in client.calls[0].items() if k != "stream_options"} == (
        client.calls[1]
    )


@pytest.mark.parametrize(
    "name, failure",
    [
        ("authentication", api_error(AuthenticationError, 401, message="bad key")),
        ("rate limit", api_error(RateLimitError, 429, message="slow down")),
        ("server fault", api_error(InternalServerError, 500, message="boom")),
        # A rejected request that names nothing: indistinguishable from
        # malformed messages, so it is not retried either. Stated cost of
        # precision, asserted so it cannot change silently.
        ("unattributed 400", api_error(BadRequestError, 400, message="Invalid request body")),
        # A `TypeError` from somewhere other than the missing keyword.
        ("unrelated TypeError", TypeError("messages must be a list")),
    ],
)
async def test_an_unrelated_failure_is_never_retried(monkeypatch, name, failure):
    """Only a refusal *of this argument* justifies a second request.

    Retrying anything else spends a call the caller never asked for and, when
    that call succeeds, hides the original failure completely — which is what
    the broad `except` did for every 401, 429, and 500.
    """
    built = fake_openai(monkeypatch, [_chunk("hi")], fail_on_stream_options=failure)

    with pytest.raises(type(failure)):
        await stream_reply(LOCAL, 0.7, [])

    assert len(built[0].calls) == 1, (name, built[0].calls)


async def test_a_transport_failure_is_not_swallowed_by_the_usage_retry(monkeypatch):
    """The retry exists for a refused argument, not for an endpoint that is down."""
    built = fake_openai(
        monkeypatch, raises=APIConnectionError(request=_REQUEST)
    )

    with pytest.raises(APIConnectionError):
        await stream_reply(LOCAL, 0.7, [])

    assert len(built[0].calls) == 1, built[0].calls


async def test_an_old_sdk_naming_the_keyword_still_retries(monkeypatch):
    """The `TypeError` branch is narrowed to the message that names the keyword."""
    refusal = TypeError("create() got an unexpected keyword argument 'stream_options'")
    built = fake_openai(monkeypatch, [_chunk("hi")], fail_on_stream_options=refusal)
    assert "".join(await items_of(await stream_reply(LOCAL, 0.7, []))) == "hi"
    assert len(built[0].calls) == 2


async def test_stream_reply_emits_usage_from_the_choiceless_final_chunk(monkeypatch):
    """The usage chunk has no choices, so it must be read before that guard."""
    fake_openai(monkeypatch, [_chunk("hi"), _usage_chunk(4242)])
    items = await items_of(await stream_reply(LOCAL, 0.7, []))
    assert ("usage", {"prompt_tokens": 4242}) in items
    assert "".join(i for i in items if isinstance(i, str)) == "hi"


async def test_reasoning_and_timings_events_are_preserved(monkeypatch):
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
    items = await items_of(await stream_reply(LOCAL, 0.7, []))
    assert ("reasoning", "hmm") in items
    assert ("timings", {"prompt_n": 12}) in items
    assert "".join(i for i in items if isinstance(i, str)) == "4"


# ------------------------------------------------------ cancellation


async def test_reply_stream_close_affects_only_its_sdk_stream(monkeypatch):
    """Closing one response must not close the pool every agent shares."""
    built = fake_openai(monkeypatch, [_chunk("hi")])
    first = await stream_reply(LOCAL, 0.7, [])
    client = built[0]

    await first.aclose()

    assert client.streams[0].closed == 1
    assert client.closed == 0

    second = await stream_reply(LOCAL, 0.7, [])
    assert built == [client]  # the same cached client answered both
    assert "".join(await items_of(second)) == "hi"
    assert client.streams[1].closed == 0


async def test_reply_stream_preserves_event_order_across_the_async_move(monkeypatch):
    """One chunk carrying everything still yields in the documented order.

    The chunk-to-``StreamItem`` loop moved from a sync generator function into
    an async adapter method; this is what would notice a reordering.
    """
    everything = SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content="4", reasoning_content="hmm"),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=11),
        timings={"prompt_n": 12},
    )
    fake_openai(monkeypatch, [everything])

    assert await items_of(await stream_reply(LOCAL, 0.7, [])) == [
        ("usage", {"prompt_tokens": 11}),
        ("reasoning", "hmm"),
        "4",
        ("timings", {"prompt_n": 12}),
    ]


# ------------------------------------------------------ concurrent access


def test_concurrent_lookups_of_one_identity_build_exactly_one_client(monkeypatch):
    """Concurrent agents share provider threads through this cache.

    A slow constructor makes the lookup-then-create window wide enough that an
    unsynchronized cache would let several threads each build a client and
    leave every one but the last orphaned.
    """
    built = fake_openai(monkeypatch)
    real = llm_client.AsyncOpenAI

    def slow(**kwargs):
        threading.Event().wait(0.02)
        return real(**kwargs)

    monkeypatch.setattr(llm_client, "AsyncOpenAI", slow)

    start = threading.Barrier(8)
    clients: list[object] = []

    def worker() -> None:
        start.wait(5)
        clients.append(make_client(CLOUD))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert len(built) == 1
    assert len(clients) == 8
    assert all(client is clients[0] for client in clients)


def test_concurrent_lookups_of_distinct_identities_stay_separate(monkeypatch):
    built = fake_openai(monkeypatch)
    results: dict[str, object] = {}

    def worker(profile: ResolvedProfile) -> None:
        results[profile.base_url] = make_client(profile)

    threads = [
        threading.Thread(target=worker, args=(profile,))
        for profile in (LOCAL, CLOUD)
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert len(built) == 2
    assert results[LOCAL.base_url] is not results[CLOUD.base_url]
    assert {client.kwargs["api_key"] for client in built} == {"none", "sk-secret"}
