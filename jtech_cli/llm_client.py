"""Thin wrapper around an OpenAI-compatible endpoint (llama-server / vLLM)."""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator
from typing import Protocol

from openai import APIStatusError, AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletionChunk

from jtech_cli.config import ResolvedProfile

# Stream items: plain content deltas (str), or tagged events:
# ("reasoning", text) for thinking tokens, ("usage", dict) for the standard
# OpenAI token counts, and ("timings", dict) for llama.cpp prompt/generation
# stats (prompt_n, prompt_ms, prompt_per_second, ...).
ReasoningEvent = tuple[str, str]
UsageEvent = tuple[str, dict]
TimingsEvent = tuple[str, dict]
StreamItem = str | ReasoningEvent | UsageEvent | TimingsEvent


class ReplyStream(Protocol):
    """Asynchronous reply items, with an awaitable close for the response.

    Asynchronous because that is the only cancellation boundary this transport
    actually has. A synchronous read of an SSE response parks in the socket,
    and closing the response from another thread does not wake it — the close
    only takes effect once the read returns on its own, which is exactly what
    a stopped generation is waiting to avoid. Awaiting each item instead makes
    the read a cancellable await, and cancelling it disconnects the response.

    These two operations are all the runtime depends on; the adapter behind
    them owns whatever its SDK calls closing a response.
    """

    def __aiter__(self) -> AsyncIterator[StreamItem]: ...

    async def aclose(self) -> None:
        """Close the provider response and release its connection."""
        ...


#: The argument that asks a server to report token usage on a stream. Named
#: once because the retry below has to recognise a refusal *of this argument*
#: in an error, and matching on a literal spelled twice is how that drifts.
USAGE_OPTION = "stream_options"


def refuses_usage_option(error: APIStatusError | TypeError) -> bool:
    """Whether ``error`` says specifically that `USAGE_OPTION` was refused.

    Exactly two things count, because exactly two things can refuse it:

    - a ``TypeError`` naming it, from an SDK too old to accept the argument at
      all — the message CPython builds for an unexpected keyword carries the
      keyword; and
    - a rejected-request status (400/422) that identifies the argument, either
      in the structured ``param`` field or by naming it in the message or body.

    Everything else is somebody else's failure. An authentication error, a rate
    limit, and a server fault say nothing about the request body: replaying the
    request without usage would spend a second call, and — if that one happened
    to succeed — hide the first failure completely. A rate limit answered by an
    immediate retry is worse than that.

    The cost of this precision is stated rather than hidden: a server that
    rejects the argument with a 400 naming nothing is indistinguishable from one
    rejecting malformed messages, so it gets no retry and the error propagates.

    Args:
        error: The exception raised by the usage-carrying request.

    Returns:
        True when the retry without `USAGE_OPTION` is worth making.
    """
    if isinstance(error, TypeError):
        return USAGE_OPTION in str(error)
    if error.status_code not in (400, 422):
        return False
    if error.param == USAGE_OPTION:
        return True
    return USAGE_OPTION in (error.message or "") or USAGE_OPTION in str(error.body or "")


# Reuse one OpenAI client (connection pool) per transport identity instead of
# building a new one per message. The credential is part of that identity: two
# profiles may share a base URL with different keys, and sharing a client
# between them would send one profile's requests with the other's credential.
_client_cache: dict[tuple[str, str], AsyncOpenAI] = {}
# Lookup-then-create is a race for any caller that is not the event loop, and
# losing it orphans a connection pool. The lock covers only that lookup and
# construction — never the stream — so agents on distinct endpoints, and on the
# same one, still consume their responses concurrently.
_cache_lock = threading.Lock()


def make_client(profile: ResolvedProfile) -> AsyncOpenAI:
    """Async OpenAI client for ``profile``, cached per (base URL, credential).

    The cached client's connection pool belongs to the event loop that first
    used it. The app runs one loop for its whole life, so that is the pool
    every agent shares; a process that ran a second loop would need its own
    cache, and tests that create a loop per case clear this one.
    """
    key = (profile.base_url, profile.api_key)
    with _cache_lock:
        client = _client_cache.get(key)
        if client is None:
            client = AsyncOpenAI(
                base_url=profile.base_url,
                api_key=profile.api_key,
                timeout=30,
                max_retries=0,
            )
            _client_cache[key] = client
    return client


class _OpenAIReplyStream:
    """One OpenAI-compatible streaming response, as `StreamItem` events.

    Wraps the SDK's async stream so that every read is an await. The consumer
    cancels the task doing the awaiting; the response is closed on the way out.
    """

    def __init__(self, source: AsyncStream[ChatCompletionChunk]) -> None:
        self._source = source

    def __aiter__(self) -> AsyncIterator[StreamItem]:
        return self._events()

    async def _events(self) -> AsyncIterator[StreamItem]:
        """Yield every event one chunk carries, in the order it carries them."""
        async for chunk in self._source:
            # Usage first. The chunk carrying it has an empty ``choices`` list —
            # that is the shape ``include_usage`` asks for — so reading it after
            # the guard below made the event unreachable on compliant servers.
            usage = getattr(chunk, "usage", None)
            if usage is not None and usage.prompt_tokens:
                yield ("usage", {"prompt_tokens": usage.prompt_tokens})
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                yield ("reasoning", reasoning)
            if delta and delta.content:
                yield delta.content
            if getattr(choice, "finish_reason", None) is not None:
                timings = getattr(chunk, "timings", None)
                if isinstance(timings, dict):
                    yield ("timings", timings)

    async def aclose(self) -> None:
        """Close this response, and only this one.

        Never the client: clients are cached and shared by transport identity,
        so closing one would take every other agent's connection pool with it.
        Disconnecting is what tells the server to stop generating; a server may
        ignore it, but the connection is released either way.
        """
        await self._source.close()


async def stream_reply(
    profile: ResolvedProfile,
    temperature: float,
    messages: list[dict],
) -> ReplyStream:
    """Open a streaming chat completion and return it as a `ReplyStream`.

    ``profile`` is the immutable identity the caller pinned for the whole turn:
    endpoint, model, and credential all come from it, and there is no default
    model to fall back on.

    The request is made here, not on first iteration, so the caller holds a
    cancellable response as soon as this returns.

    Content deltas are plain ``str``. Thinking models additionally produce
    ``("reasoning", text)`` events for their reasoning tokens. Servers honouring
    ``include_usage`` emit one ``("usage", dict)`` event carrying
    ``prompt_tokens``, and llama.cpp appends one final ``("timings", dict)``
    event with prompt/generation stats (``prompt_n``, ``prompt_ms``,
    ``prompt_per_second``, ...).
    """
    client = make_client(profile)
    request = {
        "model": profile.model,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
    }
    try:
        stream = await client.chat.completions.create(
            **request, stream_options={"include_usage": True}
        )
    except (APIStatusError, TypeError) as error:
        if not refuses_usage_option(error):
            raise
        # The retry differs from the request above in exactly one argument, so
        # it is only worth making when the failure was about that argument.
        stream = await client.chat.completions.create(**request)
    return _OpenAIReplyStream(stream)
