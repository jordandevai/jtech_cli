"""Thin wrapper around an OpenAI-compatible endpoint (llama-server / vLLM)."""

from __future__ import annotations

import threading
from collections.abc import Iterator

from openai import APIStatusError, OpenAI

from jtech_cli.config import ResolvedProfile

# Stream items: plain content deltas (str), or tagged events:
# ("reasoning", text) for thinking tokens, ("usage", dict) for the standard
# OpenAI token counts, and ("timings", dict) for llama.cpp prompt/generation
# stats (prompt_n, prompt_ms, prompt_per_second, ...).
ReasoningEvent = tuple[str, str]
UsageEvent = tuple[str, dict]
TimingsEvent = tuple[str, dict]
StreamItem = str | ReasoningEvent | UsageEvent | TimingsEvent

# Reuse one OpenAI client (connection pool) per transport identity instead of
# building a new one per message. The credential is part of that identity: two
# profiles may share a base URL with different keys, and sharing a client
# between them would send one profile's requests with the other's credential.
_client_cache: dict[tuple[str, str], OpenAI] = {}
# Concurrent agents stream from their own provider threads, so lookup-then-
# create is a race: two threads could each build a client for one identity and
# leave one connection pool orphaned. The lock covers only that lookup and
# construction — never the stream — so agents on distinct endpoints, and on the
# same one, still consume their responses in parallel.
_cache_lock = threading.Lock()


def make_client(profile: ResolvedProfile) -> OpenAI:
    """OpenAI client for ``profile``, cached per (base URL, credential)."""
    key = (profile.base_url, profile.api_key)
    with _cache_lock:
        client = _client_cache.get(key)
        if client is None:
            client = OpenAI(
                base_url=profile.base_url,
                api_key=profile.api_key,
                timeout=30,
                max_retries=0,
            )
            _client_cache[key] = client
    return client


def stream_reply(
    profile: ResolvedProfile,
    temperature: float,
    messages: list[dict],
) -> Iterator[StreamItem]:
    """Yield content deltas from a streaming chat completion.

    ``profile`` is the immutable identity the caller pinned for the whole turn:
    endpoint, model, and credential all come from it, and there is no default
    model to fall back on.

    Content deltas are plain ``str``. Thinking models additionally produce
    ``("reasoning", text)`` events for their reasoning tokens. Servers honouring
    ``include_usage`` emit one ``("usage", dict)`` event carrying
    ``prompt_tokens``, and llama.cpp appends one final ``("timings", dict)``
    event with prompt/generation stats (``prompt_n``, ``prompt_ms``,
    ``prompt_per_second``, ...).
    """
    client = make_client(profile)
    kwargs = {
        "model": profile.model,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
    }
    try:
        kwargs["stream_options"] = {"include_usage": True}
        stream = client.chat.completions.create(**kwargs)
    except (APIStatusError, TypeError):
        # The two ways asking for usage can be refused, and only those: the
        # server answered with an error status because it does not implement
        # ``stream_options``, or the installed SDK is old enough not to accept
        # the argument at all. A connection failure, a timeout, or a bug in
        # this module is not a reason to silently re-request — it now
        # propagates to the caller that knows how to report it.
        stream = client.chat.completions.create(
            model=profile.model,
            messages=messages,
            stream=True,
            temperature=temperature,
        )
    for chunk in stream:
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
