"""Thin wrapper around an OpenAI-compatible endpoint (llama-server / vLLM)."""

from __future__ import annotations

from collections.abc import Iterator

from openai import OpenAI

from jtech_cli.config import Settings

# Stream items: plain content deltas (str), or tagged events:
# ("reasoning", text) for thinking tokens, ("usage", dict) for the standard
# OpenAI token counts, and ("timings", dict) for llama.cpp prompt/generation
# stats (prompt_n, prompt_ms, prompt_per_second, ...).
ReasoningEvent = tuple[str, str]
UsageEvent = tuple[str, dict]
TimingsEvent = tuple[str, dict]
StreamItem = str | ReasoningEvent | UsageEvent | TimingsEvent

# Reuse one OpenAI client (connection pool) per base_url instead of building a
# new one per message. A changed base_url naturally gets its own entry.
_client_cache: dict[str, OpenAI] = {}


def make_client(settings: Settings) -> OpenAI:
    """OpenAI client for ``settings.base_url``, cached per base URL."""
    client = _client_cache.get(settings.base_url)
    if client is None:
        client = settings.make_client()
        _client_cache[settings.base_url] = client
    return client


def stream_reply(settings: Settings, messages: list[dict]) -> Iterator[StreamItem]:
    """Yield content deltas from a streaming chat completion.

    Content deltas are plain ``str``. Thinking models additionally produce
    ``("reasoning", text)`` events for their reasoning tokens. Servers honouring
    ``include_usage`` emit one ``("usage", dict)`` event carrying
    ``prompt_tokens``, and llama.cpp appends one final ``("timings", dict)``
    event with prompt/generation stats (``prompt_n``, ``prompt_ms``,
    ``prompt_per_second``, ...).
    """
    client = make_client(settings)
    kwargs = dict(
        model=settings.model or "default",
        messages=messages,
        stream=True,
        temperature=settings.temperature,
    )
    try:
        kwargs["stream_options"] = {"include_usage": True}
        stream = client.chat.completions.create(**kwargs)
    except Exception:
        stream = client.chat.completions.create(
            model=settings.model or "default",
            messages=messages,
            stream=True,
            temperature=settings.temperature,
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
