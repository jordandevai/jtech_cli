"""One streaming turn: reasoning, interruption, timings, batching, failures."""

import asyncio
import threading
import time

from textual.widgets import Input, Static
from textual.widgets.markdown import MarkdownStream

from jtech_cli.config import Settings
from jtech_cli.session import Session
from jtech_cli.tui import CONNECTION_ERROR, ChatApp
from jtech_cli.tui_runtime import INTERRUPTED_RESPONSE, STOPPED_LABEL

from .support import (
    BlockingStream,
    at_bottom,
    body_widgets,
    bubbles,
    chat_of,
    labels,
    local_settings,
    make_app,
    send_and_drain,
    stream_of,
    sync_stream,
    wait_until,
)


def make_settings(reasoning: str) -> Settings:
    return local_settings(reasoning=reasoning)


def reasoning_body_widget(app: ChatApp) -> Static | None:
    """The live reasoning bubble body (plain Static, not the label), if any."""
    for entry in chat_of(app)._tail:
        if "reasoning" in entry.body.classes:
            return entry.body
    return None


def reasoning_bodies(app: ChatApp) -> list[str]:
    """Reasoning text, completed or still live, in chronological order."""
    completed = [
        record.content
        for record in chat_of(app).history.records
        if record.role == "reasoning"
    ]
    live = reasoning_body_widget(app)
    return completed + ([str(live.render())] if live is not None else [])


def reason_stream(profile, temperature, messages):
    return iter([("reasoning", "hmm "), ("reasoning", "ok"), "4"])


async def test_reasoning_default_transient_shown_then_hidden(tmp_path, monkeypatch):
    """Default mode: reasoning streams in its own bubble, removed once the answer starts."""
    app = make_app(tmp_path)
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(reason_stream))
    async with app.run_test() as pilot:
        await send_and_drain(app, pilot, "what is 2+2")

        text = "\n".join(bubbles(app))
        assert "4" in text
        assert "hmm" not in text  # answer bubble never contains reasoning
        assert reasoning_bodies(app) == []  # transient bubble is gone
        assert app.session.messages == [
            {"role": "user", "content": "what is 2+2"},
            {"role": "assistant", "content": "4"},
        ]


async def test_reasoning_hidden_mode_never_shown(tmp_path, monkeypatch):
    app = make_app(tmp_path, settings=make_settings("hide"))
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(reason_stream))
    async with app.run_test() as pilot:
        await send_and_drain(app, pilot, "what is 2+2")

        text = "\n".join(bubbles(app))
        assert "4" in text
        assert "hmm" not in text
        assert not reasoning_bodies(app)
        # no reasoning widget or record anywhere in the transcript
        assert not [w for w in body_widgets(app) if "reasoning" in w.classes]
        assert not [
            r for r in chat_of(app).history.records if r.role == "reasoning"
        ]
        assert app.session.messages == [
            {"role": "user", "content": "what is 2+2"},
            {"role": "assistant", "content": "4"},
        ]


async def test_reasoning_always_kept_in_separate_bubble(tmp_path, monkeypatch):
    app = make_app(tmp_path, settings=make_settings("always"))
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(reason_stream))
    async with app.run_test() as pilot:
        await send_and_drain(app, pilot, "what is 2+2")

        answer = [b for b in bubbles(app) if "4" in b]
        assert answer and all("hmm" not in b for b in answer)
        assert reasoning_bodies(app) == ["hmm ok"]
        assert "REASONING" in labels(app)


async def test_reasoning_tail_caps_at_500_chars(tmp_path, monkeypatch):
    full = "x" * 300 + "tail-marker" + "y" * 900  # 1213 chars
    app = make_app(tmp_path, settings=make_settings("tail"))
    monkeypatch.setattr(
        "jtech_cli.tui.stream_reply", stream_of(("reasoning", full), "4")
    )
    async with app.run_test() as pilot:
        await send_and_drain(app, pilot, "what is 2+2")

        assert reasoning_bodies(app) == ["…" + full[-500:]]
        assert any("4" in b for b in bubbles(app))


async def test_waiting_label_ticks_without_tokens(tmp_path, monkeypatch):
    """The 1s timer repaints the label in real time even with a silent stream."""
    app = make_app(tmp_path)

    def fake(profile, temperature, messages):
        time.sleep(1.5)
        yield "ok"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        await send_and_drain(app, pilot, "hi")
        seen = set()
        for _ in range(40):
            await pilot.pause(0.1)
            for lab in labels(app):
                if "waiting" in lab:
                    seen.add(lab)
        assert any("waiting 1s" in lab for lab in seen)
        assert any("ok" in b for b in bubbles(app))


def ai_label(app: ChatApp) -> Static:
    """The live AI status label; only a streaming or blocked entry has one."""
    for entry in chat_of(app)._tail:
        if "ai" in entry.label.classes:
            return entry.label
    raise AssertionError("no AI label found")


def in_view(chat, widget) -> bool:
    """True when the widget's on-screen extent overlaps the chat viewport.

    ``widget.region`` is in the chat's container space: negative y is above
    the viewport, y >= container height is below it.
    """
    return widget.region.y < chat.container_size.height and (
        widget.region.y + widget.region.height > 0
    )


async def test_chat_follows_streaming_reasoning(tmp_path, monkeypatch):
    """New reasoning lines and the AI status label must stay in view mid-stream.

    The first token is delayed so the post-mount scroll settles first, like in
    a real session: without follow-scroll, the growing reasoning bubble ends
    up below the viewport (hidden until the user scrolls down).
    """
    app = make_app(tmp_path)
    gate = threading.Event()

    def slow_reason(profile, temperature, messages):
        time.sleep(0.4)  # let the mount-time scroll settle before content arrives
        yield ("reasoning", "thinking out loud " * 20)  # ~320 chars -> several lines
        gate.wait(5)
        yield ("reasoning", "more thoughts ")
        yield "4"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(slow_reason))
    async with app.run_test(size=(80, 10)) as pilot:
        await send_and_drain(app, pilot, "what is 2+2")

        chat = app.query_one("#chat")

        # wait for the first reasoning token to land (the mount-time scroll has
        # already settled by then) and for the follow-scroll to apply
        for _ in range(50):
            await pilot.pause(0.1)
            reason_body = reasoning_body_widget(app)
            if (
                reason_body is not None
                and reason_body.display
                and str(reason_body.render()).strip()
                and in_view(chat, ai_label(app))
                and in_view(chat, reason_body)
            ):
                break

        # mid-stream: reasoning is visible AND the AI status label and the
        # *newest* reasoning line (the bottom of the reasoning body) are both
        # inside the viewport
        reason_body = reasoning_body_widget(app)
        assert reason_body is not None and reason_body.display
        assert str(reason_body.render()).strip()
        assert chat.max_scroll_y > 2  # content really overflows the pane
        assert in_view(chat, ai_label(app))
        assert in_view(chat, reason_body)
        assert reason_body.region.y + reason_body.region.height - 1 < chat.container_size.height

        gate.set()
        for _ in range(10):
            await pilot.pause()

        # transient mode: reasoning removed, answer at the bottom in view
        assert reasoning_bodies(app) == []
        assert any("4" in b for b in bubbles(app))
        assert at_bottom(app.query_one("#chat"))


async def test_prompt_timings_shown_in_ai_label(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    monkeypatch.setattr(
        "jtech_cli.tui.stream_reply",
        stream_of("ok", ("timings", {"prompt_n": 170, "prompt_ms": 594.8, "prompt_per_second": 285.8})),
    )
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "hi"
        await pilot.press("enter")
        for _ in range(10):
            await pilot.pause()

        assert any(
            "170" in text and "0.6s" in text and "286 t/s" in text
            for text in labels(app)
        )


async def test_esc_idle_does_nothing(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.press("escape")
        for _ in range(5):
            await pilot.pause()

        assert app.session.messages == []
        assert not bubbles(app)
        assert app.focused is app.query_one("#input", Input)


async def test_esc_stops_stream_and_retains_marked_partial(tmp_path, monkeypatch):
    """Esc closes the response and keeps what the user actually saw.

    The provider is released by nothing but the app's own cancellation, so a
    stop that only set a flag would park this test in the blocked read.
    """
    app = make_app(tmp_path, settings=make_settings("always"))
    stream = BlockingStream()

    async def provider(profile, temperature, messages):
        return stream

    monkeypatch.setattr("jtech_cli.tui.stream_reply", provider)
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "hello"
        await pilot.press("enter")
        await wait_until(app, pilot, stream.blocked.is_set)

        await pilot.press("escape")
        await wait_until(app, pilot, lambda: not app.generating, tries=100)
        await pilot.pause()

        interrupted = f"partial \n\n{INTERRUPTED_RESPONSE}"
        assert stream.cancelled is True  # the parked read was cancelled
        assert stream.closed == 1  # and its response closed on the way out
        assert app.session.messages == [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": interrupted,
                "_model_role": "assistant",
                "_model_content": INTERRUPTED_RESPONSE,
            },
        ]
        assert app.session.messages_with_system("") == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": INTERRUPTED_RESPONSE},
        ]
        assert interrupted in bubbles(app)
        assert STOPPED_LABEL in labels(app)
        assert not any("Generation stopped" in b for b in bubbles(app))
        assert reasoning_bodies(app) == []
        assert app.query_one("#input", Input).disabled is False


async def test_startup_restores_interrupted_partial_but_context_uses_marker(
    tmp_path,
):
    """A reloaded session shows what was stopped and sends only the marker.

    The durability half of the contract: the two representations have to
    survive a restart, not just exist in memory for the rest of the turn.
    """
    path = tmp_path / "s.jsonl"
    written = Session(path)
    written.add("user", "hello")
    written.add(
        "assistant",
        f"partial answer\n\n{INTERRUPTED_RESPONSE}",
        model_role="assistant",
        model_content=INTERRUPTED_RESPONSE,
    )

    # A genuinely separate process would do exactly this and nothing else.
    reloaded = Session(path)
    reloaded.load()
    assert reloaded.messages == written.messages
    assert reloaded.messages is not written.messages

    app = make_app(
        tmp_path, session=reloaded, fetch_token_count_fn=lambda profile, text: 7
    )
    async with app.run_test() as pilot:
        await pilot.pause()

        assert f"partial answer\n\n{INTERRUPTED_RESPONSE}" in bubbles(app)

    # the model-facing override survived the JSONL round trip, not just memory
    assert reloaded.messages[1]["_model_content"] == INTERRUPTED_RESPONSE
    assert app.session.messages_with_system("") == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": INTERRUPTED_RESPONSE},
    ]


def _event_stream(*items):
    """A stream yielding exactly ``items``, counting invocations."""
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        calls["n"] += 1
        yield from items

    return fake, calls


async def test_timings_without_prompt_n_keeps_the_usage_count(tmp_path, monkeypatch):
    """A timings payload carrying no prompt_n must not zero a known count.

    Both events can arrive for one reply, in either order. Whichever carries a
    real number wins; neither may clobber the other with a zero.
    """
    app = make_app(tmp_path)
    fake, calls = _event_stream(
        "hi",
        ("usage", {"prompt_tokens": 8192}),
        ("timings", {"prompt_ms": 431.0}),  # no prompt_n
    )
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        app.query_one("#input", Input).value = "go"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: calls["n"] >= 1, tries=50)
        await wait_until(app, pilot, lambda: not app.generating, tries=50)

        assert app.status.prompt_tokens == 8192


async def test_unknown_stream_event_is_not_treated_as_timings(tmp_path, monkeypatch):
    """A new event kind must not be mistaken for timings and zero the counter."""
    app = make_app(tmp_path)
    fake, calls = _event_stream(
        "hi",
        ("usage", {"prompt_tokens": 512}),
        ("some_future_event", {"whatever": True}),
    )
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        app.query_one("#input", Input).value = "go"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: calls["n"] >= 1, tries=50)
        await wait_until(app, pilot, lambda: not app.generating, tries=50)

        assert app.status.prompt_tokens == 512


async def test_timings_with_prompt_n_still_sets_the_count(tmp_path, monkeypatch):
    """The llama.cpp path is unchanged: a real prompt_n still drives the footer."""
    app = make_app(tmp_path)
    fake, calls = _event_stream("hi", ("timings", {"prompt_n": 2048, "prompt_ms": 12.0}))
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        app.query_one("#input", Input).value = "go"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: calls["n"] >= 1, tries=50)
        await wait_until(app, pilot, lambda: not app.generating, tries=50)

        assert app.status.prompt_tokens == 2048


def record_markdown_writes(monkeypatch, gate: asyncio.Event | None = None) -> list[str]:
    """Record every fragment handed to the Markdown stream, newest last.

    With ``gate``, the first write blocks until the gate is set, so a test can
    hold the consumer open and observe what the provider produces meanwhile.
    """
    writes: list[str] = []
    real_write = MarkdownStream.write

    async def write(self, markdown_fragment: str) -> None:
        writes.append(markdown_fragment)
        if gate is not None and len(writes) == 1:
            await gate.wait()
        await real_write(self, markdown_fragment)

    monkeypatch.setattr(MarkdownStream, "write", write)
    return writes


def record_static_updates(monkeypatch) -> list[tuple[Static, object]]:
    """Record (widget, content) for every Static.update in the app."""
    calls: list[tuple[Static, object]] = []
    real_update = Static.update

    def update(self, content="", *, layout: bool = True) -> None:
        calls.append((self, content))
        real_update(self, content, layout=layout)

    monkeypatch.setattr(Static, "update", update)
    return calls


def reasoning_updates(calls: list[tuple[Static, object]]) -> list[str]:
    return [
        str(content)
        for widget, content in calls
        if "bubble" in widget.classes and "reasoning" in widget.classes
    ]


def ai_label_updates(calls: list[tuple[Static, object]]) -> list[str]:
    return [
        str(content)
        for widget, content in calls
        if "bubble-label" in widget.classes and "ai" in widget.classes
    ]


class GatedBurstStream:
    """Emits one item, waits to be told to continue, then bursts the rest."""

    def __init__(self, first, *rest):
        self.first = first
        self.rest = rest
        self.produced = asyncio.Event()  # set by the test to release the burst
        self.finished = asyncio.Event()
        self.closed = 0

    def __aiter__(self):
        return self._items()

    async def _items(self):
        yield self.first
        await self.produced.wait()
        for item in self.rest:
            yield item
        self.finished.set()

    async def aclose(self) -> None:
        self.closed += 1


async def test_chunks_produced_during_a_blocked_write_are_combined(tmp_path, monkeypatch):
    """Backlog is coalesced into the next awaited write, not one write per token.

    The provider task only queues; the consumer drains. Everything that arrives
    while a write is outstanding therefore rides in the next one.
    """
    app = make_app(tmp_path)
    release = asyncio.Event()
    writes = record_markdown_writes(monkeypatch, gate=release)
    stream = GatedBurstStream("A", "B", "C", "D")

    async def provider(profile, temperature, messages):
        return stream

    monkeypatch.setattr("jtech_cli.tui.stream_reply", provider)
    async with app.run_test() as pilot:
        app.query_one("#input", Input).value = "go"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: writes == ["A"], tries=100)

        # B, C and D are produced while the first write is still outstanding
        stream.produced.set()
        await wait_until(app, pilot, stream.finished.is_set, tries=100)
        release.set()
        await wait_until(app, pilot, lambda: not app.generating, tries=100)

        assert writes == ["A", "BCD"]
        assert app.session.messages[-1] == {"role": "assistant", "content": "ABCD"}


async def test_a_long_stream_costs_far_fewer_writes_than_deltas(tmp_path, monkeypatch):
    """Coalescing is load-bearing: 300 deltas must not cost 300 awaited writes."""
    app = make_app(tmp_path)
    chunks = [f"c{index} " for index in range(300)]
    writes = record_markdown_writes(monkeypatch)
    monkeypatch.setattr("jtech_cli.tui.stream_reply", stream_of(*chunks))
    async with app.run_test() as pilot:
        app.query_one("#input", Input).value = "go"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: not app.generating, tries=100)

        assert "".join(writes) == "".join(chunks)
        assert len(writes) < len(chunks)


async def test_markdown_writes_reproduce_the_provider_content(tmp_path, monkeypatch):
    """Whatever the batching, the writes concatenate back to the source text."""
    app = make_app(tmp_path)
    chunks = [f"chunk-{index} " for index in range(60)]
    writes = record_markdown_writes(monkeypatch)
    monkeypatch.setattr("jtech_cli.tui.stream_reply", stream_of(*chunks))
    async with app.run_test() as pilot:
        app.query_one("#input", Input).value = "go"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: not app.generating, tries=100)

        assert "".join(writes) == "".join(chunks)
        assert app.session.messages[-1]["content"] == "".join(chunks)


async def test_finalization_waits_for_a_blocked_markdown_write(tmp_path, monkeypatch):
    """Nothing is finalized or persisted while a Markdown write is outstanding."""
    app = make_app(tmp_path)
    release = asyncio.Event()
    writes = record_markdown_writes(monkeypatch, gate=release)
    monkeypatch.setattr("jtech_cli.tui.stream_reply", stream_of("held"))
    async with app.run_test() as pilot:
        app.query_one("#input", Input).value = "go"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: writes == ["held"], tries=100)
        for _ in range(5):
            await pilot.pause()

        assert app.generating
        assert app.session.messages == [{"role": "user", "content": "go"}]

        release.set()
        await wait_until(app, pilot, lambda: not app.generating, tries=100)
        assert app.session.messages[-1] == {"role": "assistant", "content": "held"}


async def test_waiting_timer_repaints_only_the_label(tmp_path, monkeypatch):
    """The 1s timer owns the label alone: no Markdown, no reasoning repaint."""
    app = make_app(tmp_path)
    writes = record_markdown_writes(monkeypatch)
    updates = record_static_updates(monkeypatch)
    gate = threading.Event()

    def silent(profile, temperature, messages):
        gate.wait(5)
        yield "spoke at last"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(silent))
    async with app.run_test() as pilot:
        app.query_one("#input", Input).value = "go"
        await pilot.press("enter")
        await wait_until(
            app,
            pilot,
            lambda: len({t for t in ai_label_updates(updates) if "waiting" in t}) >= 2,
            tries=100,
        )
        # The timer has repainted the label more than once and nothing else
        # has been touched, because nothing else has arrived yet.
        assert writes == []
        assert reasoning_updates(updates) == []

        gate.set()
        await wait_until(app, pilot, lambda: not app.generating, tries=100)
        assert writes == ["spoke at last"]


async def test_reasoning_hide_counts_without_mounting_a_bubble(tmp_path, monkeypatch):
    app = make_app(tmp_path, settings=make_settings("hide"))
    updates = record_static_updates(monkeypatch)
    monkeypatch.setattr(
        "jtech_cli.tui.stream_reply",
        stream_of(("reasoning", "hmm "), ("reasoning", "ok"), "4"),
    )
    async with app.run_test() as pilot:
        await send_and_drain(app, pilot, "what is 2+2")

        assert reasoning_updates(updates) == []
        assert not reasoning_bodies(app)
        assert any("4" in b for b in bubbles(app))


async def test_reasoning_always_renders_a_batch_once(tmp_path, monkeypatch):
    """Deltas drained together cost one repaint, not one per delta."""
    app = make_app(tmp_path, settings=make_settings("always"))
    updates = record_static_updates(monkeypatch)
    monkeypatch.setattr(
        "jtech_cli.tui.stream_reply",
        stream_of(("reasoning", "hmm "), ("reasoning", "ok"), "4"),
    )
    async with app.run_test() as pilot:
        await send_and_drain(app, pilot, "what is 2+2")

        assert reasoning_bodies(app) == ["hmm ok"]
        assert reasoning_updates(updates) == ["hmm ok"]  # two deltas, one repaint
        assert any("4" in b for b in bubbles(app))


async def test_reasoning_tail_keeps_only_the_bounded_tail(tmp_path, monkeypatch):
    full = "x" * 300 + "tail-marker" + "y" * 900
    app = make_app(tmp_path, settings=make_settings("tail"))
    updates = record_static_updates(monkeypatch)
    monkeypatch.setattr(
        "jtech_cli.tui.stream_reply",
        stream_of(("reasoning", full[:600]), ("reasoning", full[600:]), "4"),
    )
    async with app.run_test() as pilot:
        await send_and_drain(app, pilot, "what is 2+2")

        assert reasoning_bodies(app) == ["…" + full[-500:]]
        assert reasoning_updates(updates) == ["…" + full[-500:]]
        assert any("4" in b for b in bubbles(app))


async def test_reasoning_transient_drops_the_bubble_when_content_starts(
    tmp_path, monkeypatch
):
    app = make_app(tmp_path)  # transient is the default
    monkeypatch.setattr(
        "jtech_cli.tui.stream_reply",
        stream_of(("reasoning", "hmm "), ("reasoning", "ok"), "4"),
    )
    async with app.run_test() as pilot:
        await send_and_drain(app, pilot, "what is 2+2")

        assert reasoning_bodies(app) == []
        assert "REASONING" not in labels(app)
        text = "\n".join(bubbles(app))
        assert "4" in text and "hmm" not in text


async def test_usage_and_unknown_events_keep_the_usage_count(tmp_path, monkeypatch):
    """A stream carrying usage plus a future event kind still reads as usage."""
    app = make_app(tmp_path)
    monkeypatch.setattr(
        "jtech_cli.tui.stream_reply",
        stream_of(
            ("usage", {"prompt_tokens": 512}),
            "hi",
            ("some_future_event", {"whatever": True}),
        ),
    )
    async with app.run_test() as pilot:
        await send_and_drain(app, pilot, "go")

        assert app.status.prompt_tokens == 512
        assert "AI" in labels(app)  # no timings -> the plain done label


async def test_timings_still_reach_the_done_label(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    monkeypatch.setattr(
        "jtech_cli.tui.stream_reply",
        stream_of(
            "hi",
            ("timings", {"prompt_n": 2048, "prompt_ms": 594.8, "prompt_per_second": 285.8}),
        ),
    )
    async with app.run_test() as pilot:
        await send_and_drain(app, pilot, "go")

        assert app.status.prompt_tokens == 2048
        assert any("2,048" in l and "286 t/s" in l for l in labels(app))


async def test_batched_provider_error_is_reported_after_its_content(tmp_path, monkeypatch):
    """A failure enqueued behind content still lands last and still reports."""
    app = make_app(tmp_path)
    writes = record_markdown_writes(monkeypatch)

    def failing(profile, temperature, messages):
        yield "partial "
        raise RuntimeError("boom")

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(failing))
    async with app.run_test() as pilot:
        app.query_one("#input", Input).value = "go"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: not app.generating, tries=100)

        assert writes == ["partial "]
        assert any(CONNECTION_ERROR in b and "boom" in b for b in bubbles(app))
        assert app.session.messages == [{"role": "user", "content": "go"}]


async def test_manual_scroll_during_a_stream_is_not_overridden(tmp_path, monkeypatch):
    """Scrolling up mid-stream releases the follow; later chunks stay put."""
    app = make_app(tmp_path)
    gate = threading.Event()

    def fake(profile, temperature, messages):
        yield "first paragraph\n\n" * 30
        gate.wait(5)
        yield "second paragraph\n\n" * 30

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test(size=(80, 10)) as pilot:
        app.query_one("#input", Input).value = "go"
        await pilot.press("enter")
        chat = app.query_one("#chat")
        await wait_until(app, pilot, lambda: chat.max_scroll_y > 5, tries=100)

        chat.scroll_to(y=0, animate=False, immediate=True)
        await pilot.pause()
        assert chat.scroll_offset.y == 0

        gate.set()
        await wait_until(app, pilot, lambda: not app.generating, tries=100)
        for _ in range(5):
            await pilot.pause()

        assert chat.scroll_offset.y == 0
        assert not at_bottom(chat)
        assert any("second paragraph" in b for b in bubbles(app))


class _UnwritableSession(Session):
    """A session that keeps messages in memory but always fails to store them."""

    def add(self, role: str, content: str, **kwargs) -> None:
        super().add(role, content, **kwargs)  # persist=False: memory only
        raise OSError("disk full")


async def test_history_save_failure_is_reported_and_generation_continues(
    tmp_path, monkeypatch
):
    """A failed append warns in the transcript without losing the exchange."""
    session = _UnwritableSession(tmp_path / "s.jsonl", persist=False)
    app = make_app(tmp_path, session=session)
    monkeypatch.setattr("jtech_cli.tui.stream_reply", stream_of("ok"))
    async with app.run_test() as pilot:
        app.query_one("#input", Input).value = "hi"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: not app.generating, tries=100)

        assert any("Could not save history: disk full" in b for b in bubbles(app))
        assert any("ok" in b for b in bubbles(app))
        # the exchange is intact for the model, and the warning is not in it
        assert session.messages_with_system("") == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
        ]
