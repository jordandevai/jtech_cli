"""The slash-command menu and the message queue it shares Up/Enter with."""

import threading

from textual.widgets import Input, Static

from jtech_cli.tui import SettingsScreen

from .support import (
    bubbles,
    make_app,
    suggestions_box,
    suggestions_text,
    sync_stream,
    wait_until,
)


async def test_slash_prefix_lists_commands_above_input(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "/"
        await pilot.pause()

        box = suggestions_box(app)
        assert box.display
        text = suggestions_text(app)
        assert "/help" in text
        assert "/set" in text
        assert "/models" in text
        assert "/render" in text


async def test_slash_prefix_filters_matches(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "/se"
        await pilot.pause()

        text = suggestions_text(app)
        assert "/set" in text
        assert "/settings" in text
        assert "/stats" not in text
        assert "/clear" not in text


async def test_plain_text_and_no_match_hide_suggestions(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        box = suggestions_box(app)
        assert not box.display

        inp.value = "hello"
        await pilot.pause()
        assert not box.display

        inp.value = "/definitely-not-a-command"
        await pilot.pause()
        assert not box.display


async def test_arrows_cycle_and_tab_completes(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        box = suggestions_box(app)
        inp.value = "/se"
        await pilot.pause()
        assert suggestions_text(app).startswith("▸ /set")

        await pilot.press("down")
        await pilot.pause()
        assert suggestions_text(app).startswith("  /set")
        assert "/settings" in suggestions_text(app)

        await pilot.press("tab")
        await pilot.pause()
        assert inp.value == "/settings "
        assert not box.display
        assert app.focused is inp
        assert app.session.messages == []  # completion alone submits nothing


async def test_enter_completes_partial_without_submitting(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        box = suggestions_box(app)
        inp.value = "/se"
        await pilot.pause()

        # first Enter completes the highlighted (/set) instead of submitting
        await pilot.press("enter")
        await pilot.pause()
        assert inp.value == "/set "
        assert not box.display
        assert app.session.messages == []


async def test_enter_submits_exact_match_command(tmp_path):
    app = make_app(tmp_path)
    app.session.add("user", "hello")
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        box = suggestions_box(app)
        inp.value = "/clear"
        await pilot.pause()
        assert box.display

        await pilot.press("enter")
        await pilot.pause()
        assert app.session.messages == []
        assert inp.value == ""
        assert not box.display
        assert any("History cleared" in b for b in bubbles(app))


async def test_enter_after_navigating_runs_command_immediately(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "/se"
        await pilot.pause()

        await pilot.press("down")  # highlight /settings
        await pilot.pause()
        await pilot.press("enter")  # runs it, no second Enter needed
        await pilot.pause()
        assert isinstance(app.screen, SettingsScreen)
        assert inp.value == ""


async def test_enter_without_navigating_still_completes(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "/se"
        await pilot.pause()
        await pilot.press("enter")  # no scroll, not exact: complete, don't run
        await pilot.pause()
        assert inp.value == "/set "
        assert not isinstance(app.screen, SettingsScreen)
        assert app.session.messages == []


async def test_up_down_idle_are_noop(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        await pilot.press("up")
        await pilot.press("down")
        await pilot.pause()

        assert not suggestions_box(app).display
        assert app.focused is inp


async def test_esc_hides_suggestions(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        box = suggestions_box(app)
        inp.value = "/help"
        await pilot.pause()
        assert box.display

        await pilot.press("escape")
        await pilot.pause()
        assert not box.display
        assert inp.value == "/help"


async def test_stats_command_in_tui(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    app.session.add("user", "hello world")
    monkeypatch.setattr("jtech_cli.server_info.fetch_token_count", lambda s, t: 3)
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "/stats"
        await pilot.press("enter")
        for _ in range(10):
            await pilot.pause()

        text = "\n".join(bubbles(app))
        assert "messages=1" in text
        assert "history_tokens=3" in text
        assert "context_length=4096" in text
        assert "context_remaining=4093" in text


async def test_enter_while_streaming_queues_then_drains(tmp_path, monkeypatch):
    """Enter during a stream queues the message; it sends once the reply finishes."""
    app = make_app(tmp_path)
    gate = threading.Event()
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            yield "r1 "
            gate.wait(5)
            yield "r1b"
        else:
            yield "r2"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "one"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: any("r1" in b for b in bubbles(app)))
        assert app.generating

        # Second message while generating: queued, not sent
        inp.value = "two"
        await pilot.press("enter")
        await wait_until(
            app,
            pilot,
            lambda: "queue: 1" in app.query_one("#status", Static).content,
            tries=10,
            pause=0.05,
        )
        assert app.session.messages == [{"role": "user", "content": "one"}]
        text = "\n".join(bubbles(app))
        assert "Queued: two" in text

        gate.set()
        await wait_until(app, pilot, lambda: calls["n"] >= 2)
        # the transient "Queued" line is gone once the message sent
        assert not any("Queued" in b for b in bubbles(app))

    assert app.session.messages == [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "r1 r1b"},
        {"role": "user", "content": "two"},
        {"role": "assistant", "content": "r2"},
    ]
    assert app.composer.queue == []


async def test_up_recalls_queued_message_for_editing(tmp_path, monkeypatch):
    """Up with an empty input pulls the next queued message into the input.

    It is not auto-submitted: the user edits (or clears) it, then Enter sends.
    """
    app = make_app(tmp_path)
    gate = threading.Event()
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            yield "r1 "
            gate.wait(5)
            yield "r1b"
        else:
            yield "r2"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "one"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: any("r1" in b for b in bubbles(app)))

        inp.value = "two"
        await pilot.press("enter")
        inp.value = "three"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: len(app.composer.queue) == 2, tries=10, pause=0.05)

        # Up recalls the NEXT queued message ("two"), not the last
        await pilot.press("up")
        await pilot.pause()
        assert inp.value == "two"
        assert [m for m in app.composer.queue] == ["three"]
        assert "queue: 1" in app.query_one("#status", Static).content
        # the recalled message's "Queued" line is cleared; only "three" remains
        assert [b for b in bubbles(app) if "Queued" in b] == ["Queued: three"]
        # not auto-sent
        assert app.session.messages == [{"role": "user", "content": "one"}]

        # Up never clobbers unsent text in the input
        await pilot.press("up")
        await pilot.pause()
        assert inp.value == "two"
        assert [m for m in app.composer.queue] == ["three"]

        # clear it (cancel "two"), then recall the rest
        inp.value = ""
        await pilot.press("up")
        await pilot.pause()
        assert inp.value == "three"
        assert app.composer.queue == []
        assert not any("Queued" in b for b in bubbles(app))  # no stale lines

        # edit it, let the first reply finish, then submit
        inp.value = "three (edited)"
        gate.set()
        await wait_until(app, pilot, lambda: not app.generating)

        await pilot.press("enter")
        await wait_until(app, pilot, lambda: calls["n"] >= 2)

    assert app.session.messages == [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "r1 r1b"},
        {"role": "user", "content": "three (edited)"},
        {"role": "assistant", "content": "r2"},
    ]


async def test_up_with_suggestions_open_prefers_suggestions(tmp_path, monkeypatch):
    """With the command menu open, Up cycles it — it does not recall the queue."""
    app = make_app(tmp_path)
    gate = threading.Event()

    def fake(profile, temperature, messages):
        yield "r1 "
        gate.wait(5)
        yield "r1b"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "one"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: any("r1" in b for b in bubbles(app)))

        inp.value = "two"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: bool(app.composer.queue), tries=10, pause=0.05)

        inp.value = "/"
        await pilot.pause()
        assert suggestions_box(app).display
        await pilot.press("up")
        await pilot.pause()
        assert inp.value == "/"  # suggestion cycled, input untouched
        assert len(app.composer.queue) == 1  # queue untouched
        gate.set()
        await wait_until(app, pilot, lambda: not app.generating)
        assert not any("Queued" in b for b in bubbles(app))  # line cleared on drain


async def test_slash_menu_shows_prompt_commands(tmp_path):
    """The initial slash menu exposes both prompt inspection commands."""
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "/"
        await pilot.pause()

        assert "/system" in str(suggestions_box(app).render())
        assert "/prompt" in str(suggestions_box(app).render())


async def test_queue_drains_in_order(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    gate = threading.Event()
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            yield "r1 "
            gate.wait(5)
            yield "r1b"
        else:
            yield f"r{calls['n']}"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "one"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: any("r1" in b for b in bubbles(app)))

        for msg in ("two", "three"):
            inp.value = msg
            await pilot.press("enter")
        await wait_until(app, pilot, lambda: len(app.composer.queue) == 2, tries=10, pause=0.05)

        gate.set()
        await wait_until(app, pilot, lambda: calls["n"] >= 3)
        assert not any("Queued" in b for b in bubbles(app))

    assert [m["role"] for m in app.session.messages] == [
        "user", "assistant", "user", "assistant", "user", "assistant",
    ]
    assert [m["content"] for m in app.session.messages if m["role"] == "user"] == [
        "one", "two", "three",
    ]
    assert app.composer.queue == []
