"""The transcript at app level: startup history, compaction, reflow, scrolling."""

import threading

from textual.widgets import Input, Markdown, Static
from textual.widgets.markdown import MarkdownStream

from jtech_cli.cmd_tools import CmdPolicy
from jtech_cli.session import Session
from jtech_cli.tui import CONNECTION_ERROR, ChatApp
from jtech_cli.tui_app import RENDER_ERROR
from jtech_cli.tui_widgets import Transcript, TranscriptHistory

from .support import (
    BlockingStream,
    FiniteStream,
    at_bottom,
    body_widgets,
    bubbles,
    chat_of,
    command_call,
    history_lines,
    labels,
    local_settings,
    make_app,
    make_app_with_cmd,
    stream_of,
    sync_stream,
    wait_until,
)


def history_app(tmp_path, messages, settings=None):
    """An app whose session already holds ``messages``, with no network I/O."""
    session = Session(tmp_path / "s.jsonl", persist=False)
    session.messages = [dict(message) for message in messages]
    return make_app(
        tmp_path,
        settings=settings,
        session=session,
        fetch_token_count_fn=lambda s, text: 42,
    )


async def test_startup_renders_every_stored_message_in_order(tmp_path):
    app = history_app(
        tmp_path,
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"},
        ],
    )
    async with app.run_test() as pilot:
        await pilot.pause()

        assert bubbles(app) == ["first", "second", "third"]
        assert labels(app) == ["USER", "ASSISTANT", "USER"]


async def test_startup_renders_multiline_user_message_as_literal_rows(tmp_path):
    """A reloaded session needs no migration: the format is chosen at rebuild."""
    stored = [
        {"role": "user", "content": "first **literal**\n\nsecond"},
        {"role": "assistant", "content": "**formatted answer**"},
    ]
    app = history_app(tmp_path, stored)
    async with app.run_test() as pilot:
        await pilot.pause()

        assert app.session.messages == [
            {"role": "user", "content": "first **literal**\n\nsecond"},
            {"role": "assistant", "content": "**formatted answer**"},
        ]
        chat = chat_of(app)
        user, assistant = chat.history.records
        assert (user.format, user.content) == ("plain", stored[0]["content"])
        assert (assistant.format, assistant.content) == (
            "markdown",
            stored[1]["content"],
        )

        rows = [row.rstrip() for row in history_lines(app).split("\n")]
        first = rows.index("  first **literal**")
        assert rows[first + 1] == ""  # the source blank line keeps its row
        assert rows[first + 2] == "  second"
        rendered = "\n".join(rows)
        assert "formatted answer" in rendered
        assert "**formatted answer**" not in rendered

        # Replay still costs no widget per stored message.
        assert list(chat.children) == [chat.history]


async def test_startup_filters_debug_only_history_unless_debugging(tmp_path):
    stored = [
        {"role": "user", "content": "kept"},
        {"role": "system", "content": "audit", "_debug_only": True},
        {"role": "assistant", "content": "reply"},
    ]

    app = history_app(tmp_path, stored)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert bubbles(app) == ["kept", "reply"]

    debugging = local_settings(debug_level="system")
    app = history_app(tmp_path, stored, settings=debugging)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert bubbles(app) == ["kept", "audit", "reply"]


async def test_startup_mounts_no_widget_per_stored_message(tmp_path):
    """Replaying history costs no label or body widget, however long it is."""
    app = history_app(
        tmp_path,
        [{"role": "user", "content": f"stored {index}"} for index in range(20)],
    )
    async with app.run_test() as pilot:
        await pilot.pause()

        chat = chat_of(app)
        assert list(chat.children) == [chat.history]
        assert body_widgets(app) == []
        assert len(chat.history.records) == 20

        # a completed message still lands at the end and follows it
        app.push_message("system", "live")
        await pilot.pause()

        assert list(chat.children) == [chat.history]
        assert bubbles(app)[-1] == "live"
        assert at_bottom(chat)


async def test_startup_with_no_history_renders_one_empty_history(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()

        chat = chat_of(app)
        assert list(chat.children) == [chat.history]
        assert chat.history.records == ()
        assert chat.history._lines == []
        assert body_widgets(app) == []


MARKDOWN_SAMPLE = (
    "# Heading {index}\n\n"
    "Text with *emphasis*, **strong**, and `inline_code`.\n\n"
    "- one\n- two\n\n"
    "| left | right |\n| --- | --- |\n| a | b |\n\n"
    "```python\nvalue = {index}\n```\n\n"
    "Unicode: café 日本語 — [link](https://example.com/{index})\n"
)


def markdown_history(count: int) -> list[dict]:
    """``count`` stored messages, each exercising every Markdown feature."""
    return [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": MARKDOWN_SAMPLE.format(index=index),
        }
        for index in range(count)
    ]


def transcript_split(app: ChatApp) -> tuple[list[str], list[str]]:
    """(completed record contents, live tail body contents), in order."""
    chat = chat_of(app)
    return (
        [record.content for record in chat.history.records],
        [
            str(entry.body.render())
            if isinstance(entry.body, Static)
            else entry.body._markdown
            for entry in chat._tail
        ],
    )


def spy_history_renders(monkeypatch) -> list:
    """Collect every record the completed-history widget renders from now on."""
    rendered = []
    real = TranscriptHistory._render_record

    def counted(self, record, width):
        rendered.append(record)
        return real(self, record, width)

    monkeypatch.setattr(TranscriptHistory, "_render_record", counted)
    return rendered


async def test_markdown_rich_startup_mounts_one_history_widget(tmp_path):
    stored = markdown_history(30)
    app = history_app(tmp_path, stored)
    async with app.run_test() as pilot:
        await pilot.pause()

        chat = chat_of(app)
        assert list(chat.children) == [chat.history]
        assert not chat.query(Markdown)
        assert body_widgets(app) == []
        assert [r.content for r in chat.history.records] == [
            m["content"] for m in stored
        ]
        assert [r.role for r in chat.history.records] == [m["role"] for m in stored]

        body = history_lines(app)
        for index in range(30):
            assert f"Heading {index}" in body
            assert f"value = {index}" in body


async def test_every_history_shape_has_the_same_transcript_dom(tmp_path):
    shapes = {}
    for name, stored in (
        ("empty", []),
        ("simple", [{"role": "user", "content": f"line {i}"} for i in range(20)]),
        ("markdown", markdown_history(20)),
    ):
        app = history_app(tmp_path, stored)
        async with app.run_test() as pilot:
            await pilot.pause()
            chat = chat_of(app)
            shapes[name] = (len(list(chat.children)), len(chat.query("*")))

    assert len(set(shapes.values())) == 1, shapes


async def test_typing_with_a_long_history_neither_mounts_nor_re_renders(
    tmp_path, monkeypatch
):
    """Input is the hot path: it must not touch completed history at all."""
    app = history_app(tmp_path, markdown_history(30))
    async with app.run_test() as pilot:
        await pilot.pause()
        chat = chat_of(app)
        before_nodes = len(chat.query("*"))

        rendered = spy_history_renders(monkeypatch)
        await pilot.press("h", "e", "l", "l", "o")
        await pilot.pause()

        assert app.query_one("#input", Input).value == "hello"
        assert rendered == []
        assert len(chat.query("*")) == before_nodes
        assert list(chat.children) == [chat.history]


async def test_one_live_answer_leaves_no_body_widget_behind(tmp_path, monkeypatch):
    gate = threading.Event()

    def fake(profile, temperature, messages):
        yield "partial "
        gate.wait(5)
        yield "done"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        app.query_one("#input", Input).value = "go"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: any("partial" in b for b in bubbles(app)))

        chat = chat_of(app)
        assert len(chat.query(Markdown)) == 1  # exactly the live answer

        gate.set()
        await wait_until(app, pilot, lambda: not app._generating, tries=100)
        await pilot.pause()

        assert list(chat.children) == [chat.history]
        assert not chat.query(Markdown)
        assert body_widgets(app) == []
        assert [r.content for r in chat.history.records] == ["go", "partial done"]


async def test_repeated_tool_rounds_do_not_accumulate_body_widgets(
    tmp_path, monkeypatch
):
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="auto", allow=["echo:*"]))
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        calls["n"] += 1
        if calls["n"] <= 4:
            yield command_call(f"echo round-{calls['n']}")
        else:
            yield "finished"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        app.query_one("#input", Input).value = "go"
        await pilot.press("enter")
        await wait_until(
            app, pilot, lambda: any("finished" in b for b in bubbles(app)), tries=200
        )
        await pilot.pause()

        chat = chat_of(app)
        assert calls["n"] == 5
        assert list(chat.children) == [chat.history]
        assert body_widgets(app) == []

        body = history_lines(app)
        for index in range(1, 5):
            assert f"round-{index}" in body


async def test_a_queue_notice_holds_finished_messages_in_order(tmp_path, monkeypatch):
    """A drained turn waits behind the notice still on screen, then compacts.

    Two queued messages are what it takes: draining the first mounts its user
    message and answer *after* the second notice, so they cannot move into
    completed history until that notice goes.
    """
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
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "one"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: any("r1" in b for b in bubbles(app)))

        for queued in ("two", "three"):
            inp.value = queued
            await pilot.press("enter")
        await wait_until(app, pilot, lambda: len(app._queue) == 2, tries=20, pause=0.05)

        # The blocked window opens and closes inside one drain, so it is
        # sampled where it happens: as each answer finalizes.
        snapshots = []
        batches = []
        real_finalize = Transcript.finalize
        real_extend = TranscriptHistory.extend

        def finalize(self, entry, record):
            real_finalize(self, entry, record)
            snapshots.append(transcript_split(app))

        def extend(self, records):
            batch = list(records)
            batches.append([record.content for record in batch])
            return real_extend(self, batch)

        monkeypatch.setattr(Transcript, "finalize", finalize)
        monkeypatch.setattr(TranscriptHistory, "extend", extend)

        gate.set()
        await wait_until(app, pilot, lambda: calls["n"] >= 3, tries=100)
        await wait_until(app, pilot, lambda: not app._generating, tries=100)
        await pilot.pause()

        # the first answer is before both notices, so it compacts straight away
        assert (["one", "r1 r1b"], ["Queued: two", "Queued: three"]) in snapshots
        # the drained turn lands behind the notice still shown, and waits there
        assert (
            ["one", "r1 r1b"],
            ["Queued: three", "two", "r2"],
        ) in snapshots
        # and both records reach history together once that notice goes
        assert ["two", "r2"] in batches

        chat = chat_of(app)
        assert [r.content for r in chat.history.records] == [
            "one",
            "r1 r1b",
            "two",
            "r2",
            "three",
            "r3",
        ]
        assert chat._tail == []
        assert list(chat.children) == [chat.history]


async def test_a_visible_error_compacts_without_entering_session_context(
    tmp_path, monkeypatch
):
    def failing(profile, temperature, messages):
        yield "partial "
        raise RuntimeError("boom")

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(failing))
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        app.query_one("#input", Input).value = "go"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: not app._generating, tries=100)
        await pilot.pause()

        chat = chat_of(app)
        errors = [record for record in chat.history.records if record.error]
        assert len(errors) == 1
        assert CONNECTION_ERROR in errors[0].content and "boom" in errors[0].content
        assert errors[0].display_label == "AI"
        assert CONNECTION_ERROR in history_lines(app)
        assert list(chat.children) == [chat.history]
        assert body_widgets(app) == []
        assert app.session.messages == [{"role": "user", "content": "go"}]


async def test_clear_during_a_gated_stream_stays_empty_after_it_finishes(
    tmp_path, monkeypatch
):
    gate = threading.Event()

    def fake(profile, temperature, messages):
        yield "partial "
        gate.wait(5)
        yield "late"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: any("partial" in b for b in bubbles(app)))

        inp.value = "/clear"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: app.session.messages == [], tries=50)

        gate.set()
        await wait_until(app, pilot, lambda: not app._generating, tries=100)
        for _ in range(5):
            await pilot.pause()

        chat = chat_of(app)
        assert [r.content for r in chat.history.records] == ["History cleared.\n"]
        assert "partial" not in history_lines(app)
        assert "late" not in history_lines(app)
        assert list(chat.children) == [chat.history]
        # session semantics are untouched: the completed reply is still stored
        assert app.session.messages == [
            {"role": "assistant", "content": "partial late"}
        ]


async def test_resizing_reflows_completed_markdown_without_losing_content(tmp_path):
    app = history_app(tmp_path, markdown_history(5))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        chat = chat_of(app)
        wide_width = chat.history._render_width
        wide = history_lines(app)

        await pilot.resize_terminal(50, 24)
        await pilot.pause()
        narrow_width = chat.history._render_width
        narrow = history_lines(app)

        assert narrow_width < wide_width
        assert narrow != wide
        assert all(len(line) <= narrow_width for line in narrow.split("\n"))
        for index in range(5):
            assert f"Heading {index}" in narrow
            assert f"value = {index}" in narrow

        await pilot.resize_terminal(100, 24)
        await pilot.pause()

        assert chat.history._render_width == wide_width
        assert history_lines(app) == wide


async def test_a_theme_switch_reflows_completed_history_once(tmp_path, monkeypatch):
    gate = threading.Event()

    def fake(profile, temperature, messages):
        yield "streaming "
        gate.wait(5)
        yield "done"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    # Pinned dark rather than auto: the switch below has to be a real change,
    # whatever background the terminal running the suite reports.
    app = history_app(
        tmp_path,
        [{"role": "user", "content": "stored"}],
        settings=local_settings(theme="dark"),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.theme == "jtech-dark"
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await wait_until(
            app, pilot, lambda: any("streaming" in b for b in bubbles(app))
        )

        chat = chat_of(app)
        live = chat.query(Markdown)[0]
        before_style = live.rich_style
        before_records = chat.history.records

        reflows = []
        real_reflow = TranscriptHistory.reflow

        def counted(self):
            reflows.append(self)
            return real_reflow(self)

        monkeypatch.setattr(TranscriptHistory, "reflow", counted)

        inp.value = "/theme light"
        await pilot.press("enter")
        await pilot.pause()

        assert app.theme == "jtech-light"
        assert len(reflows) == 1
        assert chat.history._render_theme == "jtech-light"
        assert chat.history.records == before_records
        assert "stored" in history_lines(app)  # completed content survived
        assert live.rich_style != before_style  # CSS repainted the live bubble

        # the same theme again is not a change, so it costs no second reflow
        inp.value = "/theme light"
        await pilot.press("enter")
        await pilot.pause()
        assert len(reflows) == 1

        gate.set()
        await wait_until(app, pilot, lambda: not app._generating, tries=100)


async def test_scrolling_to_the_top_survives_later_chunks_and_compaction(
    tmp_path, monkeypatch
):
    gate = threading.Event()

    def fake(profile, temperature, messages):
        yield "first paragraph\n\n" * 30
        gate.wait(5)
        yield "second paragraph\n\n" * 30

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    app = make_app(tmp_path)
    async with app.run_test(size=(80, 10)) as pilot:
        app.query_one("#input", Input).value = "go"
        await pilot.press("enter")
        chat = chat_of(app)
        await wait_until(app, pilot, lambda: chat.max_scroll_y > 5, tries=100)

        chat.scroll_to(y=0, animate=False, immediate=True)
        await pilot.pause()
        assert chat.scroll_offset.y == 0

        gate.set()
        await wait_until(app, pilot, lambda: not app._generating, tries=100)
        for _ in range(5):
            await pilot.pause()

        assert chat.scroll_offset.y == 0
        assert not at_bottom(chat)
        assert list(chat.children) == [chat.history]  # the answer really compacted
        assert "second paragraph" in history_lines(app)


async def test_a_failing_markdown_write_ends_the_turn_instead_of_wedging_it(
    tmp_path, monkeypatch
):
    """A broken renderer must not latch _generating and strand every later send.

    Rendering moved onto the event loop, so there is no longer a provider
    thread to catch it: without cleanup around the batch loop the spinner keeps
    ticking, the reply is never recorded, and `_send_message` queues forever.
    """
    app = make_app(tmp_path)
    real_write = MarkdownStream.write
    calls = {"n": 0}

    async def failing_write(self, markdown_fragment: str) -> None:
        calls["n"] += 1
        raise RuntimeError("renderer failed")

    monkeypatch.setattr(MarkdownStream, "write", failing_write)
    monkeypatch.setattr("jtech_cli.tui.stream_reply", stream_of("unrenderable"))
    async with app.run_test() as pilot:
        app.query_one("#input", Input).value = "go"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: not app._generating, tries=100)

        assert calls["n"] == 1
        assert app._exception is None
        assert not app._generating
        assert not app._tool_rounds_active
        assert app._queue == []
        assert any(RENDER_ERROR in b and "renderer failed" in b for b in bubbles(app))
        # a partially rendered reply is not passed off as the model's turn
        assert app.session.messages == [{"role": "user", "content": "go"}]

        # the spinner timer is stopped: the final label is not overwritten
        assert labels(app)[-1] == "AI"
        for _ in range(15):
            await pilot.pause(0.1)
        assert labels(app)[-1] == "AI"

        # and the app is still usable: the next message goes all the way through
        monkeypatch.setattr(MarkdownStream, "write", real_write)
        monkeypatch.setattr("jtech_cli.tui.stream_reply", stream_of("ok"))
        app.query_one("#input", Input).value = "second"
        await pilot.press("enter")
        await wait_until(
            app, pilot, lambda: any("ok" in b for b in bubbles(app)), tries=100
        )

        assert app.session.messages == [
            {"role": "user", "content": "go"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "ok"},
        ]


async def test_a_render_failure_closes_its_response_before_the_next_turn(
    tmp_path, monkeypatch
):
    """The one exit that never sees the stream end must still close it.

    The provider would hold this response open indefinitely, so a render
    failure that merely stopped reading would leak the connection and let the
    next turn open a second, overlapping request. The task ends and its
    ``finally`` closes the response instead.
    """
    app = make_app(tmp_path)
    real_write = MarkdownStream.write
    blocked = BlockingStream("one")
    entries: list[str] = []
    closed_at_second_entry: list[bool] = []

    async def provider(profile, temperature, messages):
        entries.append(f"turn-{len(entries) + 1}")
        if len(entries) > 1:
            # Read in the second request itself: the abandoned response must
            # already be closed, not merely abandoned.
            closed_at_second_entry.append(blocked.closed == 1)
            return FiniteStream("ok")
        return blocked

    async def failing_write(self, markdown_fragment: str) -> None:
        raise RuntimeError("renderer failed")

    monkeypatch.setattr(MarkdownStream, "write", failing_write)
    monkeypatch.setattr("jtech_cli.tui.stream_reply", provider)
    async with app.run_test() as pilot:
        app.query_one("#input", Input).value = "go"
        await pilot.press("enter")
        await wait_until(
            app, pilot, lambda: any(RENDER_ERROR in b for b in bubbles(app)), tries=100
        )
        await wait_until(app, pilot, lambda: not app._generating, tries=100)

        assert blocked.closed == 1  # the response was closed, not just dropped
        assert entries == ["turn-1"]

        monkeypatch.setattr(MarkdownStream, "write", real_write)
        app.query_one("#input", Input).value = "second"
        await pilot.press("enter")
        await wait_until(
            app,
            pilot,
            lambda: any("ok" in b for b in bubbles(app)) and not app._generating,
            tries=100,
        )

        assert entries == ["turn-1", "turn-2"]  # the requests never overlapped
        assert closed_at_second_entry == [True]
        assert not app._generating
        assert app._queue == []
        assert app.session.messages == [
            {"role": "user", "content": "go"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "ok"},
        ]
