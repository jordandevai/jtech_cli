"""The composer: multiline promotion, paste handling, and editor sizing."""

import pytest
from textual import events
from textual.widgets import Input, TextArea
from textual.widgets.input import Selection as InputSelection
from textual.widgets.text_area import Selection as AreaSelection

from .support import (
    chat_of,
    clear_input_selection,
    enter_multiline,
    history_lines,
    make_app,
    settle,
    stream_of,
    type_text,
)


async def deliver_paste(app, pilot, source: str, text: str) -> None:
    """Paste ``text`` the way the terminal or Ctrl+V would deliver it."""
    if source == "terminal":
        app.post_message(events.Paste(text))
    else:
        app.copy_to_clipboard(text)
        await settle(pilot)
        await pilot.press("ctrl+v")
    await settle(pilot)


def select_input_range(inp: Input, start: int, end: int) -> None:
    """Select ``value[start:end]``; pass start > end for a reverse drag."""
    inp.selection = InputSelection(start, end)


@pytest.mark.parametrize("reverse", [False, True])
async def test_multiline_shift_enter_replaces_the_selection(tmp_path, reverse):
    """Shift+Enter is an edit: the selection becomes the newline, undoably."""
    app = make_app(tmp_path)
    async with app.run_test(size=(80, 24)) as pilot:
        ta = await enter_multiline(app, pilot, "'''")
        ta.text = "alpha beta"
        head, tail = (0, 6), (0, 10)
        ta.selection = AreaSelection(tail, head) if reverse else AreaSelection(head, tail)
        await settle(pilot)

        await pilot.press("shift+enter")
        await settle(pilot)

        assert ta.text == "alpha \n"
        assert ta.selection == AreaSelection.cursor((1, 0))
        assert app.session.messages == []

        await pilot.press("ctrl+z")
        await settle(pilot)
        assert ta.text == "alpha beta"


async def test_heredoc_multiline_sends_message(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    monkeypatch.setattr("jtech_cli.tui.stream_reply", stream_of("ok"))
    async with app.run_test() as pilot:
        ta = await enter_multiline(app, pilot, "'''")
        await type_text(pilot, "line one")
        await pilot.press("shift+enter")
        await type_text(pilot, "line two")
        await settle(pilot)
        assert ta.text == "line one\nline two"
        assert app.session.messages == []

        await pilot.press("enter")
        for _ in range(10):
            await pilot.pause()

        assert app.session.messages == [
            {"role": "user", "content": "line one\nline two"},
            {"role": "assistant", "content": "ok"},
        ]
        assert app.query_one("#input", Input) is not None
        assert not list(app.query("#multiline-input"))


async def test_write_multiline_writes_file(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        ta = await enter_multiline(app, pilot, f"/write {tmp_path / 'out.txt'}")
        await type_text(pilot, "hello")
        await pilot.press("shift+enter")
        await type_text(pilot, "world")
        await settle(pilot)
        assert ta.text == "hello\nworld"

        await pilot.press("enter")
        for _ in range(10):
            await pilot.pause()

        assert (tmp_path / "out.txt").read_text() == "hello\nworld"
        assert app.query_one("#input", Input) is not None
        assert not list(app.query("#multiline-input"))


async def test_write_multiline_esc_writes_nothing(tmp_path):
    """Esc is a cancel, not an empty submission: no create, no truncate."""
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        missing = tmp_path / "missing.txt"
        existing = tmp_path / "existing.txt"
        existing.write_text("keep me")

        await enter_multiline(app, pilot, f"/write {missing}")
        await type_text(pilot, "discarded")
        await pilot.press("escape")
        for _ in range(10):
            await pilot.pause()
        assert not missing.exists()

        await enter_multiline(app, pilot, f"/write {existing}")
        await pilot.press("escape")
        for _ in range(10):
            await pilot.pause()
        assert existing.read_text() == "keep me"

        assert app.query_one("#input", Input).display is True
        assert not list(app.query("#multiline-input"))


async def test_write_multiline_empty_submission_creates_an_empty_file(tmp_path):
    """Enter on an empty editor is a choice the TUI must pass through."""
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        target = tmp_path / "empty.txt"
        ta = await enter_multiline(app, pilot, f"/write {target}")
        assert ta.text == ""

        await pilot.press("enter")
        for _ in range(10):
            await pilot.pause()

        assert target.read_text() == ""


@pytest.mark.parametrize("source", ["terminal", "clipboard"])
async def test_multiline_paste_promotes_the_whole_draft(tmp_path, source):
    """A pasted line break expands the editor; nothing is lost and nothing sends."""
    app = make_app(tmp_path)
    async with app.run_test(size=(80, 24)) as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "prefix DROP suffix"
        inp.focus()
        # Reverse selection with a live suffix: the hard case for the caret.
        select_input_range(inp, len("prefix DROP"), len("prefix "))
        await settle(pilot)

        await deliver_paste(app, pilot, source, "one\ntwo")

        assert inp.display is False
        assert len(list(app.query("#multiline-input"))) == 1
        ta = app.query_one("#multiline-input", TextArea)
        assert ta.text == "prefix one\ntwo suffix"
        assert ta.selection == AreaSelection.cursor((1, len("two")))
        assert app.session.messages == []


async def burst(app, *events_in_order) -> None:
    """Queue events with no chance to process between, as a terminal does."""
    for event in events_in_order:
        app.post_message(event)


async def test_paste_then_character_keeps_the_typed_character(tmp_path):
    """The gap before the editor mounts must compose input, not discard it."""
    app = make_app(tmp_path)
    async with app.run_test(size=(80, 24)) as pilot:
        inp = app.query_one("#input", Input)
        inp.focus()
        clear_input_selection(inp)
        await settle(pilot)

        await burst(app, events.Paste("one\ntwo"), events.Key("x", "x"))
        await settle(pilot, 12)

        ta = app.query_one("#multiline-input", TextArea)
        assert ta.text == "one\ntwox"
        assert ta.selection == AreaSelection.cursor((1, len("twox")))
        assert app.session.messages == []


async def test_paste_then_enter_sends_the_pasted_draft_exactly_once(tmp_path, monkeypatch):
    """A racing Enter must send the composed draft, never the pre-paste one."""
    app = make_app(tmp_path)
    monkeypatch.setattr("jtech_cli.tui.stream_reply", stream_of("ok"))
    async with app.run_test(size=(80, 24)) as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "draft "
        inp.focus()
        clear_input_selection(inp)
        await settle(pilot)

        await burst(app, events.Paste("one\ntwo"), events.Key("enter", None))
        await settle(pilot, 15)

        assert app.session.messages == [
            {"role": "user", "content": "draft one\ntwo"},
            {"role": "assistant", "content": "ok"},
        ]
        assert not list(app.query("#multiline-input"))
        assert app.query_one("#input", Input).display is True


async def test_two_queued_pastes_compose_into_one_editor(tmp_path):
    """Two promotions must not race into two editors sharing one id."""
    app = make_app(tmp_path)
    async with app.run_test(size=(80, 24)) as pilot:
        inp = app.query_one("#input", Input)
        inp.focus()
        clear_input_selection(inp)
        await settle(pilot)

        await burst(
            app, events.Paste("one\ntwo"), events.Paste("three\nfour")
        )
        await settle(pilot, 12)

        assert len(list(app.query("#multiline-input"))) == 1
        ta = app.query_one("#multiline-input", TextArea)
        assert ta.text == "one\ntwothree\nfour"
        assert ta.selection == AreaSelection.cursor((2, len("four")))
        assert app.session.messages == []


async def test_single_line_terminal_paste_stays_in_the_chat_input(tmp_path):
    """Inherited Input paste still runs once — not skipped, not doubled."""
    app = make_app(tmp_path)
    async with app.run_test(size=(80, 24)) as pilot:
        inp = app.query_one("#input", Input)
        inp.focus()
        clear_input_selection(inp)
        await settle(pilot)

        await deliver_paste(app, pilot, "terminal", "one line")

        assert inp.value == "one line"
        assert inp.display is True
        assert not list(app.query("#multiline-input"))
        assert app.session.messages == []


async def test_multiline_cancel_restores_input(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await enter_multiline(app, pilot, "'''")
        await pilot.press("escape")
        for _ in range(10):
            await pilot.pause()

        assert app.session.messages == []
        assert app.query_one("#input", Input) is not None
        assert not list(app.query("#multiline-input"))


@pytest.mark.parametrize("reverse", [False, True])
async def test_shift_enter_opens_prefilled_multiline(tmp_path, monkeypatch, reverse):
    """One keypress must both expand the editor and make the newline."""
    app = make_app(tmp_path)
    monkeypatch.setattr("jtech_cli.tui.stream_reply", stream_of("ok"))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "prefix DROP suffix"
        inp.focus()
        start, end = len("prefix "), len("prefix DROP")
        select_input_range(inp, *((end, start) if reverse else (start, end)))
        await settle(pilot)
        single_outer_height = inp.outer_size.height

        await pilot.press("shift+enter")
        await settle(pilot)

        assert len(list(app.query("#multiline-input"))) == 1
        ta = app.query_one("#multiline-input", TextArea)
        assert ta.text == "prefix \n suffix"
        assert ta.selection == AreaSelection.cursor((1, 0))
        assert inp.value == ""
        assert app.session.messages == []
        # The widget binding still reaches the same action, and the editor is
        # laid out with two editable rows rather than a fixed block.
        assert ta.document.line_count == 2
        assert ta.size.height == 2
        assert ta.outer_size.height == single_outer_height + 1
        assert ta.has_focus

        await type_text(pilot, "second")
        await settle(pilot)
        assert ta.text == "prefix \nsecond suffix"

        await pilot.press("enter")
        for _ in range(10):
            await pilot.pause()

        assert app.session.messages == [
            {"role": "user", "content": "prefix \nsecond suffix"},
            {"role": "assistant", "content": "ok"},
        ]
        assert app.query_one("#input", Input) is not None
        assert not list(app.query("#multiline-input"))


@pytest.mark.parametrize("viewport_height", [10, 24])
async def test_ctrl_j_promotes_and_grows_composer_to_two_editable_rows(
    tmp_path,
    viewport_height,
) -> None:
    """Terminal LF promotes the draft and paints a second editable row.

    The key event is built by hand because a raw LF byte is what the reporting
    terminal actually sends, and Textual reports it as ``ctrl+j``. Starting at
    that public event boundary is what the suite was missing, and the height
    assertions are what prove the second line is laid out, not merely stored.
    """
    app = make_app(tmp_path)
    async with app.run_test(size=(80, viewport_height)) as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "prefix suffix"
        inp.focus()
        inp.selection = InputSelection.cursor(len("prefix "))
        await settle(pilot)
        single_outer_height = inp.outer_size.height

        app.post_message(events.Key("ctrl+j", "\n"))
        await settle(pilot, 12)

        assert len(list(app.query("#multiline-input"))) == 1
        ta = app.query_one("#multiline-input", TextArea)
        assert ta.text == "prefix \nsuffix"
        assert ta.document.line_count == 2
        assert ta.selection == AreaSelection.cursor((1, 0))
        assert ta.has_focus
        assert ta.size.height == 2
        # The border gains one row. The single-line input's bottom margin goes
        # away with it, so the reserved screen footprint is unchanged.
        assert ta.outer_size.height == single_outer_height + 1
        assert inp.display is False
        assert app.session.messages == []
        assert app._queue == []


async def test_ctrl_j_edits_and_grows_open_multiline(tmp_path, monkeypatch) -> None:
    """LF adds a line inside the open editor; CR still submits it exactly once."""
    app = make_app(tmp_path)
    monkeypatch.setattr(
        "jtech_cli.tui.stream_reply",
        stream_of("ok"),
    )
    async with app.run_test(size=(80, 24)) as pilot:
        opened = await enter_multiline(app, pilot, "'''")
        opened.text = "one\ntwothree"
        opened.selection = AreaSelection.cursor((1, len("two")))
        opened.focus()
        await settle(pilot)
        assert opened.size.height == 2

        app.post_message(events.Key("ctrl+j", "\n"))
        await settle(pilot, 12)

        assert app.query_one("#multiline-input", TextArea) is opened
        assert opened.text == "one\ntwo\nthree"
        assert opened.selection == AreaSelection.cursor((2, 0))
        assert opened.size.height == 3
        assert opened.has_focus
        assert app.session.messages == []
        assert app._queue == []

        await pilot.press("enter")
        await settle(pilot, 12)

        assert app.session.messages == [
            {"role": "user", "content": "one\ntwo\nthree"},
            {"role": "assistant", "content": "ok"},
        ]
        assert not list(app.query("#multiline-input"))

        # `bubbles()` reports source record content, so only the rendered rows
        # can show that the submitted newlines survived presentation.
        user_record = chat_of(app).history.records[0]
        assert user_record.content == "one\ntwo\nthree"
        assert user_record.format == "plain"
        rows = [row.strip() for row in history_lines(app).split("\n")]
        assert rows.count("one") == 1
        assert rows.count("two") == 1
        assert rows.count("three") == 1
        assert rows.index("one") < rows.index("two") < rows.index("three")
        assert not [row for row in rows if "one two three" in row]


async def test_multiline_editor_auto_grows_then_scrolls_at_existing_max(
    tmp_path,
) -> None:
    """Growth tracks the document, then stops at the existing six-cell footprint."""
    app = make_app(tmp_path)
    async with app.run_test(size=(80, 24)) as pilot:
        ta = await enter_multiline(app, pilot, "'''")
        await settle(pilot)

        # Short lines in a full-width app, so this measures real document
        # lines and never accidental soft wrapping.
        heights = []
        for line in range(1, 7):
            if line > 1:
                await pilot.press("ctrl+j")
            await type_text(pilot, f"l{line}")
            await settle(pilot)
            assert ta.document.line_count == line
            heights.append(ta.size.height)
            assert ta.outer_size.height <= 6

        assert heights == [1, 2, 3, 4, 4, 4]
        assert ta.outer_size.height == 6
        assert ta.virtual_size.height > ta.size.height
        assert ta.max_scroll_y > 0
        assert ta.cursor_at_last_line
        cursor = ta.cursor_screen_offset
        assert ta.content_region.contains(cursor.x, cursor.y)
