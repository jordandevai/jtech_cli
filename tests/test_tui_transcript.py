"""Focused tests for the transcript renderer, tail handles, and compaction.

These exercise `TranscriptHistory` and `Transcript` directly rather than
through `ChatApp`, so the rendering cache, the ordering invariant, and every
handle state transition can be driven deterministically.
"""

import io
from pathlib import Path

import pytest
from rich.console import Console
from rich.markdown import Markdown as RichMarkdown
from textual.app import App, ComposeResult
from textual.geometry import Offset, Size
from textual.selection import Selection
from textual.widgets import Markdown, Static

import jtech_cli
from jtech_cli.theme import JTECH_DARK, JTECH_LIGHT
from jtech_cli.tui_widgets import (
    PlainTail,
    Transcript,
    TranscriptHistory,
    TranscriptRecord,
    _TranscriptMarkdown,
)

APP_CSS = Path(jtech_cli.__file__).parent / "resources" / "styles" / "tui.css"
SIZE = Size(60, 24)


class TranscriptApp(App):
    """A bare app hosting one `Transcript` with the project themes registered."""

    CSS_PATH = APP_CSS

    def __init__(self, records=(), theme_name: str = "jtech-dark") -> None:
        super().__init__()
        self._startup_records = list(records)
        self._theme_name = theme_name
        self.mount_render_width: int | None = None
        self.mount_line_count: int | None = None
        self.opened_urls: list[str] = []

    def compose(self) -> ComposeResult:
        yield Transcript(id="chat")

    def on_mount(self) -> None:
        self.register_theme(JTECH_DARK)
        self.register_theme(JTECH_LIGHT)
        self.theme = self._theme_name
        chat = self.query_one("#chat", Transcript)
        if self._startup_records:
            chat.load(self._startup_records)
        # Sampled here because startup runs before the first layout: this is
        # the state the deferred-render path is supposed to leave behind.
        self.mount_render_width = chat.history._render_width
        self.mount_line_count = len(chat.history._lines)

    def open_url(self, url: str, *, new_tab: bool = True) -> None:
        self.opened_urls.append(url)


def transcript(app: TranscriptApp) -> Transcript:
    return app.query_one("#chat", Transcript)


def lines(history: TranscriptHistory) -> list[str]:
    """The text of every rendered completed line."""
    return [strip.text for strip in history._lines]


def text_of(history: TranscriptHistory) -> str:
    return "\n".join(lines(history))


def height(history: TranscriptHistory, width: int) -> int:
    """Ask the widget for its content height the way layout does."""
    return history.get_content_height(SIZE, SIZE, width)


def spy_rendered(monkeypatch) -> list[TranscriptRecord]:
    """Collect every record `TranscriptHistory` renders from now on."""
    rendered: list[TranscriptRecord] = []
    real = TranscriptHistory._render_record

    def counted(self, record, width):
        rendered.append(record)
        return real(self, record, width)

    monkeypatch.setattr(TranscriptHistory, "_render_record", counted)
    return rendered


def spy_extensions(monkeypatch) -> list[list[TranscriptRecord]]:
    """Collect the record batch of every `TranscriptHistory.extend()` call."""
    batches: list[list[TranscriptRecord]] = []
    real = TranscriptHistory.extend

    def counted(self, records):
        batch = list(records)
        batches.append(batch)
        return real(self, batch)

    monkeypatch.setattr(TranscriptHistory, "extend", counted)
    return batches


def tail_content(entry) -> str:
    """The visible body text of one tail entry."""
    if isinstance(entry.body, Markdown):
        return entry.body._markdown
    return str(entry.body.render())


def link_segments(history: TranscriptHistory) -> list[tuple[str, str, str]]:
    """Every rendered segment carrying a click action: text, URL, action."""
    found: list[tuple[str, str, str]] = []
    for strip in history._lines:
        for segment in strip._segments:
            style = segment.style
            if style is not None and style.meta.get("@click"):
                found.append((segment.text, style.link, style.meta["@click"]))
    return found


def styles_of(history: TranscriptHistory) -> list[tuple[str | None, str | None]]:
    """The (color, background) hex of every rendered segment, in order."""
    snapshot: list[tuple[str | None, str | None]] = []
    for strip in history._lines:
        for segment in strip._segments:
            style = segment.style
            color = style.color if style is not None else None
            bgcolor = style.bgcolor if style is not None else None
            snapshot.append(
                (
                    color.triplet.hex if color is not None and color.triplet else None,
                    (
                        bgcolor.triplet.hex
                        if bgcolor is not None and bgcolor.triplet
                        else None
                    ),
                )
            )
    return snapshot


def render_to_text(renderable, width: int = 60) -> str:
    """Rich's own rendering of ``renderable``, as plain text."""
    console = Console(width=width, file=io.StringIO(), no_color=True)
    console.print(renderable)
    return console.file.getvalue()


def backgrounds(history: TranscriptHistory) -> set[str]:
    return {bg for _, bg in styles_of(history) if bg}


def foregrounds(history: TranscriptHistory) -> set[str]:
    return {fg for fg, _ in styles_of(history) if fg}


# --- TranscriptHistory rendering -------------------------------------------


async def test_an_empty_history_has_no_records_lines_or_children():
    app = TranscriptApp()
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        chat = transcript(app)

        assert chat.history.records == ()
        assert chat.history._lines == []
        assert list(chat.history.children) == []
        assert list(chat.children) == [chat.history]


async def test_set_records_keeps_order_and_mounts_no_child_widgets():
    records = [
        TranscriptRecord("user", "first question"),
        TranscriptRecord("ai", "**second** answer"),
        TranscriptRecord("reasoning", "third *literal*", format="plain"),
    ]
    app = TranscriptApp()
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        history = transcript(app).history
        history.set_records(records)
        await pilot.pause()

        assert history.records == tuple(records)
        body = text_of(history)
        assert body.index("first question") < body.index("second answer")
        assert body.index("second answer") < body.index("third *literal*")
        assert "**second**" not in body  # Markdown really rendered
        assert "USER" in body and "AI" in body and "REASONING" in body
        assert list(history.children) == []


async def test_the_first_layout_renders_records_deferred_at_mount():
    records = [TranscriptRecord("user", f"stored {index}") for index in range(5)]
    app = TranscriptApp(records)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        history = transcript(app).history

        assert app.mount_render_width == 0  # no usable width during startup
        assert app.mount_line_count == 0
        assert history.records == tuple(records)
        assert history._render_width > 0
        assert history._lines
        assert height(history, history._render_width) == len(history._lines)
        for index in range(5):
            assert f"stored {index}" in text_of(history)


async def test_repeated_layout_at_one_width_and_theme_does_not_render_again(
    monkeypatch,
):
    app = TranscriptApp([TranscriptRecord("ai", "answer text")])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        history = transcript(app).history
        width = history._render_width
        expected = len(history._lines)

        rendered = spy_rendered(monkeypatch)
        assert height(history, width) == expected
        assert height(history, width) == expected
        assert rendered == []


async def test_a_width_change_re_renders_once_and_keeps_every_character():
    record = TranscriptRecord("ai", "word " * 40)
    app = TranscriptApp([record])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        history = transcript(app).history

        for width in (30, 100, 30):
            rendered_at_width = []
            real = TranscriptHistory._render_record

            def counted(self, rec, w, _real=real, _sink=rendered_at_width):
                _sink.append(rec)
                return _real(self, rec, w)

            TranscriptHistory._render_record = counted
            try:
                assert height(history, width) == len(history._lines)
                assert height(history, width) == len(history._lines)
            finally:
                TranscriptHistory._render_record = real

            assert rendered_at_width == [record]
            assert history._render_width == width
            assert text_of(history).count("word") == 40
            assert all(len(line) <= width for line in lines(history))


async def test_extend_renders_only_the_new_records(monkeypatch):
    first = TranscriptRecord("user", "one")
    second = TranscriptRecord("ai", "two")
    third = TranscriptRecord("user", "three")
    app = TranscriptApp([first, second])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        history = transcript(app).history
        before = len(history._lines)

        rendered = spy_rendered(monkeypatch)
        history.extend([third])
        await pilot.pause()

        assert rendered == [third]
        assert history.records == (first, second, third)
        assert len(history._lines) > before
        assert "three" in text_of(history)

        rendered.clear()
        history.extend([])
        assert rendered == []
        assert history.records == (first, second, third)


async def test_a_candidate_rendering_failure_leaves_records_and_lines_unchanged(
    monkeypatch,
):
    kept = TranscriptRecord("ai", "kept answer")
    app = TranscriptApp([kept])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        history = transcript(app).history
        before_records = history.records
        before_lines = lines(history)

        real = TranscriptHistory._render_record

        def failing(self, record, width):
            if record.content == "boom":
                raise RuntimeError("render failed")
            return real(self, record, width)

        monkeypatch.setattr(TranscriptHistory, "_render_record", failing)
        with pytest.raises(RuntimeError, match="render failed"):
            history.extend([TranscriptRecord("ai", "boom")])

        assert history.records == before_records
        assert lines(history) == before_lines

        with pytest.raises(RuntimeError, match="render failed"):
            history.set_records([kept, TranscriptRecord("ai", "boom")])

        assert history.records == before_records
        assert lines(history) == before_lines


async def test_long_unbroken_fenced_code_wraps_and_keeps_every_character():
    long_word = "z" * 300
    record = TranscriptRecord("ai", f"```\n{long_word}\nafter-line\n```\n")
    app = TranscriptApp([record])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        history = transcript(app).history

        body = text_of(history)
        assert body.count("z") == 300
        assert "after-line" in body
        assert all(len(line) <= history._render_width for line in lines(history))


async def test_rich_markdown_features_keep_their_content_and_order():
    record = TranscriptRecord(
        "ai",
        "# Heading text\n\n"
        "Some *emphasis* plus **strong** and `inline_code` here.\n\n"
        "- alpha item\n"
        "- beta item\n\n"
        "| left | right |\n| --- | --- |\n| one | two |\n\n"
        "Unicode: café — 日本語\n\n"
        "```python\nvalue = 1\n```\n",
    )
    app = TranscriptApp([record])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        body = text_of(transcript(app).history)

        order = [
            "Heading text",
            "emphasis",
            "strong",
            "inline_code",
            "alpha item",
            "beta item",
            "left",
            "one",
            "café",
            "日本語",
            "value = 1",
        ]
        positions = [body.index(part) for part in order]
        assert positions == sorted(positions)
        assert "# Heading" not in body  # rendered, not echoed as source


async def test_a_link_keeps_its_url_and_opens_it_through_a_click_action():
    url = "https://example.com/a?b=1&c=2"
    app = TranscriptApp([TranscriptRecord("ai", f"see [Example]({url}) now")])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        history = transcript(app).history

        found = link_segments(history)
        assert found == [("Example", url, f"open_link({url!r})")]

        await history.run_action(found[0][2])
        await pilot.pause()

        assert app.opened_urls == [url]


async def test_a_bare_url_is_linked_and_clickable_like_the_live_bubble():
    """A URL the model simply typed has to stay clickable once the turn ends.

    Textual's live ``Markdown`` parses with markdown-it's ``gfm-like`` preset,
    whose linkify rule links a bare URL; Rich's own parser does not. Completed
    history is parsed the live way so the link does not vanish on completion.
    """
    url = "https://example.com/bare?x=1"
    app = TranscriptApp([TranscriptRecord("ai", f"go to {url} now")])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        history = transcript(app).history

        found = link_segments(history)
        assert found == [(url, url, f"open_link({url!r})")]

        await history.run_action(found[0][2])
        await pilot.pause()

        assert app.opened_urls == [url]


async def test_completed_markdown_parses_like_rich_for_everything_but_links():
    """Only the token stream is replaced; Rich still owns the rendering.

    ``_TranscriptMarkdown`` re-derives ``markup`` and ``parsed`` and takes the
    rest from Rich's constructor. A Rich release that derived anything else
    from the source string would show up as a difference here.
    """
    document = (
        "# Heading\n\n"
        "Some *emphasis*, **strong**, ~~struck~~, and `inline_code`.\n\n"
        "- one\n- two\n\n"
        "| left | right |\n| --- | --- |\n| a | b |\n\n"
        "```python\nvalue = 1\n```\n\n"
        "An explicit [label](https://example.com/x) link.\n"
    )
    ours = _TranscriptMarkdown(document)

    assert ours.markup == document
    assert render_to_text(ours) == render_to_text(RichMarkdown(document))


async def test_selection_extracts_and_paints_without_losing_content_or_links():
    url = "https://example.com/link"
    app = TranscriptApp([TranscriptRecord("ai", f"open [Example]({url}) please")])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        history = transcript(app).history
        row = next(y for y, line in enumerate(lines(history)) if "Example" in line)
        line = lines(history)[row]

        selection = Selection(Offset(0, row), Offset(len("open Example"), row))
        extracted, ending = history.get_selection(selection)
        assert extracted == line[: len("open Example")]
        assert ending == "\n"

        unselected = history.render_line(row)
        app.screen.selections = {history: selection}
        await pilot.pause()
        selected = history.render_line(row)

        assert selected.text == unselected.text
        assert selected.cell_length == unselected.cell_length
        assert any(
            segment.style is not None and segment.style.meta.get("@click")
            for segment in selected._segments
        )
        assert any(
            segment.style is not None and segment.style.link == url
            for segment in selected._segments
        )
        # the selection colour really reaches the selected span, and only it
        painted = history.selection_style.bgcolor
        assert painted is not None
        inside = "".join(
            segment.text
            for segment in selected._segments
            if segment.style is not None and segment.style.bgcolor == painted
        )
        assert inside == "open Example"


async def test_a_multiline_selection_copies_content_without_layout_padding():
    """Copying across lines must not drag the pane-width padding along.

    Rendered lines are padded out so role backgrounds fill the pane. Pasted
    into an editor that padding turns every copied line into trailing
    whitespace, which is why extraction works on the unpadded text.
    """
    app = TranscriptApp(
        [
            TranscriptRecord("user", "hello"),
            TranscriptRecord("system", "notice", format="plain"),
        ]
    )
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        history = transcript(app).history
        last = len(history._lines) - 1

        # the rendered lines really are padded — that is what has to be dropped
        assert any(line != line.rstrip() for line in lines(history))

        selection = Selection(Offset(0, 0), Offset(history._render_width, last))
        extracted, ending = history.get_selection(selection)

        assert ending == "\n"
        copied = extracted.split("\n")
        assert copied == [line.rstrip() for line in copied]  # no trailing padding
        assert "hello" in extracted
        assert "notice" in extracted
        assert extracted.index("hello") < extracted.index("notice")


async def test_a_theme_reflow_restyles_without_changing_records_or_text():
    records = [
        TranscriptRecord("user", "user bubble"),
        TranscriptRecord("ai", "```python\nvalue = 1\n```"),
        TranscriptRecord("ai", "it failed", label="AI", error=True),
    ]
    app = TranscriptApp(records)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        chat = transcript(app)
        history = chat.history

        dark_text = lines(history)
        dark_styles = styles_of(history)
        assert JTECH_DARK.surface in backgrounds(history)
        assert JTECH_DARK.error in foregrounds(history)

        app.theme = "jtech-light"
        chat.refresh_theme()
        await pilot.pause()

        assert history.records == tuple(records)
        assert lines(history) == dark_text
        assert history._render_theme == "jtech-light"
        assert styles_of(history) != dark_styles
        assert JTECH_LIGHT.surface in backgrounds(history)
        assert JTECH_LIGHT.error in foregrounds(history)
        assert JTECH_DARK.surface not in backgrounds(history)
        assert JTECH_DARK.error not in foregrounds(history)


async def test_a_theme_without_a_required_color_is_reported_not_substituted():
    from textual.theme import Theme

    app = TranscriptApp()
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        history = transcript(app).history
        app.register_theme(Theme(name="bare", primary="#ffffff"))
        app.theme = "bare"
        await pilot.pause()

        with pytest.raises(RuntimeError, match="foreground"):
            history.set_records([TranscriptRecord("ai", "answer")])


# --- Transcript ordering and tail lifecycle --------------------------------


async def test_append_without_a_tail_adds_no_child_widgets():
    app = TranscriptApp()
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        chat = transcript(app)

        chat.append(TranscriptRecord("user", "one"))
        chat.append(TranscriptRecord("ai", "two"))
        await pilot.pause()

        assert [record.content for record in chat.history.records] == ["one", "two"]
        assert list(chat.children) == [chat.history]


async def test_load_requires_an_empty_transcript():
    app = TranscriptApp([TranscriptRecord("user", "stored")])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        chat = transcript(app)

        with pytest.raises(RuntimeError, match="empty transcript"):
            chat.load([TranscriptRecord("user", "again")])

        assert [record.content for record in chat.history.records] == ["stored"]


async def test_finalized_records_wait_behind_a_live_entry_in_visible_order():
    app = TranscriptApp()
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        chat = transcript(app)

        notice = chat.begin_plain("system", "Queued: two")
        chat.append(TranscriptRecord("user", "second question"))
        chat.append(TranscriptRecord("ai", "second answer"))
        await pilot.pause()

        assert chat.history.records == ()
        assert [entry.state for entry in chat._tail] == [
            "live",
            "finalized",
            "finalized",
        ]
        assert chat._tail[0] is notice
        assert [tail_content(entry) for entry in chat._tail] == [
            "Queued: two",
            "second question",
            "second answer",
        ]


async def test_removing_the_blocker_compacts_the_whole_prefix_in_one_extension(
    monkeypatch,
):
    app = TranscriptApp()
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        chat = transcript(app)
        notice = chat.begin_plain("system", "Queued: two")
        chat.append(TranscriptRecord("user", "second question"))
        chat.append(TranscriptRecord("ai", "second answer"))
        await pilot.pause()

        batches = spy_extensions(monkeypatch)
        chat.remove(notice)
        await pilot.pause()

        assert len(batches) == 1
        assert [record.content for record in batches[0]] == [
            "second question",
            "second answer",
        ]
        assert [record.content for record in chat.history.records] == [
            "second question",
            "second answer",
        ]
        assert chat._tail == []
        assert list(chat.children) == [chat.history]


async def test_finalizing_the_blocker_compacts_the_whole_prefix_in_one_extension(
    monkeypatch,
):
    app = TranscriptApp()
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        chat = transcript(app)
        live = chat.begin_markdown("ai", "live answer")
        chat.append(TranscriptRecord("user", "next question"))
        await pilot.pause()

        batches = spy_extensions(monkeypatch)
        chat.finalize(live, TranscriptRecord("ai", "live answer", label="AI · done"))
        await pilot.pause()

        assert len(batches) == 1
        assert [record.content for record in batches[0]] == [
            "live answer",
            "next question",
        ]
        assert [record.display_label for record in chat.history.records] == [
            "AI · done",
            "USER",
        ]
        assert chat._tail == []


async def test_the_two_notice_hard_case_preserves_order_at_every_transition():
    app = TranscriptApp()
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        chat = transcript(app)

        # 1-2: a live answer with two removable notices behind it.
        answer = chat.begin_markdown("ai", "first answer")
        first = chat.begin_plain("system", "Queued: two")
        second = chat.begin_plain("system", "Queued: three")
        await pilot.pause()
        assert chat.history.records == ()
        assert chat._tail == [answer, first, second]

        # 3: the answer finalizes and compacts, because it is before both.
        chat.finalize(answer, TranscriptRecord("ai", "first answer", label="AI"))
        await pilot.pause()
        assert [record.content for record in chat.history.records] == ["first answer"]
        assert chat._tail == [first, second]

        # 4-5: the recalled notice's turn lands behind the notice still shown.
        chat.remove(first)
        chat.append(TranscriptRecord("user", "two"))
        chat.append(TranscriptRecord("ai", "second answer"))
        await pilot.pause()
        assert [record.content for record in chat.history.records] == ["first answer"]
        assert [tail_content(entry) for entry in chat._tail] == [
            "Queued: three",
            "two",
            "second answer",
        ]

        # 6: removing the second notice makes the prefix contiguous.
        chat.remove(second)
        await pilot.pause()
        assert [record.content for record in chat.history.records] == [
            "first answer",
            "two",
            "second answer",
        ]
        assert chat._tail == []
        assert list(chat.children) == [chat.history]


async def test_clear_empties_everything_and_late_closes_are_no_ops():
    app = TranscriptApp([TranscriptRecord("ai", "stored")])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        chat = transcript(app)
        live = chat.begin_markdown("ai", "streaming")
        notice = chat.begin_plain("system", "Queued: later")
        await pilot.pause()

        chat.clear()
        await pilot.pause()

        assert chat.history.records == ()
        assert chat.history._lines == []
        assert chat._tail == []
        assert list(chat.children) == [chat.history]
        assert live.state == "cleared"
        assert notice.state == "cleared"

        chat.finalize(live, TranscriptRecord("ai", "late answer"))
        chat.remove(notice)
        chat.remove(notice)
        await pilot.pause()

        assert chat.history.records == ()
        assert chat._tail == []
        assert list(chat.children) == [chat.history]


async def test_every_invalid_owner_or_state_transition_raises():
    app = TranscriptApp()
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        chat = transcript(app)

        foreign = PlainTail(label=Static(""), body=Static(""), _owner=object())
        with pytest.raises(RuntimeError, match="another transcript"):
            chat.remove(foreign)
        with pytest.raises(RuntimeError, match="another transcript"):
            chat.finalize(foreign, TranscriptRecord("system", "x"))

        blocker = chat.begin_plain("system", "Queued: x")
        pending = chat.begin_markdown("ai", "pending")
        chat.finalize(pending, TranscriptRecord("ai", "pending"))
        await pilot.pause()
        assert pending.state == "finalized"

        with pytest.raises(RuntimeError, match="finalize a finalized"):
            chat.finalize(pending, TranscriptRecord("ai", "pending again"))
        with pytest.raises(RuntimeError, match="remove a finalized"):
            chat.remove(pending)

        chat.remove(blocker)
        await pilot.pause()
        assert pending.state == "compacted"

        with pytest.raises(RuntimeError, match="finalize a compacted"):
            chat.finalize(pending, TranscriptRecord("ai", "pending again"))
        with pytest.raises(RuntimeError, match="remove a compacted"):
            chat.remove(pending)

        gone = chat.begin_plain("system", "Queued: y")
        chat.remove(gone)
        await pilot.pause()
        assert gone.state == "removed"
        chat.remove(gone)  # documented idempotent no-op
        with pytest.raises(RuntimeError, match="finalize a removed"):
            chat.finalize(gone, TranscriptRecord("system", "x"))

        with pytest.raises(ValueError, match="Unknown transcript format"):
            chat._add_tail(
                TranscriptRecord("ai", "x", format="bogus"), live=True
            )
        with pytest.raises(ValueError, match="Unknown transcript format"):
            chat.history._render_record(
                TranscriptRecord("ai", "x", format="bogus"),
                chat.history._render_width,
            )


async def test_no_operation_drops_or_caps_records():
    stored = [
        TranscriptRecord(
            "user" if index % 2 == 0 else "ai",
            f"message {index}\n\n" + "filler " * 120,
        )
        for index in range(40)
    ]
    added = [TranscriptRecord("ai", f"message {40 + index}") for index in range(20)]
    app = TranscriptApp(stored)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        chat = transcript(app)

        for record in added:
            chat.append(record)
        await pilot.pause()

        assert chat.history.records == tuple(stored + added)
        body = text_of(chat.history)
        for index in range(60):
            assert f"message {index}" in body
        assert list(chat.children) == [chat.history]
        assert len(chat.history._lines) == height(
            chat.history, chat.history._render_width
        )
