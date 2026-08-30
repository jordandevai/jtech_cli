"""Reusable Textual widgets and messages used by the chat TUI."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import ClassVar, Literal, NamedTuple, Protocol, runtime_checkable

from markdown_it import MarkdownIt
from rich.cells import cell_len
from rich.console import Console, Group, RenderableType
from rich.errors import MarkupError
from rich.markdown import Markdown as RichMarkdown
from rich.padding import Padding
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.geometry import Size
from textual.message import Message
from textual.selection import Selection
from textual.strip import Strip
from textual.theme import Theme
from textual.widget import Widget
from textual.widgets import Input, ListItem, ListView, Markdown, Static, TextArea


@runtime_checkable
class SuggestionHost(Protocol):
    """The small app surface needed by ``_ChatInput`` completion actions."""

    @property
    def suggestion_navigated(self) -> bool: ...

    def accepted_suggestion(self) -> str | None: ...

    def apply_completion(self, completed: str) -> None: ...


class MessagePusher(Protocol):
    """The app surface needed by ``OutputSink``."""

    def push_message(self, role: str, text: str) -> None: ...


def render_menu_rows(items: Sequence[tuple[str, str]], index: int) -> Text:
    """Render menu rows, marking the selected row with a chevron."""
    parts: list[object] = []
    for i, (name, detail) in enumerate(items):
        selected = i == index
        line = Text()
        line.append("▸ " if selected else "  ")
        line.append(name, "bold" if selected else "")
        if detail:
            line.append(f"  {detail}", "dim")
        parts.append(line)
        if i < len(items) - 1:
            parts.append("\n")
    return Text.assemble(*parts)


class _PromotedDraft(NamedTuple):
    """The draft handed from ``_ChatInput`` to the multi-line editor."""

    value: str
    cursor_offset: int
    submit: bool
    """The user pressed Enter before the editor existed; submit on arrival."""


class InputToMultiline(Message):
    """Ask the app to take over the promoted draft with a text area.

    Deliberately carries no payload. A message is delivered a hop later than
    it is posted, and in that gap the terminal can still deliver keys and
    further pastes that ``_ChatInput`` applies to its own value; a snapshot
    taken at post time would be stale on arrival and would overwrite them.
    The app reads the live draft with ``_ChatInput.take_promotion()`` instead.
    """


class MultilineSubmit(Message):
    """The multi-line editor asks the app to submit its current value."""


class MultilineCancel(Message):
    """The multi-line editor asks the app to cancel editing."""


class FieldCommit(Message):
    """The in-place settings editor asks its screen to commit the value."""


class FieldCancel(Message):
    """The in-place settings editor asks its screen to discard the value."""


class _ChatInput(Input):
    """Single-line input with command completion and multi-line shortcuts."""

    # Set once a promotion is requested and cleared when the app collects the
    # draft. Until then this widget stays the one owner of the draft, so input
    # the terminal delivers in between is composed rather than lost.
    _promoting: bool = False
    _pending_submit: bool = False

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("shift+enter", "to_multiline", "Multi-line", show=False),
        Binding("enter", "submit_or_complete", "Submit", show=False),
        Binding("tab", "tab_complete", "Complete", show=False),
    ]

    def action_submit_or_complete(self) -> None:
        if self._promoting:
            # The draft is already on its way to the editor, where Enter means
            # submit. Submitting here would send the half-composed value that
            # is still sitting in this widget.
            self._pending_submit = True
            return
        app = self.app
        if isinstance(app, SuggestionHost):
            completed = app.accepted_suggestion()
            if completed is not None:
                if app.suggestion_navigated or self.value.strip() == completed.strip():
                    self.post_message(Input.Submitted(self, completed))
                    return
                app.apply_completion(completed)
                return
        self.post_message(Input.Submitted(self, self.value))

    def action_tab_complete(self) -> None:
        app = self.app
        if isinstance(app, SuggestionHost):
            completed = app.accepted_suggestion()
            if completed is not None:
                app.apply_completion(completed)
                return
        self.screen.focus_next()

    def action_to_multiline(self) -> None:
        self._promote_with("\n")

    @staticmethod
    def _has_line_break(text: str) -> bool:
        """Does this text need an editor that can show more than one line?"""
        return "\n" in text or "\r" in text

    def _promote_with(self, inserted: str) -> None:
        """Apply an edit to the draft and ask the app to take it over.

        Every promotion — the first Shift+Enter and both paste routes — is a
        real selection replacement, not merely a change of widget, so the edit
        is applied here and now. Keeping the draft in this widget until the app
        collects it is what makes a burst of terminal events safe: a second
        paste or a typed character arriving before the editor exists composes
        into the same value, and only the first promotion is announced.
        """
        start, end = sorted(self.selection)
        current = self.value
        self.value = current[:start] + inserted + current[end:]
        self.cursor_position = start + len(inserted)
        if self._promoting:
            return
        self._promoting = True
        self.post_message(InputToMultiline())

    def cancel_promotion(self) -> None:
        """Withdraw a promotion request, leaving the draft where it is.

        A promotion the app declines must still clear these flags, or Enter
        stays dead in the composer for the rest of the session.
        """
        self._promoting = False
        self._pending_submit = False

    def take_promotion(self) -> _PromotedDraft:
        """Hand the promoted draft to the app and reset to an empty composer."""
        draft = _PromotedDraft(self.value, self.cursor_position, self._pending_submit)
        self._promoting = False
        self._pending_submit = False
        self.value = ""
        return draft

    def _on_paste(self, event: events.Paste) -> None:
        """Route a bracketed terminal paste before ``Input`` truncates it.

        ``Input._on_paste()`` keeps only ``event.text.splitlines()[0]``, so a
        multi-line paste has to be caught here or its remaining lines are gone
        before the app can see them. Textual dispatches ``_on_paste`` once per
        class in the MRO, so the inherited handler is suppressed in *both*
        branches: after promotion it would truncate, and after the explicit
        ``super()`` call it would insert the same text a second time.
        """
        if self._has_line_break(event.text):
            event.stop()
            event.prevent_default()
            self._promote_with(event.text)
            return
        super()._on_paste(event)
        event.prevent_default()

    def action_paste(self) -> None:
        """Paste Textual's local clipboard, promoting multi-line content.

        Without this the extra lines are written into a widget that can only
        render the first one, so the draft silently disagrees with the screen.
        """
        clipboard = self.app.clipboard
        if self._has_line_break(clipboard):
            self._promote_with(clipboard)
            return
        super().action_paste()


class _NewlineTextArea(TextArea):
    """``TextArea`` base that makes Shift+Enter an explicit newline edit.

    Both editors bind plain ``Enter`` to submit or save, which takes away the
    newline gesture ``TextArea`` normally provides. Routing the replacement
    through the public ``replace()`` boundary — rather than the document,
    history, and cursor separately — keeps undo, ``Changed`` messages, and line
    wrapping identical to ordinary typing.
    """

    def action_insert_newline(self) -> None:
        """Replace the active selection with a newline and follow the caret."""
        if self.read_only:
            return
        start, end = self.selection
        self.replace("\n", start, end, maintain_selection_offset=False)


class _MultilineInput(_NewlineTextArea):
    """Multi-line input; Enter submits, Shift+Enter adds a line, Esc cancels."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("enter", "multiline_submit", "Submit", show=False, priority=True),
        Binding("shift+enter", "insert_newline", "Newline", show=False, priority=True),
        Binding("escape", "multiline_cancel", "Cancel", show=False, priority=True),
    ]

    def action_multiline_submit(self) -> None:
        self.post_message(MultilineSubmit())

    def action_multiline_cancel(self) -> None:
        self.post_message(MultilineCancel())


class _FieldInput(Input):
    """Single-line in-place settings editor; Enter commits and Esc cancels."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("enter", "commit_field", "Commit", show=False, priority=True),
        Binding("escape", "cancel_field", "Cancel", show=False, priority=True),
    ]

    def action_commit_field(self) -> None:
        self.post_message(FieldCommit())

    def action_cancel_field(self) -> None:
        self.post_message(FieldCancel())


class _PromptEditor(_NewlineTextArea):
    """Multi-line in-place editor for the system prompt row."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("enter", "commit_field", "Commit", show=False, priority=True),
        Binding("shift+enter", "insert_newline", "Newline", show=False, priority=True),
        Binding("escape", "cancel_field", "Cancel", show=False, priority=True),
    ]

    def action_commit_field(self) -> None:
        self.post_message(FieldCommit())

    def action_cancel_field(self) -> None:
        self.post_message(FieldCancel())


class OutputSink:
    """Render command output into the chat instead of writing to stdout."""

    def __init__(self, app: MessagePusher) -> None:
        self._app = app
        self._renderer = Console(width=100, no_color=True, highlight=False)

    def _to_text(self, obj: object) -> str:
        """Convert one printable object to plain text."""
        if isinstance(obj, str):
            return obj
        if not hasattr(obj, "__rich_console__") and not hasattr(obj, "__rich__"):
            return str(obj)
        with self._renderer.capture() as capture:
            self._renderer.print(obj)
        return capture.get().rstrip("\n")

    def print(self, *objects: object, sep: str = " ", end: str = "\n") -> None:
        text = sep.join(self._to_text(obj) for obj in objects) + end
        if not text.strip():
            return
        try:
            plain = Text.from_markup(text).plain
        except MarkupError:
            # Command handlers deliberately use Rich markup; malformed markup
            # should still be visible as literal output in the chat.
            plain = text
        self._app.push_message("system", plain)


# --- transcript ------------------------------------------------------------

TranscriptFormat = Literal["markdown", "plain"]
TailState = Literal["live", "finalized", "compacted", "removed", "cleared"]

_DIM_ROLES = frozenset({"system", "reasoning"})


@dataclass(frozen=True)
class TranscriptRecord:
    """One completed transcript entry, as presentation data only.

    This type never crosses the session or model boundary: it carries what the
    transcript needs to draw a finished message, not what the conversation
    needs to replay one.

    Args:
        role: Styling role. ``user``, ``assistant``, ``ai``, ``reasoning``, and
            ``system`` are recognized; any other stored role renders with the
            neutral foreground rather than being discarded or rewritten.
        content: The complete source string. It is never truncated.
        label: Header text. ``None`` means ``role.upper()``; an explicit label
            preserves a live completion label such as ``AI · 1.2s`` when the
            entry compacts.
        format: ``markdown`` renders Rich Markdown; ``plain`` renders the
            content literally, which is what reasoning and queue notices need.
        error: Paint the body in the active error color, without changing the
            role or the content.
    """

    role: str
    content: str
    label: str | None = None
    format: TranscriptFormat = "markdown"
    error: bool = False

    @property
    def display_label(self) -> str:
        """The header text drawn above this record's body."""
        return self.role.upper() if self.label is None else self.label


@dataclass(eq=False)
class MarkdownTail:
    """Handle for one Markdown entry in the widget tail.

    Identity, not value: ``eq=False`` keeps two handles distinct even when they
    hold equal content, so a transcript can find the one it was given. Callers
    may read ``label`` and ``body`` to update a live entry; only ``Transcript``
    may write ``state`` and ``final_record``.
    """

    label: Static
    body: Markdown
    _owner: object = field(repr=False)
    state: TailState = "live"
    final_record: TranscriptRecord | None = None


@dataclass(eq=False)
class PlainTail:
    """Handle for one literal-text entry in the widget tail.

    The value semantics and ownership rules match `MarkdownTail`; only the body
    widget differs, because reasoning and queue notices are app-generated text
    rather than model Markdown.
    """

    label: Static
    body: Static
    _owner: object = field(repr=False)
    state: TailState = "live"
    final_record: TranscriptRecord | None = None


TailEntry = MarkdownTail | PlainTail


def _theme_color(theme: Theme, name: str) -> str:
    """Return one required active-theme color.

    Raises:
        RuntimeError: if the theme does not define ``name``. A missing color is
            reported rather than substituted, so a broken theme cannot silently
            render an unreadable transcript.
    """
    value = getattr(theme, name, None)
    if not value:
        raise RuntimeError(f"The active Textual theme does not define {name!r}")
    return value


# The preset Textual's live ``Markdown`` widget parses with. Shared because a
# parser is stateless once built and every completed record wants the same one.
_MARKDOWN_PARSER = MarkdownIt("gfm-like")


class _TranscriptMarkdown(RichMarkdown):
    """Rich Markdown parsed exactly the way the live bubble parses it.

    Rich's own parser is CommonMark plus tables and strikethrough, which leaves
    a bare ``https://…`` as plain text. Textual's live ``Markdown`` uses
    markdown-it's ``gfm-like`` preset, whose linkify rule turns that URL into a
    link — so parsing completed history Rich's way would make an ordinary
    model-emitted URL stop working the moment its turn ended.

    ``markup`` and ``parsed`` are the only state Rich's constructor derives
    from the source string (rich 15.0.0), so they are the only ones replaced
    here; the base constructor still owns every other attribute. It is handed
    an empty string rather than the real source so the document is parsed once,
    not twice — this renderer re-runs on every width and theme change.
    """

    def __init__(self, markup: str, **kwargs) -> None:
        super().__init__("", **kwargs)
        self.markup = markup
        self.parsed = _MARKDOWN_PARSER.parse(markup)


def _with_link_actions(segments: Iterable[Segment]) -> Iterator[Segment]:
    """Attach a Textual click action to every hyperlink Rich emitted.

    Rich marks a Markdown link with ``Style.link`` only, which a terminal can
    follow but Textual cannot. Adding ``@click`` meta alongside the untouched
    link style keeps both routes working.
    """
    for segment in segments:
        style = segment.style
        if style is not None and style.link and not segment.control:
            action = Style.from_meta({"@click": f"open_link({style.link!r})"})
            yield Segment(segment.text, style + action, segment.control)
        else:
            yield segment


class TranscriptHistory(Widget):
    """Completed transcript content as one reflowable line-rendered widget.

    Finished messages are rendered to `Strip` lines once per width and active
    theme and then held as data, so neither the widget tree nor the layout pass
    grows with the number or Markdown complexity of completed messages. A
    keystroke, a repaint, a scroll, and a streaming update all leave the cache
    alone; only new records, a width change, or a theme change rebuild it.
    """

    def __init__(self) -> None:
        super().__init__()
        self._records: list[TranscriptRecord] = []
        self._lines: list[Strip] = []
        # 0 means no usable width has been supplied yet, so nothing is rendered.
        self._render_width = 0
        self._render_theme: str | None = None

    @property
    def records(self) -> tuple[TranscriptRecord, ...]:
        """Every completed record, in visible order."""
        return tuple(self._records)

    def set_records(self, records: Sequence[TranscriptRecord]) -> None:
        """Replace all completed content with ``records``.

        Raises:
            RuntimeError: if the active theme lacks a color the records need.
                Nothing is committed in that case.
        """
        candidate = list(records)
        width = self.size.width
        if width <= 0:
            self._records = candidate
            self._invalidate()
        else:
            theme = self.app.theme
            lines = self._render_records(candidate, width)
            self._records = candidate
            self._commit(lines, width, theme)
        self.refresh(layout=True)

    def extend(self, records: Sequence[TranscriptRecord]) -> None:
        """Append ``records`` after the existing completed content.

        With a current cache only the new records are rendered, so a message
        that has just finished costs its own render and nothing more.

        Raises:
            RuntimeError: if the active theme lacks a color the records need.
                Prior records and lines are left unchanged in that case.
        """
        new = list(records)
        if not new:
            return
        width = self.size.width
        theme = self.app.theme
        if width <= 0:
            self._records.extend(new)
            self._invalidate()
        elif width == self._render_width and theme == self._render_theme:
            lines = self._render_records(new, width)
            self._records.extend(new)
            self._lines.extend(lines)
        else:
            candidate = self._records + new
            lines = self._render_records(candidate, width)
            self._records = candidate
            self._commit(lines, width, theme)
        self.refresh(layout=True)

    def clear(self) -> None:
        """Drop every completed record and its rendered lines."""
        self._records = []
        self._invalidate()
        self.refresh(layout=True)

    def reflow(self) -> None:
        """Re-render every completed record for a new active theme.

        Width changes reach the cache through the layout pass; a theme change
        has no layout of its own, so it comes through here.
        """
        width = self.size.width
        if width <= 0:
            self._invalidate()
            return
        theme = self.app.theme
        lines = self._render_records(self._records, width)
        self._commit(lines, width, theme)
        self.refresh(layout=True)

    def get_content_height(self, container: Size, viewport: Size, width: int) -> int:
        """Report the completed content height, rendering it if needed.

        An auto-height widget must know its height during the layout pass, so
        this — not a later `Resize` — is the path that produces the first lines
        and the lines for every new width.
        """
        self._ensure_rendered(width)
        return len(self._lines)

    def render_line(self, y: int) -> Strip:
        """Return one rendered line, selection painted and offsets stamped.

        The offsets are what let Textual select inside this widget at all: the
        compositor reads ``meta["offset"]`` off the segments this returns, and
        without it every drag over completed history resolves to the whole
        widget. They are stamped last, over the final segment layout, because
        `Strip.divide()` gives both halves of a split segment the original
        segment's style — offsets applied earlier would misreport the second
        half's start.
        """
        width = self.size.width
        self._ensure_rendered(width)
        rich_style = self.rich_style
        if y >= len(self._lines):
            return Strip.blank(width, rich_style)
        line = self._lines[y].crop_extend(0, width, rich_style).apply_style(rich_style)
        selection = self.text_selection
        if selection is not None:
            span = selection.get_span(y)
            if span is not None:
                line = self._paint_span(line, span)
        # x origin 0: this widget sits inside the scrolling container and never
        # scrolls horizontally itself.
        return line.apply_offsets(0, y)

    def _paint_span(self, line: Strip, span: tuple[int, int]) -> Strip:
        """Paint the selected part of ``line``, given a character span.

        `Selection` carries character offsets into the rendered line — that is
        the coordinate system `Strip.apply_offsets()` stamps and the compositor
        converts a screen cell into. `Strip.divide()` cuts on cell positions
        instead, so the span is converted here, and only here, with Rich's own
        width measurement.
        """
        text = line.text
        start, end = span
        end = len(text) if end == -1 else end
        start = max(0, min(start, len(text)))
        end = max(start, min(end, len(text)))
        cell_start = cell_len(text[:start])
        cell_end = cell_len(text[:end])
        if cell_start == cell_end:
            return line
        before, selected, after = line.divide([cell_start, cell_end, line.cell_length])
        return Strip.join([before, self._paint(selected, self.selection_style), after])

    @staticmethod
    def _paint(strip: Strip, style: Style) -> Strip:
        """Lay ``style`` over a strip, keeping its links and click metadata.

        The selection has to win over the colors each segment already carries,
        so it goes on as a post style rather than a base one. Rich merges meta
        and keeps the existing link, so a selected hyperlink stays clickable.
        """
        return Strip(
            list(Segment.apply_style(strip, post_style=style)), strip.cell_length
        )

    @staticmethod
    def _copyable_text(strip: Strip) -> str:
        """One rendered line's text, with the right-side layout fill removed.

        A line is padded out to the pane width so role backgrounds fill it, and
        that fill arrives as its own whitespace-only trailing segments.
        Trailing spaces that came from the source stay inside the segment
        carrying the visible content, so dropping whole trailing whitespace
        segments removes the layout and keeps the content — where `rstrip()`
        would take both.

        A line that is nothing but fill has no visible difference between
        source whitespace and layout, so it reduces to ``""``: the break is
        preserved without pasting a pane-width run of spaces.
        """
        segments = list(strip)
        while segments and segments[-1].text and segments[-1].text.isspace():
            segments.pop()
        return "".join(segment.text for segment in segments if not segment.control)

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        """Extract the selected completed text, without the layout padding.

        `Selection` indexes the rendered line by character — the coordinate
        system `render_line()` stamps with `Strip.apply_offsets()` — so
        `Selection.extract()` slices it directly. Only the fill is removed
        first, and because fill is appended it shifts no index a content
        character occupies; a selection reaching into it simply clamps.

        The rendered, reflowed text is what the user selected, so that is what
        is copied; the source Markdown is not reconstructed. The whole line
        cache is walked, but only when a copy is actually requested.
        """
        text = "\n".join(self._copyable_text(strip) for strip in self._lines)
        return selection.extract(text), "\n"

    async def action_open_link(self, href: str) -> None:
        """Open a completed-history hyperlink, exactly as it was written."""
        self.app.open_url(href)

    def _invalidate(self) -> None:
        """Drop the line cache so the next layout renders from scratch."""
        self._lines = []
        self._render_width = 0
        self._render_theme = None

    def _commit(self, lines: list[Strip], width: int, theme: str) -> None:
        """Adopt a completed rendering and the cache key it was built for."""
        self._lines = lines
        self._render_width = width
        self._render_theme = theme

    def _ensure_rendered(self, width: int) -> None:
        """Rebuild the line cache when ``width`` or the theme no longer match.

        Called from layout and paint, so it never refreshes: its caller is
        already producing the frame the new lines belong to.
        """
        if width <= 0:
            return
        theme = self.app.theme
        if width == self._render_width and theme == self._render_theme:
            return
        self._commit(self._render_records(self._records, width), width, theme)

    def _render_records(
        self, records: Sequence[TranscriptRecord], width: int
    ) -> list[Strip]:
        """Render ``records`` to lines, one independently rendered record each."""
        lines: list[Strip] = []
        for record in records:
            lines.extend(self._render_record(record, width))
        return lines

    def _render_record(self, record: TranscriptRecord, width: int) -> list[Strip]:
        """Render one record: a blank line, its label, its body, a blank line.

        Raises:
            RuntimeError: if the active theme lacks a color this record needs.
            ValueError: if the record carries an unknown format.
        """
        theme = self.app.current_theme
        body_style = self._body_style(record, theme)
        body: RenderableType
        if record.format == "markdown":
            body = _TranscriptMarkdown(
                record.content,
                code_theme="monokai" if theme.dark else "friendly",
                hyperlinks=True,
                style=body_style,
            )
        elif record.format == "plain":
            body = Text(record.content, style=body_style)
        else:
            raise ValueError(f"Unknown transcript format: {record.format!r}")
        if record.role == "user":
            # The bubble padding the live user widget gets from CSS.
            body = Padding(body, (1, 2), style=body_style, expand=True)
        label_style = Style(
            color=_theme_color(theme, "foreground"), bold=True, dim=True
        )
        group = Group(
            Text(""),
            Text(record.display_label, style=label_style),
            body,
            Text(""),
        )
        console = self.app.console
        segments = _with_link_actions(
            console.render(group, console.options.update_width(width))
        )
        return Strip.from_lines(list(Segment.split_lines(segments)))

    @staticmethod
    def _body_style(record: TranscriptRecord, theme: Theme) -> Style:
        """The body style for one record; an error color outranks its role."""
        color = _theme_color(theme, "error" if record.error else "foreground")
        if record.role == "user":
            return Style(color=color, bgcolor=_theme_color(theme, "surface"))
        if record.role in _DIM_ROLES:
            return Style(color=color, dim=True)
        return Style(color=color)


class Transcript(VerticalScroll):
    """The visible transcript: a completed prefix and a live widget tail.

    The visible order is always ``history.records`` followed by the tail
    entries in list order. Only the longest consecutive finalized prefix of the
    tail may move into history, so a finished message that sits behind a still
    removable notice waits as a widget rather than jumping ahead of it.

    Nothing outside this class may mutate the tail list, move a record into
    `TranscriptHistory`, or remove a tail widget.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._history = TranscriptHistory()
        self._tail: list[TailEntry] = []
        # Identity, not equality: a handle belongs to exactly one transcript.
        self._owner_token = object()

    @property
    def history(self) -> TranscriptHistory:
        """The completed-content widget, always the first child."""
        return self._history

    def compose(self) -> ComposeResult:
        yield self._history

    def load(self, records: Sequence[TranscriptRecord]) -> None:
        """Seed startup history in one rendering, mounting no per-record widget.

        Raises:
            RuntimeError: if the transcript already holds completed or live
                content. This is a startup-only path.
        """
        if self._history.records or self._tail:
            raise RuntimeError("Transcript.load() requires an empty transcript")
        self._history.set_records(records)

    def append(self, record: TranscriptRecord) -> None:
        """Add one already-complete record at the end of the transcript."""
        if self._tail:
            # Something removable is still ahead of it, so it waits its turn.
            self._add_tail(record, live=False)
            self._compact()
        else:
            self._history.extend([record])
        self.scroll_end(animate=False)

    def begin_markdown(
        self,
        role: str,
        content: str = "",
        *,
        label: str | None = None,
        hidden: bool = False,
    ) -> MarkdownTail:
        """Open a live Markdown entry and return its handle."""
        entry = self._add_tail(
            TranscriptRecord(role, content, label, format="markdown"),
            live=True,
            hidden=hidden,
        )
        self.scroll_end(animate=False)
        assert isinstance(entry, MarkdownTail)
        return entry

    def begin_plain(
        self,
        role: str,
        content: str = "",
        *,
        label: str | None = None,
        hidden: bool = False,
    ) -> PlainTail:
        """Open a live literal-text entry and return its handle."""
        entry = self._add_tail(
            TranscriptRecord(role, content, label, format="plain"),
            live=True,
            hidden=hidden,
        )
        self.scroll_end(animate=False)
        assert isinstance(entry, PlainTail)
        return entry

    def finalize(self, entry: TailEntry, record: TranscriptRecord) -> None:
        """Close a live entry with the content it ended up holding.

        Finalizing an entry that ``clear()`` already closed is a documented
        no-op: a provider that finishes after ``/clear`` must not redraw output
        the user removed.

        Raises:
            RuntimeError: if the handle belongs to another transcript or is in
                any state other than ``live`` or ``cleared``.
        """
        self._check_owner(entry, "finalize")
        if entry.state == "cleared":
            return
        if entry.state != "live":
            raise RuntimeError(f"Cannot finalize a {entry.state} transcript entry")
        entry.final_record = record
        entry.state = "finalized"
        self._compact()

    def remove(self, entry: TailEntry) -> None:
        """Drop a live entry and its widgets.

        Removing an already removed or cleared entry is a documented no-op:
        queue cleanup and clear cleanup are both idempotent.

        Raises:
            RuntimeError: if the handle belongs to another transcript or has
                already been finalized or compacted.
        """
        self._check_owner(entry, "remove")
        if entry.state in ("removed", "cleared"):
            return
        if entry.state != "live":
            raise RuntimeError(f"Cannot remove a {entry.state} transcript entry")
        self._remove_widgets(entry)
        entry.state = "removed"
        self._tail.remove(entry)
        self._compact()

    def clear(self) -> None:
        """Empty completed and live presentation, keeping the history widget."""
        self._history.clear()
        for entry in self._tail:
            entry.state = "cleared"
            self._remove_widgets(entry)
        self._tail.clear()

    def refresh_theme(self) -> None:
        """Re-render completed history for a newly applied theme."""
        self._history.reflow()

    def _add_tail(
        self,
        record: TranscriptRecord,
        *,
        live: bool,
        hidden: bool = False,
    ) -> TailEntry:
        """Mount one label/body pair and track it as the newest tail entry.

        Raises:
            ValueError: if the record carries an unknown format. An unknown
                format is a programming error, not a reason to guess Markdown.
        """
        body_classes = f"bubble {record.role}"
        if record.error:
            body_classes += " error"
        label = Static(record.display_label, classes=f"bubble-label {record.role}")
        entry: TailEntry
        if record.format == "markdown":
            entry = MarkdownTail(
                label=label,
                body=Markdown(record.content, classes=body_classes),
                _owner=self._owner_token,
            )
        elif record.format == "plain":
            entry = PlainTail(
                label=label,
                body=Static(record.content, classes=body_classes),
                _owner=self._owner_token,
            )
        else:
            raise ValueError(f"Unknown transcript format: {record.format!r}")
        if not live:
            entry.state = "finalized"
            entry.final_record = record
        self.mount(entry.label, entry.body)
        if hidden:
            entry.label.display = False
            entry.body.display = False
        self._tail.append(entry)
        return entry

    def _compact(self) -> None:
        """Move the longest finalized tail prefix into completed history.

        Ordering is the whole point: the prefix stops at the first entry that
        is still live, because moving a later record past it would reorder the
        transcript.
        """
        prefix: list[TailEntry] = []
        for entry in self._tail:
            if entry.state != "finalized":
                break
            prefix.append(entry)
        if not prefix:
            return
        records: list[TranscriptRecord] = []
        for entry in prefix:
            if entry.final_record is None:
                raise RuntimeError("A finalized transcript entry has no record")
            records.append(entry.final_record)
        with self.app.batch_update():
            # Rendering first: a failure here must leave the widgets that are
            # still showing this content mounted and the tail untouched.
            self._history.extend(records)
            for entry in prefix:
                self._remove_widgets(entry)
                entry.state = "compacted"
            del self._tail[: len(prefix)]

    def _check_owner(self, entry: TailEntry, operation: str) -> None:
        """Refuse to operate on a handle this transcript did not create."""
        if entry._owner is not self._owner_token:
            raise RuntimeError(
                f"Cannot {operation} a transcript entry owned by another transcript"
            )

    @staticmethod
    def _remove_widgets(entry: TailEntry) -> None:
        """Take both widgets of one tail entry out of the DOM.

        ``parent``, not ``is_mounted``: Textual finishes mounting on a later
        pass of the loop, so a fast turn can close an entry before its widgets
        report themselves mounted. They are in the DOM from the mount call
        onward, and that is what has to be undone.
        """
        for widget in (entry.label, entry.body):
            if widget.parent is not None:
                widget.remove()


# --- agent workspace -------------------------------------------------------

AgentStatus = Literal["idle", "running", "waiting", "completed", "failed"]

# Status is carried by shape, not color: a theme change, a monochrome terminal,
# or a color-blind reader must not erase it. This mapping is the single source
# of that signal for both validation and rendering.
_AGENT_STATUS_GLYPHS: dict[str, str] = {
    "idle": "○",
    "running": "●",
    "waiting": "◌",
    "completed": "✓",
    "failed": "!",
}

_SELECTED_MARKER = "▸ "
_UNSELECTED_MARKER = "  "
# Two columns past the agent marker, so a task label lands exactly two columns
# deeper than its agent label without a tree connector or a second cursor.
_TASK_MARKER = "    "


@dataclass(frozen=True, slots=True)
class AgentTaskSummary:
    """One task row beneath its owning agent, as presentation data only.

    Args:
        task_id: Opaque orchestration-owned identity, unique within its agent.
        label: The single line drawn for this task; it is ellipsized, never
            wrapped, at the sidebar edge.
        status: One of the five declared `AgentStatus` values.
    """

    task_id: str
    label: str
    status: AgentStatus


@dataclass(frozen=True, slots=True)
class AgentSummary:
    """One agent row and its tasks, as presentation data only.

    This type crosses the orchestration-to-presentation boundary and nothing
    else: it carries no prompt, credential, session, command policy, task
    output, or mutable runtime object.

    Args:
        agent_id: Opaque orchestration-owned identity, unique within one
            workspace and stable for that workspace's lifetime. No DOM id is
            ever derived from it.
        label: The single line drawn for this agent.
        status: One of the five declared `AgentStatus` values.
        tasks: Presentation-only task rows, drawn in the supplied order. The
            workspace neither sorts nor re-parents them.
    """

    agent_id: str
    label: str
    status: AgentStatus
    tasks: tuple[AgentTaskSummary, ...] = ()


def _status_glyph(status: str) -> str:
    """Return the fixed glyph for ``status``.

    Raises:
        ValueError: if ``status`` is not one of the five declared values. An
            unknown status is reported rather than drawn as a blank or
            substituted glyph, which would silently erase the status signal.
    """
    try:
        return _AGENT_STATUS_GLYPHS[status]
    except KeyError:
        raise ValueError(f"Unknown agent status: {status!r}") from None


def _check_single_line(value: str, what: str) -> None:
    """Reject a label that would not occupy exactly one sidebar row.

    A row is one line for the agent plus one line per task, so a label carrying
    its own line break forges a row the orchestration layer never declared —
    a fake agent or a fake task, drawn indistinguishably from a real one.
    ``splitlines()`` is the predicate because it is exactly what splits the
    rendered text: it catches an interior break, a trailing one, and the
    carriage returns and Unicode separators a terminal would break on too.

    Raises:
        ValueError: if ``value`` is anything other than one line.
    """
    if value.splitlines() != [value]:
        raise ValueError(f"{what} must be a single line: {value!r}")


def _validate_agent_summary(summary: AgentSummary) -> None:
    """Check one summary completely, before anything is mutated for it.

    This is the only validation on the orchestration-to-presentation boundary,
    shared by insertion and update, so neither path can accept data the other
    would reject. Nothing here strips, generates, repairs, or substitutes
    caller data — a label with a line break is rejected, never silently
    flattened.

    Raises:
        ValueError: if an id or label is empty, a label spans more than one
            line, a status is unknown, or two tasks of this agent share a task
            id.
    """
    if not summary.agent_id:
        raise ValueError("An agent summary needs a non-empty agent id")
    if not summary.label:
        raise ValueError(f"Agent {summary.agent_id!r} needs a non-empty label")
    _check_single_line(summary.label, f"The label of agent {summary.agent_id!r}")
    _status_glyph(summary.status)
    seen: set[str] = set()
    for task in summary.tasks:
        if not task.task_id:
            raise ValueError(
                f"A task of agent {summary.agent_id!r} needs a non-empty task id"
            )
        if not task.label:
            raise ValueError(
                f"Task {task.task_id!r} of agent {summary.agent_id!r} needs a "
                "non-empty label"
            )
        _check_single_line(
            task.label,
            f"The label of task {task.task_id!r} of agent {summary.agent_id!r}",
        )
        _status_glyph(task.status)
        if task.task_id in seen:
            raise ValueError(
                f"Agent {summary.agent_id!r} has a duplicate task id: "
                f"{task.task_id!r}"
            )
        seen.add(task.task_id)


def render_agent_summary(summary: AgentSummary, *, selected: bool) -> Text:
    """Render one agent block: its own line, then one line per task.

    Pure by design — no I/O and no app state — so the hierarchy and the status
    glyphs are testable without mounting a TUI. Every agent line starts at the
    same column whether or not it is selected, so the two-column marker never
    shifts the hierarchy; every task line is indented two columns further.

    Args:
        summary: The agent to draw. Its tasks are drawn in the supplied order.
        selected: Whether this agent's activity stream is the visible one. This
            is the selected marker, not the ListView keyboard highlight.

    Raises:
        ValueError: if the agent or one of its tasks carries an unknown status.
    """
    marker = _SELECTED_MARKER if selected else _UNSELECTED_MARKER
    lines = [f"{marker}{_status_glyph(summary.status)} {summary.label}"]
    lines.extend(
        f"{_TASK_MARKER}{_status_glyph(task.status)} {task.label}"
        for task in summary.tasks
    )
    return Text("\n".join(lines))


class _AgentListItem(ListItem):
    """One selectable sidebar row: an agent plus its non-selectable tasks.

    Tasks are lines in this row's single renderable rather than child
    ``ListItem``s, so Textual's own list navigation selects agents only and no
    second task cursor has to exist.
    """

    def __init__(self, summary: AgentSummary, *, selected: bool) -> None:
        super().__init__(classes="agent-list-item")
        self._summary = summary
        self._text = Static(
            render_agent_summary(summary, selected=selected),
            classes="agent-list-text",
        )

    @property
    def agent_id(self) -> str:
        """The opaque agent id this row was created for and never leaves."""
        return self._summary.agent_id

    def compose(self) -> ComposeResult:
        yield self._text

    def update_summary(self, summary: AgentSummary, *, selected: bool) -> None:
        """Repaint this row from ``summary``.

        Raises:
            ValueError: if ``summary`` describes a different agent. A row is
                bound to one agent for its lifetime; rebinding it would move an
                agent's tasks and status under another agent's identity.
        """
        if summary.agent_id != self._summary.agent_id:
            raise ValueError(
                f"Cannot rebind the sidebar row for {self._summary.agent_id!r} "
                f"to {summary.agent_id!r}"
            )
        self._summary = summary
        self._text.update(render_agent_summary(summary, selected=selected))


class AgentWorkspace(Horizontal):
    """The activity/sidebar split: one stable `Transcript` per agent.

    Every registered agent keeps its own transcript for the workspace lifetime.
    Selecting an agent changes only which transcript is displayed — it copies no
    records, reloads no history, rebuilds no live stream handle, and retargets
    nothing. Each agent therefore keeps its own content, live tail, and scroll
    position while another agent is visible.

    Orchestration owns agents, tasks, and statuses; this widget owns navigation
    and visibility only. It never infers an agent from prose, polls a file, or
    inspects a worker.
    """

    class AgentSelected(Message):
        """The visible activity stream changed to ``agent_id``."""

        def __init__(self, workspace: AgentWorkspace, agent_id: str) -> None:
            super().__init__()
            self.workspace = workspace
            self.agent_id = agent_id

        @property
        def control(self) -> AgentWorkspace:
            return self.workspace

    def __init__(
        self,
        primary: AgentSummary,
        primary_transcript: Transcript,
        **kwargs,
    ) -> None:
        """Compose the workspace around an already-created Primary transcript.

        Args:
            primary: The Primary agent summary. It is selected on mount and is
                the only agent present until orchestration registers more.
            primary_transcript: The caller's own Primary `Transcript`, so the
                stable ``#chat`` id and every selector built on it stay valid.

        Raises:
            ValueError: if ``primary`` fails boundary validation.
        """
        super().__init__(**kwargs)
        _validate_agent_summary(primary)
        self._summaries: dict[str, AgentSummary] = {primary.agent_id: primary}
        self._activities: dict[str, Transcript] = {
            primary.agent_id: primary_transcript
        }
        self._items: dict[str, _AgentListItem] = {
            primary.agent_id: _AgentListItem(primary, selected=True)
        }
        # Agent ids claimed by a registration that has not finished yet.
        # ``add_agent()`` awaits two DOM mutations, so the registry alone cannot
        # answer "is this id taken?" across that gap.
        self._pending: set[str] = set()
        self._primary_agent_id = primary.agent_id
        self._selected_agent_id = primary.agent_id

    def compose(self) -> ComposeResult:
        with Vertical(id="activity-stack"):
            yield self._activities[self._primary_agent_id]
        with Vertical(id="agent-sidebar"):
            yield Static("Agents", id="agent-sidebar-title")
            yield ListView(self._items[self._primary_agent_id], id="agent-list")

    @property
    def primary_agent_id(self) -> str:
        """The agent whose composer, session, and runtime the app owns."""
        return self._primary_agent_id

    @property
    def selected_agent_id(self) -> str:
        """The agent whose activity stream is currently displayed."""
        return self._selected_agent_id

    @property
    def selected_activity(self) -> Transcript:
        """The displayed transcript. Presentation only: it owns nothing."""
        return self._activities[self._selected_agent_id]

    def summary_for(self, agent_id: str) -> AgentSummary:
        """The current summary for ``agent_id``.

        Raises:
            KeyError: if no such agent is registered.
        """
        try:
            return self._summaries[agent_id]
        except KeyError:
            raise KeyError(f"Unknown agent id: {agent_id!r}") from None

    def activity_for(self, agent_id: str) -> Transcript:
        """The stable transcript registered for ``agent_id``.

        Raises:
            KeyError: if no such agent is registered.
        """
        try:
            return self._activities[agent_id]
        except KeyError:
            raise KeyError(f"Unknown agent id: {agent_id!r}") from None

    async def add_agent(
        self,
        summary: AgentSummary,
        records: Sequence[TranscriptRecord] = (),
    ) -> Transcript:
        """Register one agent and return the transcript the caller must retain.

        The returned object is that agent's activity stream for the workspace
        lifetime; every later update for the agent goes to it directly, never to
        whichever transcript happens to be selected.

        Adding an agent does not select or focus it, and it does not disturb the
        current selection. Call this on Textual's event loop, with the workspace
        already mounted; a violation of that contract surfaces as the normal
        mount error rather than being queued or dropped.

        Registration spans two awaited DOM mutations, so the id is claimed
        before the first of them: two concurrent registrations of one id would
        otherwise both pass a check made only against the finished registry,
        both mount, and leave one transcript orphaned under a sidebar row the
        registry cannot reach. A registration that fails or is cancelled after
        mounting takes its widgets back out and releases the claim, so the
        workspace is left exactly as it was and the id is free to retry.

        Args:
            summary: The new agent's presentation data.
            records: Completed activity to seed, rendered in one pass.

        Raises:
            ValueError: if ``summary`` fails boundary validation, or if its
                agent id is already registered or currently being registered.
        """
        _validate_agent_summary(summary)
        agent_id = summary.agent_id
        if agent_id in self._summaries:
            raise ValueError(f"Agent {agent_id!r} is already registered")
        if agent_id in self._pending:
            raise ValueError(f"Agent {agent_id!r} is already being registered")
        # Claimed synchronously, before any await can yield to another caller.
        self._pending.add(agent_id)
        transcript = Transcript(classes="agent-activity")
        transcript.display = False
        item = _AgentListItem(summary, selected=False)
        try:
            try:
                await self.query_one("#activity-stack", Vertical).mount(transcript)
                await self.query_one("#agent-list", ListView).append(item)
                if records:
                    transcript.load(records)
            except BaseException:
                # BaseException, because cancellation between the mounts is the
                # likeliest way this unwinds and it must not leave a widget the
                # registry never learned about. ``parent``, not ``is_mounted``:
                # Textual finishes mounting on a later pass of the loop, but a
                # widget is in the DOM from the mount call onward. The base
                # method is named explicitly because ``Transcript.remove()``
                # means "drop this tail entry", not "leave the DOM".
                for widget in (transcript, item):
                    if widget.parent is not None:
                        Widget.remove(widget)
                raise
            # Synchronous from here: the three registries commit together.
            self._summaries[agent_id] = summary
            self._activities[agent_id] = transcript
            self._items[agent_id] = item
        finally:
            self._pending.discard(agent_id)
        return transcript

    def update_agent(self, summary: AgentSummary) -> None:
        """Replace one agent's stored summary and repaint its sidebar row.

        This is presentation only: it never replaces, reloads, clears, scrolls,
        shows, or hides a transcript, and it never changes the selection.

        Raises:
            ValueError: if ``summary`` fails boundary validation.
            KeyError: if its agent id is not registered.
        """
        _validate_agent_summary(summary)
        if summary.agent_id not in self._summaries:
            raise KeyError(f"Unknown agent id: {summary.agent_id!r}")
        self._summaries[summary.agent_id] = summary
        self._items[summary.agent_id].update_summary(
            summary, selected=summary.agent_id == self._selected_agent_id
        )

    def select_agent(self, agent_id: str) -> None:
        """Show ``agent_id``'s activity stream and mark it selected.

        Selecting the already selected agent is an idempotent no-op, so it
        cannot discard a selection the user just made inside the visible
        transcript.

        A real change clears the screen-level arbitrary text selection first:
        that selection lives in `Screen.selections`, not in a transcript, so
        leaving it intact would let a copy pull text out of an attached but
        invisible agent. Nothing else is cleared — not the composer value, not
        its widget-owned selection, not any transcript content.

        Raises:
            KeyError: if no such agent is registered. There is no fallback to
                Primary or to the first agent.
        """
        if agent_id not in self._summaries:
            raise KeyError(f"Unknown agent id: {agent_id!r}")
        previous_id = self._selected_agent_id
        if agent_id == previous_id:
            return
        self.screen.clear_selection()
        self._activities[previous_id].display = False
        self._activities[agent_id].display = True
        self._selected_agent_id = agent_id
        self._items[previous_id].update_summary(
            self._summaries[previous_id], selected=False
        )
        self._items[agent_id].update_summary(
            self._summaries[agent_id], selected=True
        )
        self.post_message(self.AgentSelected(self, agent_id))

    def refresh_theme(self) -> None:
        """Re-render completed history for every agent, hidden ones included.

        A hidden transcript that skipped a theme change would reveal stale
        colors the moment it is selected.
        """
        for transcript in self._activities.values():
            transcript.refresh_theme()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Commit a keyboard or mouse selection to the visible activity stream.

        Only ``Selected`` is acted on: ``Highlighted`` is keyboard navigation,
        not commitment, so arrowing through agents never changes what is shown.

        Raises:
            TypeError: if the sidebar somehow holds a row this workspace did not
                create, which would mean its agent identity is unknown.
        """
        item = event.item
        if not isinstance(item, _AgentListItem):
            raise TypeError(f"The agent sidebar holds an unexpected row: {item!r}")
        self.select_agent(item.agent_id)
