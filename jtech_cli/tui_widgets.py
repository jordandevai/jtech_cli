"""Reusable Textual widgets and messages used by the chat TUI."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import ClassVar, Literal, Protocol, runtime_checkable

from markdown_it import MarkdownIt
from rich.console import Console, Group, RenderableType
from rich.errors import MarkupError
from rich.markdown import Markdown as RichMarkdown
from rich.padding import Padding
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.geometry import Size
from textual.message import Message
from textual.selection import Selection
from textual.strip import Strip
from textual.theme import Theme
from textual.widget import Widget
from textual.widgets import Input, Markdown, Static, TextArea


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


class InputToMultiline(Message):
    """Request that the app replace the single-line editor with a text area."""

    def __init__(self, value: str) -> None:
        super().__init__()
        self.value = value


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

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+enter", "submit_input", "Submit", show=False),
        Binding("shift+enter", "to_multiline", "Multi-line", show=False),
        Binding("enter", "submit_or_complete", "Submit", show=False),
        Binding("tab", "tab_complete", "Complete", show=False),
    ]

    def action_submit_input(self) -> None:
        self.post_message(Input.Submitted(self, self.value))

    def action_submit_or_complete(self) -> None:
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
        self.post_message(InputToMultiline(self.value))


class _MultilineInput(TextArea):
    """Multi-line input; Ctrl+Enter submits and Esc cancels."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+enter", "multiline_submit", "Submit", show=False),
        Binding("escape", "multiline_cancel", "Cancel", show=False),
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


class _PromptEditor(TextArea):
    """Multi-line in-place editor for the system prompt row."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("enter", "commit_field", "Commit", show=False, priority=True),
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
        """Return one rendered line, painting the active selection over it."""
        width = self.size.width
        self._ensure_rendered(width)
        rich_style = self.rich_style
        if y >= len(self._lines):
            return Strip.blank(width, rich_style)
        line = self._lines[y].crop_extend(0, width, rich_style).apply_style(rich_style)
        selection = self.text_selection
        if selection is None:
            return line
        span = selection.get_span(y)
        if span is None:
            return line
        start, end = span
        length = line.cell_length
        end = length if end == -1 else end
        start = max(0, min(start, length))
        end = max(start, min(end, length))
        if start == end:
            return line
        before, selected, after = line.divide([start, end, length])
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

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        """Extract the selected completed text, without the layout padding.

        Rendered lines are padded out to the pane width so role backgrounds
        fill it. That padding is layout, not content: copied across several
        lines it would bury pasted prose and code in trailing spaces. Dropping
        only trailing space leaves every column offset inside a line intact, so
        the extracted text still lines up with the visual selection.
        """
        text = "\n".join(strip.text.rstrip() for strip in self._lines)
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
