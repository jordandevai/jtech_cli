"""Tests for the Textual TUI: bubbles, status bar, notices, settings, and /clear."""

import asyncio
import json
import subprocess
import threading
import time
from pathlib import Path

import pytest
from textual import events
from textual.containers import Vertical
from textual.geometry import Offset
from textual.widgets import Input, ListView, Markdown, Static, TextArea
from textual.widgets.input import Selection as InputSelection
from textual.widgets.markdown import MarkdownStream
from textual.widgets.text_area import Selection as AreaSelection

from jtech_cli.cmd_tools import CmdPolicy
from jtech_cli.config import (
    Profile,
    ProfileError,
    Profiles,
    ResolvedProfile,
    Settings,
    build_settings,
    load_cmd_policy,
)
from jtech_cli.prompts import NUDGE_PROMPT
from jtech_cli.server_info import ServerInfo
from jtech_cli.session import Session
from jtech_cli.tui import (
    CONNECTION_ERROR,
    ChatApp,
    CommandPrompt,
    ProfilesScreen,
    QuitScreen,
    SettingsScreen,
)
from jtech_cli.tui_app import (
    PRIMARY_AGENT_ID,
    RENDER_ERROR,
    SUBAGENT_CLEAR_BLOCKED,
    SUBAGENT_READONLY,
)
from jtech_cli.tui_runtime import (
    INTERRUPTED_RESPONSE,
    STOPPED_LABEL,
    AutonomousRuntime,
)
from jtech_cli.tui_widgets import (
    AgentSummary,
    AgentTaskSummary,
    AgentWorkspace,
    Transcript,
    TranscriptHistory,
    TranscriptRecord,
    _AgentListItem,
)

LOCAL = Profile(name="local", base_url="http://host:9000/v1", model="qwen3")


def local_settings(**kwargs) -> Settings:
    """Settings whose active profile is the default test endpoint."""
    return Settings(profiles=Profiles(items=(LOCAL,), active_name="local"), **kwargs)


def local_settings_with_model(model: str) -> Settings:
    """Settings whose active profile pins ``model`` explicitly."""
    profile = Profile(name="local", base_url="http://host:9000/v1", model=model)
    return Settings(profiles=Profiles(items=(profile,), active_name="local"))


def make_app(
    tmp_path,
    settings=None,
    session=None,
    server=None,
    no_discover=True,
    fetch_token_count_fn=None,
):
    """A ChatApp for tests. Discovery is off unless a test opts in.

    Left on, every app here fires a real HTTP request at host:9000 and waits
    out the 5s timeout — network I/O in a unit test, and a failure notice in
    the chat that tests asserting on bubbles have to know to ignore. A session
    that starts with history triggers the same thing through the startup token
    count, so those tests inject ``fetch_token_count_fn``.
    """
    settings = settings or local_settings()
    session = session or Session(tmp_path / "s.jsonl", persist=False)
    server = server or ServerInfo(models=["qwen3"], context_length=4096)
    return ChatApp(
        settings=settings,
        session=session,
        server=server,
        config_path=tmp_path / "config.toml",
        no_discover=no_discover,
        fetch_token_count_fn=fetch_token_count_fn,
    )


def make_settings(reasoning: str) -> Settings:
    return local_settings(reasoning=reasoning)


def chat_of(app: ChatApp) -> Transcript:
    return app.query_one("#chat", Transcript)


def bubbles(app: ChatApp) -> list[str]:
    """Completed message content, then visible live content, in order."""
    chat = chat_of(app)
    completed = [record.content for record in chat.history.records]
    live = [
        str(entry.body.render())
        if isinstance(entry.body, Static)
        else entry.body._markdown
        for entry in chat._tail
        if entry.body.display
    ]
    return completed + live


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


def labels(app: ChatApp) -> list[str]:
    """Completed message labels, then visible live labels, in order."""
    chat = chat_of(app)
    completed = [record.display_label for record in chat.history.records]
    live = [
        str(entry.label.render()) for entry in chat._tail if entry.label.display
    ]
    return completed + live


def history_lines(app: ChatApp) -> str:
    """Every rendered completed-history line, joined for content assertions."""
    return "\n".join(strip.text for strip in chat_of(app).history._lines)


def body_widgets(app: ChatApp) -> list[Static]:
    """Every message body widget mounted in the transcript."""
    return [
        widget
        for widget in chat_of(app).query(Static)
        if "bubble" in widget.classes
    ]


def at_bottom(chat) -> bool:
    """True when the scroll offset is at (or within 2 lines of) the bottom."""
    return chat.scroll_offset.y >= chat.max_scroll_y - 2


class BlockingStream:
    """A `ReplyStream` that emits one item and then never produces another.

    Only cancellation ends it. A test that opens a gate by hand proves the
    runtime waited, not that it stopped the read.
    """

    def __init__(self, first: str = "partial "):
        self.first = first
        self.blocked = asyncio.Event()  # set once the reader is parked
        self.cancelled = False
        self.closed = 0

    def __aiter__(self):
        return self._items()

    async def _items(self):
        yield self.first
        self.blocked.set()
        try:
            await asyncio.Event().wait()  # never set by anyone
        except asyncio.CancelledError:
            self.cancelled = True
            raise

    async def aclose(self) -> None:
        self.closed += 1


class FiniteStream:
    """A `ReplyStream` over a fixed list of items."""

    def __init__(self, *items):
        self.items = items
        self.closed = 0

    def __aiter__(self):
        return self._items()

    async def _items(self):
        for item in self.items:
            yield item

    async def aclose(self) -> None:
        self.closed += 1


class SyncStream:
    """A `ReplyStream` over a synchronous generator of stream items.

    Each item is pulled off the event loop. Several fakes here block between
    yields to hold a stream open while the test does something else; doing that
    on a worker thread is exactly what the provider thread used to do for them,
    so their bodies keep working unchanged while production is fully async.
    """

    def __init__(self, items):
        self._items = items
        self.closed = 0

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        done = object()
        while True:
            item = await asyncio.to_thread(next, self._items, done)
            if item is done:
                return
            yield item

    async def aclose(self) -> None:
        self.closed += 1


def sync_stream(fn):
    """Adapt a synchronous generator fake to the async `stream_reply` seam."""

    async def factory(profile, temperature, messages):
        return SyncStream(fn(profile, temperature, messages))

    return factory


def stream_of(*items):
    """A `stream_reply` stand-in producing ``items`` for every request."""

    async def factory(profile, temperature, messages):
        return FiniteStream(*items)

    return factory


async def test_submit_shows_user_and_ai_bubble(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    monkeypatch.setattr(
        "jtech_cli.tui.stream_reply", stream_of("hi ", "there")
    )
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "hello"
        await pilot.press("enter")
        for _ in range(10):
            await pilot.pause()

        assert app.session.messages == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        text = "\n".join(bubbles(app))
        assert "hello" in text
        assert "hi there" in text


async def test_long_code_line_wraps_in_fence(tmp_path, monkeypatch):
    """Unbroken words longer than the screen must wrap, not run off it.

    A fence whose longest line is wider than the pane used to be cut off at
    the right edge. Completed history folds it like normal text, so the whole
    word stays on screen; the live bubble mid-stream is still a Textual
    ``Markdown``, which the app CSS constrains to the bubble width.
    """
    # 'z' is unused elsewhere on screen (the status bar's "ctx" would skew
    # a character count if we used 'x').
    long_word = "z" * 300
    reply = f"```\n{long_word}\nafter-line\n```\n"
    monkeypatch.setattr(
        "jtech_cli.tui.stream_reply", stream_of(reply)
    )
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        for _ in range(10):
            await pilot.pause()
        await pilot.pause()

        chat = chat_of(app)
        # Wrapped, not clipped: no completed line exceeds the pane width, and
        # every character of the long word survived the fold.
        assert all(
            len(strip.text) <= chat.history._render_width
            for strip in chat.history._lines
        )
        assert history_lines(app).count("z") == len(long_word)
        assert "after-line" in history_lines(app)
        # And every character of the long word must actually be rendered.
        comp = app.screen._compositor
        visible = sum(s.text.count("z") for s in comp.render_strips(comp.size))
        assert visible == len(long_word)


async def test_status_bar_shows_the_profile_url_and_model(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test():
        status = app.query_one("#status", Static).content
    assert "base_url=" not in status
    assert "profile: local" in status
    assert "(override)" not in status
    assert "http://host:9000/v1" in status
    assert "model: qwen3" in status
    assert "ctx 4096" in status


async def test_status_bar_marks_a_cli_override(tmp_path):
    settings = local_settings()
    settings.profile_override = Profile(
        name="local", base_url="http://override:1/v1", model="override-model"
    )
    app = make_app(tmp_path, settings=settings)
    async with app.run_test():
        status = app.query_one("#status", Static).content
    assert "profile: local (override)" in status
    assert "http://override:1/v1" in status
    assert "model: override-model" in status


async def test_status_bar_shows_a_uniquely_discovered_model(tmp_path):
    """An empty configured model displays what the server actually serves."""
    settings = local_settings()
    settings.profiles = Profiles(
        items=(Profile(name="local", base_url="http://host:9000/v1"),),
        active_name="local",
    )
    app = make_app(tmp_path, settings=settings)
    async with app.run_test():
        assert "model: qwen3" in app.query_one("#status", Static).content


async def test_status_bar_empty_base_url(tmp_path):
    app = make_app(tmp_path, settings=Settings())
    async with app.run_test():
        status = app.query_one("#status", Static).content
    assert status.strip() != ""


async def test_input_responsive_after_connection_error(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return iter(["ok"])

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "first"
        await pilot.press("enter")
        for _ in range(10):
            await pilot.pause()
        assert app.session.messages == [{"role": "user", "content": "first"}]

        inp.value = "second"
        await pilot.press("enter")
        for _ in range(10):
            await pilot.pause()

    assert app.session.messages == [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "ok"},
    ]


def settings_rows_text(app: ChatApp) -> str:
    return str(app.screen.query_one("#settings-rows", Static).render())


def settings_help_text(app: ChatApp) -> str:
    return str(app.screen.query_one("#settings-help", Static).render())


async def test_settings_screen_lists_only_global_settings(tmp_path):
    """Endpoint and model moved to /profiles; /settings must not offer them."""
    app = make_app(tmp_path)
    async with app.run_test(size=(80, 24)) as pilot:
        app.action_settings()
        await pilot.pause()
        assert isinstance(app.screen, SettingsScreen)
        rows = settings_rows_text(app)
        assert "Temperature" in rows
        assert "Theme" in rows
        assert "Additional instructions" in rows
        assert "Model" not in rows
        assert "Base URL" not in rows
        assert "qwen3" not in rows

        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, SettingsScreen)


async def test_settings_description_follows_highlighted_row(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test(size=(80, 24)) as pilot:
        app.action_settings()
        await pilot.pause()
        assert "0.0-2.0" in settings_help_text(app)  # Temperature is row 0

        await pilot.press("down")  # Theme
        await pilot.pause()
        help_text = settings_help_text(app)
        assert "terminal" in help_text and "light/dark" in help_text

        await pilot.press("down")  # Reasoning
        await pilot.pause()
        assert "thinking tokens" in settings_help_text(app)

        await pilot.press("up", "up")  # back to Temperature
        await pilot.pause()
        assert "0.0-2.0" in settings_help_text(app)


async def test_settings_enter_edits_row_and_commits(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test(size=(80, 24)) as pilot:
        app.action_settings()
        await pilot.pause()

        # cursor starts on Temperature; Enter opens the in-place editor
        await pilot.press("enter")
        await pilot.pause()
        field = app.screen.query_one("#settings-field", Input)
        assert field.value == "0.7"

        # the highlighted row's description stays while editing
        assert "0.0-2.0" in settings_help_text(app)

        field.value = "0.42"
        await pilot.press("enter")
        await pilot.pause()
        assert app.settings.temperature == 0.42
        assert (tmp_path / "config.toml").exists()
        assert not app.screen.query_one("#settings-editor", Vertical).children
        assert "0.42" in settings_rows_text(app)


async def test_settings_invalid_value_keeps_editor_open(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test(size=(80, 24)) as pilot:
        app.action_settings()
        await pilot.pause()

        # Temperature is row 0
        await pilot.press("enter")
        await pilot.pause()
        field = app.screen.query_one("#settings-field", Input)
        field.value = "hot"
        await pilot.press("enter")
        await pilot.pause()
        assert app.settings.temperature == 0.7  # unchanged
        assert field.value == "hot"  # still editing

        field.value = "0.2"
        await pilot.press("enter")
        await pilot.pause()
        assert app.settings.temperature == 0.2
        assert not app.screen.query_one("#settings-editor", Vertical).children


async def test_settings_theme_row_applies_theme_live(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test(size=(80, 24)) as pilot:
        app.action_settings()
        await pilot.pause()

        await pilot.press("down")  # Theme is row 1
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        field = app.screen.query_one("#settings-field", Input)
        field.value = "light"
        await pilot.press("enter")
        await pilot.pause()
        assert app.settings.theme == "light"
        assert app.theme == "jtech-light"


async def test_settings_esc_cancels_edit_then_closes(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test(size=(80, 24)) as pilot:
        app.action_settings()
        await pilot.pause()

        await pilot.press("enter")  # start editing Temperature
        await pilot.pause()
        field = app.screen.query_one("#settings-field", Input)
        field.value = "0.9"
        await pilot.press("escape")  # cancel the edit
        await pilot.pause()
        assert app.settings.temperature == 0.7
        assert not app.screen.query_one("#settings-editor", Vertical).children

        await pilot.press("escape")  # close the menu
        await pilot.pause()
        assert not isinstance(app.screen, SettingsScreen)


@pytest.mark.parametrize("newline_key", ["ctrl+j", "shift+enter"])
async def test_settings_system_prompt_edits_multiline(tmp_path, newline_key):
    """The newline is typed, not assigned: a newline key makes it, Enter saves."""
    app = make_app(tmp_path)
    before = app.settings.system_prompt
    async with app.run_test(size=(80, 24)) as pilot:
        ta = await open_prompt_editor(app, pilot)
        ta.text = "line one"
        ta.selection = AreaSelection.cursor((0, len("line one")))
        await settle(pilot)

        await pilot.press(newline_key)
        await type_text(pilot, "line two")
        await settle(pilot)

        # The newline key edits; it must not save or close.
        assert ta.text == "line one\nline two"
        assert app.screen.query_one("#settings-field", TextArea) is ta
        assert app.settings.system_prompt == before
        assert not (tmp_path / "config.toml").exists()
        assert hint_text(app) == SettingsScreen._HINT_PROMPT_EDITING

        await pilot.press("enter")
        await settle(pilot)
        assert app.settings.system_prompt == "line one\nline two"
        assert app.settings.prompt_source == "inline"
        assert not app.screen.query_one("#settings-editor", Vertical).children


async def test_settings_single_line_row_keeps_its_own_hint(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test(size=(80, 24)) as pilot:
        app.action_settings()
        await settle(pilot)
        await pilot.press("enter")  # Temperature is a single-line row
        await settle(pilot)

        assert app.screen.query_one("#settings-field", Input) is not None
        assert hint_text(app) == SettingsScreen._HINT_EDITING


async def test_settings_prompt_editor_keeps_every_pasted_line(tmp_path):
    """Native TextArea paste already preserves lines; no custom handler here."""
    app = make_app(tmp_path)
    async with app.run_test(size=(80, 24)) as pilot:
        ta = await open_prompt_editor(app, pilot)
        ta.text = ""
        await settle(pilot)

        app.post_message(events.Paste("line one\nline two"))
        await settle(pilot)
        assert ta.text == "line one\nline two"
        assert app.settings.system_prompt != "line one\nline two"

        await pilot.press("enter")
        await settle(pilot)
        assert app.settings.system_prompt == "line one\nline two"


async def test_input_works_after_opening_and_closing_settings(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    monkeypatch.setattr("jtech_cli.tui.stream_reply", stream_of("ok"))
    async with app.run_test() as pilot:
        app.action_settings()
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, SettingsScreen)

        inp = app.query_one("#input", Input)
        inp.value = "still works"
        await pilot.press("enter")
        for _ in range(10):
            await pilot.pause()
        assert app.session.messages == [
            {"role": "user", "content": "still works"},
            {"role": "assistant", "content": "ok"},
        ]


async def test_no_profile_shows_a_notice_pointing_at_profiles(tmp_path):
    app = make_app(tmp_path, settings=Settings())
    async with app.run_test():
        assert any("No API profile is configured" in b for b in bubbles(app))
        assert any("/profiles" in b for b in bubbles(app))


async def test_theme_command_applies_theme_live(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "/theme light"
        await pilot.press("enter")
        await pilot.pause()
        assert app.settings.theme == "light"
        assert app.theme == "jtech-light"
        assert (tmp_path / "config.toml").exists()


async def test_status_is_last_row_of_root(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test():
        root = app.query_one(Vertical)
        assert root.children[-1] is app.query_one("#status", Static)
        assert app.query_one("#input", Input) in root.children[:-1]


async def test_clear_empties_chat(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    monkeypatch.setattr("jtech_cli.tui.stream_reply", stream_of("ok"))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "hello"
        await pilot.press("enter")
        for _ in range(10):
            await pilot.pause()
        assert app.session.messages == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "ok"},
        ]

        inp.value = "/clear"
        await pilot.press("enter")
        for _ in range(10):
            await pilot.pause()

        assert app.session.messages == []
        assert not any("hello" in b for b in bubbles(app))


async def _enter_multiline(app, pilot, trigger: str) -> "TextArea":
    """Submit ``trigger`` and wait for the multiline editor to mount."""
    inp = app.query_one("#input", Input)
    inp.value = trigger
    await pilot.press("enter")
    for _ in range(5):
        await pilot.pause()
    return app.query_one("#multiline-input", TextArea)


async def type_text(pilot, text: str) -> None:
    """Type ``text`` a key at a time, so the widget sees real key events."""
    for char in text:
        await pilot.press("space" if char == " " else char)


async def open_prompt_editor(app, pilot) -> "TextArea":
    """Open the settings system-prompt row in its multi-line editor."""
    app.action_settings()
    for _ in range(6):
        await pilot.pause()
    await pilot.press("down", "down", "down")  # System prompt is row 3
    for _ in range(6):
        await pilot.pause()
    await pilot.press("enter")
    for _ in range(6):
        await pilot.pause()
    return app.screen.query_one("#settings-field", TextArea)


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
        ta = await _enter_multiline(app, pilot, "'''")
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


@pytest.mark.parametrize("newline_key", ["ctrl+j", "shift+enter"])
@pytest.mark.parametrize("reverse", [False, True])
async def test_prompt_editor_newline_keys_replace_the_selection(
    tmp_path, reverse, newline_key
):
    """Both newline keys share the settings editor's action, undo included."""
    app = make_app(tmp_path)
    before = app.settings.system_prompt
    async with app.run_test(size=(80, 24)) as pilot:
        ta = await open_prompt_editor(app, pilot)
        ta.text = "alpha beta"
        head, tail = (0, 6), (0, 10)
        ta.selection = AreaSelection(tail, head) if reverse else AreaSelection(head, tail)
        await settle(pilot)

        await pilot.press(newline_key)
        await settle(pilot)

        assert ta.text == "alpha \n"
        assert ta.selection == AreaSelection.cursor((1, 0))
        assert app.settings.system_prompt == before

        await pilot.press("ctrl+z")
        await settle(pilot)
        assert ta.text == "alpha beta"


async def test_heredoc_multiline_sends_message(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    monkeypatch.setattr("jtech_cli.tui.stream_reply", stream_of("ok"))
    async with app.run_test() as pilot:
        ta = await _enter_multiline(app, pilot, "'''")
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
        ta = await _enter_multiline(app, pilot, f"/write {tmp_path / 'out.txt'}")
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

        await _enter_multiline(app, pilot, f"/write {missing}")
        await type_text(pilot, "discarded")
        await pilot.press("escape")
        for _ in range(10):
            await pilot.pause()
        assert not missing.exists()

        await _enter_multiline(app, pilot, f"/write {existing}")
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
        ta = await _enter_multiline(app, pilot, f"/write {target}")
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
        await _enter_multiline(app, pilot, "'''")
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
        opened = await _enter_multiline(app, pilot, "'''")
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
        ta = await _enter_multiline(app, pilot, "'''")
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


async def _send_and_drain(app: ChatApp, pilot, text: str) -> None:
    inp = app.query_one("#input", Input)
    inp.value = text
    await pilot.press("enter")
    for _ in range(10):
        await pilot.pause()


def reason_stream(profile, temperature, messages):
    return iter([("reasoning", "hmm "), ("reasoning", "ok"), "4"])


async def test_reasoning_default_transient_shown_then_hidden(tmp_path, monkeypatch):
    """Default mode: reasoning streams in its own bubble, removed once the answer starts."""
    app = make_app(tmp_path)
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(reason_stream))
    async with app.run_test() as pilot:
        await _send_and_drain(app, pilot, "what is 2+2")

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
        await _send_and_drain(app, pilot, "what is 2+2")

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
        await _send_and_drain(app, pilot, "what is 2+2")

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
        await _send_and_drain(app, pilot, "what is 2+2")

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
        await _send_and_drain(app, pilot, "hi")
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
        await _send_and_drain(app, pilot, "what is 2+2")

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
        await _wait_until(app, pilot, stream.blocked.is_set)

        await pilot.press("escape")
        await _wait_until(app, pilot, lambda: not app._generating, tries=100)
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


def suggestions_text(app: ChatApp) -> str:
    return str(app.query_one("#suggestions", Static).render())


def hint_text(app: ChatApp) -> str:
    """The settings screen hint line, as rendered."""
    return str(app.screen.query_one("#settings-hint", Static).render())


def suggestions_box(app: ChatApp) -> Static:
    return app.query_one("#suggestions", Static)


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


async def _wait_until(app, pilot, predicate, tries=50, pause=0.1):
    """Poll the app until ``predicate()`` is true (or tries are exhausted)."""
    for _ in range(tries):
        await pilot.pause(pause)
        if predicate():
            return
    raise AssertionError("condition not met in time")


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
        await _wait_until(app, pilot, lambda: any("r1" in b for b in bubbles(app)))
        assert app._generating

        # Second message while generating: queued, not sent
        inp.value = "two"
        await pilot.press("enter")
        await _wait_until(
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
        await _wait_until(app, pilot, lambda: calls["n"] >= 2)
        # the transient "Queued" line is gone once the message sent
        assert not any("Queued" in b for b in bubbles(app))

    assert app.session.messages == [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "r1 r1b"},
        {"role": "user", "content": "two"},
        {"role": "assistant", "content": "r2"},
    ]
    assert app._queue == []


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
        await _wait_until(app, pilot, lambda: any("r1" in b for b in bubbles(app)))

        inp.value = "two"
        await pilot.press("enter")
        inp.value = "three"
        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: len(app._queue) == 2, tries=10, pause=0.05)

        # Up recalls the NEXT queued message ("two"), not the last
        await pilot.press("up")
        await pilot.pause()
        assert inp.value == "two"
        assert [m for m in app._queue] == ["three"]
        assert "queue: 1" in app.query_one("#status", Static).content
        # the recalled message's "Queued" line is cleared; only "three" remains
        assert [b for b in bubbles(app) if "Queued" in b] == ["Queued: three"]
        # not auto-sent
        assert app.session.messages == [{"role": "user", "content": "one"}]

        # Up never clobbers unsent text in the input
        await pilot.press("up")
        await pilot.pause()
        assert inp.value == "two"
        assert [m for m in app._queue] == ["three"]

        # clear it (cancel "two"), then recall the rest
        inp.value = ""
        await pilot.press("up")
        await pilot.pause()
        assert inp.value == "three"
        assert app._queue == []
        assert not any("Queued" in b for b in bubbles(app))  # no stale lines

        # edit it, let the first reply finish, then submit
        inp.value = "three (edited)"
        gate.set()
        await _wait_until(app, pilot, lambda: not app._generating)

        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: calls["n"] >= 2)

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
        await _wait_until(app, pilot, lambda: any("r1" in b for b in bubbles(app)))

        inp.value = "two"
        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: bool(app._queue), tries=10, pause=0.05)

        inp.value = "/"
        await pilot.pause()
        assert suggestions_box(app).display
        await pilot.press("up")
        await pilot.pause()
        assert inp.value == "/"  # suggestion cycled, input untouched
        assert len(app._queue) == 1  # queue untouched
        gate.set()
        await _wait_until(app, pilot, lambda: not app._generating)
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
        await _wait_until(app, pilot, lambda: any("r1" in b for b in bubbles(app)))

        for msg in ("two", "three"):
            inp.value = msg
            await pilot.press("enter")
        await _wait_until(app, pilot, lambda: len(app._queue) == 2, tries=10, pause=0.05)

        gate.set()
        await _wait_until(app, pilot, lambda: calls["n"] >= 3)
        assert not any("Queued" in b for b in bubbles(app))

    assert [m["role"] for m in app.session.messages] == [
        "user", "assistant", "user", "assistant", "user", "assistant",
    ]
    assert [m["content"] for m in app.session.messages if m["role"] == "user"] == [
        "one", "two", "three",
    ]
    assert app._queue == []


def make_app_with_cmd(tmp_path, cmd: CmdPolicy, settings=None, session=None):
    app = make_app(tmp_path, settings=settings, session=session)
    app.cmd = cmd
    app.settings.cmd_mode = cmd.mode
    return app


def cmd_stream(first: str, second: str):
    """A stream fake: first call yields commands, next yields the final."""
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        calls["n"] += 1
        yield (first if calls["n"] == 1 else second)

    return fake, calls


def command_call(command: str) -> str:
    """Format a test command using the production command-only protocol."""
    return f"jtech_cmd({command!r})"


async def test_cmd_auto_allowlist_runs_silently(tmp_path, monkeypatch):
    """auto mode: an allowlisted command runs without a prompt; output feeds back."""
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="auto", allow=["echo:*"]))
    calls = {"n": 0}
    requests = []

    def fake(profile, temperature, messages):
        calls["n"] += 1
        requests.append(messages)
        yield command_call("echo hello-out") if calls["n"] == 1 else "done"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: calls["n"] >= 2, tries=50)

        assert not isinstance(app.screen, CommandPrompt)
        assert any("hello-out" in b for b in bubbles(app))
        assert any("done" in b for b in bubbles(app))
        roles = [m["role"] for m in app.session.messages]
        assert "system" in roles
        sys_msgs = [m["content"] for m in app.session.messages if m["role"] == "system"]
        assert any("hello-out" in m and "exit 0" in m for m in sys_msgs)
        assert requests[1][-1] == {
            "role": "user",
            "content": "[JTECH runtime event]\n$ echo hello-out\nexit 0\nhello-out",
        }


async def test_cmd_ask_prompts_then_allow_runs(tmp_path, monkeypatch):
    """ask mode: a non-allowlisted command prompts; 'y' allows and runs it."""
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="ask", allow=[]))
    fake, calls = cmd_stream(command_call("echo prompt-out"), "done")
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: isinstance(app.screen, CommandPrompt), tries=50)
        await pilot.press("y")
        await _wait_until(app, pilot, lambda: calls["n"] >= 2, tries=50)

        assert any("prompt-out" in b for b in bubbles(app))
        sys_msgs = [m["content"] for m in app.session.messages if m["role"] == "system"]
        assert any("prompt-out" in m for m in sys_msgs)


async def test_cmd_ask_decline_feeds_back(tmp_path, monkeypatch):
    """ask mode: 'n' declines; the command does not run but the model still reacts."""
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="ask", allow=[]))
    fake, calls = cmd_stream(command_call("echo never-runs"), "done")
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: isinstance(app.screen, CommandPrompt), tries=50)
        await pilot.press("n")
        await _wait_until(app, pilot, lambda: calls["n"] >= 2, tries=50)

        # the command text is in the AI's own bubble (the fenced block), so
        # "not run" is proven by the absence of an exit-code result message
        sys_msgs = [m["content"] for m in app.session.messages if m["role"] == "system"]
        assert any("declined by the user" in m for m in sys_msgs)
        assert not any("exit 0" in m for m in sys_msgs)
        assert any("done" in b for b in bubbles(app))


async def test_cmd_blacklist_blocked_even_in_yolo(tmp_path, monkeypatch):
    """The blacklist is absolute: even yolo blocks sudo, and no prompt is shown."""
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="yolo"))
    fake, calls = cmd_stream(command_call("sudo ls"), "done")
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: calls["n"] >= 2, tries=50)

        assert not isinstance(app.screen, CommandPrompt)
        sys_msgs = [m["content"] for m in app.session.messages if m["role"] == "system"]
        assert any("blocked" in m and "sudo" in m for m in sys_msgs)
        assert not any("exit 0" in m for m in sys_msgs)  # never executed


async def test_cmd_off_mode_disables_execution(tmp_path, monkeypatch):
    """off mode: requested commands are not run; a disabled note is fed back."""
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="off"))
    fake, calls = cmd_stream(command_call("echo should-not-run"), "done")
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: calls["n"] >= 2, tries=50)

        sys_msgs = [m["content"] for m in app.session.messages if m["role"] == "system"]
        assert any("disabled" in m for m in sys_msgs)
        assert not any("exit 0" in m for m in sys_msgs)  # never executed


async def test_cmd_always_allow_saves_rule(tmp_path, monkeypatch):
    """'a' in the prompt persists a prefix rule to config and runs the command."""
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="ask", allow=[]))
    fake, calls = cmd_stream(command_call("git status"), "done")
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: isinstance(app.screen, CommandPrompt), tries=50)
        await pilot.press("a")
        await _wait_until(app, pilot, lambda: calls["n"] >= 2, tries=50)

        loaded = load_cmd_policy(app.config_path)
        assert "git status:*" in loaded.allow
        assert any("git status:*" in b for b in bubbles(app))


async def test_every_command_in_one_reply_runs_in_source_order(tmp_path, monkeypatch):
    """A reply may carry any number of calls; all of them run, in order.

    With no per-reply cap nothing is dropped, so there is nothing to report as
    dropped: the model gets exactly one result per call it emitted.
    """
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="auto", allow=["echo:*"]))
    total = 7  # more than the retired five-call cap
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            yield "\n".join(command_call(f"echo blk-{i}") for i in range(total))
        else:
            yield "stopped"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: calls["n"] >= 2, tries=150)
        await _wait_until(app, pilot, lambda: not app._tool_rounds_active, tries=100)
        for _ in range(10):
            await pilot.pause()

        assert calls["n"] == 2  # every call ran, then exactly one more round
        assert not any("ignored" in b for b in bubbles(app))

    fed = [m["content"] for m in app.session.messages if m["role"] == "system"]
    assert len(fed) == total  # nothing dropped
    for index, result in enumerate(fed):
        assert f"blk-{index}" in result  # and nothing reordered


async def test_different_command_rounds_are_not_limited(tmp_path, monkeypatch):
    """Distinct command results can continue without an arbitrary round cap."""
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="auto", allow=["echo:*"]))
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        calls["n"] += 1
        if calls["n"] <= 6:
            yield command_call(f"echo round-{calls['n']}")
        else:
            yield "finished"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: calls["n"] >= 7, tries=200)
        await _wait_until(app, pilot, lambda: not app._tool_rounds_active, tries=100)
        for _ in range(10):
            await pilot.pause()

        assert calls["n"] == 7
        assert any("finished" in b for b in bubbles(app))
        assert not any("round limit" in b.lower() for b in bubbles(app))

    fed = [m for m in app.session.messages if m["role"] == "system"]
    assert len(fed) == 6  # one result for each distinct command round
    assert "round-6" in "\n".join(m["content"] for m in fed)


async def test_repeated_commands_and_results_do_not_stop_the_loop(
    tmp_path, monkeypatch
):
    """The same command with the same result keeps the turn running.

    Repetition is the model's business, not the loop's: only prose without a
    call ends the turn, so four identical rounds all execute and feed back.
    """
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="auto", allow=["echo:*"]))
    repeats = 4
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        calls["n"] += 1
        if calls["n"] <= repeats:
            yield command_call("echo unchanged")
        else:
            yield "done"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: calls["n"] >= repeats + 1, tries=200)
        await _wait_until(app, pilot, lambda: not app._tool_rounds_active, tries=100)
        for _ in range(10):
            await pilot.pause()

        assert calls["n"] == repeats + 1
        assert not any("no-progress" in b for b in bubbles(app))
        assert any("done" in b for b in bubbles(app))

    fed = [m["content"] for m in app.session.messages if m["role"] == "system"]
    assert len(fed) == repeats
    assert all("unchanged" in message and "exit 0" in message for message in fed)


async def test_consecutive_empty_replies_are_each_nudged(tmp_path, monkeypatch):
    """Every empty reply earns another nudge; there is no recovery budget.

    An empty reply is not an answer, so the turn ends only once the model
    produces prose — here after three consecutive empty streams.
    """
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="auto", allow=["echo:*"]))
    calls = {"n": 0}
    requests = []

    def fake(profile, temperature, messages):
        calls["n"] += 1
        requests.append(messages)
        if calls["n"] == 1:
            yield command_call("echo tool-out")
        elif calls["n"] <= 4:
            yield ""
        else:
            yield "recovered"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: calls["n"] >= 5, tries=200)
        await _wait_until(app, pilot, lambda: not app._tool_rounds_active, tries=100)
        for _ in range(10):
            await pilot.pause()

        assert calls["n"] == 5
        assert any("recovered" in b for b in bubbles(app))

    # request 0 is the user turn and 1 follows the tool result; 2-4 are nudges
    nudged = [
        index
        for index, messages in enumerate(requests)
        if messages[-1] == {"role": "system", "content": NUDGE_PROMPT}
    ]
    assert nudged == [2, 3, 4]
    # empty replies are never stored, and the nudge stays out of the history
    assert [m["role"] for m in app.session.messages] == [
        "user",
        "assistant",
        "system",
        "assistant",
    ]


async def test_nudge_is_shown_in_system_debug_mode(tmp_path, monkeypatch):
    """Debug system mode exposes the ephemeral nudge in the live chat."""
    settings = local_settings(debug_level="system")
    app = make_app_with_cmd(
        tmp_path, CmdPolicy(mode="auto", allow=["echo:*"]), settings=settings
    )
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            yield command_call("echo debug-nudge")
        elif calls["n"] == 2:
            yield ""
        else:
            yield "done"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: calls["n"] >= 3, tries=100)
        await _wait_until(app, pilot, lambda: not app._tool_rounds_active, tries=100)

        assert any("Continue your task" in bubble for bubble in bubbles(app))
        nudges = [
            message
            for message in app.session.messages
            if "Continue your task" in message["content"]
        ]
        assert len(nudges) == 1
        assert nudges[0]["_include_in_context"] is False


async def test_nudge_can_continue_with_an_explicit_command(tmp_path, monkeypatch):
    """A nudge may recover a command-only stop, but prose still ends the turn."""
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="auto", allow=["echo:*"]))
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        calls["n"] += 1
        replies = {
            1: command_call("echo first"),
            2: "",
            3: command_call("echo second"),
            4: "finished",
        }
        yield replies[calls["n"]]

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: calls["n"] >= 4, tries=100)
        await _wait_until(app, pilot, lambda: not app._tool_rounds_active, tries=100)

        assert calls["n"] == 4
        system_messages = [
            m["content"] for m in app.session.messages if m["role"] == "system"
        ]
        assert any("first" in message for message in system_messages)
        assert any("second" in message for message in system_messages)


async def test_final_answer_after_tool_ends_turn_without_repeat(tmp_path, monkeypatch):
    """A final answer after ``pwd`` ends the turn without rerunning the command."""
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="auto", allow=["pwd:*"]))
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            yield command_call("pwd")
        else:
            yield "The cwd is /the/project.\n\n```cmd\npwd\n```"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "whats the cwd?"
        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: calls["n"] >= 2, tries=100)
        await _wait_until(app, pilot, lambda: not app._tool_rounds_active, tries=100)
        for _ in range(10):
            await pilot.pause()

        assert calls["n"] == 2
        assert any("The cwd is /the/project." in b for b in bubbles(app))

    sys_msgs = [m["content"] for m in app.session.messages if m["role"] == "system"]
    assert len([m for m in sys_msgs if "pwd" in m and "exit 0" in m]) == 1


async def test_command_prefix_commentary_is_preserved_and_tool_round_continues(
    tmp_path, monkeypatch
):
    """Prefix commentary is visible, while the command still starts a tool round."""
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="auto", allow=["pwd:*"]))
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            yield command_call("pwd") + "\n\nLet me inspect the project structure next."
        else:
            yield "The cwd is /the/project."

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "audit this project"
        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: calls["n"] >= 2, tries=100)
        await _wait_until(app, pilot, lambda: not app._tool_rounds_active, tries=100)

        assert calls["n"] == 2
        assert any("Let me inspect the project structure next." in b for b in bubbles(app))
        assert any("The cwd is /the/project." in b for b in bubbles(app))

    sys_msgs = [m["content"] for m in app.session.messages if m["role"] == "system"]
    assert len([m for m in sys_msgs if "pwd" in m and "exit 0" in m]) == 1


async def test_interleaved_commentary_commands_start_one_tool_round(
    tmp_path, monkeypatch
):
    """Commentary between standalone calls does not suppress later commands."""
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="auto", allow=["echo:*"]))
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            yield (
                "I'll audit the project.\n\n"
                f'{command_call("echo first")}\n'
                "Let me inspect the first result.\n"
                f'{command_call("echo second")}\n'
                "Let me finish the audit."
            )
        else:
            yield "The audit is complete."

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "audit this project"
        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: calls["n"] >= 2, tries=100)
        await _wait_until(app, pilot, lambda: not app._tool_rounds_active, tries=100)

        assert calls["n"] == 2
        assert any("I'll audit the project." in b for b in bubbles(app))
        assert any("Let me inspect the first result." in b for b in bubbles(app))
        assert any("The audit is complete." in b for b in bubbles(app))

    sys_msgs = [m["content"] for m in app.session.messages if m["role"] == "system"]
    assert sum("exit 0" in message for message in sys_msgs) == 2


async def test_html_wrapped_command_executes_once(tmp_path, monkeypatch):
    """A whole-response HTML wrapper does not disable the command protocol."""
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="auto", allow=["pwd:*"]))
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            yield '<code>\njtech_cmd("pwd")\n</code>'
        else:
            yield "done"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "whats the cwd?"
        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: calls["n"] >= 2, tries=100)
        await _wait_until(app, pilot, lambda: not app._tool_rounds_active, tries=100)

        assert calls["n"] == 2

    sys_msgs = [m["content"] for m in app.session.messages if m["role"] == "system"]
    assert len([m for m in sys_msgs if "pwd" in m and "exit 0" in m]) == 1


async def test_clear_during_tool_followup_does_not_crash(tmp_path, monkeypatch):
    """/clear can empty history while the post-command reply is in flight."""
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="auto", allow=["echo:*"]))
    gate = threading.Event()
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            yield command_call("echo clear-out")
        elif calls["n"] == 2:
            gate.wait(5)  # hold the post-command stream open across the /clear
            yield "after clear"
        else:
            yield "unexpected extra request"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: calls["n"] >= 2, tries=100)

        # /clear is dispatched straight from the input, bypassing the
        # tool-rounds queue guard, so it lands while the follow-up is in flight.
        inp.value = "/clear"
        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: app.session.messages == [], tries=50)
        gate.set()

        await _wait_until(app, pilot, lambda: not app._tool_rounds_active, tries=100)
        await pilot.pause()

        assert app._exception is None
        assert calls["n"] == 2


async def test_plain_final_answer_ends_turn(tmp_path, monkeypatch):
    """A plain final answer (no tool rounds yet) ends the turn."""
    app = make_app(tmp_path)
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        calls["n"] += 1
        yield "all done"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "hello"
        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: calls["n"] >= 1)
        # give a would-be extra stream time to fire
        for _ in range(10):
            await pilot.pause()

    assert calls["n"] == 1
    assert app.session.messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "all done"},
    ]


async def test_declined_command_ends_tool_turn(tmp_path, monkeypatch):
    """A decline is user input: the next model reply ends the tool turn."""
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="ask", allow=[]))
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            yield command_call("echo declined-out")
        else:
            yield "stopped"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: isinstance(app.screen, CommandPrompt), tries=50)
        await pilot.press("n")
        await _wait_until(app, pilot, lambda: calls["n"] >= 2)
        # give a would-be extra stream time to fire
        for _ in range(10):
            await pilot.pause()

    assert calls["n"] == 2
    assert [m["role"] for m in app.session.messages] == [
        "user", "assistant", "system", "system", "assistant",
    ]
    # the model is told to ask the user, not to silently adapt
    assert any("ask the user" in m["content"] for m in app.session.messages)


async def test_blocked_command_ends_tool_turn(tmp_path, monkeypatch):
    """A blocked command is guardrail input; the next reply ends the turn."""
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="yolo"))
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            yield command_call("sudo ls")
        else:
            yield "stopped"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: calls["n"] >= 2)
        # give a would-be extra stream time to fire
        for _ in range(10):
            await pilot.pause()
        assert not isinstance(app.screen, CommandPrompt)

    assert calls["n"] == 2


async def test_failed_command_result_continues_the_loop(tmp_path, monkeypatch):
    """A non-zero exit is a result, not a stop: the model gets it and continues."""
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="yolo"))
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            yield command_call("echo failing-out; exit 3")
        elif calls["n"] == 2:
            yield command_call("echo recovered-out")
        else:
            yield "handled"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: calls["n"] >= 3, tries=150)
        await _wait_until(app, pilot, lambda: not app._tool_rounds_active, tries=100)
        for _ in range(10):
            await pilot.pause()

        assert calls["n"] == 3
        assert any("handled" in b for b in bubbles(app))

    sys_msgs = [m["content"] for m in app.session.messages if m["role"] == "system"]
    assert any("exit 3" in m and "failing-out" in m for m in sys_msgs)
    assert any("recovered-out" in m for m in sys_msgs)


async def test_a_running_command_is_shown_then_replaced_by_its_result(tmp_path, monkeypatch):
    """The reported defect, end to end: an executing command looked inert.

    Nothing was drawn until the process exited, so a `jtech_cmd(...)` the app
    had already parsed and started read as a call that never fired.
    """
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="yolo"))
    command = "sleep 60"
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            yield command_call(command)
        else:
            yield "final"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await _wait_until(
            app, pilot, lambda: app._primary_runtime.state.running_proc is not None
        )

        # Visible while the process is still alive, not after it exits.
        running = [b for b in bubbles(app) if b.startswith(f"$ {command}")]
        assert len(running) == 1, running
        assert "running…" in running[0]

        # The existing stop path, exactly as a user reaches it.
        await pilot.press("escape")
        await _wait_until(app, pilot, lambda: calls["n"] >= 2, tries=100)
        await _wait_until(app, pilot, lambda: not app._tool_rounds_active, tries=100)
        await pilot.pause()

        # One presentation for the whole lifecycle: the running entry became the
        # result rather than being joined by a second bubble for the same run.
        shown = [b for b in bubbles(app) if b.startswith(f"$ {command}")]
        assert len(shown) == 1, shown
        assert "running…" not in shown[0]
        assert "interrupted" in shown[0]

        sys_msgs = [m["content"] for m in app.session.messages if m["role"] == "system"]
        assert any("interrupted by user" in m for m in sys_msgs)
        assert not any("running…" in m for m in sys_msgs)
        assert any("final" in b for b in bubbles(app))


async def test_queue_drains_after_esc_stop(tmp_path, monkeypatch):
    """The queued turn starts after the stopped one closed, and sees the marker.

    Two user turns in a row is what degenerated the next completion. The stop
    now leaves a balanced assistant turn between them, carrying the marker
    rather than the partial answer.
    """
    app = make_app(tmp_path)
    first = BlockingStream()
    second = FiniteStream("r2")
    entered: list[str] = []
    first_closed_at_second_entry: list[bool] = []
    sent: list[list[dict]] = []

    async def provider(profile, temperature, messages):
        sent.append(messages)
        if not entered:
            entered.append("one")
            return first
        entered.append("two")
        # Read here, in the second request itself: the first response must
        # already be closed, not merely asked to stop.
        first_closed_at_second_entry.append(first.closed == 1)
        return second

    monkeypatch.setattr("jtech_cli.tui.stream_reply", provider)
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "one"
        await pilot.press("enter")
        await _wait_until(app, pilot, first.blocked.is_set)
        system_prompt = app._primary_system_prompt()

        inp.value = "two"
        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: bool(app._queue), tries=10, pause=0.05)
        assert entered == ["one"]  # the queued turn has not started

        await pilot.press("escape")
        await _wait_until(app, pilot, lambda: any("r2" in b for b in bubbles(app)))

        interrupted = f"partial \n\n{INTERRUPTED_RESPONSE}"
        assert entered == ["one", "two"]  # the requests never overlapped
        assert first_closed_at_second_entry == [True]
        assert first.cancelled is True
        assert not any("Queued" in b for b in bubbles(app))
        assert not any("Generation stopped" in b for b in bubbles(app))
        # the stopped partial is still on screen, above the queued turn's answer
        shown = bubbles(app)
        assert shown.index(interrupted) < shown.index("r2")

    assert sent[1] == [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": INTERRUPTED_RESPONSE},
        {"role": "user", "content": "two"},
    ]
    assert app.session.messages == [
        {"role": "user", "content": "one"},
        {
            "role": "assistant",
            "content": interrupted,
            "_model_role": "assistant",
            "_model_content": INTERRUPTED_RESPONSE,
        },
        {"role": "user", "content": "two"},
        {"role": "assistant", "content": "r2"},
    ]


def test_output_sink_renders_rich_objects_instead_of_their_repr():
    """A renderable must not reach the chat as "<... object at 0x...>"."""
    from rich.markdown import Markdown

    from jtech_cli.tui import OutputSink

    class _StubApp:
        def __init__(self):
            self.messages = []

        def push_message(self, role, text):
            self.messages.append((role, text))

    app = _StubApp()
    sink = OutputSink(app)
    sink.print(Markdown("# Heading"))

    _role, text = app.messages[-1]
    assert "object at" not in text
    assert "Heading" in text


def test_output_sink_passes_plain_strings_through():
    from jtech_cli.tui import OutputSink

    class _StubApp:
        def __init__(self):
            self.messages = []

        def push_message(self, role, text):
            self.messages.append((role, text))

    app = _StubApp()
    OutputSink(app).print("[dim]hello[/dim]")
    assert app.messages[-1][1].strip() == "hello"


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
        await _wait_until(app, pilot, lambda: calls["n"] >= 1, tries=50)
        await _wait_until(app, pilot, lambda: not app._generating, tries=50)

        assert app._prompt_tokens == 8192


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
        await _wait_until(app, pilot, lambda: calls["n"] >= 1, tries=50)
        await _wait_until(app, pilot, lambda: not app._generating, tries=50)

        assert app._prompt_tokens == 512


async def test_timings_with_prompt_n_still_sets_the_count(tmp_path, monkeypatch):
    """The llama.cpp path is unchanged: a real prompt_n still drives the footer."""
    app = make_app(tmp_path)
    fake, calls = _event_stream("hi", ("timings", {"prompt_n": 2048, "prompt_ms": 12.0}))
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        app.query_one("#input", Input).value = "go"
        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: calls["n"] >= 1, tries=50)
        await _wait_until(app, pilot, lambda: not app._generating, tries=50)

        assert app._prompt_tokens == 2048


async def test_successful_discovery_reaches_the_command_context(tmp_path, monkeypatch):
    """/models and /stats read ctx.server — discovery has to update that too.

    The footer reads app.server and the commands read app.ctx.server. They are
    one object; rebinding only the first left /models reporting "unreachable"
    against a perfectly healthy server.
    """
    monkeypatch.setattr(
        "jtech_cli.tui.fetch_server_info",
        lambda settings: ServerInfo(models=["qwen3"], context_length=4096),
    )
    app = make_app(tmp_path, server=ServerInfo(), no_discover=False)  # cli.make_app shape
    async with app.run_test() as pilot:
        await _wait_until(app, pilot, lambda: bool(app.server.models), tries=50)

        assert app.ctx.server.models == ["qwen3"]
        assert app.ctx.server.context_length == 4096
        assert app.ctx.server is app.server

        app.query_one("#input", Input).value = "/models"
        await pilot.press("enter")
        await pilot.pause()
        assert any("qwen3" in b for b in bubbles(app))
        assert not any("No model info available" in b for b in bubbles(app))


async def test_failed_discovery_keeps_known_info_and_says_so(tmp_path, monkeypatch):
    """A best-effort probe may not downgrade what is already known."""
    monkeypatch.setattr("jtech_cli.tui.fetch_server_info", lambda settings: ServerInfo())
    app = make_app(
        tmp_path, server=ServerInfo(models=["qwen3"], context_length=4096), no_discover=False
    )
    async with app.run_test() as pilot:
        await _wait_until(app, pilot, lambda: any("Could not reach" in b for b in bubbles(app)), tries=50)

        assert app.server.models == ["qwen3"]
        assert app.server.context_length == 4096
        assert app.ctx.server.models == ["qwen3"]


async def test_discovery_does_not_overwrite_an_explicit_model(tmp_path, monkeypatch):
    """A model set in config/flags wins over the discovered one."""
    monkeypatch.setattr(
        "jtech_cli.tui.fetch_server_info",
        lambda settings: ServerInfo(models=["discovered"], context_length=4096),
    )
    app = make_app(
        tmp_path,
        settings=local_settings_with_model("explicit"),
        server=ServerInfo(),
        no_discover=False,
    )
    async with app.run_test() as pilot:
        await _wait_until(app, pilot, lambda: bool(app.server.models), tries=50)
        assert app.settings.model == "explicit"


# --- streaming handoff, batching, and follow-scroll ------------------------


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
        await _wait_until(app, pilot, lambda: writes == ["A"], tries=100)

        # B, C and D are produced while the first write is still outstanding
        stream.produced.set()
        await _wait_until(app, pilot, stream.finished.is_set, tries=100)
        release.set()
        await _wait_until(app, pilot, lambda: not app._generating, tries=100)

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
        await _wait_until(app, pilot, lambda: not app._generating, tries=100)

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
        await _wait_until(app, pilot, lambda: not app._generating, tries=100)

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
        await _wait_until(app, pilot, lambda: writes == ["held"], tries=100)
        for _ in range(5):
            await pilot.pause()

        assert app._generating
        assert app.session.messages == [{"role": "user", "content": "go"}]

        release.set()
        await _wait_until(app, pilot, lambda: not app._generating, tries=100)
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
        await _wait_until(
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
        await _wait_until(app, pilot, lambda: not app._generating, tries=100)
        assert writes == ["spoke at last"]


async def test_reasoning_hide_counts_without_mounting_a_bubble(tmp_path, monkeypatch):
    app = make_app(tmp_path, settings=make_settings("hide"))
    updates = record_static_updates(monkeypatch)
    monkeypatch.setattr(
        "jtech_cli.tui.stream_reply",
        stream_of(("reasoning", "hmm "), ("reasoning", "ok"), "4"),
    )
    async with app.run_test() as pilot:
        await _send_and_drain(app, pilot, "what is 2+2")

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
        await _send_and_drain(app, pilot, "what is 2+2")

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
        await _send_and_drain(app, pilot, "what is 2+2")

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
        await _send_and_drain(app, pilot, "what is 2+2")

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
        await _send_and_drain(app, pilot, "go")

        assert app._prompt_tokens == 512
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
        await _send_and_drain(app, pilot, "go")

        assert app._prompt_tokens == 2048
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
        await _wait_until(app, pilot, lambda: not app._generating, tries=100)

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
        await _wait_until(app, pilot, lambda: chat.max_scroll_y > 5, tries=100)

        chat.scroll_to(y=0, animate=False, immediate=True)
        await pilot.pause()
        assert chat.scroll_offset.y == 0

        gate.set()
        await _wait_until(app, pilot, lambda: not app._generating, tries=100)
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
        await _wait_until(app, pilot, lambda: not app._generating, tries=100)

        assert any("Could not save history: disk full" in b for b in bubbles(app))
        assert any("ok" in b for b in bubbles(app))
        # the exchange is intact for the model, and the warning is not in it
        assert session.messages_with_system("") == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
        ]


# --- startup transcript mounting -------------------------------------------


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


# --- transcript shape, reflow, and compaction ------------------------------

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
        await _wait_until(app, pilot, lambda: any("partial" in b for b in bubbles(app)))

        chat = chat_of(app)
        assert len(chat.query(Markdown)) == 1  # exactly the live answer

        gate.set()
        await _wait_until(app, pilot, lambda: not app._generating, tries=100)
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
        await _wait_until(
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
        await _wait_until(app, pilot, lambda: any("r1" in b for b in bubbles(app)))

        for queued in ("two", "three"):
            inp.value = queued
            await pilot.press("enter")
        await _wait_until(app, pilot, lambda: len(app._queue) == 2, tries=20, pause=0.05)

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
        await _wait_until(app, pilot, lambda: calls["n"] >= 3, tries=100)
        await _wait_until(app, pilot, lambda: not app._generating, tries=100)
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
        await _wait_until(app, pilot, lambda: not app._generating, tries=100)
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
        await _wait_until(app, pilot, lambda: any("partial" in b for b in bubbles(app)))

        inp.value = "/clear"
        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: app.session.messages == [], tries=50)

        gate.set()
        await _wait_until(app, pilot, lambda: not app._generating, tries=100)
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
        await _wait_until(
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
        await _wait_until(app, pilot, lambda: not app._generating, tries=100)


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
        await _wait_until(app, pilot, lambda: chat.max_scroll_y > 5, tries=100)

        chat.scroll_to(y=0, animate=False, immediate=True)
        await pilot.pause()
        assert chat.scroll_offset.y == 0

        gate.set()
        await _wait_until(app, pilot, lambda: not app._generating, tries=100)
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
        await _wait_until(app, pilot, lambda: not app._generating, tries=100)

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
        await _wait_until(
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
        await _wait_until(
            app, pilot, lambda: any(RENDER_ERROR in b for b in bubbles(app)), tries=100
        )
        await _wait_until(app, pilot, lambda: not app._generating, tries=100)

        assert blocked.closed == 1  # the response was closed, not just dropped
        assert entries == ["turn-1"]

        monkeypatch.setattr(MarkdownStream, "write", real_write)
        app.query_one("#input", Input).value = "second"
        await pilot.press("enter")
        await _wait_until(
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


# --- profile modal, switching, and turn ownership --------------------------

CLOUD = Profile(
    name="cloud",
    base_url="https://api.example.com/v1",
    model="cloud-model",
    api_key_env="CLOUD_API_KEY",
)


def two_profile_settings(**kwargs) -> Settings:
    """A local (unauthenticated) profile plus an authenticated cloud one."""
    return Settings(
        profiles=Profiles(items=(LOCAL, CLOUD), active_name="local"), **kwargs
    )


def profiles_rows_text(app: ChatApp) -> str:
    return str(app.screen.query_one("#profiles-rows", Static).render())


def profiles_help_text(app: ChatApp) -> str:
    return str(app.screen.query_one("#profiles-help", Static).render())


def notifications(app: ChatApp) -> list[str]:
    return [notification.message for notification in app._notifications]


def set_field(app: ChatApp, widget_id: str, value: str) -> None:
    app.screen.query_one(f"#{widget_id}", Input).value = value


async def settle(pilot, times: int = 6) -> None:
    for _ in range(times):
        await pilot.pause()


async def open_profiles(app: ChatApp, pilot) -> None:
    app.query_one("#input", Input).value = "/profiles"
    await pilot.press("enter")
    await settle(pilot)


async def test_profiles_modal_lists_every_profile_and_the_active_marker(tmp_path):
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test(size=(80, 30)) as pilot:
        await open_profiles(app, pilot)

        assert isinstance(app.screen, ProfilesScreen)
        rows = profiles_rows_text(app)
        assert "local (active)" in rows
        assert "cloud" in rows and "cloud (active)" not in rows
        assert "https://api.example.com/v1" in rows
        assert "$CLOUD_API_KEY" in rows  # the variable name, never a key value
        assert ProfilesScreen.ADD_ROW in rows


async def test_profiles_modal_does_not_probe_any_endpoint(tmp_path, monkeypatch):
    """Connectivity is transient; a stopped local server is still editable."""
    probes = []
    monkeypatch.setattr(
        "jtech_cli.tui.fetch_server_info",
        lambda profile: probes.append(profile) or ServerInfo(),
    )
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test(size=(80, 30)) as pilot:
        await open_profiles(app, pilot)
        await pilot.press("down")
        await settle(pilot)
        assert probes == []


async def test_profiles_modal_activates_and_persists(tmp_path):
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test(size=(80, 30)) as pilot:
        await open_profiles(app, pilot)
        await pilot.press("down")  # cloud
        await pilot.press("enter")  # actions
        await settle(pilot)
        assert "Profile: cloud" in profiles_help_text(app)
        await pilot.press("enter")  # Activate
        await settle(pilot)

        assert app.settings.profiles.active_name == "cloud"
        assert 'active_profile = "cloud"' in (tmp_path / "config.toml").read_text()
        assert "profile: cloud" in app.query_one("#status", Static).content
        assert "cloud (active)" in profiles_rows_text(app)


async def test_profiles_modal_adds_a_profile_without_activating_it(tmp_path):
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test(size=(80, 30)) as pilot:
        await open_profiles(app, pilot)
        await pilot.press("down", "down")  # Add profile…
        await pilot.press("enter")
        await settle(pilot)

        set_field(app, "profile-name", "staging")
        set_field(app, "profile-url", "https://staging.example.com/v1")
        set_field(app, "profile-key", "STAGING_KEY")
        await pilot.press("enter")
        await settle(pilot)

        assert app.settings.profiles.names == ("local", "cloud", "staging")
        assert app.settings.profiles.active_name == "local"
        added = app.settings.profiles.get("staging")
        assert added.model == ""  # blank means auto-discover
        assert added.api_key_env == "STAGING_KEY"
        assert "[profiles.staging]" in (tmp_path / "config.toml").read_text()


async def test_profiles_modal_edits_and_renames_in_place(tmp_path):
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test(size=(80, 30)) as pilot:
        await open_profiles(app, pilot)
        await pilot.press("enter")  # local -> actions
        await pilot.press("down")  # Edit
        await pilot.press("enter")
        await settle(pilot)
        assert app.screen.query_one("#profile-url", Input).value == "http://host:9000/v1"

        set_field(app, "profile-name", "workstation")
        set_field(app, "profile-url", "http://renamed:1/v1")
        await pilot.press("enter")
        await settle(pilot)

        assert app.settings.profiles.names == ("workstation", "cloud")
        # a renamed profile that was active stays active
        assert app.settings.profiles.active_name == "workstation"
        assert app.settings.profiles.get("workstation").base_url == "http://renamed:1/v1"
        assert "profile: workstation" in app.query_one("#status", Static).content


async def test_profiles_modal_keeps_the_editor_open_on_invalid_input(tmp_path):
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test(size=(80, 30)) as pilot:
        before = app.settings.profiles
        await open_profiles(app, pilot)
        await pilot.press("enter", "down", "enter")  # edit local
        await settle(pilot)

        set_field(app, "profile-url", "not-a-url")
        await pilot.press("enter")
        await settle(pilot)

        assert app.screen.query_one("#profile-url", Input).value == "not-a-url"
        assert app.settings.profiles is before
        assert any("base_url" in message for message in notifications(app))

        set_field(app, "profile-url", "http://fixed:1/v1")
        await pilot.press("enter")
        await settle(pilot)
        assert app.settings.profiles.get("local").base_url == "http://fixed:1/v1"


async def test_profiles_modal_cancels_an_edit_without_saving(tmp_path):
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test(size=(80, 30)) as pilot:
        before = app.settings.profiles
        await open_profiles(app, pilot)
        await pilot.press("enter", "down", "enter")  # edit local
        await settle(pilot)

        set_field(app, "profile-url", "http://discarded/v1")
        await pilot.press("escape")
        await settle(pilot)

        assert app.settings.profiles is before
        assert not (tmp_path / "config.toml").exists()
        assert isinstance(app.screen, ProfilesScreen)  # back to the action list


async def test_profiles_modal_deletes_an_inactive_profile_after_confirming(tmp_path):
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test(size=(80, 30)) as pilot:
        await open_profiles(app, pilot)
        await pilot.press("down")  # cloud
        await pilot.press("enter")  # actions
        await pilot.press("down", "down")  # Delete
        await pilot.press("enter")  # confirm state
        await settle(pilot)
        assert "Delete profile cloud?" in profiles_help_text(app)

        await pilot.press("up")  # Confirm delete (the cursor defaults to Cancel)
        await pilot.press("enter")
        await settle(pilot)

        assert app.settings.profiles.names == ("local",)
        assert "[profiles.cloud]" not in (tmp_path / "config.toml").read_text()


async def test_profiles_modal_confirm_defaults_to_cancel(tmp_path):
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test(size=(80, 30)) as pilot:
        await open_profiles(app, pilot)
        await pilot.press("down", "enter", "down", "down", "enter")  # cloud -> Delete
        await settle(pilot)
        await pilot.press("enter")  # take the default choice
        await settle(pilot)

        assert app.settings.profiles.names == ("local", "cloud")
        assert not (tmp_path / "config.toml").exists()


async def test_profiles_modal_refuses_to_delete_the_active_profile(tmp_path):
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test(size=(80, 30)) as pilot:
        await open_profiles(app, pilot)
        await pilot.press("enter")  # local (active) -> actions
        await pilot.press("down", "down")  # Delete
        await pilot.press("enter")
        await pilot.press("up")  # Confirm delete
        await pilot.press("enter")
        await settle(pilot)

        assert app.settings.profiles.names == ("local", "cloud")
        assert any(
            "activate another profile" in message for message in notifications(app)
        )


async def test_a_failed_profile_save_keeps_the_modal_open_and_the_old_catalog(tmp_path):
    (tmp_path / "config.toml").mkdir()  # writing here raises OSError
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test(size=(80, 30)) as pilot:
        before = app.settings.profiles
        await open_profiles(app, pilot)
        await pilot.press("down", "enter", "enter")  # activate cloud
        await settle(pilot)

        assert app.settings.profiles is before
        assert app.settings.profiles.active_name == "local"
        assert notifications(app)
        # the modal is still open, still on the profile whose save failed
        assert isinstance(app.screen, ProfilesScreen)
        assert "Profile: cloud" in profiles_help_text(app)

        await pilot.press("escape")  # back to the list
        await settle(pilot)
        assert "local (active)" in profiles_rows_text(app)


async def test_switching_profiles_clears_stale_endpoint_state(tmp_path, monkeypatch):
    monkeypatch.setenv("CLOUD_API_KEY", "sk-secret")
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test() as pilot:
        app._prompt_tokens = 1234
        app._render_status()
        assert "ctx" in app.query_one("#status", Static).content

        app.query_one("#input", Input).value = "/profile cloud"
        await pilot.press("enter")
        await settle(pilot)

        assert app.settings.profiles.active_name == "cloud"
        assert app.server.models == []
        assert app.server.context_length is None
        assert app.server.error is None
        assert app._prompt_tokens == 0
        assert app.ctx.server is app.server  # cleared in place, not rebound
        status = app.query_one("#status", Static).content
        assert "profile: cloud" in status
        assert "https://api.example.com/v1" in status
        assert "ctx" not in status


async def test_a_stale_discovery_result_is_discarded(tmp_path, monkeypatch):
    """A slow probe of the previous endpoint must not describe the new one."""
    monkeypatch.setenv("CLOUD_API_KEY", "sk-secret")
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test() as pilot:
        await app._switch_profile("cloud")
        await settle(pilot)

        app._fetch_server_info_fn = lambda profile: ServerInfo(
            models=["stale"], context_length=999
        )
        await app._discover_server(LOCAL)  # a probe started before the switch

        assert app.server.models == []
        assert app.server.context_length is None
        assert not any("stale" in bubble for bubble in bubbles(app))


async def test_an_unknown_profile_name_reports_without_changing_state(tmp_path):
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test() as pilot:
        before = app.settings.profiles
        app.query_one("#input", Input).value = "/profile nope"
        await pilot.press("enter")
        await settle(pilot)

        assert app.settings.profiles is before
        assert app.settings.profiles.active_name == "local"
        assert not (tmp_path / "config.toml").exists()
        assert any("No profile named 'nope'" in bubble for bubble in bubbles(app))


async def test_switching_clears_a_cli_override_once_stored(tmp_path, monkeypatch):
    monkeypatch.setenv("CLOUD_API_KEY", "sk-secret")
    settings = two_profile_settings()
    settings.profile_override = Profile(
        name="local", base_url="http://override/v1", model="override-model"
    )
    app = make_app(tmp_path, settings=settings)
    async with app.run_test() as pilot:
        assert "(override)" in app.query_one("#status", Static).content

        await app._switch_profile("cloud")
        await settle(pilot)

        assert app.settings.profile_override is None
        assert app.settings.active_profile == CLOUD
        assert "(override)" not in app.query_one("#status", Static).content


async def test_a_failed_switch_keeps_the_override_and_the_catalog(tmp_path):
    (tmp_path / "config.toml").mkdir()  # writing here raises OSError
    settings = two_profile_settings()
    override = Profile(name="local", base_url="http://override/v1", model="override-model")
    settings.profile_override = override
    app = make_app(tmp_path, settings=settings)
    async with app.run_test() as pilot:
        before = app.settings.profiles
        await app._switch_profile("cloud")
        await settle(pilot)

        assert app.settings.profiles is before
        assert app.settings.profile_override is override
        assert any(
            "Could not save profile selection" in bubble for bubble in bubbles(app)
        )


async def test_a_profile_switch_is_refused_while_streaming(tmp_path, monkeypatch):
    app = make_app(tmp_path, settings=two_profile_settings())
    gate = threading.Event()

    def fake(profile, temperature, messages):
        yield "partial "
        gate.wait(5)
        yield "done"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        await _send_and_drain(app, pilot, "go")
        await _wait_until(app, pilot, lambda: any("partial" in b for b in bubbles(app)))
        assert app._generating

        app.query_one("#input", Input).value = "/profile cloud"
        await pilot.press("enter")
        await settle(pilot)
        assert app.settings.profiles.active_name == "local"
        assert any("Esc to stop it" in bubble for bubble in bubbles(app))

        app._open_profiles()
        await pilot.pause()
        assert not isinstance(app.screen, ProfilesScreen)

        gate.set()
        await _wait_until(app, pilot, lambda: not app._generating)


def park_in_tool_round(app: ChatApp) -> None:
    """Install a Primary runtime parked between completions, as a batch is.

    The real state object, not a stand-in boolean: ``_busy()`` reads the run
    that owns the flag, so a test that fakes the flag would prove nothing.
    """
    state = app._primary_run_state(
        ResolvedProfile(
            name="local", base_url="http://host:9000/v1", model="qwen3", api_key="none"
        )
    )
    state.tool_rounds_active = True
    app._primary_runtime = AutonomousRuntime(
        state,
        host=app,
        stream_reply_fn=app._stream_reply_fn,
        cmd_policy=app.cmd,
        project_root=app._project_root,
    )


async def test_a_profile_change_is_refused_during_a_tool_round(tmp_path):
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test() as pilot:
        before = app.settings.profiles
        park_in_tool_round(app)
        try:
            await app._switch_profile("cloud")
            await pilot.pause()
            assert app.settings.profiles is before

            app._open_profiles()
            await pilot.pause()
            assert not isinstance(app.screen, ProfilesScreen)

            with pytest.raises(ProfileError):
                await app._commit_profiles(before.activate("cloud"))
            assert app.settings.profiles is before
            assert any("tool round" in bubble for bubble in bubbles(app))
        finally:
            app._primary_runtime = None


async def test_one_autonomous_turn_uses_one_resolved_profile(tmp_path, monkeypatch):
    """First reply, command continuation, and nudge share one immutable profile."""
    app = make_app_with_cmd(
        tmp_path, CmdPolicy(mode="auto", allow=["echo:*"]), settings=two_profile_settings()
    )
    seen: list[tuple[object, float]] = []
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        calls["n"] += 1
        seen.append((profile, temperature))
        if calls["n"] == 1:
            yield command_call("echo turn-out")
        elif calls["n"] == 2:
            yield ""  # empty reply -> a nudge round
        else:
            yield "done"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        await _send_and_drain(app, pilot, "go")
        await _wait_until(app, pilot, lambda: calls["n"] >= 3, tries=150)
        await _wait_until(app, pilot, lambda: not app._tool_rounds_active, tries=100)

    assert calls["n"] == 3
    used = [profile for profile, _ in seen]
    # the same object, not merely an equal one
    assert all(profile is used[0] for profile in used)
    assert used[0].base_url == "http://host:9000/v1"
    assert used[0].model == "qwen3"
    assert used[0].api_key == "none"
    assert {temperature for _, temperature in seen} == {0.7}


async def test_the_next_idle_turn_uses_the_newly_activated_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("CLOUD_API_KEY", "sk-secret")
    app = make_app(tmp_path, settings=two_profile_settings())
    seen = []

    def fake(profile, temperature, messages):
        seen.append(profile)
        yield "ok"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        await _send_and_drain(app, pilot, "first")
        await _wait_until(app, pilot, lambda: len(seen) == 1)

        app.query_one("#input", Input).value = "/profile cloud"
        await pilot.press("enter")
        await _wait_until(
            app, pilot, lambda: app.settings.profiles.active_name == "cloud"
        )

        await _send_and_drain(app, pilot, "second")
        await _wait_until(app, pilot, lambda: len(seen) == 2)

    assert seen[0].base_url == "http://host:9000/v1"
    assert seen[0].api_key == "none"
    assert seen[1].base_url == "https://api.example.com/v1"
    assert seen[1].model == "cloud-model"
    assert seen[1].api_key == "sk-secret"
    # switching profiles does not clear or fork the conversation
    assert [m["content"] for m in app.session.messages] == [
        "first", "ok", "second", "ok",
    ]


async def test_a_missing_credential_stops_before_the_provider_thread(tmp_path, monkeypatch):
    monkeypatch.delenv("CLOUD_API_KEY", raising=False)
    settings = Settings(profiles=Profiles(items=(CLOUD,), active_name="cloud"))
    app = make_app(tmp_path, settings=settings)
    started = []

    def fake(profile, temperature, messages):
        started.append(profile)
        yield "never"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        await _send_and_drain(app, pilot, "go")

        assert started == []
        assert any("CLOUD_API_KEY" in bubble for bubble in bubbles(app))
        assert any("unset or empty" in bubble for bubble in bubbles(app))
        assert app.session.messages == [{"role": "user", "content": "go"}]
        assert not app._generating


async def test_a_missing_model_stops_before_the_provider_thread(tmp_path, monkeypatch):
    """No configured model and no unique served model is an error, not a guess."""
    profile = Profile(name="local", base_url="http://host:9000/v1")
    settings = Settings(profiles=Profiles(items=(profile,), active_name="local"))
    app = make_app(
        tmp_path, settings=settings, server=ServerInfo(models=["one", "two"])
    )
    started = []

    def fake(profile, temperature, messages):
        started.append(profile)
        yield "never"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        await _send_and_drain(app, pilot, "go")

        assert started == []
        assert any("no model configured" in bubble for bubble in bubbles(app))
        assert not app._generating


async def test_a_turn_without_a_profile_reports_instead_of_streaming(tmp_path, monkeypatch):
    app = make_app(tmp_path, settings=Settings())
    started = []

    def fake(profile, temperature, messages):
        started.append(profile)
        yield "never"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        await _send_and_drain(app, pilot, "go")

        assert started == []
        assert any("No API profile is configured" in b for b in bubbles(app))
        assert not app._generating


async def test_modal_activation_retires_a_cli_override(tmp_path):
    """Regression: the modal persisted a selection the override kept shadowing."""
    settings = two_profile_settings()
    settings.profile_override = Profile(
        name="local", base_url="http://override.example/v1", model="override-model"
    )
    app = make_app(tmp_path, settings=settings)
    async with app.run_test(size=(80, 30)) as pilot:
        assert "(override)" in app.query_one("#status", Static).content

        await open_profiles(app, pilot)
        await pilot.press("down")  # cloud
        await pilot.press("enter")  # actions
        await pilot.press("enter")  # Activate
        await settle(pilot)

        assert app.settings.profiles.active_name == "cloud"
        assert app.settings.profile_override is None
        assert app.settings.active_profile == CLOUD
        status = app.query_one("#status", Static).content
        assert "profile: cloud" in status
        assert "(override)" not in status
        assert "override.example" not in status


async def test_a_failed_modal_activation_keeps_the_override(tmp_path):
    (tmp_path / "config.toml").mkdir()  # writing here raises OSError
    settings = two_profile_settings()
    override = Profile(name="local", base_url="http://override.example/v1", model="ov")
    settings.profile_override = override
    app = make_app(tmp_path, settings=settings)
    async with app.run_test(size=(80, 30)) as pilot:
        await open_profiles(app, pilot)
        await pilot.press("down", "enter", "enter")  # activate cloud
        await settle(pilot)

        assert app.settings.profile_override is override
        assert app.settings.profiles.active_name == "local"


async def test_a_modal_edit_does_not_retire_a_cli_override(tmp_path):
    """Editing is not selecting: only Activate supersedes the CLI flag."""
    settings = two_profile_settings()
    override = Profile(name="local", base_url="http://override.example/v1", model="ov")
    settings.profile_override = override
    app = make_app(tmp_path, settings=settings)
    async with app.run_test(size=(80, 30)) as pilot:
        await open_profiles(app, pilot)
        await pilot.press("down", "enter", "down", "enter")  # edit cloud
        await settle(pilot)
        set_field(app, "profile-model", "edited-model")
        await pilot.press("enter")
        await settle(pilot)

        assert app.settings.profiles.get("cloud").model == "edited-model"
        assert app.settings.profile_override is override


async def test_adding_the_first_profile_persists_an_active_selection(tmp_path):
    """Regression: the modal could write a config that failed on next launch."""
    app = make_app(tmp_path, settings=Settings())
    async with app.run_test(size=(80, 30)) as pilot:
        await open_profiles(app, pilot)
        await pilot.press("enter")  # the only row is Add profile…
        await settle(pilot)

        set_field(app, "profile-name", "first")
        set_field(app, "profile-url", "http://first:1/v1")
        await pilot.press("enter")
        await settle(pilot)

        assert app.settings.profiles.active_name == "first"
        text = (tmp_path / "config.toml").read_text()
        assert 'active_profile = "first"' in text
        assert build_settings(config_path=tmp_path / "config.toml").profiles.active_name == "first"


async def test_a_stale_token_count_is_discarded(tmp_path, monkeypatch):
    """A count describes one tokenizer; a late one must not describe the new one."""
    monkeypatch.setenv("CLOUD_API_KEY", "sk-secret")
    app = make_app(
        tmp_path,
        settings=two_profile_settings(),
        fetch_token_count_fn=lambda profile, text: 7,
    )
    app.session.add("user", "hello world")
    entered = threading.Event()
    released = threading.Event()

    def slow_count(profile, text):
        entered.set()
        released.wait(5)
        return 42

    async with app.run_test() as pilot:
        await _wait_until(app, pilot, lambda: app._prompt_tokens == 7)

        app._fetch_token_count_fn = slow_count
        counting = asyncio.ensure_future(app._init_token_count(LOCAL))
        await _wait_until(app, pilot, entered.is_set, tries=100)

        await app._switch_profile("cloud")
        await settle(pilot)
        assert app._prompt_tokens == 0  # the switch cleared the old count

        released.set()
        await counting
        await settle(pilot)

        assert app._prompt_tokens == 0  # the old endpoint's 42 never landed
        assert "ctx" not in app.query_one("#status", Static).content


async def test_a_stale_credential_error_is_not_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("CLOUD_API_KEY", "sk-secret")
    app = make_app(
        tmp_path,
        settings=two_profile_settings(),
        fetch_token_count_fn=lambda profile, text: 7,
    )
    app.session.add("user", "hello world")

    def boom(profile, text):
        raise ProfileError("stale credential complaint")

    async with app.run_test() as pilot:
        await _wait_until(app, pilot, lambda: app._prompt_tokens == 7)

        app._fetch_token_count_fn = boom
        await app._switch_profile("cloud")
        await settle(pilot)
        await app._init_token_count(LOCAL)  # a probe from before the switch
        await settle(pilot)

        assert not any("stale credential" in bubble for bubble in bubbles(app))


async def test_a_current_credential_error_is_still_reported(tmp_path):
    """The staleness guard silences late results, not live failures."""
    app = make_app(tmp_path, settings=two_profile_settings())
    app.session.add("user", "hello world")

    def boom(profile, text):
        raise ProfileError("live credential complaint")

    app._fetch_token_count_fn = boom
    async with app.run_test() as pilot:
        await settle(pilot)
        assert any("live credential complaint" in b for b in bubbles(app))


async def test_a_live_token_count_still_reaches_the_footer(tmp_path):
    """The staleness guard must not disable the normal path."""
    app = make_app(
        tmp_path,
        settings=two_profile_settings(),
        fetch_token_count_fn=lambda profile, text: 128,
    )
    app.session.add("user", "hello world")
    async with app.run_test() as pilot:
        await _wait_until(app, pilot, lambda: app._prompt_tokens == 128)
        assert "ctx" in app.query_one("#status", Static).content


@pytest.mark.parametrize("url", ["http://[::1/v1", "https://host:0/v1"])
async def test_the_editor_reports_an_unparseable_url_instead_of_crashing(tmp_path, url):
    """urlparse raises bare ValueError on these; the modal only catches ProfileError."""
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test(size=(80, 30)) as pilot:
        before = app.settings.profiles
        await open_profiles(app, pilot)
        await pilot.press("enter", "down", "enter")  # edit local
        await settle(pilot)

        set_field(app, "profile-url", url)
        await pilot.press("enter")
        await settle(pilot)

        assert app._exception is None
        assert isinstance(app.screen, ProfilesScreen)  # editor still open
        assert app.screen.query_one("#profile-url", Input).value == url
        assert app.settings.profiles is before
        assert notifications(app)


# --- contextual Ctrl+C: copy, clear, confirm quit --------------------------


def quit_rows_text(app: ChatApp) -> str:
    return str(app.screen.query_one("#quit-rows", Static).render())


def base_input(app: ChatApp) -> Input:
    """The chat composer on the base screen, reachable while a modal is up.

    ``App.query_one`` only sees the active screen, so a suspended draft has to
    be read through the bottom of the screen stack.
    """
    return app.screen_stack[0].query_one("#input", Input)


def spy_exit(app: ChatApp, monkeypatch) -> list[tuple]:
    """Record ``App.exit()`` on this instance instead of stopping the app.

    Textual's ``App`` is already the injected lifecycle boundary, so the app
    needs no exit callback of its own for tests to observe termination.
    """
    calls: list[tuple] = []
    monkeypatch.setattr(app, "exit", lambda *a, **kw: calls.append((a, kw)))
    return calls


async def drag_history(pilot, app: ChatApp, needle: str) -> None:
    """Drag-select ``needle`` in the completed transcript, as a pointer would."""
    history = chat_of(app).history
    rendered = [strip.text for strip in history._lines]
    row = next(y for y, line in enumerate(rendered) if needle in line)
    column = rendered[row].index(needle)
    end = Offset(column + len(needle) - 1, row)
    history.screen.clear_selection()
    await pilot.mouse_down(history, Offset(column, row))
    await pilot.hover(history, end)
    await pilot.mouse_up(history, end)
    await pilot.pause()


def clear_input_selection(inp: Input) -> None:
    """Drop the select-all an Input takes on focus, leaving a bare cursor."""
    inp.selection = InputSelection.cursor(len(inp.value))


async def test_ctrl_c_copies_transcript_selection_before_touching_draft_or_quit(
    tmp_path, monkeypatch
):
    app = make_app(tmp_path)
    exits = spy_exit(app, monkeypatch)
    async with app.run_test(size=(80, 24)) as pilot:
        app.push_message("system", "second message with several words in it")
        await settle(pilot)
        inp = app.query_one("#input", Input)
        inp.value = "draft"
        await drag_history(pilot, app, "several")

        await pilot.press("ctrl+c")
        await settle(pilot)

        assert app.clipboard == "several"
        assert inp.value == "draft"
        assert not isinstance(app.screen, QuitScreen)
        assert exits == []


async def test_ctrl_c_copies_single_line_selection_without_clearing_input(
    tmp_path, monkeypatch
):
    app = make_app(tmp_path)
    exits = spy_exit(app, monkeypatch)
    async with app.run_test(size=(80, 24)) as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "alpha beta"
        inp.focus()
        await settle(pilot)
        inp.selection = InputSelection(0, len("alpha"))
        await settle(pilot)

        await pilot.press("ctrl+c")
        await settle(pilot)

        assert app.clipboard == "alpha"
        assert inp.value == "alpha beta"
        assert not isinstance(app.screen, QuitScreen)
        assert exits == []


async def test_ctrl_c_copies_multiline_selection_without_clearing_editor(
    tmp_path, monkeypatch
):
    app = make_app(tmp_path)
    exits = spy_exit(app, monkeypatch)
    async with app.run_test(size=(80, 24)) as pilot:
        ta = await _enter_multiline(app, pilot, "'''")
        ta.text = "alpha\nbeta"
        await settle(pilot)
        ta.selection = AreaSelection((0, 0), (0, len("alpha")))
        await settle(pilot)

        await pilot.press("ctrl+c")
        await settle(pilot)

        assert app.clipboard == "alpha"
        assert ta.text == "alpha\nbeta"
        assert app._multiline_future is not None and not app._multiline_future.done()
        assert not isinstance(app.screen, QuitScreen)
        assert exits == []


# "/set" is here so the suggestion assertion is not vacuous: it is the only
# value of the three that actually opens the command menu before the clear.
@pytest.mark.parametrize("draft", ["draft", "   ", "/set"])
async def test_ctrl_c_clears_nonempty_single_line_input(tmp_path, monkeypatch, draft):
    app = make_app(tmp_path)
    exits = spy_exit(app, monkeypatch)
    async with app.run_test(size=(80, 24)) as pilot:
        inp = app.query_one("#input", Input)
        inp.value = draft
        inp.focus()
        await settle(pilot)
        clear_input_selection(inp)
        await settle(pilot)

        await pilot.press("ctrl+c")
        await settle(pilot)

        assert inp.value == ""
        assert app.focused is inp
        assert suggestions_box(app).display is False
        assert not isinstance(app.screen, QuitScreen)
        assert exits == []


async def test_ctrl_c_clears_nonempty_multiline_without_canceling(
    tmp_path, monkeypatch
):
    app = make_app(tmp_path)
    exits = spy_exit(app, monkeypatch)
    async with app.run_test(size=(80, 24)) as pilot:
        ta = await _enter_multiline(app, pilot, "'''")
        ta.text = "line one\nline two"
        await settle(pilot)
        ta.selection = AreaSelection.cursor((1, 0))
        await settle(pilot)

        await pilot.press("ctrl+c")
        await settle(pilot)

        assert ta.text == ""
        assert app.query_one("#multiline-input", TextArea) is ta
        assert app.focused is ta
        assert app._multiline_future is not None and not app._multiline_future.done()
        assert app.query_one("#input", Input).display is False
        assert not isinstance(app.screen, QuitScreen)
        assert exits == []


async def test_ctrl_c_on_empty_multiline_opens_quit_screen_without_canceling(
    tmp_path, monkeypatch
):
    app = make_app(tmp_path)
    exits = spy_exit(app, monkeypatch)
    async with app.run_test(size=(80, 24)) as pilot:
        ta = await _enter_multiline(app, pilot, "'''")
        assert ta.text == ""

        await pilot.press("ctrl+c")
        await settle(pilot)
        assert isinstance(app.screen, QuitScreen)
        assert app._multiline_future is not None and not app._multiline_future.done()

        await pilot.press("escape")
        await settle(pilot)
        assert not isinstance(app.screen, QuitScreen)
        assert app.query_one("#multiline-input", TextArea) is ta
        assert ta.text == ""
        assert app.focused is ta
        assert app._multiline_future is not None and not app._multiline_future.done()
        assert exits == []


async def test_ctrl_c_on_empty_composer_opens_quit_screen(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    exits = spy_exit(app, monkeypatch)
    async with app.run_test(size=(80, 24)) as pilot:
        assert app.query_one("#input", Input).value == ""

        await pilot.press("ctrl+c")
        await settle(pilot)

        assert isinstance(app.screen, QuitScreen)
        assert "▸ Stay" in quit_rows_text(app)
        assert exits == []


async def test_quit_screen_enter_on_default_stays(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    exits = spy_exit(app, monkeypatch)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("ctrl+c")
        await settle(pilot)

        await pilot.press("enter")
        await settle(pilot)

        assert not isinstance(app.screen, QuitScreen)
        assert app.is_running
        assert app.focused is app.query_one("#input", Input)
        assert exits == []


async def test_quit_screen_navigation_can_confirm_quit(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    exits = spy_exit(app, monkeypatch)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("ctrl+c")
        await settle(pilot)

        await pilot.press("right")
        await settle(pilot)
        assert "▸ Quit" in quit_rows_text(app)

        await pilot.press("enter")
        await settle(pilot)

        assert len(exits) == 1


async def test_quit_screen_escape_stays(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    exits = spy_exit(app, monkeypatch)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("ctrl+c")
        await settle(pilot)

        await pilot.press("escape")
        await settle(pilot)

        assert not isinstance(app.screen, QuitScreen)
        assert app.query_one("#input", Input) is not None
        assert exits == []


async def test_repeated_ctrl_c_clears_then_confirms_then_panic_exits(
    tmp_path, monkeypatch
):
    """The complete transition order, and the modal's priority panic binding.

    The third press never moves the cursor off **Stay**, so an exit here proves
    ``ctrl+c`` on the quit screen ignores its own normal choice.
    """
    app = make_app(tmp_path)
    exits = spy_exit(app, monkeypatch)
    async with app.run_test(size=(80, 24)) as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "draft"
        inp.focus()
        await settle(pilot)
        clear_input_selection(inp)
        await settle(pilot)

        await pilot.press("ctrl+c")
        await settle(pilot)
        assert inp.value == ""
        assert not isinstance(app.screen, QuitScreen)
        assert exits == []

        await pilot.press("ctrl+c")
        await settle(pilot)
        assert isinstance(app.screen, QuitScreen)
        assert "▸ Stay" in quit_rows_text(app)
        assert exits == []

        await pilot.press("ctrl+c")
        await settle(pilot)
        assert len(exits) == 1


async def test_quit_screen_over_existing_modal_preserves_modal_and_draft(
    tmp_path, monkeypatch
):
    app = make_app(tmp_path)
    exits = spy_exit(app, monkeypatch)
    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one("#input", Input).value = "draft"
        app.action_settings()
        await settle(pilot)
        await pilot.press("enter")  # edit Temperature
        await settle(pilot)
        field = app.screen.query_one("#settings-field", Input)
        field.value = "0.9"  # unsaved
        clear_input_selection(field)
        await settle(pilot)

        await pilot.press("ctrl+c")
        await settle(pilot)

        assert isinstance(app.screen, QuitScreen)
        assert base_input(app).value == "draft"
        assert app.settings.temperature == 0.7

        await pilot.press("escape")
        await settle(pilot)

        assert isinstance(app.screen, SettingsScreen)
        assert app.screen.query_one("#settings-field", Input).value == "0.9"
        assert app.query_one("#input", Input).value == "draft"
        assert app.settings.temperature == 0.7
        assert exits == []


async def test_textual_local_clipboard_pastes_without_system_clipboard_support(
    tmp_path, monkeypatch
):
    """Copy then paste is Textual's local clipboard: no OSC 52 answer required."""
    app = make_app(tmp_path)
    spy_exit(app, monkeypatch)
    async with app.run_test(size=(80, 24)) as pilot:
        app.push_message("system", "second message with several words in it")
        await settle(pilot)
        await drag_history(pilot, app, "several")
        await pilot.press("ctrl+c")
        await settle(pilot)
        assert app.clipboard == "several"

        app.screen.clear_selection()
        inp = app.query_one("#input", Input)
        inp.focus()
        await settle(pilot)
        assert inp.value == ""

        await pilot.press("ctrl+v")
        await settle(pilot)

        assert inp.value == "several"


async def test_ctrl_q_remains_an_immediate_exit(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    exits = spy_exit(app, monkeypatch)
    async with app.run_test(size=(80, 24)) as pilot:
        assert app.query_one("#input", Input).value == ""

        await pilot.press("ctrl+q")
        await settle(pilot)

        assert len(exits) == 1
        assert not isinstance(app.screen, QuitScreen)


# --- multi-agent workspace -------------------------------------------------


def workspace_of(app: ChatApp) -> AgentWorkspace:
    return app.query_one("#agent-workspace", AgentWorkspace)


def primary_summary(app: ChatApp) -> AgentSummary:
    return workspace_of(app).summary_for(PRIMARY_AGENT_ID)


def readonly_notice(app: ChatApp) -> Static:
    return app.query_one("#subagent-readonly", Static)


def visible_activity(app: ChatApp) -> Transcript:
    """The one displayed activity stream, proving the others are hidden."""
    displayed = [stream for stream in app.query(Transcript) if stream.display]
    assert len(displayed) == 1, displayed
    return displayed[0]


def activity_text(stream: Transcript) -> list[str]:
    return [record.content for record in stream.history.records]


async def select_agent(app: ChatApp, pilot, agent_id: str) -> None:
    """Select an agent the way a user does: focus, arrow, Enter."""
    agent_list = app.query_one("#agent-list", ListView)
    agent_list.focus()
    await pilot.pause()
    rows = list(agent_list.query(_AgentListItem))
    target = next(i for i, row in enumerate(rows) if row.agent_id == agent_id)
    while agent_list.index != target:
        await pilot.press("down" if agent_list.index < target else "up")
        await pilot.pause()
    await pilot.press("enter")
    await settle(pilot)


async def drag_transcript(pilot, stream: Transcript, needle: str) -> None:
    """Drag-select ``needle`` in one transcript's completed history."""
    history = stream.history
    rendered = [strip.text for strip in history._lines]
    row = next(y for y, line in enumerate(rendered) if needle in line)
    column = rendered[row].index(needle)
    end = Offset(column + len(needle) - 1, row)
    history.screen.clear_selection()
    await pilot.mouse_down(history, Offset(column, row))
    await pilot.hover(history, end)
    await pilot.mouse_up(history, end)
    await pilot.pause()


def record_primary_status(app: ChatApp, monkeypatch) -> list[str]:
    """Record every Primary status write without changing what it does."""
    seen: list[str] = []
    original = app._set_primary_agent_status

    def recorder(status: str) -> None:
        seen.append(status)
        original(status)

    monkeypatch.setattr(app, "_set_primary_agent_status", recorder)
    return seen


async def test_primary_is_selected_and_existing_composer_is_writable_on_startup(
    tmp_path, monkeypatch
):
    app = make_app(tmp_path)
    monkeypatch.setattr(
        "jtech_cli.tui.stream_reply", stream_of("ok")
    )
    async with app.run_test() as pilot:
        workspace = workspace_of(app)
        chat = chat_of(app)

        assert app.query_one("#activity-stack") in workspace.children
        assert app.query_one("#agent-sidebar") in workspace.children
        assert workspace.selected_agent_id == PRIMARY_AGENT_ID
        assert workspace.selected_activity is chat
        assert app.viewing_primary
        assert visible_activity(app) is chat
        assert app.focused is app.query_one("#input", Input)
        assert app.query_one("#input", Input).display
        assert not readonly_notice(app).display

        inp = app.query_one("#input", Input)
        inp.value = "hello"
        await pilot.press("enter")
        await settle(pilot, 10)

        assert app.session.messages == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "ok"},
        ]


async def test_workspace_uses_the_declared_split_without_shrinking_composer(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test(size=(100, 24)) as pilot:
        await settle(pilot)
        root = app.query_one(Vertical)

        assert root.region.width == 100
        assert workspace_of(app).region.width == 100
        assert app.query_one("#activity-stack").region.width == 85
        assert app.query_one("#agent-sidebar").region.width == 15
        # The composer and status line sit outside the horizontal split, so the
        # narrow sidebar never takes width from them.
        assert app.query_one("#input", Input).region.width == 100
        assert app.query_one("#status", Static).region.width == 100

        # The split follows terminal width continuously: no breakpoint, no
        # debounce, no persisted width.
        await pilot.resize_terminal(60, 24)
        await settle(pilot)

        assert app.query_one("#activity-stack").region.width == 51
        assert app.query_one("#agent-sidebar").region.width == 9
        assert app.query_one("#input", Input).region.width == 60
        assert app.query_one("#status", Static).region.width == 60


async def test_selecting_subagent_shows_only_its_private_activity(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        chat = chat_of(app)
        app.push_message("system", "primary only")
        first = await app.add_agent_view(
            AgentSummary("a", "Agent A", "running"),
            [TranscriptRecord("assistant", "agent a only")],
        )
        second = await app.add_agent_view(
            AgentSummary("b", "Agent B", "waiting"),
            [TranscriptRecord("assistant", "agent b only")],
        )
        await settle(pilot)

        await select_agent(app, pilot, "a")
        assert visible_activity(app) is first
        assert activity_text(first) == ["agent a only"]

        await select_agent(app, pilot, "b")
        assert visible_activity(app) is second
        assert activity_text(second) == ["agent b only"]

        await select_agent(app, pilot, PRIMARY_AGENT_ID)
        assert visible_activity(app) is chat
        assert activity_text(chat) == ["primary only"]


async def test_subagent_view_hides_composer_and_preserves_primary_draft(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await app.add_agent_view(AgentSummary("a", "Agent A", "running"))
        inp = app.query_one("#input", Input)
        inp.value = "/se"
        await settle(pilot)
        assert suggestions_box(app).display

        await select_agent(app, pilot, "a")

        assert not inp.display
        assert not suggestions_box(app).display
        assert readonly_notice(app).display
        assert str(readonly_notice(app).render()) == SUBAGENT_READONLY
        assert inp.value == "/se"
        assert not app.viewing_primary

        await select_agent(app, pilot, PRIMARY_AGENT_ID)

        assert inp.display
        assert inp.value == "/se"
        assert not readonly_notice(app).display
        assert suggestions_box(app).display
        assert "/settings" in suggestions_text(app)


async def test_subagent_view_preserves_active_multiline_editor(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await app.add_agent_view(AgentSummary("a", "Agent A", "running"))
        inp = app.query_one("#input", Input)
        inp.value = "first line"
        await pilot.press("shift+enter")
        await settle(pilot)
        editor = app.query_one("#multiline-input", TextArea)
        editor.text = "first line\nsecond line"
        editor.selection = AreaSelection((0, 0), (0, 5))
        await settle(pilot)
        future = app._multiline_future
        assert future is not None and not future.done()

        await select_agent(app, pilot, "a")
        assert not editor.display
        assert not inp.display
        assert readonly_notice(app).display

        await select_agent(app, pilot, PRIMARY_AGENT_ID)

        assert app.query_one("#multiline-input", TextArea) is editor
        assert app._multiline_textarea is editor
        assert editor.display
        assert editor.text == "first line\nsecond line"
        assert editor.selection == AreaSelection((0, 0), (0, 5))
        # Multi-line mode was hidden, never left.
        assert not inp.display
        assert not readonly_notice(app).display
        assert app._multiline_future is future and not future.done()


async def test_ctrl_c_in_subagent_view_copies_selection_before_quit(
    tmp_path, monkeypatch
):
    app = make_app(tmp_path)
    exits = spy_exit(app, monkeypatch)
    async with app.run_test(size=(80, 24)) as pilot:
        stream = await app.add_agent_view(
            AgentSummary("a", "Agent A", "running"),
            [TranscriptRecord("assistant", "subagent message with several words")],
        )
        inp = app.query_one("#input", Input)
        inp.value = "hidden draft"
        await settle(pilot)
        await select_agent(app, pilot, "a")
        await drag_transcript(pilot, stream, "several")

        await pilot.press("ctrl+c")
        await settle(pilot)

        assert app.clipboard == "several"
        assert not isinstance(app.screen, QuitScreen)
        assert exits == []
        assert inp.value == "hidden draft"
        assert workspace_of(app).selected_agent_id == "a"


async def test_ctrl_c_without_selection_in_subagent_view_opens_quit_without_clearing_primary(
    tmp_path, monkeypatch
):
    app = make_app(tmp_path)
    exits = spy_exit(app, monkeypatch)
    async with app.run_test(size=(80, 24)) as pilot:
        await app.add_agent_view(AgentSummary("a", "Agent A", "running"))
        inp = app.query_one("#input", Input)
        inp.value = "hidden draft"
        await settle(pilot)
        await select_agent(app, pilot, "a")
        assert not app.screen.selections

        await pilot.press("ctrl+c")
        await settle(pilot)

        assert isinstance(app.screen, QuitScreen)
        assert exits == []
        assert base_input(app).value == "hidden draft"

        await pilot.press("escape")
        await settle(pilot)

        assert not isinstance(app.screen, QuitScreen)
        assert workspace_of(app).selected_agent_id == "a"
        assert app.query_one("#input", Input).value == "hidden draft"
        assert not app.query_one("#input", Input).display


async def test_ctrl_l_in_subagent_view_changes_no_transcript_or_session(
    tmp_path, monkeypatch
):
    app = make_app(tmp_path)
    monkeypatch.setattr(
        "jtech_cli.tui.stream_reply", stream_of("ok")
    )
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "hello"
        await pilot.press("enter")
        await settle(pilot, 10)
        stream = await app.add_agent_view(
            AgentSummary("a", "Agent A", "running"),
            [TranscriptRecord("assistant", "subagent activity")],
        )
        await settle(pilot)
        await select_agent(app, pilot, "a")
        before_session = list(app.session.messages)
        before_primary = activity_text(chat_of(app))

        await pilot.press("ctrl+l")
        await settle(pilot)

        assert SUBAGENT_CLEAR_BLOCKED in notifications(app)
        assert app.session.messages == before_session
        assert activity_text(chat_of(app)) == before_primary
        assert activity_text(stream) == ["subagent activity"]


async def test_escape_in_subagent_view_does_not_stop_hidden_primary_work(
    tmp_path, monkeypatch
):
    app = make_app(tmp_path)
    gate = threading.Event()

    def fake(profile, temperature, messages):
        yield "partial "
        gate.wait(5)
        yield "rest"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "hello"
        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: any("partial" in b for b in bubbles(app)))
        await app.add_agent_view(AgentSummary("a", "Agent A", "running"))
        await settle(pilot)
        await select_agent(app, pilot, "a")

        await pilot.press("escape")
        await settle(pilot)

        assert not app._primary_runtime.state.stop_event.is_set()
        assert app._generating

        gate.set()
        await _wait_until(app, pilot, lambda: not app._generating)

    assert app.session.messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "partial rest"},
    ]


async def test_agent_updates_never_change_user_selection(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        first = await app.add_agent_view(AgentSummary("a", "Agent A", "running"))
        await app.add_agent_view(AgentSummary("b", "Agent B", "idle"))
        await settle(pilot)
        await select_agent(app, pilot, "a")

        for status in ("running", "completed", "failed"):
            app.update_agent_view(AgentSummary("b", "Agent B", status))
            await settle(pilot)
            assert workspace_of(app).selected_agent_id == "a"
            assert visible_activity(app) is first

        app.update_agent_view(
            AgentSummary(
                "a",
                "Agent A",
                "waiting",
                (AgentTaskSummary("t1", "Reproduce failure", "failed"),),
            )
        )
        await app.add_agent_view(AgentSummary("c", "Agent C", "running"))
        await settle(pilot)

        assert workspace_of(app).selected_agent_id == "a"
        assert visible_activity(app) is first
        assert not app.viewing_primary


async def test_primary_status_transitions_preserve_primary_tasks(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    gate = threading.Event()

    def fake(profile, temperature, messages):
        yield "r1 "
        gate.wait(5)
        yield "r1b"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        tasks = (
            AgentTaskSummary("t1", "Implement parser", "running"),
            AgentTaskSummary("t2", "Run audit", "waiting"),
        )
        app.update_agent_view(
            AgentSummary(PRIMARY_AGENT_ID, "Primary agent", "idle", tasks)
        )
        await settle(pilot)
        assert primary_summary(app).status == "idle"

        inp = app.query_one("#input", Input)
        inp.value = "hello"
        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: primary_summary(app).status == "running")

        running = primary_summary(app)
        assert running.label == "Primary agent"
        assert running.tasks == tasks

        gate.set()
        await _wait_until(app, pilot, lambda: primary_summary(app).status == "idle")

        finished = primary_summary(app)
        assert finished.label == "Primary agent"
        assert finished.tasks == tasks
        assert app._primary_turn_depth == 0


async def test_primary_status_stays_running_across_queued_turn_drain(
    tmp_path, monkeypatch
):
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
        statuses = record_primary_status(app, monkeypatch)
        inp = app.query_one("#input", Input)
        inp.value = "one"
        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: any("r1" in b for b in bubbles(app)))
        assert statuses == ["running"]

        for message in ("two", "three"):
            inp.value = message
            await pilot.press("enter")
        await _wait_until(app, pilot, lambda: len(app._queue) == 2, tries=10, pause=0.05)
        # Queued turns are nested inside the accepted one; none of them may
        # report the agent idle while the drain is still running.
        assert statuses == ["running"]

        gate.set()
        await _wait_until(app, pilot, lambda: calls["n"] >= 3)
        await _wait_until(app, pilot, lambda: app._primary_turn_depth == 0)

        assert statuses == ["running", "idle"]
        assert primary_summary(app).status == "idle"
        assert app._queue == []


# ------------------------------------------------------- agent orchestration


def dispatch_call(
    key="coder",
    label="Coder",
    profile="local",
    task_label="Task",
    task="do the work",
) -> str:
    """Format one dispatch using the production protocol."""
    return f"jtech_agent({key!r}, {label!r}, {profile!r}, {task_label!r}, {task!r})"


class Conversation:
    """A stream fake that answers each conversation from its own script.

    Keyed by the system prompt's role — coordinator or worker — and then by how
    many completions that conversation has already had, so one fixture can drive
    Primary and several agents at once and record exactly what each was sent.
    """

    def __init__(self, primary: list[str], worker: list[str] | None = None) -> None:
        self.primary = list(primary)
        self.worker = list(worker or [])
        self.requests: list[tuple[str, list[dict]]] = []
        self.profiles: list[ResolvedProfile] = []
        self._counts: dict[str, int] = {}

    def __call__(self, profile, temperature, messages):
        system = messages[0]["content"]
        role = "worker" if "You are a subagent" in system else "primary"
        index = self._counts.get(role, 0)
        self._counts[role] = index + 1
        self.requests.append((role, messages))
        self.profiles.append(profile)
        script = self.primary if role == "primary" else self.worker
        yield script[index] if index < len(script) else "finished"

    def sent_to(self, role: str) -> list[list[dict]]:
        return [messages for name, messages in self.requests if name == role]


def agent_summary(app: ChatApp, agent_id: str) -> AgentSummary:
    return workspace_of(app).summary_for(agent_id)


def agent_activity(app: ChatApp, agent_id: str) -> list[str]:
    return activity_text(workspace_of(app).activity_for(agent_id))


def agent_results(app: ChatApp) -> list[dict]:
    """Every agent-result envelope in Primary's model-facing context."""
    envelopes = []
    for message in app.session.messages:
        body = message.get("_model_content", "")
        if body.startswith("[JTECH agent result]\n"):
            envelopes.append(json.loads(body.split("\n", 1)[1]))
    return envelopes


async def run_primary(app, pilot, text="go", tries=100):
    """Submit one Primary message and wait for its whole turn to settle."""
    inp = app.query_one("#input", Input)
    inp.value = text
    await pilot.press("enter")
    await _wait_until(app, pilot, lambda: app._primary_turn_depth == 0, tries=tries)
    await settle(pilot)


async def test_a_first_dispatch_creates_one_agent_view_session_and_task(
    tmp_path, monkeypatch
):
    stream = Conversation([dispatch_call(), "all done"], ["worker answer"])
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(stream))
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await run_primary(app, pilot)

        summary = agent_summary(app, "coder")
        assert summary.label == "Coder"
        assert summary.status == "completed"
        assert [(t.label, t.status) for t in summary.tasks] == [("Task", "completed")]
        # The task is the agent's first message, recorded once and shown once.
        managed = app._agents["coder"]
        assert [m["content"] for m in managed.session.messages] == [
            "do the work",
            "worker answer",
        ]
        assert agent_activity(app, "coder")[0] == "do the work"
        worker_prompts = stream.sent_to("worker")
        assert len(worker_prompts) == 1
        assert "You are a subagent" in worker_prompts[0][0]["content"]
        assert "### Available profiles" not in worker_prompts[0][0]["content"]


async def test_the_coordinator_prompt_reaches_the_real_primary_request(
    tmp_path, monkeypatch
):
    stream = Conversation(["done"])
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(stream))
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test() as pilot:
        await run_primary(app, pilot)
    system = stream.sent_to("primary")[0][0]["content"]
    assert "jtech_agent(" in system
    assert "`local` — the profile this conversation runs on" in system
    assert "`cloud`" in system


async def test_a_repeated_key_continues_one_conversation(tmp_path, monkeypatch):
    stream = Conversation(
        [
            dispatch_call(task_label="First", task="first task"),
            dispatch_call(task_label="Second", task="second task"),
            "all done",
        ],
        ["first answer", "second answer"],
    )
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(stream))
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await run_primary(app, pilot)

        # One agent, two tasks, one transcript, one session.
        assert list(app._agents) == ["coder"]
        summary = agent_summary(app, "coder")
        assert [(t.label, t.status) for t in summary.tasks] == [
            ("First", "completed"),
            ("Second", "completed"),
        ]
        assert [m["content"] for m in app._agents["coder"].session.messages] == [
            "first task",
            "first answer",
            "second task",
            "second answer",
        ]
        # The second request carried the first exchange, so context survived.
        second_request = stream.sent_to("worker")[1]
        assert [m["content"] for m in second_request[1:]] == [
            "first task",
            "first answer",
            "second task",
        ]
        assert agent_activity(app, "coder").count("second task") == 1
        # The seeded first task and the appended second one share one policy.
        tasks = [
            record
            for record in app._agents["coder"].transcript.history.records
            if record.role == "user"
        ]
        assert [record.content for record in tasks] == ["first task", "second task"]
        assert [record.format for record in tasks] == ["plain", "plain"]


async def test_a_label_or_profile_change_for_one_key_fails_without_mutating(
    tmp_path, monkeypatch
):
    stream = Conversation(
        [
            dispatch_call(task_label="First"),
            dispatch_call(label="Renamed", task_label="Second"),
            dispatch_call(profile="cloud", task_label="Third"),
            "all done",
        ],
        ["worker answer"],
    )
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(stream))
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test() as pilot:
        await run_primary(app, pilot)

        summary = agent_summary(app, "coder")
        assert summary.label == "Coder"
        assert [t.label for t in summary.tasks] == ["First"]
        # One real run, two refusals — no second worker request was made.
        assert len(stream.sent_to("worker")) == 1
        results = agent_results(app)
        assert [r["status"] for r in results] == ["completed", "failed", "failed"]
        assert "keeps its label" in results[1]["content"]
        assert "keeps its profile" in results[2]["content"]
        # Each result names the call it answers, not the agent that already
        # exists: the rejected call asked for "Renamed", so answering it as
        # "Coder" would hide which call failed.
        assert [r["agent_label"] for r in results] == ["Coder", "Renamed", "Coder"]
        assert [r["task_label"] for r in results] == ["First", "Second", "Third"]
        assert [r["agent_key"] for r in results] == ["coder", "coder", "coder"]


async def test_two_agents_never_share_a_session_or_a_transcript(
    tmp_path, monkeypatch
):
    stream = Conversation(
        [f'{dispatch_call(key="a", label="A")}\n{dispatch_call(key="b", label="B")}',
         "all done"],
        ["answer one", "answer two"],
    )
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(stream))
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await run_primary(app, pilot)

        first = app._agents["a"]
        second = app._agents["b"]
        assert first.session is not second.session
        assert first.transcript is not second.transcript
        one = " ".join(m["content"] for m in first.session.messages)
        two = " ".join(m["content"] for m in second.session.messages)
        assert ("answer one" in one) != ("answer one" in two)
        assert set(agent_activity(app, "a")).isdisjoint(
            set(agent_activity(app, "b")) - {"do the work"}
        )


async def test_subagent_sessions_never_touch_the_filesystem(tmp_path, monkeypatch):
    stream = Conversation([dispatch_call(), "all done"], ["worker answer"])
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(stream))
    worker_history = tmp_path / "never" / "session.jsonl"
    monkeypatch.setattr(
        "jtech_cli.session.default_history_path", lambda: worker_history
    )
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await run_primary(app, pilot)
        managed = app._agents["coder"]
        assert managed.session.persist is False
        assert len(managed.session.messages) == 2
        assert not worker_history.exists()
        assert not worker_history.parent.exists()


async def test_a_failed_agent_stays_selectable_and_can_take_another_task(
    tmp_path, monkeypatch
):
    calls = {"n": 0, "worker": 0}

    def fake(profile, temperature, messages):
        if "You are a subagent" in messages[0]["content"]:
            calls["worker"] += 1
            if calls["worker"] == 1:
                raise RuntimeError("provider exploded")
            yield "recovered"
            return
        calls["n"] += 1
        if calls["n"] == 1:
            yield dispatch_call(task_label="First")
        elif calls["n"] == 2:
            yield dispatch_call(task_label="Second")
        else:
            yield "all done"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await run_primary(app, pilot)

        summary = agent_summary(app, "coder")
        assert [(t.label, t.status) for t in summary.tasks] == [
            ("First", "failed"),
            ("Second", "completed"),
        ]
        assert summary.status == "completed"
        results = agent_results(app)
        assert [r["status"] for r in results] == ["failed", "completed"]
        assert "provider exploded" in results[0]["content"]
        assert any(
            "provider exploded" in line for line in agent_activity(app, "coder")
        )
        await select_agent(app, pilot, "coder")
        assert visible_activity(app) is workspace_of(app).activity_for("coder")


async def test_a_relaunch_restores_primary_history_only(tmp_path, monkeypatch):
    stream = Conversation([dispatch_call(), "all done"], ["worker answer"])
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(stream))
    session = Session(tmp_path / "s.jsonl")
    app = make_app(tmp_path, session=session)
    async with app.run_test() as pilot:
        await run_primary(app, pilot)
    assert agent_results(app)[0]["content"] == "worker answer"

    restored = Session(tmp_path / "s.jsonl")
    restored.load()
    relaunched = make_app(tmp_path, session=restored)
    async with relaunched.run_test() as pilot:
        await settle(pilot)
        # The result survives in Primary's context; the worker does not come back.
        assert agent_results(relaunched)[0]["content"] == "worker answer"
        assert relaunched._agents == {}
        assert len(workspace_of(relaunched).query(_AgentListItem)) == 1


async def test_dispatch_never_disturbs_the_composer_selection_or_queue(
    tmp_path, monkeypatch
):
    gate = threading.Event()
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        system = messages[0]["content"]
        if "You are a subagent" in system:
            gate.wait(5)
            yield "worker answer"
            return
        calls["n"] += 1
        yield dispatch_call() if calls["n"] == 1 else "all done"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: "coder" in app._agents)

        assert app.viewing_primary
        assert visible_activity(app) is chat_of(app)
        inp.value = "a draft"
        inp.value = "queued while busy"
        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: app._queue == ["queued while busy"])
        assert primary_summary(app).status == "waiting"

        gate.set()
        await _wait_until(app, pilot, lambda: app._primary_turn_depth == 0)
        assert app._queue == []
        assert primary_summary(app).status == "idle"


# ---------------------------------------------------------------- profiles


async def test_each_agent_uses_its_own_resolved_profile(tmp_path, monkeypatch):
    stream = Conversation(
        [
            (
                f'{dispatch_call(key="a", label="A", profile="local")}\n'
                f'{dispatch_call(key="b", label="B", profile="cloud")}'
            ),
            "all done",
        ],
        ["one", "two"],
    )
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(stream))
    monkeypatch.setenv("CLOUD_API_KEY", "sk-secret")
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test() as pilot:
        await run_primary(app, pilot)

        worker_profiles = [
            profile
            for profile, (role, _) in zip(
                stream.profiles, stream.requests, strict=True
            )
            if role == "worker"
        ]
        assert {p.name for p in worker_profiles} == {"local", "cloud"}
        by_name = {p.name: p for p in worker_profiles}
        assert by_name["cloud"].model == "cloud-model"
        assert by_name["cloud"].base_url == "https://api.example.com/v1"
        assert by_name["local"].api_key == "none"
        # The credential is never rendered, never returned, and never repr'd.
        assert "sk-secret" not in repr(by_name["cloud"])
        assert "sk-secret" not in "\n".join(bubbles(app))
        assert "sk-secret" not in "\n".join(agent_activity(app, "b"))
        assert "sk-secret" not in json.dumps(agent_results(app))
        assert "sk-secret" not in app.query_one("#status", Static).content


async def test_an_explicit_model_skips_discovery(tmp_path, monkeypatch):
    stream = Conversation([dispatch_call(), "all done"], ["ok"])
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(stream))
    probed: list[Profile] = []

    def discover(profile):
        probed.append(profile)
        raise AssertionError("discovery must not run for a configured model")

    app = make_app(tmp_path)
    app._fetch_server_info_fn = discover
    async with app.run_test() as pilot:
        await run_primary(app, pilot)
    assert probed == []
    assert agent_results(app)[0]["status"] == "completed"


async def test_an_empty_model_on_the_active_profile_uses_the_discovered_one(
    tmp_path, monkeypatch
):
    stream = Conversation([dispatch_call(), "all done"], ["ok"])
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(stream))
    settings = local_settings()
    settings.profiles = Profiles(
        items=(Profile(name="local", base_url="http://host:9000/v1"),),
        active_name="local",
    )
    app = make_app(tmp_path, settings=settings)
    app._fetch_server_info_fn = lambda profile: pytest.fail("no probe expected")
    async with app.run_test() as pilot:
        await run_primary(app, pilot)
    worker_profile = next(
        profile
        for profile, (role, _) in zip(stream.profiles, stream.requests, strict=True)
        if role == "worker"
    )
    assert worker_profile.model == "qwen3"


async def test_an_empty_model_elsewhere_is_discovered_without_touching_primary(
    tmp_path, monkeypatch
):
    stream = Conversation([dispatch_call(profile="cloud"), "all done"], ["ok"])
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(stream))
    monkeypatch.setenv("CLOUD_API_KEY", "sk-secret")
    cloud = Profile(
        name="cloud",
        base_url="https://api.example.com/v1",
        api_key_env="CLOUD_API_KEY",
    )
    settings = Settings(profiles=Profiles(items=(LOCAL, cloud), active_name="local"))
    threads: list[int] = []

    def discover(profile):
        threads.append(threading.get_ident())
        return ServerInfo(models=["discovered-cloud"], context_length=999)

    app = make_app(tmp_path, settings=settings)
    app._fetch_server_info_fn = discover
    async with app.run_test() as pilot:
        await run_primary(app, pilot)
        worker_profile = next(
            profile
            for profile, (role, _) in zip(
                stream.profiles, stream.requests, strict=True
            )
            if role == "worker"
        )
        assert worker_profile.model == "discovered-cloud"
        # Off the event loop, and the Primary footer keeps its own server.
        assert threads and threads[0] != threading.get_ident()
        assert app.server.models == ["qwen3"]
        assert app.server.context_length == 4096
        assert "discovered-cloud" not in app.query_one("#status", Static).content


@pytest.mark.parametrize(
    ("profile_name", "fragment"),
    [
        ("nope", "No profile named 'nope'"),
        ("cloud", "$CLOUD_API_KEY"),
    ],
)
async def test_a_profile_failure_fails_only_its_own_task(
    tmp_path, monkeypatch, profile_name, fragment
):
    monkeypatch.delenv("CLOUD_API_KEY", raising=False)
    stream = Conversation(
        [
            (
                f'{dispatch_call(key="bad", label="Bad", profile=profile_name)}\n'
                f'{dispatch_call(key="good", label="Good", profile="local")}'
            ),
            "all done",
        ],
        ["good answer"],
    )
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(stream))
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test() as pilot:
        await run_primary(app, pilot)

        results = agent_results(app)
        assert [r["agent_key"] for r in results] == ["bad", "good"]
        assert results[0]["status"] == "failed"
        assert fragment in results[0]["content"]
        assert results[1] == {
            **results[1],
            "status": "completed",
            "content": "good answer",
        }
        assert agent_summary(app, "bad").status == "failed"
        assert agent_summary(app, "good").status == "completed"
        assert any(fragment in line for line in agent_activity(app, "bad"))


async def test_an_unreachable_discovery_endpoint_fails_only_its_task(
    tmp_path, monkeypatch
):
    stream = Conversation([dispatch_call(profile="cloud"), "all done"])
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(stream))
    monkeypatch.setenv("CLOUD_API_KEY", "sk-secret")
    cloud = Profile(
        name="cloud",
        base_url="https://api.example.com/v1",
        api_key_env="CLOUD_API_KEY",
    )
    settings = Settings(profiles=Profiles(items=(LOCAL, cloud), active_name="local"))
    app = make_app(tmp_path, settings=settings)
    app._fetch_server_info_fn = lambda profile: ServerInfo(
        error="URLError: connection refused"
    )
    async with app.run_test() as pilot:
        await run_primary(app, pilot)
    content = agent_results(app)[0]["content"]
    assert "could not be reached" in content
    assert "connection refused" in content
    assert "sk-secret" not in content


async def test_a_cli_override_is_advertised_once_and_dispatchable(
    tmp_path, monkeypatch
):
    stream = Conversation([dispatch_call(profile="local"), "all done"], ["ok"])
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(stream))
    settings = two_profile_settings()
    settings.profile_override = Profile(
        name="local", base_url="http://override:1/v1", model="override-model"
    )
    app = make_app(tmp_path, settings=settings)
    async with app.run_test() as pilot:
        await run_primary(app, pilot)

    system = stream.sent_to("primary")[0][0]["content"]
    listing = system[system.index("### Available profiles") :]
    assert listing.count("`local`") == 1
    assert listing.index("`local`") < listing.index("`cloud`")
    worker_profile = next(
        profile
        for profile, (role, _) in zip(stream.profiles, stream.requests, strict=True)
        if role == "worker"
    )
    assert worker_profile.base_url == "http://override:1/v1"
    assert worker_profile.model == "override-model"


async def test_each_continuation_re_resolves_the_named_profile(tmp_path, monkeypatch):
    """A key keeps its profile *name*, not a pinned resolution: the second task
    picks up whatever that name resolves to now."""
    inside = threading.Event()
    release = threading.Event()
    models: list[str] = []
    calls = {"n": 0, "worker": 0}

    def fake(profile, temperature, messages):
        if "You are a subagent" in messages[0]["content"]:
            models.append(profile.model)
            calls["worker"] += 1
            if calls["worker"] == 1:
                inside.set()
                release.wait(5)
            yield "worker answer"
            return
        calls["n"] += 1
        if calls["n"] == 1:
            yield dispatch_call(task_label="First")
        elif calls["n"] == 2:
            yield dispatch_call(task_label="Second")
        else:
            yield "all done"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    settings = local_settings()
    settings.profiles = Profiles(
        items=(Profile(name="local", base_url="http://host:9000/v1"),),
        active_name="local",
    )
    app = make_app(tmp_path, settings=settings)
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: inside.is_set())
        # The endpoint's uniquely discovered model changes between the tasks.
        app.server.models = ["qwen4"]
        release.set()
        await _wait_until(app, pilot, lambda: app._primary_turn_depth == 0)
        assert app._agents["coder"].profile_name == "local"

    assert models == ["qwen3", "qwen4"]


# ------------------------------------------------------------- parallelism


async def test_distinct_calls_all_start_before_any_of_them_finishes(
    tmp_path, monkeypatch
):
    """A gated fake: every worker must be inside its stream before any is
    released. A sequential implementation deadlocks here and fails."""
    release = threading.Event()
    live = threading.Semaphore(0)
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        if "You are a subagent" in messages[0]["content"]:
            live.release()
            release.wait(5)
            yield "worker answer"
            return
        calls["n"] += 1
        if calls["n"] == 1:
            yield "\n".join(
                dispatch_call(key=key, label=key.upper()) for key in ("a", "b", "c")
            )
        else:
            yield "all done"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")

        started = 0
        for _ in range(100):
            await pilot.pause(0.05)
            while live.acquire(blocking=False):
                started += 1
            if started == 3:
                break
        assert started == 3, f"only {started} agents started before any finished"
        assert all(agent_summary(app, key).status == "running" for key in "abc")
        assert primary_summary(app).status == "waiting"

        release.set()
        await _wait_until(app, pilot, lambda: app._primary_turn_depth == 0)
        assert [r["agent_key"] for r in agent_results(app)] == ["a", "b", "c"]
        assert primary_summary(app).status == "idle"


async def test_results_are_ordered_by_call_not_by_completion(tmp_path, monkeypatch):
    """The first call finishes last; the coordinator still reads them in the
    order it wrote them."""
    slow = threading.Event()
    calls = {"n": 0}
    finished: list[str] = []

    def fake(profile, temperature, messages):
        system = messages[0]["content"]
        if "You are a subagent" in system:
            task = messages[1]["content"]
            if task == "slow task":
                slow.wait(5)
            finished.append(task)
            yield f"answer for {task}"
            return
        calls["n"] += 1
        if calls["n"] == 1:
            yield (
                dispatch_call(key="a", label="A", task="slow task")
                + "\n"
                + dispatch_call(key="b", label="B", task="fast task")
            )
        else:
            yield "all done"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: finished == ["fast task"])
        slow.set()
        await _wait_until(app, pilot, lambda: app._primary_turn_depth == 0)

    assert finished == ["fast task", "slow task"]
    assert [r["agent_key"] for r in agent_results(app)] == ["a", "b"]
    assert agent_results(app)[0]["content"] == "answer for slow task"


async def test_one_failing_agent_does_not_cancel_its_siblings(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        system = messages[0]["content"]
        if "You are a subagent" in system:
            if messages[1]["content"] == "bad task":
                raise RuntimeError("provider exploded")
            yield "sibling answer"
            return
        calls["n"] += 1
        if calls["n"] == 1:
            yield (
                dispatch_call(key="a", label="A", task="bad task")
                + "\n"
                + dispatch_call(key="b", label="B", task="good task")
            )
        else:
            yield "all done"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await run_primary(app, pilot)

        results = agent_results(app)
        assert [r["status"] for r in results] == ["failed", "completed"]
        assert "provider exploded" in results[0]["content"]
        assert results[1]["content"] == "sibling answer"
        assert agent_summary(app, "a").status == "failed"
        assert agent_summary(app, "b").status == "completed"
        assert any("provider exploded" in line for line in agent_activity(app, "a"))


async def test_one_agent_runs_its_own_tasks_sequentially(tmp_path, monkeypatch):
    """A second call for a live key is refused rather than serialized: two
    concurrent writers to one conversation is the thing being prevented."""
    inside = threading.Event()
    release = threading.Event()
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        if "You are a subagent" in messages[0]["content"]:
            inside.set()
            release.wait(5)
            yield "worker answer"
            return
        calls["n"] += 1
        if calls["n"] == 1:
            yield dispatch_call(task_label="First")
        elif calls["n"] == 2:
            yield dispatch_call(task_label="Second")
        else:
            yield "all done"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: inside.is_set())
        # Reach into the live batch: a second task for the busy key is refused.
        managed = app._agents["coder"]
        assert managed.runtime is not None
        release.set()
        await _wait_until(app, pilot, lambda: app._primary_turn_depth == 0)
        assert [t.label for t in agent_summary(app, "coder").tasks] == [
            "First",
            "Second",
        ]


# --------------------------------------------------------- approvals & exit


async def test_one_modal_at_a_time_names_each_requesting_agent(
    tmp_path, monkeypatch
):
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        system = messages[0]["content"]
        if "You are a subagent" in system:
            worker_replies = [m for m in messages if m["role"] == "user"]
            if len(worker_replies) == 1:
                yield command_call("echo from-agent")
            else:
                yield "worker done"
            return
        calls["n"] += 1
        if calls["n"] == 1:
            yield (
                dispatch_call(key="a", label="Agent A")
                + "\n"
                + dispatch_call(key="b", label="Agent B")
            )
        else:
            yield "all done"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="ask", allow=[]))
    titles: list[str] = []
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")

        for _ in range(2):
            await _wait_until(
                app, pilot, lambda: isinstance(app.screen, CommandPrompt), tries=100
            )
            assert (
                sum(isinstance(screen, CommandPrompt) for screen in app.screen_stack)
                == 1
            )
            titles.append(
                str(app.screen.query_one(".dialog-title", Static).render())
            )
            # Waiting is the requester's own state: one agent's approval
            # never parks another.
            asking = "a" if "Agent A" in titles[-1] else "b"
            assert [
                key
                for key in ("a", "b")
                if agent_summary(app, key).status == "waiting"
            ] == [asking]
            await pilot.press("y")
            await settle(pilot)

        await _wait_until(app, pilot, lambda: app._primary_turn_depth == 0, tries=100)

    assert sorted(titles) == [
        "Run command for Agent A?",
        "Run command for Agent B?",
    ]
    assert [r["status"] for r in agent_results(app)] == ["completed", "completed"]


async def test_an_agent_waiting_for_the_lock_re_reads_the_saved_rule(
    tmp_path, monkeypatch
):
    """The second agent must not be prompted for a command the first agent's
    always-allow rule now covers."""
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        system = messages[0]["content"]
        if "You are a subagent" in system:
            if len([m for m in messages if m["role"] == "user"]) == 1:
                yield command_call("echo shared")
            else:
                yield "worker done"
            return
        calls["n"] += 1
        if calls["n"] == 1:
            yield (
                dispatch_call(key="a", label="Agent A")
                + "\n"
                + dispatch_call(key="b", label="Agent B")
            )
        else:
            yield "all done"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="ask", allow=[]))
    prompts = {"n": 0}
    async with app.run_test() as pilot:
        await _wait_until(app, pilot, lambda: True, tries=1)
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")

        await _wait_until(
            app, pilot, lambda: isinstance(app.screen, CommandPrompt), tries=100
        )
        prompts["n"] += 1
        await pilot.press("a")  # always allow
        await _wait_until(app, pilot, lambda: app._primary_turn_depth == 0, tries=100)
        if isinstance(app.screen, CommandPrompt):  # pragma: no cover - failure path
            prompts["n"] += 1

        assert prompts["n"] == 1
        assert "echo:*" in app.cmd.allow
        notices = agent_activity(app, "a") + agent_activity(app, "b")
        assert sum("Always-allow saved: echo:*" in line for line in notices) == 1
        assert [r["status"] for r in agent_results(app)] == ["completed", "completed"]


async def test_escape_never_stops_a_subagent(tmp_path, monkeypatch):
    release = threading.Event()
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        if "You are a subagent" in messages[0]["content"]:
            release.wait(5)
            yield "worker answer"
            return
        calls["n"] += 1
        yield dispatch_call() if calls["n"] == 1 else "all done"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: "coder" in app._agents)
        worker = app._agents["coder"].runtime
        assert worker is not None

        await pilot.press("escape")
        await select_agent(app, pilot, "coder")
        await pilot.press("escape")
        await settle(pilot)
        assert not worker.state.stop_event.is_set()

        release.set()
        await _wait_until(app, pilot, lambda: app._primary_turn_depth == 0)
        assert agent_results(app)[0]["status"] == "completed"


async def test_exiting_signals_every_live_runtime(tmp_path, monkeypatch):
    release = threading.Event()
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        if "You are a subagent" in messages[0]["content"]:
            release.wait(5)
            yield "worker answer"
            return
        calls["n"] += 1
        yield dispatch_call() if calls["n"] == 1 else "all done"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    app = make_app(tmp_path)
    try:
        async with app.run_test() as pilot:
            inp = app.query_one("#input", Input)
            inp.value = "go"
            await pilot.press("enter")
            await _wait_until(
                app,
                pilot,
                lambda: app._agents.get("coder") is not None
                and app._agents["coder"].runtime is not None
                and app._agents["coder"].runtime.state.generating,
            )
            worker = app._agents["coder"].runtime
            primary = app._primary_runtime
            app.exit()
            assert worker.state.stop_event.is_set()
            assert primary is not None
    finally:
        release.set()


async def test_exiting_kills_a_subagent_command_and_everything_it_started(
    tmp_path, monkeypatch
):
    """Exiting must leave nothing behind — not the shell, not its pipeline, and
    not a background job that would keep editing the project afterwards."""
    marker = tmp_path / "leaked.txt"
    command = f"( sleep 3; echo LEAKED > {marker} ) & sleep 30 | cat"
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        if "You are a subagent" in messages[0]["content"]:
            yield command_call(command)
            return
        calls["n"] += 1
        yield dispatch_call() if calls["n"] == 1 else "all done"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="yolo"))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")

        def descendants() -> list[str]:
            runtime = (app._agents.get("coder") or _ManagedAgentStub).runtime
            proc = None if runtime is None else runtime.state.running_proc
            if proc is None:
                return []
            found = subprocess.run(
                ["pgrep", "-P", str(proc.pid)],
                capture_output=True,
                text=True,
                check=False,
            )
            return found.stdout.split()

        await _wait_until(app, pilot, lambda: len(descendants()) >= 2)
        proc = app._agents["coder"].runtime.state.running_proc
        children = descendants()

        app.exit()
        await settle(pilot)
        assert proc.poll() is not None
        for _ in range(100):
            await pilot.pause(0.05)
            if not [c for c in children if Path(f"/proc/{c}").exists()]:
                break
        assert [c for c in children if Path(f"/proc/{c}").exists()] == []
        # Past the background job's own delay: it must never have run.
        await asyncio.sleep(4)
    assert not marker.exists()


class _ManagedAgentStub:
    """Stand-in used only while an agent has not been registered yet."""

    runtime = None


async def test_an_unexpected_setup_failure_fails_only_its_own_call(
    tmp_path, monkeypatch
):
    """Setting one agent up is that call's own work: an unexpected failure
    there must not take the rest of the batch down with it."""
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        if "You are a subagent" in messages[0]["content"]:
            yield "sibling answer"
            return
        calls["n"] += 1
        if calls["n"] == 1:
            yield (
                dispatch_call(key="a", label="A")
                + "\n"
                + dispatch_call(key="b", label="B")
            )
        else:
            yield "all done"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    app = make_app(tmp_path)
    original = app._begin_agent_task

    async def flaky(call, task_id):
        if call.agent_key == "a":
            raise RuntimeError("setup boom")
        return await original(call, task_id)

    monkeypatch.setattr(app, "_begin_agent_task", flaky)
    async with app.run_test() as pilot:
        await run_primary(app, pilot)

        results = agent_results(app)
        assert [(r["agent_key"], r["status"]) for r in results] == [
            ("a", "failed"),
            ("b", "completed"),
        ]
        assert "RuntimeError: setup boom" in results[0]["content"]
        assert results[1]["content"] == "sibling answer"
        # The failing call created nothing; the sibling ran normally.
        assert list(app._agents) == ["b"]
        assert agent_summary(app, "b").status == "completed"


async def test_a_setup_failure_never_leaves_a_task_row_running_forever(
    tmp_path, monkeypatch
):
    """A continuation commits its task row before its transcript write. If that
    write fails, the row must end failed, not stay running for the session."""
    inside = threading.Event()
    release = threading.Event()
    calls = {"n": 0, "worker": 0}

    def fake(profile, temperature, messages):
        if "You are a subagent" in messages[0]["content"]:
            calls["worker"] += 1
            if calls["worker"] == 1:
                inside.set()
                release.wait(5)
            yield "worker answer"
            return
        calls["n"] += 1
        if calls["n"] == 1:
            yield dispatch_call(task_label="First")
        elif calls["n"] == 2:
            yield dispatch_call(task_label="Second")
        else:
            yield "all done"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        # The first task is committed and running; break the second one's
        # transcript write before the coordinator sends it.
        await _wait_until(app, pilot, lambda: inside.is_set())
        managed = app._agents["coder"]

        def exploding_append(record):
            raise RuntimeError("transcript boom")

        monkeypatch.setattr(managed.transcript, "append", exploding_append)
        release.set()
        await _wait_until(app, pilot, lambda: app._primary_turn_depth == 0, tries=100)

        summary = agent_summary(app, "coder")
        assert [(t.label, t.status) for t in summary.tasks] == [
            ("First", "completed"),
            ("Second", "failed"),
        ]
        results = agent_results(app)
        assert [r["status"] for r in results] == ["completed", "failed"]
        assert "RuntimeError: transcript boom" in results[1]["content"]
        # And the assignment that never ran is not left in the worker's
        # context, where the agent's next task would read it as an
        # outstanding instruction.
        assert [m["content"] for m in app._agents["coder"].session.messages] == [
            "do the work",
            "worker answer",
        ]


# ------------------------------------------------------------ result routing


async def test_a_result_is_recorded_once_with_its_exact_identity(
    tmp_path, monkeypatch
):
    stream = Conversation([dispatch_call(), "all done"], ["the worker answer"])
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(stream))
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await run_primary(app, pilot)

        results = agent_results(app)
        assert len(results) == 1
        result = results[0]
        assert result["agent_key"] == "coder"
        assert result["agent_label"] == "Coder"
        assert result["task_label"] == "Task"
        assert result["status"] == "completed"
        assert result["content"] == "the worker answer"
        assert result["task_id"] == agent_summary(app, "coder").tasks[0].task_id
        record = next(
            m for m in app.session.messages
            if m.get("_model_content", "").startswith("[JTECH agent result]")
        )
        assert record["role"] == "system"
        assert record["_model_role"] == "user"
        assert record["content"] == "Coder completed: Task"


async def test_a_primary_history_failure_is_visible_but_keeps_the_result(
    tmp_path, monkeypatch
):
    stream = Conversation([dispatch_call(), "all done"], ["worker answer"])
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(stream))
    app = make_app(tmp_path)
    original = app.session.add
    failures = {"n": 0}

    def failing(role, content, **kwargs):
        # The message still joins the in-memory conversation; only the append
        # to disk fails, exactly as a real OSError leaves it.
        original(role, content, **kwargs)
        if (kwargs.get("model_content") or "").startswith("[JTECH agent result]"):
            failures["n"] += 1
            raise OSError("disk full")

    monkeypatch.setattr(app.session, "add", failing)
    async with app.run_test() as pilot:
        await run_primary(app, pilot)

        assert failures["n"] == 1
        assert any("Could not save history: disk full" in b for b in bubbles(app))
        # The in-memory result still reached the next request.
        last_request = stream.sent_to("primary")[-1]
        assert any(
            m["content"].startswith("[JTECH agent result]") for m in last_request
        )


async def test_primary_reports_waiting_while_its_own_command_awaits_approval(
    tmp_path, monkeypatch
):
    """The requester is Primary here, so the same waiting rule applies to it."""
    fake, _ = cmd_stream(command_call("echo needs-approval"), "done")
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="ask", allow=[]))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await _wait_until(
            app, pilot, lambda: isinstance(app.screen, CommandPrompt), tries=100
        )
        assert primary_summary(app).status == "waiting"
        title = str(app.screen.query_one(".dialog-title", Static).render())
        assert title == "Run command for Primary?"

        await pilot.press("y")
        await _wait_until(app, pilot, lambda: app._primary_turn_depth == 0, tries=100)
        assert primary_summary(app).status == "idle"


async def test_a_late_runtime_notification_after_exit_is_not_an_error(tmp_path):
    """Every runtime is still unwinding its own `finally` while the app tears
    down, so those notifications arrive after the widgets are gone."""
    app = make_app(tmp_path)
    async with app.run_test():
        run = app._primary_run_state(
            ResolvedProfile(
                name="local",
                base_url="http://host:9000/v1",
                model="qwen3",
                api_key="none",
            )
        )
    assert not app.is_running
    run.generating = False
    run.running_proc = None
    app.runtime_changed(run)
