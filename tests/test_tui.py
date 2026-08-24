"""Tests for the Textual TUI: bubbles, status bar, notices, settings, and /clear."""

import threading
import time

from textual.containers import Vertical
from textual.widgets import Input, Markdown, Static, TextArea

from jtech_cli.cmd_tools import CmdPolicy
from jtech_cli.config import Settings, load_cmd_policy
from jtech_cli.prompts import NUDGE_PROMPT
from jtech_cli.server_info import ServerInfo
from jtech_cli.session import Session
from jtech_cli.tui import ChatApp, CommandPrompt, SettingsScreen


def make_app(tmp_path, settings=None, session=None, server=None, no_discover=True):
    """A ChatApp for tests. Discovery is off unless a test opts in.

    Left on, every app here fires a real HTTP request at host:9000 and waits
    out the 5s timeout — network I/O in a unit test, and a failure notice in
    the chat that tests asserting on bubbles have to know to ignore.
    """
    settings = settings or Settings(base_url="http://host:9000/v1", model="qwen3")
    session = session or Session(tmp_path / "s.jsonl", persist=False)
    server = server or ServerInfo(models=["qwen3"], context_length=4096)
    return ChatApp(
        settings=settings,
        session=session,
        server=server,
        config_path=tmp_path / "config.toml",
        no_discover=no_discover,
    )


def make_settings(reasoning: str) -> Settings:
    return Settings(base_url="http://host:9000/v1", model="qwen3", reasoning=reasoning)


def bubbles(app: ChatApp) -> list[str]:
    chat = app.query_one("#chat")
    return [w._markdown for w in chat.children if isinstance(w, Markdown)]


def reasoning_body_widget(app: ChatApp) -> Static | None:
    """The mounted reasoning bubble body (plain Static, not the label), if any."""
    for w in app.query_one("#chat").children:
        if isinstance(w, Static) and "bubble" in w.classes and "reasoning" in w.classes:
            return w
    return None


def reasoning_bodies(app: ChatApp) -> list[str]:
    """Text of mounted reasoning bubble bodies (plain Statics, not the label)."""
    w = reasoning_body_widget(app)
    return [str(w.render())] if w is not None else []


def labels(app: ChatApp) -> list[str]:
    return [
        str(w.render()) for w in app.query_one("#chat").children if isinstance(w, Static)
    ]


def at_bottom(chat) -> bool:
    """True when the scroll offset is at (or within 2 lines of) the bottom."""
    return chat.scroll_offset.y >= chat.max_scroll_y - 2


async def test_submit_shows_user_and_ai_bubble(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    monkeypatch.setattr(
        "jtech_cli.tui.stream_reply", lambda settings, messages: iter(["hi ", "there"])
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

    MarkdownFence's Label defaults to content width with its horizontal
    scrollbar hidden, so long code lines used to be cut off at the right
    edge. The app CSS constrains the label to the bubble width so the
    content folds like normal text — the whole word stays visible.
    """
    from textual.widgets._markdown import MarkdownFence

    # 'z' is unused elsewhere on screen (the status bar's "ctx" would skew
    # a character count if we used 'x').
    long_word = "z" * 300
    reply = f"```\n{long_word}\nafter-line\n```\n"
    monkeypatch.setattr(
        "jtech_cli.tui.stream_reply", lambda settings, messages: iter([reply])
    )
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        for _ in range(10):
            await pilot.pause()
        await pilot.pause()

        fence = app.query_one("#chat").query_one(MarkdownFence)
        label = fence.query_one("#code-content", Static)
        # The label must be constrained to the bubble width (it wraps);
        # before the fix it was as wide as the longest line (300+ cells).
        assert label.size.width <= fence.size.width
        assert label.size.height > 1
        # And every character of the long word must actually be rendered.
        comp = app.screen._compositor
        visible = sum(s.text.count("z") for s in comp.render_strips(comp.size))
        assert visible == len(long_word)


async def test_status_bar_omits_base_url_prefix(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test():
        status = app.query_one("#status", Static).content
    assert "base_url=" not in status
    assert "http://host:9000/v1" in status
    assert "model: qwen3" in status
    assert "ctx 4096" in status


async def test_status_bar_empty_base_url(tmp_path):
    app = make_app(tmp_path, settings=Settings())
    async with app.run_test():
        status = app.query_one("#status", Static).content
    assert status.strip() != ""


async def test_input_responsive_after_connection_error(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    calls = {"n": 0}

    def fake(settings, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return iter(["ok"])

    monkeypatch.setattr("jtech_cli.tui.stream_reply", fake)
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


async def test_settings_screen_opens_and_lists_rows(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test(size=(80, 24)) as pilot:
        app.action_settings()
        await pilot.pause()
        assert isinstance(app.screen, SettingsScreen)
        rows = settings_rows_text(app)
        assert "Model" in rows
        assert "Base URL" in rows
        assert "Additional instructions" in rows
        assert "qwen3" in rows  # current model value shown on its row

        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, SettingsScreen)


async def test_settings_description_follows_highlighted_row(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test(size=(80, 24)) as pilot:
        app.action_settings()
        await pilot.pause()
        assert "model" in settings_help_text(app).lower()  # Model is row 0

        await pilot.press("down")  # Base URL
        await pilot.pause()
        assert "endpoint" in settings_help_text(app)

        await pilot.press("down", "down")  # Theme
        await pilot.pause()
        help_text = settings_help_text(app)
        assert "terminal" in help_text and "light/dark" in help_text

        await pilot.press("up")  # back to Temperature
        await pilot.pause()
        assert "0.0-2.0" in settings_help_text(app)


async def test_settings_enter_edits_row_and_commits(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test(size=(80, 24)) as pilot:
        app.action_settings()
        await pilot.pause()

        # cursor starts on Model; Enter opens the in-place editor
        await pilot.press("enter")
        await pilot.pause()
        field = app.screen.query_one("#settings-field", Input)
        assert field.value == "qwen3"

        # the highlighted row's description stays while editing
        assert "model" in settings_help_text(app).lower()

        field.value = "qwen9"
        await pilot.press("enter")
        await pilot.pause()
        assert app.settings.model == "qwen9"
        assert (tmp_path / "config.toml").exists()
        assert not app.screen.query_one("#settings-editor", Vertical).children
        assert "qwen9" in settings_rows_text(app)


async def test_settings_invalid_value_keeps_editor_open(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test(size=(80, 24)) as pilot:
        app.action_settings()
        await pilot.pause()

        await pilot.press("down", "down")  # Temperature is row 2
        await pilot.pause()
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

        await pilot.press("down", "down", "down")  # Theme is row 3
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

        await pilot.press("enter")  # start editing Model
        await pilot.pause()
        field = app.screen.query_one("#settings-field", Input)
        field.value = "changed"
        await pilot.press("escape")  # cancel the edit
        await pilot.pause()
        assert app.settings.model == "qwen3"
        assert not app.screen.query_one("#settings-editor", Vertical).children

        await pilot.press("escape")  # close the menu
        await pilot.pause()
        assert not isinstance(app.screen, SettingsScreen)


async def test_settings_system_prompt_edits_multiline(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test(size=(80, 24)) as pilot:
        app.action_settings()
        await pilot.pause()

        await pilot.press("down", "down", "down", "down", "down")  # System prompt is row 5
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        ta = app.screen.query_one("#settings-field", TextArea)
        ta.text = "line one\nline two"
        await pilot.press("enter")
        await pilot.pause()
        assert app.settings.system_prompt == "line one\nline two"
        assert app.settings.prompt_source == "inline"
        assert not app.screen.query_one("#settings-editor", Vertical).children


async def test_input_works_after_opening_and_closing_settings(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    monkeypatch.setattr("jtech_cli.tui.stream_reply", lambda settings, messages: iter(["ok"]))
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


async def test_empty_base_url_shows_notice(tmp_path):
    app = make_app(tmp_path, settings=Settings())
    async with app.run_test():
        assert any("No server configured" in b for b in bubbles(app))


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
    monkeypatch.setattr("jtech_cli.tui.stream_reply", lambda settings, messages: iter(["ok"]))
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


async def test_heredoc_multiline_sends_message(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    monkeypatch.setattr("jtech_cli.tui.stream_reply", lambda s, m: iter(["ok"]))
    async with app.run_test() as pilot:
        ta = await _enter_multiline(app, pilot, "'''")
        ta.text = "line one\nline two\n'''"
        await pilot.press("ctrl+enter")
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
        ta.text = "hello\nworld\nEND"
        await pilot.press("ctrl+enter")
        for _ in range(10):
            await pilot.pause()

        assert (tmp_path / "out.txt").read_text() == "hello\nworld"
        assert app.query_one("#input", Input) is not None
        assert not list(app.query("#multiline-input"))


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


async def test_shift_enter_opens_prefilled_multiline(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    monkeypatch.setattr("jtech_cli.tui.stream_reply", lambda s, m: iter(["ok"]))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "first line"
        await pilot.press("shift+enter")
        for _ in range(5):
            await pilot.pause()
        ta = app.query_one("#multiline-input", TextArea)
        assert ta.text == "first line"
        assert inp.value == ""

        ta.text = "first line\nsecond line\n'''"
        await pilot.press("ctrl+enter")
        for _ in range(10):
            await pilot.pause()

        assert app.session.messages == [
            {"role": "user", "content": "first line\nsecond line"},
            {"role": "assistant", "content": "ok"},
        ]
        assert app.query_one("#input", Input) is not None
        assert not list(app.query("#multiline-input"))


async def _send_and_drain(app: ChatApp, pilot, text: str) -> None:
    inp = app.query_one("#input", Input)
    inp.value = text
    await pilot.press("enter")
    for _ in range(10):
        await pilot.pause()


def reason_stream(s, m):
    return iter([("reasoning", "hmm "), ("reasoning", "ok"), "4"])


async def test_reasoning_default_transient_shown_then_hidden(tmp_path, monkeypatch):
    """Default mode: reasoning streams in its own bubble, removed once the answer starts."""
    app = make_app(tmp_path)
    monkeypatch.setattr("jtech_cli.tui.stream_reply", reason_stream)
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
    monkeypatch.setattr("jtech_cli.tui.stream_reply", reason_stream)
    async with app.run_test() as pilot:
        await _send_and_drain(app, pilot, "what is 2+2")

        text = "\n".join(bubbles(app))
        assert "4" in text
        assert "hmm" not in text
        assert not reasoning_bodies(app)
        # no reasoning widgets mounted at all
        chat = app.query_one("#chat")
        assert not [
            w for w in chat.children if isinstance(w, Static) and "reasoning" in w.classes
        ]
        assert app.session.messages == [
            {"role": "user", "content": "what is 2+2"},
            {"role": "assistant", "content": "4"},
        ]


async def test_reasoning_always_kept_in_separate_bubble(tmp_path, monkeypatch):
    app = make_app(tmp_path, settings=make_settings("always"))
    monkeypatch.setattr("jtech_cli.tui.stream_reply", reason_stream)
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
        "jtech_cli.tui.stream_reply", lambda s, m: iter([("reasoning", full), "4"])
    )
    async with app.run_test() as pilot:
        await _send_and_drain(app, pilot, "what is 2+2")

        assert reasoning_bodies(app) == ["…" + full[-500:]]
        assert any("4" in b for b in bubbles(app))


async def test_waiting_label_ticks_without_tokens(tmp_path, monkeypatch):
    """The 1s timer repaints the label in real time even with a silent stream."""
    app = make_app(tmp_path)

    def fake(settings, messages):
        time.sleep(1.5)
        yield "ok"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", fake)
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
    for w in app.query_one("#chat").children:
        if isinstance(w, Static) and "bubble-label" in w.classes and "ai" in w.classes:
            return w
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

    def slow_reason(settings, messages):
        time.sleep(0.4)  # let the mount-time scroll settle before content arrives
        yield ("reasoning", "thinking out loud " * 20)  # ~320 chars -> several lines
        gate.wait(5)
        yield ("reasoning", "more thoughts ")
        yield "4"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", slow_reason)
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
        lambda s, m: iter(
            ["ok", ("timings", {"prompt_n": 170, "prompt_ms": 594.8, "prompt_per_second": 285.8})]
        ),
    )
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "hi"
        await pilot.press("enter")
        for _ in range(10):
            await pilot.pause()

        labels = [
            str(w.render()) for w in app.query_one("#chat").children if isinstance(w, Static)
        ]
        assert any("170" in l and "0.6s" in l and "286 t/s" in l for l in labels)


async def test_esc_idle_does_nothing(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.press("escape")
        for _ in range(5):
            await pilot.pause()

        assert app.session.messages == []
        assert not bubbles(app)
        assert app.focused is app.query_one("#input", Input)


async def test_esc_stops_stream_and_discards_partial(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    gate = threading.Event()

    def fake(settings, messages):
        yield "partial "
        gate.wait(5)
        yield "never"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", fake)
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "hello"
        await pilot.press("enter")
        for _ in range(50):
            await pilot.pause()
            if any("partial" in b for b in bubbles(app)):
                break

        await pilot.press("escape")
        gate.set()
        for _ in range(50):
            await pilot.pause()
            if not any("partial" in b for b in bubbles(app)):
                break

        assert app.session.messages == [{"role": "user", "content": "hello"}]
        assert not any("partial" in b for b in bubbles(app))
        assert not any("never" in b for b in bubbles(app))
        assert any("Generation stopped" in b for b in bubbles(app))
        assert app.query_one("#input", Input) is not None


def suggestions_text(app: ChatApp) -> str:
    return str(app.query_one("#suggestions", Static).render())


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

    def fake(settings, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            yield "r1 "
            gate.wait(5)
            yield "r1b"
        else:
            yield "r2"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", fake)
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

    def fake(settings, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            yield "r1 "
            gate.wait(5)
            yield "r1b"
        else:
            yield "r2"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", fake)
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

    def fake(settings, messages):
        yield "r1 "
        gate.wait(5)
        yield "r1b"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", fake)
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

    def fake(settings, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            yield "r1 "
            gate.wait(5)
            yield "r1b"
        else:
            yield f"r{calls['n']}"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", fake)
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

    def fake(settings, messages):
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

    def fake(settings, messages):
        calls["n"] += 1
        requests.append(messages)
        yield command_call("echo hello-out") if calls["n"] == 1 else "done"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", fake)
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
    monkeypatch.setattr("jtech_cli.tui.stream_reply", fake)
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
    monkeypatch.setattr("jtech_cli.tui.stream_reply", fake)
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
    monkeypatch.setattr("jtech_cli.tui.stream_reply", fake)
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
    monkeypatch.setattr("jtech_cli.tui.stream_reply", fake)
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
    monkeypatch.setattr("jtech_cli.tui.stream_reply", fake)
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

    def fake(settings, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            yield "\n".join(command_call(f"echo blk-{i}") for i in range(total))
        else:
            yield "stopped"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", fake)
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

    def fake(settings, messages):
        calls["n"] += 1
        if calls["n"] <= 6:
            yield command_call(f"echo round-{calls['n']}")
        else:
            yield "finished"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", fake)
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

    def fake(settings, messages):
        calls["n"] += 1
        if calls["n"] <= repeats:
            yield command_call("echo unchanged")
        else:
            yield "done"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", fake)
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

    def fake(settings, messages):
        calls["n"] += 1
        requests.append(messages)
        if calls["n"] == 1:
            yield command_call("echo tool-out")
        elif calls["n"] <= 4:
            yield ""
        else:
            yield "recovered"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", fake)
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
    settings = Settings(
        base_url="http://host:9000/v1", model="qwen3", debug_level="system"
    )
    app = make_app_with_cmd(
        tmp_path, CmdPolicy(mode="auto", allow=["echo:*"]), settings=settings
    )
    calls = {"n": 0}

    def fake(settings, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            yield command_call("echo debug-nudge")
        elif calls["n"] == 2:
            yield ""
        else:
            yield "done"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", fake)
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

    def fake(settings, messages):
        calls["n"] += 1
        replies = {
            1: command_call("echo first"),
            2: "",
            3: command_call("echo second"),
            4: "finished",
        }
        yield replies[calls["n"]]

    monkeypatch.setattr("jtech_cli.tui.stream_reply", fake)
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

    def fake(settings, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            yield command_call("pwd")
        else:
            yield "The cwd is /the/project.\n\n```cmd\npwd\n```"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", fake)
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

    def fake(settings, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            yield command_call("pwd") + "\n\nLet me inspect the project structure next."
        else:
            yield "The cwd is /the/project."

    monkeypatch.setattr("jtech_cli.tui.stream_reply", fake)
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

    def fake(settings, messages):
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

    monkeypatch.setattr("jtech_cli.tui.stream_reply", fake)
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

    def fake(settings, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            yield '<code>\njtech_cmd("pwd")\n</code>'
        else:
            yield "done"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", fake)
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

    def fake(settings, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            yield command_call("echo clear-out")
        elif calls["n"] == 2:
            gate.wait(5)  # hold the post-command stream open across the /clear
            yield "after clear"
        else:
            yield "unexpected extra request"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", fake)
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

    def fake(settings, messages):
        calls["n"] += 1
        yield "all done"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", fake)
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

    def fake(settings, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            yield command_call("echo declined-out")
        else:
            yield "stopped"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", fake)
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

    def fake(settings, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            yield command_call("sudo ls")
        else:
            yield "stopped"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", fake)
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

    def fake(settings, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            yield command_call("echo failing-out; exit 3")
        elif calls["n"] == 2:
            yield command_call("echo recovered-out")
        else:
            yield "handled"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", fake)
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


async def test_cmd_timeout_feeds_partial_output(tmp_path, monkeypatch):
    """A command killed by the timeout feeds its partial output to the model."""
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="yolo", timeout=1))
    calls = {"n": 0}

    def fake(settings, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            yield command_call("echo got-far; sleep 5")
        else:
            yield "final"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", fake)
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: calls["n"] >= 2, tries=100)

        sys_msgs = [m["content"] for m in app.session.messages if m["role"] == "system"]
        assert any("timed out" in m and "got-far" in m for m in sys_msgs)
        assert any("timed out" in b for b in bubbles(app))

        # let the tool turn finish before teardown: a still-live
        # stream would race _render_status against the unmounting widgets
        await _wait_until(app, pilot, lambda: not app._tool_rounds_active, tries=100)
        await pilot.pause()


async def test_queue_drains_after_esc_stop(tmp_path, monkeypatch):
    """Esc-stopping the in-flight reply still unblocks the queue."""
    app = make_app(tmp_path)
    gate = threading.Event()
    calls = {"n": 0}

    def fake(settings, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            yield "partial "
            gate.wait(5)
        else:
            yield "r2"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", fake)
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "one"
        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: any("partial" in b for b in bubbles(app)))

        inp.value = "two"
        await pilot.press("enter")
        await _wait_until(app, pilot, lambda: bool(app._queue), tries=10, pause=0.05)

        await pilot.press("escape")
        gate.set()
        await _wait_until(app, pilot, lambda: calls["n"] >= 2)
        assert any("Generation stopped" in b for b in bubbles(app))
        assert not any("Queued" in b for b in bubbles(app))

    # partial reply discarded; queued "two" still sent afterwards
    assert app.session.messages == [
        {"role": "user", "content": "one"},
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

    def fake(settings, messages):
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
    monkeypatch.setattr("jtech_cli.tui.stream_reply", fake)
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
    monkeypatch.setattr("jtech_cli.tui.stream_reply", fake)
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
    monkeypatch.setattr("jtech_cli.tui.stream_reply", fake)
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
        settings=Settings(base_url="http://host:9000/v1", model="explicit"),
        server=ServerInfo(),
        no_discover=False,
    )
    async with app.run_test() as pilot:
        await _wait_until(app, pilot, lambda: bool(app.server.models), tries=50)
        assert app.settings.model == "explicit"
