"""App shell: startup, status bar, discovery, settings, theme, and /clear."""

import pytest
from textual import events
from textual.containers import Vertical
from textual.widgets import Input, Static, TextArea
from textual.widgets.text_area import Selection as AreaSelection

from jtech_cli.config import Profile, Profiles, Settings
from jtech_cli.server_info import ServerInfo
from jtech_cli.tui import ChatApp, SettingsScreen

from .support import (
    bubbles,
    chat_of,
    history_lines,
    local_settings,
    make_app,
    settle,
    stream_of,
    sync_stream,
    type_text,
    wait_until,
)


def local_settings_with_model(model: str) -> Settings:
    """Settings whose active profile pins ``model`` explicitly."""
    profile = Profile(name="local", base_url="http://host:9000/v1", model=model)
    return Settings(profiles=Profiles(items=(profile,), active_name="local"))


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


def hint_text(app: ChatApp) -> str:
    """The settings screen hint line, as rendered."""
    return str(app.screen.query_one("#settings-hint", Static).render())


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
        await wait_until(app, pilot, lambda: bool(app.server.models), tries=50)

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
        await wait_until(app, pilot, lambda: any("Could not reach" in b for b in bubbles(app)), tries=50)

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
        await wait_until(app, pilot, lambda: bool(app.server.models), tries=50)
        assert app.settings.model == "explicit"
