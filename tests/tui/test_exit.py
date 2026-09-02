"""Contextual Ctrl+C: copy, clear, confirm quit."""

import pytest
from textual.geometry import Offset
from textual.widgets import Input, Static, TextArea
from textual.widgets.input import Selection as InputSelection
from textual.widgets.text_area import Selection as AreaSelection

from jtech_cli.tui import ChatApp, QuitScreen, SettingsScreen

from .support import (
    base_input,
    chat_of,
    clear_input_selection,
    enter_multiline,
    make_app,
    settle,
    spy_exit,
    suggestions_box,
)


def quit_rows_text(app: ChatApp) -> str:
    return str(app.screen.query_one("#quit-rows", Static).render())


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
        ta = await enter_multiline(app, pilot, "'''")
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
        ta = await enter_multiline(app, pilot, "'''")
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
        ta = await enter_multiline(app, pilot, "'''")
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
