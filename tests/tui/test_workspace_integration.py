"""The agent workspace at app level: selection, isolation, primary status."""

import threading

from textual.containers import Vertical
from textual.geometry import Offset
from textual.widgets import Input, Static, TextArea
from textual.widgets.text_area import Selection as AreaSelection

from jtech_cli.tui import ChatApp, QuitScreen
from jtech_cli.tui_app import (
    PRIMARY_AGENT_ID,
    SUBAGENT_CLEAR_BLOCKED,
    SUBAGENT_READONLY,
)
from jtech_cli.tui_widgets import (
    AgentSummary,
    AgentTaskSummary,
    Transcript,
    TranscriptRecord,
)

from .support import (
    activity_text,
    base_input,
    bubbles,
    chat_of,
    make_app,
    notifications,
    primary_summary,
    select_agent,
    settle,
    spy_exit,
    stream_of,
    suggestions_box,
    suggestions_text,
    sync_stream,
    visible_activity,
    wait_until,
    workspace_of,
)


def readonly_notice(app: ChatApp) -> Static:
    return app.query_one("#subagent-readonly", Static)


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
        await wait_until(app, pilot, lambda: any("partial" in b for b in bubbles(app)))
        await app.add_agent_view(AgentSummary("a", "Agent A", "running"))
        await settle(pilot)
        await select_agent(app, pilot, "a")

        await pilot.press("escape")
        await settle(pilot)

        assert not app._primary_runtime.state.stop_event.is_set()
        assert app._generating

        gate.set()
        await wait_until(app, pilot, lambda: not app._generating)

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
        await wait_until(app, pilot, lambda: primary_summary(app).status == "running")

        running = primary_summary(app)
        assert running.label == "Primary agent"
        assert running.tasks == tasks

        gate.set()
        await wait_until(app, pilot, lambda: primary_summary(app).status == "idle")

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
        await wait_until(app, pilot, lambda: any("r1" in b for b in bubbles(app)))
        assert statuses == ["running"]

        for message in ("two", "three"):
            inp.value = message
            await pilot.press("enter")
        await wait_until(app, pilot, lambda: len(app._queue) == 2, tries=10, pause=0.05)
        # Queued turns are nested inside the accepted one; none of them may
        # report the agent idle while the drain is still running.
        assert statuses == ["running"]

        gate.set()
        await wait_until(app, pilot, lambda: calls["n"] >= 3)
        await wait_until(app, pilot, lambda: app._primary_turn_depth == 0)

        assert statuses == ["running", "idle"]
        assert primary_summary(app).status == "idle"
        assert app._queue == []
