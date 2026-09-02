"""Tests for the agent workspace: sidebar rendering, selection, and isolation.

These are focused component tests. ``tests/tui/test_workspace_integration.py``
owns the app-level contract; this module owns the workspace's own rendering
and selection rules, which are testable with a minimal host and, for the pure
helper, with no host at all.
"""

import asyncio
from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.geometry import Offset
from textual.selection import Selection
from textual.widgets import ListItem, ListView, Static
from textual.widgets.input import Selection as InputSelection

from jtech_cli.theme import JTECH_DARK, JTECH_LIGHT
from jtech_cli.tui_widgets import (
    AgentSummary,
    AgentTaskSummary,
    AgentWorkspace,
    Transcript,
    TranscriptRecord,
    _AgentListItem,
    _ChatInput,
    render_agent_summary,
)

CSS_PATH = (
    Path(__file__).resolve().parent.parent
    / "jtech_cli"
    / "resources"
    / "styles"
    / "tui.css"
)

PRIMARY = AgentSummary("primary", "Primary", "idle")

# Every glyph the sidebar may draw, and the only status signal there is.
STATUS_GLYPHS = {
    "idle": "○",
    "running": "●",
    "waiting": "◌",
    "completed": "✓",
    "failed": "!",
}


class WorkspaceApp(App):
    """A minimal host for one `AgentWorkspace`.

    It carries the real stylesheet and the real themes, so widths, clipping,
    and completed-history rendering behave exactly as they do in the chat app.
    The single-line input exists only so a widget-owned selection can be shown
    to survive an agent switch.
    """

    CSS_PATH = CSS_PATH

    def __init__(self, primary: AgentSummary = PRIMARY) -> None:
        super().__init__()
        self._primary = primary

    def compose(self) -> ComposeResult:
        yield AgentWorkspace(
            self._primary,
            Transcript(id="chat", classes="agent-activity"),
            id="agent-workspace",
        )
        yield _ChatInput(id="input")

    def on_mount(self) -> None:
        self.register_theme(JTECH_DARK)
        self.register_theme(JTECH_LIGHT)
        self.theme = JTECH_DARK.name


def workspace_of(app: WorkspaceApp) -> AgentWorkspace:
    return app.query_one("#agent-workspace", AgentWorkspace)


def sidebar_rows(app: WorkspaceApp) -> list[str]:
    """One string per sidebar row, in list order, as the user sees them."""
    return [
        str(item.query_one(".agent-list-text", Static).render())
        for item in app.query(_AgentListItem)
    ]


def records(*contents: str) -> list[TranscriptRecord]:
    return [TranscriptRecord("assistant", content) for content in contents]


# --- pure rendering --------------------------------------------------------


def test_agent_summary_render_keeps_agents_flat_and_tasks_two_spaces_deeper():
    primary = AgentSummary(
        "primary",
        "Primary",
        "running",
        (
            AgentTaskSummary("t1", "Implement parser", "running"),
            AgentTaskSummary("t2", "Run audit", "waiting"),
        ),
    )
    agent_a = AgentSummary(
        "a",
        "Agent A",
        "completed",
        (AgentTaskSummary("t3", "Inspect tests", "completed"),),
    )

    selected = render_agent_summary(primary, selected=True).plain.split("\n")
    unselected = render_agent_summary(agent_a, selected=False).plain.split("\n")

    assert selected == [
        "▸ ● Primary",
        "    ● Implement parser",
        "    ◌ Run audit",
    ]
    assert unselected == ["  ✓ Agent A", "    ✓ Inspect tests"]
    # Selected and unselected agents share one base column: the marker never
    # shifts the hierarchy, and no agent is nested under another.
    assert selected[0].index("Primary") == unselected[0].index("Agent A") == 4
    # Every task label sits exactly two columns deeper than its agent label.
    assert selected[1].index("Implement parser") == 6
    assert selected[2].index("Run audit") == 6
    assert unselected[1].index("Inspect tests") == 6

    every_line = selected + unselected
    for connector in ("│", "├", "└", "─", "┬", "▾", "▶", "·"):
        assert not any(connector in line for line in every_line)
    # The selected marker belongs to the agent line alone.
    assert "▸" not in "".join(selected[1:] + unselected)


def test_status_glyphs_are_complete_and_distinct_without_color():
    agent_glyphs = {}
    task_glyphs = {}
    for status in STATUS_GLYPHS:
        agent = render_agent_summary(
            AgentSummary("a", "Agent", status), selected=False
        )
        # No spans: the glyph, not a color, is the whole status signal.
        assert agent.spans == []
        agent_glyphs[status] = agent.plain[2]

        task_line = render_agent_summary(
            AgentSummary(
                "a", "Agent", "idle", (AgentTaskSummary("t", "Task", status),)
            ),
            selected=False,
        ).plain.split("\n")[1]
        task_glyphs[status] = task_line[4]

    assert agent_glyphs == STATUS_GLYPHS
    assert task_glyphs == STATUS_GLYPHS
    assert len(set(STATUS_GLYPHS.values())) == 5


# --- mounted workspace -----------------------------------------------------


async def test_workspace_starts_with_primary_selected_and_only_primary_visible():
    app = WorkspaceApp()
    async with app.run_test() as pilot:
        workspace = workspace_of(app)
        chat = app.query_one("#chat", Transcript)

        assert workspace.primary_agent_id == "primary"
        assert workspace.selected_agent_id == "primary"
        assert workspace.selected_activity is chat
        assert chat.display
        assert sidebar_rows(app) == ["▸ ○ Primary"]

        await workspace.add_agent(AgentSummary("a", "Agent A", "running"))
        await pilot.pause()

        assert workspace.selected_activity is chat
        assert [t.display for t in app.query(Transcript)] == [True, False]
        assert sidebar_rows(app) == ["▸ ○ Primary", "  ● Agent A"]


async def test_tasks_are_not_list_items_or_selection_targets():
    app = WorkspaceApp()
    async with app.run_test() as pilot:
        workspace = workspace_of(app)
        await workspace.add_agent(
            AgentSummary(
                "a",
                "Agent A",
                "running",
                (
                    AgentTaskSummary("t1", "One", "running"),
                    AgentTaskSummary("t2", "Two", "waiting"),
                ),
            )
        )
        await workspace.add_agent(AgentSummary("b", "Agent B", "idle"))
        await pilot.pause()

        agent_list = app.query_one("#agent-list", ListView)
        # Three agents, five rendered rows, three selection targets.
        assert len(agent_list.query(ListItem)) == 3
        assert sum(len(row.split("\n")) for row in sidebar_rows(app)) == 5

        agent_list.focus()
        await pilot.pause()
        await pilot.press("down")
        await pilot.pause()
        assert agent_list.index == 1
        assert agent_list.highlighted_child.agent_id == "a"
        await pilot.press("down")
        await pilot.pause()
        # Down moved past Agent A's two tasks in one step, straight to Agent B.
        assert agent_list.index == 2
        assert agent_list.highlighted_child.agent_id == "b"

        # A click on a rendered task line selects its owning agent: the task is
        # text inside that agent's one row, never a selection target of its own.
        task_row = next(
            row for row in app.query(_AgentListItem) if row.agent_id == "a"
        )
        await pilot.click(task_row.query_one(".agent-list-text", Static), offset=(2, 1))
        await pilot.pause()
        assert workspace.selected_agent_id == "a"
        assert agent_list.index == 1


async def test_highlight_does_not_select_until_enter():
    app = WorkspaceApp()
    async with app.run_test() as pilot:
        workspace = workspace_of(app)
        chat = app.query_one("#chat", Transcript)
        activity = await workspace.add_agent(AgentSummary("a", "Agent A", "running"))
        await pilot.pause()

        app.query_one("#agent-list", ListView).focus()
        await pilot.pause()
        await pilot.press("down")
        await pilot.pause()

        assert workspace.selected_agent_id == "primary"
        assert workspace.selected_activity is chat
        assert chat.display and not activity.display
        assert sidebar_rows(app) == ["▸ ○ Primary", "  ● Agent A"]

        await pilot.press("enter")
        await pilot.pause()

        assert workspace.selected_agent_id == "a"
        assert workspace.selected_activity is activity
        assert activity.display and not chat.display
        assert sidebar_rows(app) == ["  ○ Primary", "▸ ● Agent A"]


async def test_agent_selection_preserves_transcript_identity_content_and_scroll():
    app = WorkspaceApp()
    async with app.run_test(size=(80, 12)) as pilot:
        workspace = workspace_of(app)
        chat = app.query_one("#chat", Transcript)
        chat.load(records(*[f"primary line {i}" for i in range(40)]))
        activity = await workspace.add_agent(
            AgentSummary("a", "Agent A", "running"),
            records(*[f"agent line {i}" for i in range(40)]),
        )
        await pilot.pause()

        chat.scroll_to(y=7, animate=False)
        await pilot.pause()
        primary_scroll = chat.scroll_offset.y
        assert primary_scroll > 0

        workspace.select_agent("a")
        await pilot.pause()
        activity.scroll_to(y=13, animate=False)
        await pilot.pause()
        agent_scroll = activity.scroll_offset.y
        assert agent_scroll > 0 and agent_scroll != primary_scroll

        workspace.select_agent("primary")
        await pilot.pause()
        assert workspace.selected_activity is chat
        assert app.query_one("#chat", Transcript) is chat
        assert [r.content for r in chat.history.records][:1] == ["primary line 0"]
        assert chat.scroll_offset.y == primary_scroll

        workspace.select_agent("a")
        await pilot.pause()
        assert workspace.selected_activity is activity
        assert workspace.activity_for("a") is activity
        assert [r.content for r in activity.history.records][:1] == ["agent line 0"]
        assert activity.scroll_offset.y == agent_scroll


async def test_agent_selection_clears_only_screen_level_text_selection():
    app = WorkspaceApp()
    async with app.run_test(size=(80, 12)) as pilot:
        workspace = workspace_of(app)
        chat = app.query_one("#chat", Transcript)
        chat.load(records("primary content"))
        activity = await workspace.add_agent(AgentSummary("a", "Agent A", "running"))
        await pilot.pause()

        chat_input = app.query_one("#input", _ChatInput)
        chat_input.value = "an unsent draft"
        chat_input.selection = InputSelection(0, 3)
        span = Selection.from_offsets(Offset(0, 0), Offset(5, 0))
        app.screen.selections = {chat.history: span}
        await pilot.pause()

        workspace.select_agent("a")
        await pilot.pause()

        # A hidden transcript must never remain a copy source.
        assert app.screen.selections == {}
        assert app.screen.get_selected_text() is None
        # The composer is untouched: value and widget-owned selection both.
        assert chat_input.value == "an unsent draft"
        assert chat_input.selection == InputSelection(0, 3)
        assert [r.content for r in chat.history.records] == ["primary content"]

        # Reselecting the visible agent is a no-op, so a selection made inside
        # it survives.
        app.screen.selections = {activity.history: span}
        workspace.select_agent("a")
        await pilot.pause()
        assert app.screen.selections == {activity.history: span}


async def test_agent_update_preserves_selection_and_transcript_identity():
    app = WorkspaceApp()
    async with app.run_test() as pilot:
        workspace = workspace_of(app)
        chat = app.query_one("#chat", Transcript)
        activity = await workspace.add_agent(AgentSummary("a", "Agent A", "idle"))
        await pilot.pause()
        workspace.select_agent("a")
        await pilot.pause()

        workspace.update_agent(
            AgentSummary(
                "a",
                "Agent A",
                "running",
                (AgentTaskSummary("t1", "Inspect tests", "running"),),
            )
        )
        workspace.update_agent(AgentSummary("primary", "Primary", "failed"))
        await pilot.pause()

        assert sidebar_rows(app) == [
            "  ! Primary",
            "▸ ● Agent A\n    ● Inspect tests",
        ]
        assert workspace.selected_agent_id == "a"
        assert workspace.selected_activity is activity
        assert workspace.activity_for("primary") is chat
        assert activity.display and not chat.display
        assert workspace.summary_for("primary").status == "failed"


async def test_adding_an_agent_does_not_steal_selection():
    app = WorkspaceApp()
    async with app.run_test() as pilot:
        workspace = workspace_of(app)
        first = await workspace.add_agent(AgentSummary("a", "Agent A", "running"))
        await pilot.pause()
        workspace.select_agent("a")
        await pilot.pause()

        second = await workspace.add_agent(AgentSummary("b", "Agent B", "idle"))
        await pilot.pause()

        assert workspace.selected_agent_id == "a"
        assert workspace.selected_activity is first
        assert first.display and not second.display
        assert sidebar_rows(app) == ["  ○ Primary", "▸ ● Agent A", "  ○ Agent B"]


async def test_invalid_duplicate_and_unknown_agent_data_fails_explicitly():
    app = WorkspaceApp()
    async with app.run_test() as pilot:
        workspace = workspace_of(app)
        chat = app.query_one("#chat", Transcript)
        await workspace.add_agent(AgentSummary("a", "Agent A", "running"))
        await pilot.pause()
        before = sidebar_rows(app)

        with pytest.raises(ValueError, match="non-empty agent id"):
            await workspace.add_agent(AgentSummary("", "Agent B", "idle"))
        with pytest.raises(ValueError, match="non-empty label"):
            await workspace.add_agent(AgentSummary("b", "", "idle"))
        with pytest.raises(ValueError, match="Unknown agent status"):
            await workspace.add_agent(AgentSummary("b", "Agent B", "busy"))
        with pytest.raises(ValueError, match="non-empty task id"):
            await workspace.add_agent(
                AgentSummary("b", "Agent B", "idle", (AgentTaskSummary("", "T", "idle"),))
            )
        with pytest.raises(ValueError, match="non-empty label"):
            await workspace.add_agent(
                AgentSummary("b", "Agent B", "idle", (AgentTaskSummary("t", "", "idle"),))
            )
        with pytest.raises(ValueError, match="duplicate task id"):
            await workspace.add_agent(
                AgentSummary(
                    "b",
                    "Agent B",
                    "idle",
                    (
                        AgentTaskSummary("t", "One", "idle"),
                        AgentTaskSummary("t", "Two", "idle"),
                    ),
                )
            )
        with pytest.raises(ValueError, match="already registered"):
            await workspace.add_agent(AgentSummary("a", "Agent A again", "idle"))
        with pytest.raises(KeyError, match="Unknown agent id"):
            workspace.update_agent(AgentSummary("ghost", "Ghost", "idle"))
        with pytest.raises(KeyError, match="Unknown agent id"):
            workspace.select_agent("ghost")
        with pytest.raises(KeyError, match="Unknown agent id"):
            workspace.summary_for("ghost")
        with pytest.raises(KeyError, match="Unknown agent id"):
            workspace.activity_for("ghost")
        await pilot.pause()

        # Nothing was half-applied by any rejection.
        assert sidebar_rows(app) == before
        assert list(workspace._summaries) == ["primary", "a"]
        assert len(app.query(Transcript)) == 2
        assert workspace.selected_agent_id == "primary"
        assert workspace.selected_activity is chat


async def test_concurrent_registration_of_one_id_registers_exactly_one_agent():
    """Registration spans two awaits, so the id must be claimed before them."""
    app = WorkspaceApp()
    async with app.run_test() as pilot:
        workspace = workspace_of(app)

        results = await asyncio.gather(
            workspace.add_agent(AgentSummary("same", "First", "running")),
            workspace.add_agent(AgentSummary("same", "Second", "idle")),
            return_exceptions=True,
        )
        await pilot.pause()

        granted = [r for r in results if isinstance(r, Transcript)]
        refused = [r for r in results if isinstance(r, ValueError)]
        assert len(granted) == 1
        assert len(refused) == 1
        assert "already being registered" in str(refused[0])

        # The winner's handle is the one the registry hands out, and the loser
        # left neither a transcript nor a sidebar row behind.
        assert workspace.activity_for("same") is granted[0]
        assert list(workspace._summaries) == ["primary", "same"]
        assert len(app.query(Transcript)) == 2
        assert len(app.query(_AgentListItem)) == 2
        assert workspace._pending == set()


async def test_a_cancelled_registration_leaves_no_widget_and_frees_its_id():
    app = WorkspaceApp()
    async with app.run_test() as pilot:
        workspace = workspace_of(app)
        before_streams = len(app.query(Transcript))
        before_rows = len(app.query(_AgentListItem))

        registration = asyncio.ensure_future(
            workspace.add_agent(AgentSummary("ghost", "Ghost", "idle"))
        )
        # Yield once so it reaches its first awaited DOM mutation, then cancel
        # it the way an orchestrator's ``wait_for`` or task cancellation would.
        await asyncio.sleep(0)
        registration.cancel()
        with pytest.raises(asyncio.CancelledError):
            await registration
        await pilot.pause()

        assert list(workspace._summaries) == ["primary"]
        assert len(app.query(Transcript)) == before_streams
        assert len(app.query(_AgentListItem)) == before_rows
        assert workspace._pending == set()

        # A cancelled registration is not a claim: the id retries cleanly.
        retried = await workspace.add_agent(AgentSummary("ghost", "Ghost", "idle"))
        await pilot.pause()
        assert workspace.activity_for("ghost") is retried
        assert len(app.query(Transcript)) == before_streams + 1
        assert len(app.query(_AgentListItem)) == before_rows + 1


async def test_a_label_may_not_forge_a_sidebar_row():
    """A line break in a label would draw a row nobody declared."""
    forged_agent = AgentSummary("x", "Agent A\n  ! Forged agent", "idle")
    forged_task = AgentSummary(
        "y",
        "Agent B",
        "idle",
        (AgentTaskSummary("t", "Task\n  ● Forged", "idle"),),
    )
    trailing_agent = AgentSummary("z", "Agent C\n", "idle")
    carriage_task = AgentSummary(
        "w", "Agent D", "idle", (AgentTaskSummary("t", "Task\r  ● Forged", "idle"),)
    )

    # The constructor is a boundary too, not just add/update.
    with pytest.raises(ValueError, match="must be a single line"):
        AgentWorkspace(forged_agent, Transcript())

    app = WorkspaceApp()
    async with app.run_test() as pilot:
        workspace = workspace_of(app)
        await workspace.add_agent(AgentSummary("a", "Agent A", "idle"))
        await pilot.pause()
        before = sidebar_rows(app)

        for summary in (forged_agent, forged_task, trailing_agent, carriage_task):
            with pytest.raises(ValueError, match="must be a single line"):
                await workspace.add_agent(summary)
        with pytest.raises(ValueError, match="must be a single line"):
            workspace.update_agent(AgentSummary("a", "Agent A\n  ! Forged", "idle"))
        with pytest.raises(ValueError, match="must be a single line"):
            workspace.update_agent(
                AgentSummary(
                    "a",
                    "Agent A",
                    "idle",
                    (AgentTaskSummary("t", "Task\n  ● Forged", "idle"),),
                )
            )
        await pilot.pause()

        # Every row still comes from a summary the caller actually declared.
        assert sidebar_rows(app) == before
        assert list(workspace._summaries) == ["primary", "a"]
        assert workspace._pending == set()


async def test_theme_refresh_reaches_hidden_transcripts(monkeypatch):
    app = WorkspaceApp()
    async with app.run_test() as pilot:
        workspace = workspace_of(app)
        chat = app.query_one("#chat", Transcript)
        hidden_a = await workspace.add_agent(AgentSummary("a", "Agent A", "running"))
        hidden_b = await workspace.add_agent(AgentSummary("b", "Agent B", "idle"))
        await pilot.pause()
        assert not hidden_a.display and not hidden_b.display

        refreshed: list[Transcript] = []
        monkeypatch.setattr(
            Transcript,
            "refresh_theme",
            lambda self: refreshed.append(self),
        )
        workspace.refresh_theme()

        assert refreshed == [chat, hidden_a, hidden_b]
