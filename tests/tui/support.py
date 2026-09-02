"""Helpers shared by more than one module of the TUI test package."""

import asyncio
import json

from textual.widgets import Input, ListView, Static, TextArea
from textual.widgets.input import Selection as InputSelection

from jtech_cli.cmd_tools import CmdPolicy
from jtech_cli.config import Profile, Profiles, ResolvedProfile, Settings
from jtech_cli.server_info import ServerInfo
from jtech_cli.session import Session
from jtech_cli.tui import ChatApp
from jtech_cli.tui_app import PRIMARY_AGENT_ID
from jtech_cli.tui_widgets import (
    AgentSummary,
    AgentWorkspace,
    Transcript,
    _AgentListItem,
)


LOCAL = Profile(name="local", base_url="http://host:9000/v1", model="qwen3")


def local_settings(**kwargs) -> Settings:
    """Settings whose active profile is the default test endpoint."""
    return Settings(profiles=Profiles(items=(LOCAL,), active_name="local"), **kwargs)


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


async def enter_multiline(app, pilot, trigger: str) -> "TextArea":
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


async def send_and_drain(app: ChatApp, pilot, text: str) -> None:
    inp = app.query_one("#input", Input)
    inp.value = text
    await pilot.press("enter")
    for _ in range(10):
        await pilot.pause()


def suggestions_text(app: ChatApp) -> str:
    return str(app.query_one("#suggestions", Static).render())


def suggestions_box(app: ChatApp) -> Static:
    return app.query_one("#suggestions", Static)


async def wait_until(app, pilot, predicate, tries=50, pause=0.1):
    """Poll the app until ``predicate()`` is true (or tries are exhausted)."""
    for _ in range(tries):
        await pilot.pause(pause)
        if predicate():
            return
    raise AssertionError("condition not met in time")


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


def notifications(app: ChatApp) -> list[str]:
    return [notification.message for notification in app._notifications]


async def settle(pilot, times: int = 6) -> None:
    for _ in range(times):
        await pilot.pause()


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


def clear_input_selection(inp: Input) -> None:
    """Drop the select-all an Input takes on focus, leaving a bare cursor."""
    inp.selection = InputSelection.cursor(len(inp.value))


def workspace_of(app: ChatApp) -> AgentWorkspace:
    return app.query_one("#agent-workspace", AgentWorkspace)


def primary_summary(app: ChatApp) -> AgentSummary:
    return workspace_of(app).summary_for(PRIMARY_AGENT_ID)


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
    await wait_until(app, pilot, lambda: app._primary_turn_depth == 0, tries=tries)
    await settle(pilot)
