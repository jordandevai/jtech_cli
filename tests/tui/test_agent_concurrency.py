"""Agent parallelism, approval locking, shutdown, and setup failures."""

import asyncio
import subprocess
import threading
from pathlib import Path

from textual.widgets import Input, Static

from jtech_cli.cmd_tools import CmdPolicy
from jtech_cli.tui import CommandPrompt

from .support import (
    agent_activity,
    agent_results,
    agent_summary,
    command_call,
    dispatch_call,
    make_app,
    make_app_with_cmd,
    primary_summary,
    result_call,
    run_primary,
    select_agent,
    settle,
    sync_stream,
    wait_until,
)


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
            yield result_call("completed", "worker answer")
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
        await wait_until(app, pilot, lambda: app._primary_turn_depth == 0)
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
            yield result_call("completed", f"answer for {task}")
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
        await wait_until(app, pilot, lambda: finished == ["fast task"])
        slow.set()
        await wait_until(app, pilot, lambda: app._primary_turn_depth == 0)

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
            yield result_call("completed", "sibling answer")
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
            yield result_call("completed", "worker answer")
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
        await wait_until(app, pilot, lambda: inside.is_set())
        # Reach into the live batch: a second task for the busy key is refused.
        managed = app.agents["coder"]
        assert managed.runtime is not None
        release.set()
        await wait_until(app, pilot, lambda: app._primary_turn_depth == 0)
        assert [t.label for t in agent_summary(app, "coder").tasks] == [
            "First",
            "Second",
        ]


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
                yield result_call("completed", "worker done")
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
            await wait_until(
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

        await wait_until(app, pilot, lambda: app._primary_turn_depth == 0, tries=100)

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
                yield result_call("completed", "worker done")
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
        await wait_until(app, pilot, lambda: True, tries=1)
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")

        await wait_until(
            app, pilot, lambda: isinstance(app.screen, CommandPrompt), tries=100
        )
        prompts["n"] += 1
        await pilot.press("a")  # always allow
        await wait_until(app, pilot, lambda: app._primary_turn_depth == 0, tries=100)
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
            yield result_call("completed", "worker answer")
            return
        calls["n"] += 1
        yield dispatch_call() if calls["n"] == 1 else "all done"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: "coder" in app.agents)
        worker = app.agents["coder"].runtime
        assert worker is not None

        await pilot.press("escape")
        await select_agent(app, pilot, "coder")
        await pilot.press("escape")
        await settle(pilot)
        assert not worker.state.stop_event.is_set()

        release.set()
        await wait_until(app, pilot, lambda: app._primary_turn_depth == 0)
        assert agent_results(app)[0]["status"] == "completed"


async def test_exiting_signals_every_live_runtime(tmp_path, monkeypatch):
    release = threading.Event()
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        if "You are a subagent" in messages[0]["content"]:
            release.wait(5)
            yield result_call("completed", "worker answer")
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
            await wait_until(
                app,
                pilot,
                lambda: app.agents.get("coder") is not None
                and app.agents["coder"].runtime is not None
                and app.agents["coder"].runtime.state.generating,
            )
            worker = app.agents["coder"].runtime
            primary = app.primary_runtime
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
            runtime = (app.agents.get("coder") or _ManagedAgentStub).runtime
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

        await wait_until(app, pilot, lambda: len(descendants()) >= 2)
        proc = app.agents["coder"].runtime.state.running_proc
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
            yield result_call("completed", "sibling answer")
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
    original = app.agents._begin_agent_task

    async def flaky(call, task_id):
        if call.agent_key == "a":
            raise RuntimeError("setup boom")
        return await original(call, task_id)

    monkeypatch.setattr(app.agents, "_begin_agent_task", flaky)
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
        assert list(app.agents) == ["b"]
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
            yield result_call("completed", "worker answer")
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
        await wait_until(app, pilot, lambda: inside.is_set())
        managed = app.agents["coder"]

        def exploding_append(record):
            raise RuntimeError("transcript boom")

        monkeypatch.setattr(managed.transcript, "append", exploding_append)
        release.set()
        await wait_until(app, pilot, lambda: app._primary_turn_depth == 0, tries=100)

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
        assert [m["content"] for m in app.agents["coder"].session.messages] == [
            "do the work",
            result_call("completed", "worker answer"),
        ]
