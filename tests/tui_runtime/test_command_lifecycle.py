"""One command's transcript entry, from `running…` to its result."""

import asyncio
import os
import threading
from unittest import mock

import pytest

from jtech_cli import tui_runtime
from jtech_cli.cmd_tools import CmdPolicy, truncate_output
from jtech_cli.session import Session
from jtech_cli.tui_widgets import Transcript

from .support import (
    Harness,
    command_call,
    live_entries,
    make_runtime,
    model_messages,
    scripted_stream,
    transcript_text,
    wait_for_shell,
)


class _GatedPipe:
    """A command's stdout that stays open until the test releases it.

    ``read`` accepts a size and nothing else, so a production deadline passed
    down to the pipe would raise ``TypeError`` here rather than turning this
    into a slow test that must outlive a timer to prove one is gone.
    """

    def __init__(self, released: threading.Event, output: str) -> None:
        self._released = released
        self._output = output
        self._spent = False

    def read(self, size, **kwargs):
        assert not kwargs, f"the drain passed {sorted(kwargs)} to the pipe"
        assert size > 0
        if self._spent:
            return ""
        assert self._released.wait(20), "the test never released the process"
        self._spent = True
        return self._output

    def close(self):
        self._spent = True

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


class _GatedProc:
    """A process whose exit the test releases, exposing only what the drain uses."""

    def __init__(self, released: threading.Event, output: str) -> None:
        self.pid = os.getpid()
        self.returncode = 0
        self.stdout = _GatedPipe(released, output)

    def wait(self):
        return self.returncode

    def poll(self):
        return self.returncode


async def test_an_authorized_command_is_visible_while_it_runs(tmp_path):
    """The reported defect: nothing in the stream acknowledged the command
    until the process exited, so an executed call looked like it never fired."""
    gate = tmp_path / "gate"
    command = f"until [ -e {gate} ]; do sleep 0.05; done; echo released"
    stream, _ = scripted_stream(command_call(command), "done")
    async with Harness().run_test() as pilot:
        runtime, _ = make_runtime(pilot.app, stream, root=tmp_path)
        task = asyncio.create_task(runtime.run())
        await wait_for_shell(runtime, pilot)

        running = [
            (label, body)
            for label, body in live_entries(pilot.app)
            if command in body
        ]
        assert len(running) == 1, running
        assert running[0][0] == "SYSTEM"
        assert "running…" in running[0][1]

        gate.write_text("go")
        outcome = await asyncio.wait_for(task, timeout=20)
        await pilot.pause()

        assert outcome.final_text == "done"
        # One lifecycle, one entry: the running placeholder became the result
        # rather than being joined by a second presentation of the same run.
        shown = [
            record.content
            for record in pilot.app.query_one("#chat", Transcript).history.records
            if record.content.startswith(f"$ {command}")
        ]
        assert len(shown) == 1, shown
        assert "running…" not in shown[0]
        assert "exit 0" in shown[0]
        assert "released" in shown[0]


async def test_a_command_has_no_elapsed_time_deadline(tmp_path):
    """No duration turns a running command into a result: it waits for the exit."""
    released = threading.Event()
    proc = _GatedProc(released, "late output")
    session = Session(persist=False)
    stream, _ = scripted_stream(command_call("build"), "done")
    async with Harness().run_test() as pilot:
        runtime, _ = make_runtime(pilot.app, stream, session=session, root=tmp_path)
        with mock.patch.object(tui_runtime.subprocess, "Popen", lambda *a, **k: proc):
            task = asyncio.create_task(runtime.run())
            await wait_for_shell(runtime, pilot)
            for _ in range(5):
                await pilot.pause(0.05)
            assert not task.done(), "the run stopped waiting for a live command"
            assert "running…" in "".join(
                body for _, body in live_entries(pilot.app)
            )

            released.set()
            outcome = await asyncio.wait_for(task, timeout=20)
        await pilot.pause()

    assert outcome.final_text == "done"
    assert any("late output" in message["content"] for message in model_messages(session))


async def test_a_cancelled_run_leaves_no_running_placeholder(tmp_path):
    """A placeholder is presentation state; it leaves with the run that opened
    it rather than being finalized into a result the command never reported."""
    stream, _ = scripted_stream(command_call("sleep 60"), "done")
    async with Harness().run_test() as pilot:
        runtime, _ = make_runtime(pilot.app, stream, root=tmp_path)
        task = asyncio.create_task(runtime.run())
        await wait_for_shell(runtime, pilot)
        assert any("running…" in body for _, body in live_entries(pilot.app))

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await pilot.pause()

        assert live_entries(pilot.app) == []
        assert "running…" not in transcript_text(pilot.app)


async def test_a_finished_command_redraws_even_while_compaction_waits(tmp_path):
    """Finalizing records the value; only the body update redraws it.

    A still-removable notice ahead of the command holds the whole finalized
    prefix in the tail, so an implementation that skipped the update would go
    on showing ``running…`` after the process had already exited.
    """
    stream, _ = scripted_stream(command_call("echo done-out"), "done")
    async with Harness().run_test() as pilot:
        chat = pilot.app.query_one("#chat", Transcript)
        notice = chat.begin_plain("system", "Queued", label="Queued")

        runtime, _ = make_runtime(pilot.app, stream, root=tmp_path)
        await runtime.run()
        await pilot.pause()

        drawn = [body for _, body in live_entries(pilot.app)]
        assert "running…" not in "\n".join(drawn)
        assert any("done-out" in body and "exit 0" in body for body in drawn)

        chat.remove(notice)
        await pilot.pause()
        assert "done-out" in transcript_text(pilot.app)


async def test_a_subagent_command_entry_stays_in_its_own_transcript(tmp_path):
    """The entry is written to this run's injected transcript and nowhere else."""
    stream, _ = scripted_stream(command_call("echo agent-out"), "done")
    async with Harness().run_test() as pilot:
        agent_chat = Transcript(id="agent")
        await pilot.app.mount(agent_chat)

        runtime, _ = make_runtime(pilot.app, stream, kind="subagent", root=tmp_path)
        runtime.state.transcript = agent_chat
        await runtime.run()
        await pilot.pause()

        agent_text = "\n".join(r.content for r in agent_chat.history.records)
        assert "$ echo agent-out" in agent_text
        assert "agent-out" in agent_text
        assert "agent-out" not in transcript_text(pilot.app)
        assert live_entries(pilot.app) == []


async def test_a_verbose_command_is_captured_within_the_cap(tmp_path):
    """The deadline was the only thing bounding capture; removing it unbounded it.

    ``communicate()`` returns the whole stream and the cap was applied to the
    result, so a command that printed for long enough could take the CLI down
    with it. Retention is asserted on the collector, not on process RSS, which
    no test can pin down.
    """
    limit = 100
    total = 2_000_000
    peak = {"retained": 0, "seen": 0}

    class _Watched(tui_runtime.BoundedOutput):
        def add(self, chunk):
            super().add(chunk)
            peak["seen"] += len(chunk)
            peak["retained"] = max(peak["retained"], self._head_len + len(self._tail))

    command = f"yes {'x' * 63} | head -c {total}"
    session = Session(persist=False)
    stream, _ = scripted_stream(command_call(command), "done")
    async with Harness().run_test() as pilot:
        runtime, _ = make_runtime(
            pilot.app,
            stream,
            cmd=CmdPolicy(mode="yolo", max_output=limit),
            session=session,
            root=tmp_path,
        )
        with mock.patch.object(tui_runtime, "BoundedOutput", _Watched):
            outcome = await runtime.run()
        await pilot.pause()

        assert outcome.final_text == "done"
        # Every character passed through the collector, and almost none of it
        # was kept. Both halves matter: a runtime that never drains at all would
        # leave `retained` at zero and pass a one-sided assertion.
        assert peak["seen"] == total, peak["seen"]
        assert 0 < peak["retained"] <= 2 * limit, peak["retained"]

        # Bounded capture, identical result: the model sees what full buffering
        # would have given it.
        expected, _ = truncate_output("x" * 63 + "\n" + ("x" * 63 + "\n") * 31248 + "x" * 63, limit)
        shown = [
            record.content
            for record in pilot.app.query_one("#chat", Transcript).history.records
            if record.content.startswith(f"$ {command}")
        ]
        assert len(shown) == 1, shown
        assert "chars truncated" in shown[0]
        assert expected in shown[0]


async def test_clear_during_execution_never_redraws_the_command(tmp_path):
    """`/clear` can land while the process is alive, not just during a stream.

    The entry's body is unmounted by then, so the completion must not repaint
    it — while the result still has to reach the model, which never saw the
    clear.
    """
    gate = tmp_path / "gate"
    command = f"until [ -e {gate} ]; do sleep 0.05; done; echo cleared-out"
    session = Session(persist=False)
    stream, _ = scripted_stream(command_call(command), "done")
    async with Harness().run_test() as pilot:
        chat = pilot.app.query_one("#chat", Transcript)
        runtime, _ = make_runtime(pilot.app, stream, session=session, root=tmp_path)
        task = asyncio.create_task(runtime.run())
        await wait_for_shell(runtime, pilot)
        assert any("running…" in body for _, body in live_entries(pilot.app))

        chat.clear()
        await pilot.pause()

        gate.write_text("go")
        outcome = await asyncio.wait_for(task, timeout=20)
        await pilot.pause()

        assert outcome.final_text == "done"
        # Presentation the user removed stays removed, live or finished.
        assert "cleared-out" not in transcript_text(pilot.app)
        assert "running…" not in transcript_text(pilot.app)
        assert live_entries(pilot.app) == []
        # The autonomous loop is unaffected: the model still got the result.
        assert any("cleared-out" in m["content"] for m in model_messages(session))


async def test_a_launch_failure_becomes_that_entrys_result(tmp_path):
    """`OSError` from `Popen` lands after the running entry has opened.

    It is a command result, not a runtime crash, so the same entry has to carry
    it rather than being stranded on ``running…``.
    """
    session = Session(persist=False)
    stream, _ = scripted_stream(command_call("nope"), "done")

    def boom(*args, **kwargs):
        raise OSError("cannot allocate memory")

    async with Harness().run_test() as pilot:
        runtime, host = make_runtime(pilot.app, stream, session=session, root=tmp_path)
        with mock.patch.object(tui_runtime.subprocess, "Popen", boom):
            outcome = await runtime.run()
        await pilot.pause()

        assert outcome.final_text == "done"
        assert host.phases[-1] == "completed"
        shown = [
            record.content
            for record in pilot.app.query_one("#chat", Transcript).history.records
            if record.content.startswith("$ nope")
        ]
        assert len(shown) == 1, shown
        assert "running…" not in shown[0]
        assert "exit 127" in shown[0]
        assert "cannot allocate memory" in shown[0]
        assert live_entries(pilot.app) == []
        assert any(
            "cannot allocate memory" in m["content"] for m in model_messages(session)
        )
