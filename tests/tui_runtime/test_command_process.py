"""Process ownership: a command's group, its kills, and its launch window."""

import asyncio
import os
import pathlib
import subprocess
import threading
import time
from unittest import mock

import pytest

from jtech_cli import tui_runtime

from .support import (
    Harness,
    command_call,
    make_runtime,
    scripted_stream,
    wait_for_shell,
)


def descendants(pid: int) -> list[int]:
    """Every direct child pid of ``pid`` still visible to the OS."""
    found = subprocess.run(
        ["pgrep", "-P", str(pid)], capture_output=True, text=True, check=False
    )
    return [int(line) for line in found.stdout.split()]


def alive(pids: list[int]) -> list[int]:
    return [pid for pid in pids if pathlib.Path(f"/proc/{pid}").exists()]


async def wait_for_command(runtime, pilot, tries=100):
    """Wait until the run owns a live shell with at least one descendant."""
    for _ in range(tries):
        await pilot.pause(0.05)
        proc = runtime.state.running_proc
        if proc is not None and descendants(proc.pid):
            return proc
    raise AssertionError("the command never started a descendant")


async def test_stopping_a_command_kills_its_whole_pipeline(tmp_path):
    """`proc.kill()` alone reaps the shell and orphans the pipeline: the
    grandchildren keep running and keep the stdout pipe open."""
    stream, _ = scripted_stream(command_call("sleep 30 | cat"), "done")
    async with Harness().run_test() as pilot:
        runtime, _ = make_runtime(pilot.app, stream, root=tmp_path)
        task = asyncio.create_task(runtime.run())
        proc = await wait_for_command(runtime, pilot)
        children = descendants(proc.pid)
        assert len(children) >= 2, children

        runtime.request_stop()
        for _ in range(100):
            await pilot.pause(0.05)
            if not alive(children):
                break
        assert alive(children) == []
        # And the run continues rather than hanging on a pipe nobody closed.
        outcome = await asyncio.wait_for(task, timeout=20)
    assert outcome.final_text == "done"


async def test_a_stopped_command_cannot_leave_a_background_job_writing(tmp_path):
    """The failure that matters: a detached job still editing the project
    after the user stopped the command."""
    marker = tmp_path / "leaked.txt"
    command = f"( sleep 3; echo LEAKED > {marker} ) & sleep 30"
    stream, _ = scripted_stream(command_call(command), "done")
    async with Harness().run_test() as pilot:
        runtime, _ = make_runtime(pilot.app, stream, root=tmp_path)
        task = asyncio.create_task(runtime.run())
        await wait_for_command(runtime, pilot)
        runtime.request_stop()
        await asyncio.wait_for(task, timeout=20)
        # Past the background job's own delay, so the file would exist by now.
        await asyncio.sleep(4)
    assert not marker.exists()


async def test_a_command_owns_its_own_process_group(tmp_path):
    """The group this runtime signals contains the command and nothing else.

    ``bash -c "sleep 30"`` execs the single command, so the shell pid *is* the
    command: it must still lead its own group, separate from the app's.
    """
    stream, _ = scripted_stream(command_call("sleep 30"), "done")
    async with Harness().run_test() as pilot:
        runtime, _ = make_runtime(pilot.app, stream, root=tmp_path)
        task = asyncio.create_task(runtime.run())
        proc = await wait_for_shell(runtime, pilot)
        assert os.getpgid(proc.pid) == proc.pid
        assert os.getpgid(proc.pid) != os.getpgid(0)
        runtime.request_stop()
        await asyncio.wait_for(task, timeout=20)


async def test_cancelling_during_process_creation_still_kills_the_command(tmp_path):
    """Cancelling `asyncio.to_thread` abandons the await, not the thread: the
    Popen still completes, so an exit landing inside the launch window would
    otherwise leave a command with nobody responsible for it."""
    marker = tmp_path / "orphan.txt"
    entered = threading.Event()
    launched: dict[str, subprocess.Popen] = {}
    real_popen = subprocess.Popen

    def slow_popen(*args, **kwargs):
        entered.set()
        # Wide enough for a cancellation to land mid-launch, deterministically.
        time.sleep(1.0)
        proc = real_popen(*args, **kwargs)
        launched["proc"] = proc
        return proc

    command = f"( sleep 2; echo LEAKED > {marker} ) & sleep 60"
    stream, _ = scripted_stream(command_call(command), "done")
    async with Harness().run_test() as pilot:
        runtime, _ = make_runtime(pilot.app, stream, root=tmp_path)
        with mock.patch.object(tui_runtime.subprocess, "Popen", slow_popen):
            task = asyncio.create_task(runtime.run())
            for _ in range(200):
                await pilot.pause(0.02)
                if entered.is_set():
                    break
            assert entered.is_set(), "the launch never started"
            assert runtime.state.running_proc is None, "the window closed too early"

            runtime.request_stop()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            for _ in range(200):
                await pilot.pause(0.02)
                if "proc" in launched and launched["proc"].poll() is not None:
                    break
        proc = launched.get("proc")
        assert proc is not None, "the launch thread never produced a process"
        assert proc.poll() is not None, "the command outlived its cancelled owner"
        # Past the background job's own delay: it must never have run.
        await asyncio.sleep(3)
    assert not marker.exists()


async def test_cancelling_a_running_command_kills_it_without_a_prior_stop(tmp_path):
    """Plain task cancellation, as Textual delivers it to a worker.

    `_exec_command()`'s own `finally` releases ownership of the process, so by
    the time `run()` gets to call `request_stop()` there is nothing left for it
    to find. Stopping the command cannot be delegated to whoever cancelled the
    task: the command phase has to own it.
    """
    marker = tmp_path / "orphan.txt"
    command = f"( sleep 3; echo LEAKED > {marker} ) & sleep 60 | cat"
    stream, _ = scripted_stream(command_call(command), "done")
    async with Harness().run_test() as pilot:
        runtime, _ = make_runtime(pilot.app, stream, root=tmp_path)
        task = asyncio.create_task(runtime.run())
        proc = await wait_for_command(runtime, pilot)
        children = descendants(proc.pid)
        assert len(children) >= 2, children

        # No request_stop() and no app exit: cancellation alone.
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert runtime.state.running_proc is None
        assert proc.poll() is not None, "the command outlived its cancelled run"
        for _ in range(100):
            await pilot.pause(0.05)
            if not alive(children):
                break
        assert alive(children) == []
        # Past the background job's own delay: it must never have run.
        await asyncio.sleep(4)
    assert not marker.exists()
