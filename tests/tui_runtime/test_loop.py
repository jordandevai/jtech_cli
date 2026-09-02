"""The run loop: its outcome type, what ends a turn, and what it keeps per run."""

import asyncio

import pytest

from jtech_cli.prompts import NUDGE_PROMPT
from jtech_cli.session import Session
from jtech_cli.tui_runtime import AgentOutcome, RunOutcome, _CompletionOutcome
from jtech_cli.tui_widgets import Transcript

from .support import (
    BlockingReplyStream,
    FakeHost,
    Harness,
    ScriptedReplyStream,
    command_call,
    dispatch_call,
    make_runtime,
    reply_stream_factory,
    scripted_stream,
    transcript_text,
    wait_for,
)


def test_run_outcome_rejects_an_empty_completion():
    with pytest.raises(ValueError, match="must carry its final response"):
        RunOutcome("completed")


def test_run_outcome_rejects_a_reasonless_failure():
    with pytest.raises(ValueError, match="must carry its failure description"):
        RunOutcome("failed")


def test_run_outcome_rejects_a_stop_carrying_content():
    with pytest.raises(ValueError, match="carries no response and no error"):
        RunOutcome("stopped", final_text="x")


def test_completion_outcome_allows_an_empty_reply():
    """An empty response is the nudge trigger, so the loop — not the type —
    decides what it means."""
    assert _CompletionOutcome("reply").text == ""


def test_completion_outcome_rejects_a_reasonless_failure():
    with pytest.raises(ValueError, match="must carry its failure description"):
        _CompletionOutcome("failed")


async def test_final_prose_is_the_only_normal_exit():
    stream, calls = scripted_stream("all done")
    async with Harness().run_test() as pilot:
        runtime, host = make_runtime(pilot.app, stream)
        outcome = await runtime.run()
    assert outcome == RunOutcome("completed", final_text="all done")
    assert calls["n"] == 1
    assert host.phases[-1] == "completed"


async def test_an_empty_reply_is_nudged_not_completed():
    stream, calls = scripted_stream("", "   ", "finally")
    async with Harness().run_test() as pilot:
        runtime, _ = make_runtime(pilot.app, stream)
        outcome = await runtime.run()
    assert outcome.final_text == "finally"
    assert calls["n"] == 3


async def test_the_nudge_never_joins_the_stored_conversation():
    stream, _ = scripted_stream("", "done")
    session = Session(persist=False)
    async with Harness().run_test() as pilot:
        runtime, _ = make_runtime(pilot.app, stream, session=session)
        await runtime.run()
    assert all(NUDGE_PROMPT not in m["content"] for m in session.messages)


async def test_the_same_command_repeats_without_any_round_budget():
    """Fifty identical command rounds: no counter, cap, or repetition guard
    exists to end them, only the model's own final prose."""
    replies = [command_call("echo n")] * 50 + ["done"]
    stream, calls = scripted_stream(*replies)
    async with Harness().run_test() as pilot:
        runtime, host = make_runtime(pilot.app, stream)
        outcome = await runtime.run()
    assert outcome.status == "completed"
    assert calls["n"] == 51
    assert len(host.authorized) == 50


async def test_repeated_dispatch_rounds_have_no_budget_either():
    replies = [dispatch_call()] * 20 + ["done"]
    stream, _ = scripted_stream(*replies)
    outcome_stub = AgentOutcome("coder", "Coder", "task-1", "t", "completed", "ok")
    host = FakeHost(outcomes=(outcome_stub,))
    async with Harness().run_test() as pilot:
        runtime, _ = make_runtime(pilot.app, stream, host=host)
        result = await runtime.run()
    assert result.status == "completed"
    assert len(host.dispatched) == 20


async def test_a_history_write_failure_is_visible_and_does_not_end_the_run():
    class _Failing(Session):
        def add(self, *args, **kwargs):
            raise OSError("disk full")

    stream, _ = scripted_stream("done")
    async with Harness().run_test() as pilot:
        runtime, _ = make_runtime(pilot.app, stream, session=_Failing(persist=False))
        outcome = await runtime.run()
        assert outcome.final_text == "done"
        assert "Could not save history: disk full" in transcript_text(pilot.app)


async def test_each_run_records_only_into_its_own_session():
    first = Session(persist=False)
    second = Session(persist=False)
    stream_a, _ = scripted_stream("answer A")
    stream_b, _ = scripted_stream("answer B")
    async with Harness().run_test() as pilot:
        runtime_a, _ = make_runtime(pilot.app, stream_a, session=first)
        runtime_b, _ = make_runtime(pilot.app, stream_b, session=second)
        await runtime_a.run()
        await runtime_b.run()
    assert [m["content"] for m in first.messages] == ["answer A"]
    assert [m["content"] for m in second.messages] == ["answer B"]


async def test_two_runs_keep_separate_stop_signals_and_processes():
    blocking = BlockingReplyStream()
    idle, _ = scripted_stream("x")
    async with Harness().run_test() as pilot:
        one, _ = make_runtime(pilot.app, reply_stream_factory(blocking))
        two, _ = make_runtime(pilot.app, idle)
        task = asyncio.create_task(one.run())
        await wait_for(pilot, blocking.blocked.is_set)

        one.request_stop()
        assert one.state.stop_event.is_set()
        assert not two.state.stop_event.is_set()
        await asyncio.wait_for(task, 10)


async def test_the_system_prompt_is_evaluated_per_completion():
    seen: list[str] = []
    prompts = iter(["FIRST", "SECOND"])
    calls = {"n": 0}

    async def fake(profile, temperature, messages):
        calls["n"] += 1
        seen.append(messages[0]["content"])
        return ScriptedReplyStream(
            command_call("echo x") if calls["n"] == 1 else "done"
        )

    async with Harness().run_test() as pilot:
        runtime, _ = make_runtime(pilot.app, fake)
        runtime.state.system_prompt = lambda: next(prompts)
        await runtime.run()
    assert seen == ["FIRST", "SECOND"]


async def test_the_provider_task_never_sees_a_live_message_list():
    """The snapshot is taken before the task starts, so a mutation afterwards
    cannot reach the request already in flight."""
    session = Session(persist=False)
    session.add("user", "original")
    captured: list[list[dict]] = []

    async def fake(profile, temperature, messages):
        captured.append(messages)
        session.messages.append({"role": "user", "content": "injected"})
        return ScriptedReplyStream("done")

    async with Harness().run_test() as pilot:
        runtime, _ = make_runtime(pilot.app, fake, session=session)
        await runtime.run()
    assert [m["content"] for m in captured[0]] == ["SYSTEM", "original"]


async def test_repeated_tool_rounds_do_not_accumulate_live_widgets():
    replies = [command_call("echo n")] * 6 + ["done"]
    stream, _ = scripted_stream(*replies)
    async with Harness().run_test() as pilot:
        runtime, _ = make_runtime(pilot.app, stream)
        await runtime.run()
        await pilot.pause()
        assert pilot.app.query_one("#chat", Transcript)._tail == []


async def test_the_phase_sequence_of_a_command_round_is_observable():
    stream, _ = scripted_stream(command_call("echo n"), "done")
    async with Harness().run_test() as pilot:
        runtime, host = make_runtime(pilot.app, stream)
        await runtime.run()
    assert "command" in host.phases
    assert host.phases[-1] == "completed"
