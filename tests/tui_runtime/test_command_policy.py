"""Command authorization: every host verdict, as the model receives it."""

from jtech_cli.session import Session
from jtech_cli.tui_runtime import CommandAuthorization, RunOutcome

from .support import (
    FakeHost,
    Harness,
    command_call,
    make_runtime,
    model_messages,
    result_call,
    scripted_stream,
)


async def test_a_blocked_command_reaches_the_model_and_the_run_continues():
    """Policy is the host's decision; the runtime only reports and continues."""
    stream, _ = scripted_stream(command_call("echo x"), "adapted")
    host = FakeHost(
        authorization=CommandAuthorization("blocked", "on the absolute blacklist")
    )
    session = Session(persist=False)
    async with Harness().run_test() as pilot:
        runtime, _ = make_runtime(pilot.app, stream, host=host, session=session)
        outcome = await runtime.run()
    assert outcome.final_text == "adapted"
    observation = model_messages(session)[-2]["content"]
    assert "blocked — on the absolute blacklist" in observation


async def test_a_declined_command_adds_the_decline_prompt():
    stream, _ = scripted_stream(command_call("echo hi"), "adapted")
    host = FakeHost(
        authorization=CommandAuthorization("declined", "declined by the user")
    )
    session = Session(persist=False)
    async with Harness().run_test() as pilot:
        runtime, _ = make_runtime(pilot.app, stream, host=host, session=session)
        await runtime.run()
    contents = [m["content"] for m in model_messages(session)]
    assert any("declined by the user" in c for c in contents)
    assert any("Do not retry it" in c for c in contents)


async def test_a_real_command_result_reaches_the_run_session(tmp_path):
    (tmp_path / "marker.txt").write_text("hello-from-disk")
    stream, _ = scripted_stream(command_call("cat marker.txt"), "read it")
    session = Session(persist=False)
    async with Harness().run_test() as pilot:
        runtime, _ = make_runtime(pilot.app, stream, session=session, root=tmp_path)
        outcome = await runtime.run()
    assert outcome.final_text == "read it"
    assert any(
        "hello-from-disk" in m["content"] for m in model_messages(session)
    )


async def test_an_empty_command_is_reported_and_never_run():
    stream, _ = scripted_stream(command_call("   "), "moved on")
    host = FakeHost()
    session = Session(persist=False)
    async with Harness().run_test() as pilot:
        runtime, _ = make_runtime(pilot.app, stream, host=host, session=session)
        await runtime.run()
    assert host.authorized == []
    assert "empty command" in model_messages(session)[-2]["content"]


class _ScriptedVerdicts(FakeHost):
    """A host answering each command with the next verdict in a list.

    ``FakeHost`` holds one verdict for the whole run, which cannot express the
    case that matters here: a subagent meeting two different refusals before it
    decides the assignment is lost.
    """

    def __init__(self, *verdicts: CommandAuthorization) -> None:
        super().__init__()
        self._verdicts = verdicts

    async def authorize_command(self, run, command):
        self.authorized.append(command)
        return self._verdicts[len(self.authorized) - 1]


async def test_a_blocked_command_and_an_explicit_failure_fail_the_run():
    """Blocking a command does not end a subagent's turn; declaring failure does."""
    report = "The only command I had was blocked, so nothing was changed."
    stream, _ = scripted_stream(
        command_call("echo x"), result_call("failed", report)
    )
    host = FakeHost(
        authorization=CommandAuthorization("blocked", "on the absolute blacklist")
    )
    async with Harness().run_test() as pilot:
        runtime, _ = make_runtime(pilot.app, stream, host=host, kind="subagent")
        outcome = await runtime.run()
    assert outcome == RunOutcome("failed", error=report)


async def test_a_declined_command_and_an_explicit_failure_fail_the_run():
    report = "The user declined the one command that would have finished this."
    stream, _ = scripted_stream(
        command_call("echo x"), result_call("failed", report)
    )
    host = FakeHost(
        authorization=CommandAuthorization("declined", "declined by the user")
    )
    async with Harness().run_test() as pilot:
        runtime, _ = make_runtime(pilot.app, stream, host=host, kind="subagent")
        outcome = await runtime.run()
    assert outcome == RunOutcome("failed", error=report)


async def test_a_failing_command_the_subagent_adapts_around_still_completes(tmp_path):
    """An intermediate failure is information, not a verdict: only the terminal
    result decides, so a subagent that recovers still reports completion."""
    (tmp_path / "marker.txt").write_text("hello-from-disk")
    report = "The first path did not exist; read marker.txt instead."
    stream, _ = scripted_stream(
        command_call("cat missing.txt"),
        command_call("cat marker.txt"),
        result_call("completed", report),
    )
    session = Session(persist=False)
    async with Harness().run_test() as pilot:
        runtime, _ = make_runtime(
            pilot.app, stream, kind="subagent", session=session, root=tmp_path
        )
        outcome = await runtime.run()
    assert outcome == RunOutcome("completed", final_text=report)
    contents = [m["content"] for m in model_messages(session)]
    assert any("missing.txt" in content for content in contents)
    assert any("hello-from-disk" in content for content in contents)


async def test_every_refusal_reaches_the_model_before_the_result_is_declared():
    """The subagent's own context must show why it gave up, in the order it
    happened, or its report is the only surviving account of the run."""
    report = "Neither command was permitted."
    stream, _ = scripted_stream(
        command_call("echo one"),
        command_call("echo two"),
        result_call("failed", report),
    )
    host = _ScriptedVerdicts(
        CommandAuthorization("blocked", "on the absolute blacklist"),
        CommandAuthorization("declined", "declined by the user"),
    )
    session = Session(persist=False)
    async with Harness().run_test() as pilot:
        runtime, _ = make_runtime(
            pilot.app, stream, host=host, kind="subagent", session=session
        )
        outcome = await runtime.run()
    assert outcome == RunOutcome("failed", error=report)
    assert host.authorized == ["echo one", "echo two"]
    contents = [m["content"] for m in model_messages(session)]
    blocked = next(i for i, c in enumerate(contents) if "on the absolute blacklist" in c)
    declined = next(i for i, c in enumerate(contents) if "declined by the user" in c)
    declared = next(i for i, c in enumerate(contents) if report in c)
    assert blocked < declined < declared
