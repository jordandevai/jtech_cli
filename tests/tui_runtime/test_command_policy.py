"""Command authorization: every host verdict, as the model receives it."""

from jtech_cli.session import Session
from jtech_cli.tui_runtime import CommandAuthorization

from .support import (
    FakeHost,
    Harness,
    command_call,
    make_runtime,
    model_messages,
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
