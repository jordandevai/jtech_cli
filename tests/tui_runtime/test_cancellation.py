"""Stopping a run: closing the response, teardown, and reporting cleanup."""

import asyncio
import logging

import pytest
from textual.widgets.markdown import MarkdownStream

from jtech_cli import tui_runtime
from jtech_cli.session import Session
from jtech_cli.tui_runtime import (
    INTERRUPTED_RESPONSE,
    RENDER_ERROR,
    STOPPED_LABEL,
    STREAM_CANCEL_ERROR,
    RunOutcome,
    StreamCloseAborted,
)
from jtech_cli.tui_widgets import Transcript

from .support import (
    BlockingReplyStream,
    Harness,
    command_call,
    make_runtime,
    model_messages,
    reply_stream_factory,
    scripted_stream,
    wait_for,
)


async def test_a_provider_failure_is_a_typed_outcome_that_releases_the_flags():
    stream, _ = scripted_stream(RuntimeError("boom"))
    async with Harness().run_test() as pilot:
        runtime, _ = make_runtime(pilot.app, stream)
        outcome = await runtime.run()
        assert outcome.status == "failed"
        assert "boom" in outcome.error
        assert runtime.state.generating is False
        assert runtime.state.tool_rounds_active is False
        assert runtime.state.phase == "failed"


async def test_a_stopped_run_closes_provider_and_records_balanced_context():
    """Esc must close the response and leave both sides of the turn recorded.

    Nothing here releases the provider but the runtime's own cancellation, and
    the partial answer deliberately contains a complete-looking protocol block:
    a stopped completion carries no text out, so nothing can parse or run it.
    """
    partial = f"partial\n{command_call('echo forbidden')}"
    stream = BlockingReplyStream(partial)
    session = Session(persist=False)
    session.add("user", "inspect it")
    async with Harness().run_test() as pilot:
        runtime, host = make_runtime(
            pilot.app, reply_stream_factory(stream), session=session
        )
        task = asyncio.create_task(runtime.run())
        await wait_for(pilot, stream.blocked.is_set)

        runtime.request_stop()
        outcome = await asyncio.wait_for(task, 10)

        chat = pilot.app.query_one("#chat", Transcript)
        records = list(chat.history.records)
        tail = list(chat._tail)

    assert stream.cancelled is True  # the parked read was cancelled, not waited out
    assert stream.closed == 1  # and its response was closed on the way out
    assert outcome == RunOutcome("stopped")
    assert runtime.state.generating is False
    assert runtime.state.tool_rounds_active is False
    assert runtime.state.phase == "stopped"

    interrupted = f"{partial}\n\n{INTERRUPTED_RESPONSE}"
    assert session.messages == [
        {"role": "user", "content": "inspect it"},
        {
            "role": "assistant",
            "content": interrupted,
            "_model_role": "assistant",
            "_model_content": INTERRUPTED_RESPONSE,
        },
    ]
    assert model_messages(session) == [
        {"role": "user", "content": "inspect it"},
        {"role": "assistant", "content": INTERRUPTED_RESPONSE},
    ]

    stopped = [record for record in records if record.display_label == STOPPED_LABEL]
    assert [record.content for record in stopped] == [interrupted]
    assert not any("Generation stopped" in record.content for record in records)
    assert tail == []  # nothing left live

    assert host.authorized == []
    assert host.dispatched == []


async def test_stop_during_request_creation_cancels_before_any_iteration():
    """A stop landing while the request is still being opened cancels it.

    The request itself is an await inside the cancellable task, so there is no
    window where a stop can be dropped for arriving too early, and no stream to
    read once it is cancelled.
    """
    stream = BlockingReplyStream()
    creating = asyncio.Event()
    release = asyncio.Event()

    async def factory(profile, temperature, messages):
        creating.set()
        await release.wait()
        return stream

    session = Session(persist=False)
    async with Harness().run_test() as pilot:
        runtime, _ = make_runtime(pilot.app, factory, session=session)
        task = asyncio.create_task(runtime.run())
        await wait_for(pilot, creating.is_set)

        runtime.request_stop()
        release.set()
        outcome = await asyncio.wait_for(task, 10)

    assert stream.iterated is False  # the stream was never read
    assert outcome == RunOutcome("stopped")
    assert runtime.state.generating is False
    assert [m["content"] for m in session.messages] == [INTERRUPTED_RESPONSE]


async def test_reasoning_only_stop_never_records_reasoning():
    """Hidden reasoning is discarded, in both the durable and model-facing forms."""
    stream = BlockingReplyStream(("reasoning", "secret thoughts"))
    session = Session(persist=False)
    async with Harness().run_test() as pilot:
        runtime, _ = make_runtime(
            pilot.app,
            reply_stream_factory(stream),
            session=session,
            reasoning="always",
        )
        task = asyncio.create_task(runtime.run())
        await wait_for(pilot, stream.blocked.is_set)

        runtime.request_stop()
        outcome = await asyncio.wait_for(task, 10)

        records = list(pilot.app.query_one("#chat", Transcript).history.records)

    assert outcome == RunOutcome("stopped")
    assert session.messages == [
        {
            "role": "assistant",
            "content": INTERRUPTED_RESPONSE,
            "_model_role": "assistant",
            "_model_content": INTERRUPTED_RESPONSE,
        }
    ]
    assert model_messages(session) == [
        {"role": "assistant", "content": INTERRUPTED_RESPONSE}
    ]
    assert not any("secret thoughts" in m["content"] for m in session.messages)
    assert not any("secret thoughts" in record.content for record in records)
    assert not any(record.role == "reasoning" for record in records)


class _SlowClosingStream(BlockingReplyStream):
    """A stream whose response takes a while to close, and may refuse to."""

    def __init__(self, first="partial", *, close_delay=0.2, close_error=None):
        super().__init__(first)
        self._close_delay = close_delay
        self._close_error = close_error
        self.close_started = asyncio.Event()

    async def aclose(self):
        self.close_started.set()
        await asyncio.sleep(self._close_delay)
        if self._close_error is not None:
            raise self._close_error
        self.closed += 1


async def test_a_second_stop_cannot_interrupt_the_response_close():
    """Esc pressed again mid-cleanup must not abandon an unclosed response.

    The second stop used to land on whatever was cleaning up after the first,
    cancelling the close and releasing the turn with the connection still open.
    """
    stream = _SlowClosingStream()
    session = Session(persist=False)
    async with Harness().run_test() as pilot:
        runtime, _ = make_runtime(
            pilot.app, reply_stream_factory(stream), session=session
        )
        task = asyncio.create_task(runtime.run())
        await wait_for(pilot, stream.blocked.is_set)

        runtime.request_stop()
        await wait_for(pilot, stream.close_started.is_set)
        assert not task.done()  # still closing

        runtime.request_stop()  # the second Esc, mid-close
        runtime.request_stop()  # and a third, for good measure
        outcome = await asyncio.wait_for(task, 10)

    assert stream.closed == 1  # the close completed rather than being cancelled
    assert outcome == RunOutcome("stopped")
    assert runtime.state.generating is False


async def test_a_close_failure_is_logged_and_reported_not_passed_off_as_clean(
    caplog,
):
    """A response the CLI could not release is never reported as a clean stop."""
    caplog.set_level(logging.ERROR, logger="jtech_cli.tui_runtime")
    stream = _SlowClosingStream(close_delay=0, close_error=RuntimeError("close failed"))
    session = Session(persist=False)
    async with Harness().run_test() as pilot:
        runtime, _ = make_runtime(
            pilot.app, reply_stream_factory(stream), session=session
        )
        task = asyncio.create_task(runtime.run())
        await wait_for(pilot, stream.blocked.is_set)

        runtime.request_stop()
        outcome = await asyncio.wait_for(task, 10)
        records = list(pilot.app.query_one("#chat", Transcript).history.records)

    assert outcome == RunOutcome("stopped")
    logged = [r for r in caplog.records if r.levelno == logging.ERROR and r.exc_info]
    assert logged, caplog.records
    assert "close failed" in str(logged[0].exc_info[1])

    warnings = [r for r in records if STREAM_CANCEL_ERROR in r.content]
    assert len(warnings) == 1
    assert "close failed" in warnings[0].content
    # visible only: the model sees the interruption, not the socket trouble
    assert not any(STREAM_CANCEL_ERROR in m["content"] for m in session.messages)


async def test_teardown_cancellation_survives_a_blocked_close():
    """Cleanup is protected, but the caller's cancellation is not discarded.

    Cancelling the run while the response is closing must still finish the
    close and still cancel the run. Swallowing it returned a completed outcome
    for a turn the app was shutting down.
    """
    stream = _SlowClosingStream(close_delay=0.3)
    session = Session(persist=False)
    async with Harness().run_test() as pilot:
        runtime, _ = make_runtime(
            pilot.app, reply_stream_factory(stream), session=session
        )
        task = asyncio.create_task(runtime.run())
        await wait_for(pilot, stream.blocked.is_set)

        runtime.request_stop()
        await wait_for(pilot, stream.close_started.is_set)

        task.cancel()  # application teardown, landing mid-close
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 10)

    assert stream.closed == 1  # the close still completed
    assert runtime.state.generating is False


async def test_a_cancelled_close_is_a_cleanup_failure_not_a_clean_stop(caplog):
    """A close that never completed says so; it is not silently a success."""
    caplog.set_level(logging.ERROR, logger="jtech_cli.tui_runtime")

    class _SelfCancellingClose(BlockingReplyStream):
        async def aclose(self):
            raise asyncio.CancelledError

    stream = _SelfCancellingClose()
    session = Session(persist=False)
    async with Harness().run_test() as pilot:
        runtime, _ = make_runtime(
            pilot.app, reply_stream_factory(stream), session=session
        )
        task = asyncio.create_task(runtime.run())
        await wait_for(pilot, stream.blocked.is_set)

        runtime.request_stop()
        outcome = await asyncio.wait_for(task, 10)
        records = list(pilot.app.query_one("#chat", Transcript).history.records)

    assert outcome == RunOutcome("stopped")
    logged = [r for r in caplog.records if r.levelno == logging.ERROR and r.exc_info]
    assert logged, caplog.records
    assert isinstance(logged[0].exc_info[1], StreamCloseAborted)
    assert any(STREAM_CANCEL_ERROR in r.content for r in records)


async def test_teardown_close_failure_is_logged_but_not_drawn(caplog):
    """Teardown has no reader left, so its report is the log and nothing else."""
    caplog.set_level(logging.ERROR, logger="jtech_cli.tui_runtime")
    stream = _SlowClosingStream(close_delay=0, close_error=RuntimeError("close failed"))
    session = Session(persist=False)
    async with Harness().run_test() as pilot:
        runtime, _ = make_runtime(
            pilot.app, reply_stream_factory(stream), session=session
        )
        task = asyncio.create_task(runtime.run())
        await wait_for(pilot, stream.blocked.is_set)

        task.cancel()  # teardown, with no stop requested first
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 10)
        records = list(pilot.app.query_one("#chat", Transcript).history.records)

    logged = [r for r in caplog.records if r.levelno == logging.ERROR and r.exc_info]
    assert logged, caplog.records
    assert "close failed" in str(logged[0].exc_info[1])
    assert not any(STREAM_CANCEL_ERROR in r.content for r in records)
    assert not any(STREAM_CANCEL_ERROR in m["content"] for m in session.messages)
    assert runtime.state.generating is False


class _GatedCloseStream(BlockingReplyStream):
    """A stream whose response close finishes only when the test says so.

    The gate is what makes the interleaving deterministic: releasing it and
    cancelling the run in the same synchronous block puts both on the same pass
    of the loop, which is the collision these tests are about.
    """

    def __init__(self, first="partial", *, close_error=None):
        super().__init__(first)
        self._close_error = close_error
        self.close_started = asyncio.Event()
        self.release_close = asyncio.Event()

    async def aclose(self):
        self.close_started.set()
        await self.release_close.wait()
        if self._close_error is not None:
            raise self._close_error
        self.closed += 1


async def test_teardown_cancellation_survives_a_close_that_lands_with_it():
    """A close finishing alongside teardown is not an answer to teardown.

    Provenance used to be inferred from whether the awaited close had
    finished, so a close completing in the same tick as the cancel request
    read as the close's own — and the run reported a completed turn for an app
    that was shutting down.
    """
    stream = _GatedCloseStream()
    session = Session(persist=False)
    async with Harness().run_test() as pilot:
        runtime, _ = make_runtime(
            pilot.app, reply_stream_factory(stream), session=session
        )
        task = asyncio.create_task(runtime.run())
        await wait_for(pilot, stream.blocked.is_set)

        runtime.request_stop()
        await wait_for(pilot, stream.close_started.is_set)

        # Nothing awaits between these two, so the close completes and the run
        # is cancelled on the same pass of the loop.
        stream.release_close.set()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 10)

    assert stream.closed == 1  # the close still ran to completion
    assert runtime.state.generating is False


async def test_a_close_failure_lands_in_the_log_even_when_teardown_lands_on_it(caplog):
    """Cleanup reports its result before handing the cancellation back.

    Restoring the cancellation first skipped the report, and the ``finally``
    then found the release already done and skipped it too — so a response the
    CLI could not close went unmentioned anywhere.
    """
    caplog.set_level(logging.ERROR, logger="jtech_cli.tui_runtime")
    stream = _GatedCloseStream(close_error=RuntimeError("close failed"))
    session = Session(persist=False)
    async with Harness().run_test() as pilot:
        runtime, _ = make_runtime(
            pilot.app, reply_stream_factory(stream), session=session
        )
        task = asyncio.create_task(runtime.run())
        await wait_for(pilot, stream.blocked.is_set)

        runtime.request_stop()
        await wait_for(pilot, stream.close_started.is_set)

        stream.release_close.set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 10)
        records = list(pilot.app.query_one("#chat", Transcript).history.records)

    logged = [r for r in caplog.records if r.levelno == logging.ERROR and r.exc_info]
    assert logged, caplog.records
    assert "close failed" in str(logged[0].exc_info[1])
    # The run is unwinding, so the log is the whole report.
    assert not any(STREAM_CANCEL_ERROR in r.content for r in records)
    assert not any(STREAM_CANCEL_ERROR in m["content"] for m in session.messages)
    assert runtime.state.generating is False


async def test_a_render_failure_reports_its_close_failure_to_the_log_alone(
    caplog, monkeypatch
):
    """A dead turn is reported once, by the failure that killed it.

    Cleanup here is triggered by the renderer, and the turn already ends in a
    visible ``RENDER_ERROR`` bubble saying so. Adding a close warning beside it
    reports the same dead turn twice; the log is where that detail belongs.
    """
    caplog.set_level(logging.ERROR, logger="jtech_cli.tui_runtime")

    async def failing_write(self, markdown_fragment: str) -> None:
        raise RuntimeError("renderer failed")

    monkeypatch.setattr(MarkdownStream, "write", failing_write)
    stream = _SlowClosingStream(close_delay=0, close_error=RuntimeError("close failed"))
    session = Session(persist=False)
    async with Harness().run_test() as pilot:
        runtime, _ = make_runtime(
            pilot.app, reply_stream_factory(stream), session=session
        )
        outcome = await asyncio.wait_for(runtime.run(), 10)
        records = list(pilot.app.query_one("#chat", Transcript).history.records)

    assert outcome.status == "failed"
    assert RENDER_ERROR in outcome.error
    logged = [r for r in caplog.records if r.levelno == logging.ERROR and r.exc_info]
    assert logged, caplog.records
    assert "close failed" in str(logged[0].exc_info[1])
    assert not any(STREAM_CANCEL_ERROR in r.content for r in records)
    assert not any(STREAM_CANCEL_ERROR in m["content"] for m in session.messages)


async def test_stream_events_coalesce_a_burst_behind_one_drain():
    """A burst queued while the consumer was busy comes back as one batch."""
    events = tui_runtime._StreamEvents()
    for item in ["a", ("reasoning", "r"), "b"]:
        events.put(item)

    assert await events.drain() == ["a", ("reasoning", "r"), "b"]
    assert not events.finished

    events.put("c")
    events.close()
    assert await events.drain() == ["c"]
    assert events.finished
