"""The provider transport boundary, with no test double in the path."""

import asyncio
import socket
import threading

from jtech_cli import llm_client
from jtech_cli.config import ResolvedProfile
from jtech_cli.session import Session
from jtech_cli.tui_runtime import INTERRUPTED_RESPONSE, RunOutcome

from .support import (
    Harness,
    live_entries,
    make_runtime,
    wait_for,
)


class _OpenSSEServer:
    """A real HTTP endpoint that streams one token and then never finishes.

    The point of this fixture is what it refuses to do: after the first chunk
    it holds the response open and sends nothing further, so the only thing
    that can end a read of it is the client disconnecting. A server that
    politely closed would let a broken cancellation look like a working one.
    """

    def __init__(self) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(("127.0.0.1", 0))
        self._socket.listen(1)
        self.port = self._socket.getsockname()[1]
        self.disconnected = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._socket.close()
        self._thread.join(5)

    @property
    def profile(self) -> ResolvedProfile:
        return ResolvedProfile(
            name="open",
            base_url=f"http://127.0.0.1:{self.port}/v1",
            model="m",
            api_key="none",
        )

    def _serve(self) -> None:
        try:
            conn, _ = self._socket.accept()
        except OSError:
            return
        try:
            conn.recv(65536)
            conn.sendall(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/event-stream\r\n"
                b"Transfer-Encoding: chunked\r\n\r\n"
            )
            body = (
                b'data: {"id":"1","object":"chat.completion.chunk","created":1,'
                b'"model":"m","choices":[{"index":0,"delta":{"content":"partial"},'
                b'"finish_reason":null}]}\n\n'
            )
            conn.sendall(b"%x\r\n" % len(body) + body + b"\r\n")
            conn.settimeout(30)
            while True:  # only a client disconnect ends this
                if not conn.recv(1):
                    self.disconnected.set()
                    return
        except OSError:
            self.disconnected.set()
        finally:
            conn.close()


async def test_stop_disconnects_a_provider_response_that_never_finishes():
    """The transport boundary, with no test double anywhere in the path.

    A real SSE response emits one token and then stays open forever. Stopping
    must disconnect it and finish the run, which is only possible if the
    cancellation reaches the read itself rather than a flag the reader checks
    between items. Closing the response from another thread does not do this;
    that is what this test exists to keep proving.
    """
    llm_client._client_cache.clear()
    session = Session(persist=False)
    with _OpenSSEServer() as server:
        async with Harness().run_test() as pilot:
            runtime, _ = make_runtime(
                pilot.app,
                llm_client.stream_reply,
                session=session,
                profile=server.profile,
            )
            task = asyncio.create_task(runtime.run())
            await wait_for(
                pilot,
                lambda: any("partial" in body for _, body in live_entries(pilot.app)),
                tries=200,
            )
            # The token has arrived and been rendered, so the reader is now
            # parked in a response that will never produce another one.
            await pilot.pause(0.2)
            assert not task.done()

            runtime.request_stop()
            outcome = await asyncio.wait_for(task, 5)

    assert outcome == RunOutcome("stopped")
    assert server.disconnected.wait(5), "the provider response was never closed"
    assert runtime.state.generating is False
    assert [m["content"] for m in session.messages] == [
        f"partial\n\n{INTERRUPTED_RESPONSE}"
    ]
