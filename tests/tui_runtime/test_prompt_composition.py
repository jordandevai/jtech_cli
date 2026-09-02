"""System-prompt failures, which land before any live state exists."""

import pytest

from jtech_cli.prompts import PromptResourceError, PromptSourceError
from jtech_cli.tui_runtime import PROMPT_ERROR
from jtech_cli.tui_widgets import Transcript

from .support import (
    Harness,
    make_runtime,
    scripted_stream,
)


@pytest.mark.parametrize(
    "error",
    [
        PromptSourceError("The selected prompt file has not been loaded"),
        PromptResourceError("Prompt resource 'coordinator.md' could not be loaded"),
    ],
)
async def test_a_prompt_failure_never_latches_the_run(error):
    """Composing the prompt happens before any live state exists, so a failure
    cannot strand a "live" bubble or a generating flag nothing will release."""

    def boom() -> str:
        raise error

    async with Harness().run_test() as pilot:
        runtime, _ = make_runtime(pilot.app, scripted_stream("never")[0])
        runtime.state.system_prompt = boom
        outcome = await runtime.run()

        assert outcome.status == "failed"
        assert PROMPT_ERROR in outcome.error
        assert str(error) in outcome.error
        assert runtime.state.generating is False
        assert runtime.state.phase == "failed"
        chat = pilot.app.query_one("#chat", Transcript)
        assert chat._tail == []
        assert any(PROMPT_ERROR in r.content for r in chat.history.records)


async def test_a_prompt_failure_makes_no_provider_request():
    requests = []

    def fake(profile, temperature, messages):
        requests.append(messages)
        yield "never"

    async with Harness().run_test() as pilot:
        runtime, _ = make_runtime(pilot.app, fake)
        runtime.state.system_prompt = lambda: (_ for _ in ()).throw(
            PromptSourceError("no prompt")
        )
        await runtime.run()
    assert requests == []
