"""Command policy, approval, tool rounds, and nudging."""

import threading

from textual.widgets import Input

from jtech_cli.cmd_tools import CmdPolicy
from jtech_cli.config import load_cmd_policy
from jtech_cli.prompts import NUDGE_PROMPT
from jtech_cli.tui import CommandPrompt
from jtech_cli.tui_runtime import INTERRUPTED_RESPONSE

from .support import (
    BlockingStream,
    FiniteStream,
    bubbles,
    cmd_stream,
    command_call,
    local_settings,
    make_app,
    make_app_with_cmd,
    sync_stream,
    wait_until,
)


async def test_cmd_auto_allowlist_runs_silently(tmp_path, monkeypatch):
    """auto mode: an allowlisted command runs without a prompt; output feeds back."""
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="auto", allow=["echo:*"]))
    calls = {"n": 0}
    requests = []

    def fake(profile, temperature, messages):
        calls["n"] += 1
        requests.append(messages)
        yield command_call("echo hello-out") if calls["n"] == 1 else "done"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: calls["n"] >= 2, tries=50)

        assert not isinstance(app.screen, CommandPrompt)
        assert any("hello-out" in b for b in bubbles(app))
        assert any("done" in b for b in bubbles(app))
        roles = [m["role"] for m in app.session.messages]
        assert "system" in roles
        sys_msgs = [m["content"] for m in app.session.messages if m["role"] == "system"]
        assert any("hello-out" in m and "exit 0" in m for m in sys_msgs)
        assert requests[1][-1] == {
            "role": "user",
            "content": "[JTECH runtime event]\n$ echo hello-out\nexit 0\nhello-out",
        }


async def test_cmd_ask_prompts_then_allow_runs(tmp_path, monkeypatch):
    """ask mode: a non-allowlisted command prompts; 'y' allows and runs it."""
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="ask", allow=[]))
    fake, calls = cmd_stream(command_call("echo prompt-out"), "done")
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: isinstance(app.screen, CommandPrompt), tries=50)
        await pilot.press("y")
        await wait_until(app, pilot, lambda: calls["n"] >= 2, tries=50)

        assert any("prompt-out" in b for b in bubbles(app))
        sys_msgs = [m["content"] for m in app.session.messages if m["role"] == "system"]
        assert any("prompt-out" in m for m in sys_msgs)


async def test_cmd_ask_decline_feeds_back(tmp_path, monkeypatch):
    """ask mode: 'n' declines; the command does not run but the model still reacts."""
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="ask", allow=[]))
    fake, calls = cmd_stream(command_call("echo never-runs"), "done")
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: isinstance(app.screen, CommandPrompt), tries=50)
        await pilot.press("n")
        await wait_until(app, pilot, lambda: calls["n"] >= 2, tries=50)

        # the command text is in the AI's own bubble (the fenced block), so
        # "not run" is proven by the absence of an exit-code result message
        sys_msgs = [m["content"] for m in app.session.messages if m["role"] == "system"]
        assert any("declined by the user" in m for m in sys_msgs)
        assert not any("exit 0" in m for m in sys_msgs)
        assert any("done" in b for b in bubbles(app))


async def test_cmd_blacklist_blocked_even_in_yolo(tmp_path, monkeypatch):
    """The blacklist is absolute: even yolo blocks sudo, and no prompt is shown."""
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="yolo"))
    fake, calls = cmd_stream(command_call("sudo ls"), "done")
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: calls["n"] >= 2, tries=50)

        assert not isinstance(app.screen, CommandPrompt)
        sys_msgs = [m["content"] for m in app.session.messages if m["role"] == "system"]
        assert any("blocked" in m and "sudo" in m for m in sys_msgs)
        assert not any("exit 0" in m for m in sys_msgs)  # never executed


async def test_cmd_off_mode_disables_execution(tmp_path, monkeypatch):
    """off mode: requested commands are not run; a disabled note is fed back."""
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="off"))
    fake, calls = cmd_stream(command_call("echo should-not-run"), "done")
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: calls["n"] >= 2, tries=50)

        sys_msgs = [m["content"] for m in app.session.messages if m["role"] == "system"]
        assert any("disabled" in m for m in sys_msgs)
        assert not any("exit 0" in m for m in sys_msgs)  # never executed


async def test_cmd_always_allow_saves_rule(tmp_path, monkeypatch):
    """'a' in the prompt persists a prefix rule to config and runs the command."""
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="ask", allow=[]))
    fake, calls = cmd_stream(command_call("git status"), "done")
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: isinstance(app.screen, CommandPrompt), tries=50)
        await pilot.press("a")
        await wait_until(app, pilot, lambda: calls["n"] >= 2, tries=50)

        loaded = load_cmd_policy(app.config_path)
        assert "git status:*" in loaded.allow
        assert any("git status:*" in b for b in bubbles(app))


async def test_every_command_in_one_reply_runs_in_source_order(tmp_path, monkeypatch):
    """A reply may carry any number of calls; all of them run, in order.

    With no per-reply cap nothing is dropped, so there is nothing to report as
    dropped: the model gets exactly one result per call it emitted.
    """
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="auto", allow=["echo:*"]))
    total = 7  # more than the retired five-call cap
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            yield "\n".join(command_call(f"echo blk-{i}") for i in range(total))
        else:
            yield "stopped"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: calls["n"] >= 2, tries=150)
        await wait_until(app, pilot, lambda: not app.tool_rounds_active, tries=100)
        for _ in range(10):
            await pilot.pause()

        assert calls["n"] == 2  # every call ran, then exactly one more round
        assert not any("ignored" in b for b in bubbles(app))

    fed = [m["content"] for m in app.session.messages if m["role"] == "system"]
    assert len(fed) == total  # nothing dropped
    for index, result in enumerate(fed):
        assert f"blk-{index}" in result  # and nothing reordered


async def test_different_command_rounds_are_not_limited(tmp_path, monkeypatch):
    """Distinct command results can continue without an arbitrary round cap."""
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="auto", allow=["echo:*"]))
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        calls["n"] += 1
        if calls["n"] <= 6:
            yield command_call(f"echo round-{calls['n']}")
        else:
            yield "finished"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: calls["n"] >= 7, tries=200)
        await wait_until(app, pilot, lambda: not app.tool_rounds_active, tries=100)
        for _ in range(10):
            await pilot.pause()

        assert calls["n"] == 7
        assert any("finished" in b for b in bubbles(app))
        assert not any("round limit" in b.lower() for b in bubbles(app))

    fed = [m for m in app.session.messages if m["role"] == "system"]
    assert len(fed) == 6  # one result for each distinct command round
    assert "round-6" in "\n".join(m["content"] for m in fed)


async def test_repeated_commands_and_results_do_not_stop_the_loop(
    tmp_path, monkeypatch
):
    """The same command with the same result keeps the turn running.

    Repetition is the model's business, not the loop's: only prose without a
    call ends the turn, so four identical rounds all execute and feed back.
    """
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="auto", allow=["echo:*"]))
    repeats = 4
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        calls["n"] += 1
        if calls["n"] <= repeats:
            yield command_call("echo unchanged")
        else:
            yield "done"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: calls["n"] >= repeats + 1, tries=200)
        await wait_until(app, pilot, lambda: not app.tool_rounds_active, tries=100)
        for _ in range(10):
            await pilot.pause()

        assert calls["n"] == repeats + 1
        assert not any("no-progress" in b for b in bubbles(app))
        assert any("done" in b for b in bubbles(app))

    fed = [m["content"] for m in app.session.messages if m["role"] == "system"]
    assert len(fed) == repeats
    assert all("unchanged" in message and "exit 0" in message for message in fed)


async def test_consecutive_empty_replies_are_each_nudged(tmp_path, monkeypatch):
    """Every empty reply earns another nudge; there is no recovery budget.

    An empty reply is not an answer, so the turn ends only once the model
    produces prose — here after three consecutive empty streams.
    """
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="auto", allow=["echo:*"]))
    calls = {"n": 0}
    requests = []

    def fake(profile, temperature, messages):
        calls["n"] += 1
        requests.append(messages)
        if calls["n"] == 1:
            yield command_call("echo tool-out")
        elif calls["n"] <= 4:
            yield ""
        else:
            yield "recovered"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: calls["n"] >= 5, tries=200)
        await wait_until(app, pilot, lambda: not app.tool_rounds_active, tries=100)
        for _ in range(10):
            await pilot.pause()

        assert calls["n"] == 5
        assert any("recovered" in b for b in bubbles(app))

    # request 0 is the user turn and 1 follows the tool result; 2-4 are nudges
    nudged = [
        index
        for index, messages in enumerate(requests)
        if messages[-1] == {"role": "system", "content": NUDGE_PROMPT}
    ]
    assert nudged == [2, 3, 4]
    # empty replies are never stored, and the nudge stays out of the history
    assert [m["role"] for m in app.session.messages] == [
        "user",
        "assistant",
        "system",
        "assistant",
    ]


async def test_nudge_is_shown_in_system_debug_mode(tmp_path, monkeypatch):
    """Debug system mode exposes the ephemeral nudge in the live chat."""
    settings = local_settings(debug_level="system")
    app = make_app_with_cmd(
        tmp_path, CmdPolicy(mode="auto", allow=["echo:*"]), settings=settings
    )
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            yield command_call("echo debug-nudge")
        elif calls["n"] == 2:
            yield ""
        else:
            yield "done"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: calls["n"] >= 3, tries=100)
        await wait_until(app, pilot, lambda: not app.tool_rounds_active, tries=100)

        assert any("Continue your task" in bubble for bubble in bubbles(app))
        nudges = [
            message
            for message in app.session.messages
            if "Continue your task" in message["content"]
        ]
        assert len(nudges) == 1
        assert nudges[0]["_include_in_context"] is False


async def test_nudge_can_continue_with_an_explicit_command(tmp_path, monkeypatch):
    """A nudge may recover a command-only stop, but prose still ends the turn."""
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="auto", allow=["echo:*"]))
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        calls["n"] += 1
        replies = {
            1: command_call("echo first"),
            2: "",
            3: command_call("echo second"),
            4: "finished",
        }
        yield replies[calls["n"]]

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: calls["n"] >= 4, tries=100)
        await wait_until(app, pilot, lambda: not app.tool_rounds_active, tries=100)

        assert calls["n"] == 4
        system_messages = [
            m["content"] for m in app.session.messages if m["role"] == "system"
        ]
        assert any("first" in message for message in system_messages)
        assert any("second" in message for message in system_messages)


async def test_final_answer_after_tool_ends_turn_without_repeat(tmp_path, monkeypatch):
    """A final answer after ``pwd`` ends the turn without rerunning the command."""
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="auto", allow=["pwd:*"]))
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            yield command_call("pwd")
        else:
            yield "The cwd is /the/project.\n\n```cmd\npwd\n```"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "whats the cwd?"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: calls["n"] >= 2, tries=100)
        await wait_until(app, pilot, lambda: not app.tool_rounds_active, tries=100)
        for _ in range(10):
            await pilot.pause()

        assert calls["n"] == 2
        assert any("The cwd is /the/project." in b for b in bubbles(app))

    sys_msgs = [m["content"] for m in app.session.messages if m["role"] == "system"]
    assert len([m for m in sys_msgs if "pwd" in m and "exit 0" in m]) == 1


async def test_command_prefix_commentary_is_preserved_and_tool_round_continues(
    tmp_path, monkeypatch
):
    """Prefix commentary is visible, while the command still starts a tool round."""
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="auto", allow=["pwd:*"]))
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            yield command_call("pwd") + "\n\nLet me inspect the project structure next."
        else:
            yield "The cwd is /the/project."

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "audit this project"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: calls["n"] >= 2, tries=100)
        await wait_until(app, pilot, lambda: not app.tool_rounds_active, tries=100)

        assert calls["n"] == 2
        assert any("Let me inspect the project structure next." in b for b in bubbles(app))
        assert any("The cwd is /the/project." in b for b in bubbles(app))

    sys_msgs = [m["content"] for m in app.session.messages if m["role"] == "system"]
    assert len([m for m in sys_msgs if "pwd" in m and "exit 0" in m]) == 1


async def test_interleaved_commentary_commands_start_one_tool_round(
    tmp_path, monkeypatch
):
    """Commentary between standalone calls does not suppress later commands."""
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="auto", allow=["echo:*"]))
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            yield (
                "I'll audit the project.\n\n"
                f'{command_call("echo first")}\n'
                "Let me inspect the first result.\n"
                f'{command_call("echo second")}\n'
                "Let me finish the audit."
            )
        else:
            yield "The audit is complete."

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "audit this project"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: calls["n"] >= 2, tries=100)
        await wait_until(app, pilot, lambda: not app.tool_rounds_active, tries=100)

        assert calls["n"] == 2
        assert any("I'll audit the project." in b for b in bubbles(app))
        assert any("Let me inspect the first result." in b for b in bubbles(app))
        assert any("The audit is complete." in b for b in bubbles(app))

    sys_msgs = [m["content"] for m in app.session.messages if m["role"] == "system"]
    assert sum("exit 0" in message for message in sys_msgs) == 2


async def test_html_wrapped_command_executes_once(tmp_path, monkeypatch):
    """A whole-response HTML wrapper does not disable the command protocol."""
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="auto", allow=["pwd:*"]))
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            yield '<code>\njtech_cmd("pwd")\n</code>'
        else:
            yield "done"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "whats the cwd?"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: calls["n"] >= 2, tries=100)
        await wait_until(app, pilot, lambda: not app.tool_rounds_active, tries=100)

        assert calls["n"] == 2

    sys_msgs = [m["content"] for m in app.session.messages if m["role"] == "system"]
    assert len([m for m in sys_msgs if "pwd" in m and "exit 0" in m]) == 1


async def test_clear_during_tool_followup_does_not_crash(tmp_path, monkeypatch):
    """/clear can empty history while the post-command reply is in flight."""
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="auto", allow=["echo:*"]))
    gate = threading.Event()
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            yield command_call("echo clear-out")
        elif calls["n"] == 2:
            gate.wait(5)  # hold the post-command stream open across the /clear
            yield "after clear"
        else:
            yield "unexpected extra request"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: calls["n"] >= 2, tries=100)

        # /clear is dispatched straight from the input, bypassing the
        # tool-rounds queue guard, so it lands while the follow-up is in flight.
        inp.value = "/clear"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: app.session.messages == [], tries=50)
        gate.set()

        await wait_until(app, pilot, lambda: not app.tool_rounds_active, tries=100)
        await pilot.pause()

        assert app._exception is None
        assert calls["n"] == 2


async def test_plain_final_answer_ends_turn(tmp_path, monkeypatch):
    """A plain final answer (no tool rounds yet) ends the turn."""
    app = make_app(tmp_path)
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        calls["n"] += 1
        yield "all done"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "hello"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: calls["n"] >= 1)
        # give a would-be extra stream time to fire
        for _ in range(10):
            await pilot.pause()

    assert calls["n"] == 1
    assert app.session.messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "all done"},
    ]


async def test_declined_command_ends_tool_turn(tmp_path, monkeypatch):
    """A decline is user input: the next model reply ends the tool turn."""
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="ask", allow=[]))
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            yield command_call("echo declined-out")
        else:
            yield "stopped"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: isinstance(app.screen, CommandPrompt), tries=50)
        await pilot.press("n")
        await wait_until(app, pilot, lambda: calls["n"] >= 2)
        # give a would-be extra stream time to fire
        for _ in range(10):
            await pilot.pause()

    assert calls["n"] == 2
    assert [m["role"] for m in app.session.messages] == [
        "user", "assistant", "system", "system", "assistant",
    ]
    # The model is told not to retry and how to end if it cannot adapt. The
    # guidance is role-neutral: a subagent runs this same prompt and cannot
    # talk to the user at all, so it must never be told to ask them.
    contents = [m["content"] for m in app.session.messages]
    assert any("Do not retry it" in content for content in contents)
    assert any(
        "adapt and continue with a permitted approach" in content
        for content in contents
    )
    assert any('jtech_result("failed"' in content for content in contents)
    assert not any("ask the user" in content for content in contents)


async def test_blocked_command_ends_tool_turn(tmp_path, monkeypatch):
    """A blocked command is guardrail input; the next reply ends the turn."""
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="yolo"))
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            yield command_call("sudo ls")
        else:
            yield "stopped"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: calls["n"] >= 2)
        # give a would-be extra stream time to fire
        for _ in range(10):
            await pilot.pause()
        assert not isinstance(app.screen, CommandPrompt)

    assert calls["n"] == 2


async def test_failed_command_result_continues_the_loop(tmp_path, monkeypatch):
    """A non-zero exit is a result, not a stop: the model gets it and continues."""
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="yolo"))
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            yield command_call("echo failing-out; exit 3")
        elif calls["n"] == 2:
            yield command_call("echo recovered-out")
        else:
            yield "handled"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: calls["n"] >= 3, tries=150)
        await wait_until(app, pilot, lambda: not app.tool_rounds_active, tries=100)
        for _ in range(10):
            await pilot.pause()

        assert calls["n"] == 3
        assert any("handled" in b for b in bubbles(app))

    sys_msgs = [m["content"] for m in app.session.messages if m["role"] == "system"]
    assert any("exit 3" in m and "failing-out" in m for m in sys_msgs)
    assert any("recovered-out" in m for m in sys_msgs)


async def test_a_running_command_is_shown_then_replaced_by_its_result(tmp_path, monkeypatch):
    """The reported defect, end to end: an executing command looked inert.

    Nothing was drawn until the process exited, so a `jtech_cmd(...)` the app
    had already parsed and started read as a call that never fired.
    """
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="yolo"))
    command = "sleep 60"
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            yield command_call(command)
        else:
            yield "final"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await wait_until(
            app, pilot, lambda: app.primary_runtime.state.running_proc is not None
        )

        # Visible while the process is still alive, not after it exits.
        running = [b for b in bubbles(app) if b.startswith(f"$ {command}")]
        assert len(running) == 1, running
        assert "running…" in running[0]

        # The existing stop path, exactly as a user reaches it.
        await pilot.press("escape")
        await wait_until(app, pilot, lambda: calls["n"] >= 2, tries=100)
        await wait_until(app, pilot, lambda: not app.tool_rounds_active, tries=100)
        await pilot.pause()

        # One presentation for the whole lifecycle: the running entry became the
        # result rather than being joined by a second bubble for the same run.
        shown = [b for b in bubbles(app) if b.startswith(f"$ {command}")]
        assert len(shown) == 1, shown
        assert "running…" not in shown[0]
        assert "interrupted" in shown[0]

        sys_msgs = [m["content"] for m in app.session.messages if m["role"] == "system"]
        assert any("interrupted by user" in m for m in sys_msgs)
        assert not any("running…" in m for m in sys_msgs)
        assert any("final" in b for b in bubbles(app))


async def test_queue_drains_after_esc_stop(tmp_path, monkeypatch):
    """The queued turn starts after the stopped one closed, and sees the marker.

    Two user turns in a row is what degenerated the next completion. The stop
    now leaves a balanced assistant turn between them, carrying the marker
    rather than the partial answer.
    """
    app = make_app(tmp_path)
    first = BlockingStream()
    second = FiniteStream("r2")
    entered: list[str] = []
    first_closed_at_second_entry: list[bool] = []
    sent: list[list[dict]] = []

    async def provider(profile, temperature, messages):
        sent.append(messages)
        if not entered:
            entered.append("one")
            return first
        entered.append("two")
        # Read here, in the second request itself: the first response must
        # already be closed, not merely asked to stop.
        first_closed_at_second_entry.append(first.closed == 1)
        return second

    monkeypatch.setattr("jtech_cli.tui.stream_reply", provider)
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "one"
        await pilot.press("enter")
        await wait_until(app, pilot, first.blocked.is_set)
        system_prompt = app._primary_system_prompt()

        inp.value = "two"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: bool(app.composer.queue), tries=10, pause=0.05)
        assert entered == ["one"]  # the queued turn has not started

        await pilot.press("escape")
        await wait_until(app, pilot, lambda: any("r2" in b for b in bubbles(app)))

        interrupted = f"partial \n\n{INTERRUPTED_RESPONSE}"
        assert entered == ["one", "two"]  # the requests never overlapped
        assert first_closed_at_second_entry == [True]
        assert first.cancelled is True
        assert not any("Queued" in b for b in bubbles(app))
        assert not any("Generation stopped" in b for b in bubbles(app))
        # the stopped partial is still on screen, above the queued turn's answer
        shown = bubbles(app)
        assert shown.index(interrupted) < shown.index("r2")

    assert sent[1] == [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": INTERRUPTED_RESPONSE},
        {"role": "user", "content": "two"},
    ]
    assert app.session.messages == [
        {"role": "user", "content": "one"},
        {
            "role": "assistant",
            "content": interrupted,
            "_model_role": "assistant",
            "_model_content": INTERRUPTED_RESPONSE,
        },
        {"role": "user", "content": "two"},
        {"role": "assistant", "content": "r2"},
    ]
