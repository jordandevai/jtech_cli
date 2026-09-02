"""Agent dispatch: creation, continuation, and result routing."""

import threading

from textual.widgets import Input, Static

from jtech_cli.cmd_tools import CmdPolicy
from jtech_cli.config import ResolvedProfile
from jtech_cli.session import Session
from jtech_cli.tui import CommandPrompt
from jtech_cli.tui_widgets import _AgentListItem

from .support import (
    Conversation,
    agent_activity,
    agent_results,
    agent_summary,
    bubbles,
    chat_of,
    cmd_stream,
    command_call,
    dispatch_call,
    make_app,
    make_app_with_cmd,
    primary_summary,
    run_primary,
    select_agent,
    settle,
    sync_stream,
    two_profile_settings,
    visible_activity,
    wait_until,
    workspace_of,
)


async def test_a_first_dispatch_creates_one_agent_view_session_and_task(
    tmp_path, monkeypatch
):
    stream = Conversation([dispatch_call(), "all done"], ["worker answer"])
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(stream))
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await run_primary(app, pilot)

        summary = agent_summary(app, "coder")
        assert summary.label == "Coder"
        assert summary.status == "completed"
        assert [(t.label, t.status) for t in summary.tasks] == [("Task", "completed")]
        # The task is the agent's first message, recorded once and shown once.
        managed = app._agents["coder"]
        assert [m["content"] for m in managed.session.messages] == [
            "do the work",
            "worker answer",
        ]
        assert agent_activity(app, "coder")[0] == "do the work"
        worker_prompts = stream.sent_to("worker")
        assert len(worker_prompts) == 1
        assert "You are a subagent" in worker_prompts[0][0]["content"]
        assert "### Available profiles" not in worker_prompts[0][0]["content"]


async def test_the_coordinator_prompt_reaches_the_real_primary_request(
    tmp_path, monkeypatch
):
    stream = Conversation(["done"])
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(stream))
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test() as pilot:
        await run_primary(app, pilot)
    system = stream.sent_to("primary")[0][0]["content"]
    assert "jtech_agent(" in system
    assert "`local` — the profile this conversation runs on" in system
    assert "`cloud`" in system


async def test_a_repeated_key_continues_one_conversation(tmp_path, monkeypatch):
    stream = Conversation(
        [
            dispatch_call(task_label="First", task="first task"),
            dispatch_call(task_label="Second", task="second task"),
            "all done",
        ],
        ["first answer", "second answer"],
    )
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(stream))
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await run_primary(app, pilot)

        # One agent, two tasks, one transcript, one session.
        assert list(app._agents) == ["coder"]
        summary = agent_summary(app, "coder")
        assert [(t.label, t.status) for t in summary.tasks] == [
            ("First", "completed"),
            ("Second", "completed"),
        ]
        assert [m["content"] for m in app._agents["coder"].session.messages] == [
            "first task",
            "first answer",
            "second task",
            "second answer",
        ]
        # The second request carried the first exchange, so context survived.
        second_request = stream.sent_to("worker")[1]
        assert [m["content"] for m in second_request[1:]] == [
            "first task",
            "first answer",
            "second task",
        ]
        assert agent_activity(app, "coder").count("second task") == 1
        # The seeded first task and the appended second one share one policy.
        tasks = [
            record
            for record in app._agents["coder"].transcript.history.records
            if record.role == "user"
        ]
        assert [record.content for record in tasks] == ["first task", "second task"]
        assert [record.format for record in tasks] == ["plain", "plain"]


async def test_a_label_or_profile_change_for_one_key_fails_without_mutating(
    tmp_path, monkeypatch
):
    stream = Conversation(
        [
            dispatch_call(task_label="First"),
            dispatch_call(label="Renamed", task_label="Second"),
            dispatch_call(profile="cloud", task_label="Third"),
            "all done",
        ],
        ["worker answer"],
    )
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(stream))
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test() as pilot:
        await run_primary(app, pilot)

        summary = agent_summary(app, "coder")
        assert summary.label == "Coder"
        assert [t.label for t in summary.tasks] == ["First"]
        # One real run, two refusals — no second worker request was made.
        assert len(stream.sent_to("worker")) == 1
        results = agent_results(app)
        assert [r["status"] for r in results] == ["completed", "failed", "failed"]
        assert "keeps its label" in results[1]["content"]
        assert "keeps its profile" in results[2]["content"]
        # Each result names the call it answers, not the agent that already
        # exists: the rejected call asked for "Renamed", so answering it as
        # "Coder" would hide which call failed.
        assert [r["agent_label"] for r in results] == ["Coder", "Renamed", "Coder"]
        assert [r["task_label"] for r in results] == ["First", "Second", "Third"]
        assert [r["agent_key"] for r in results] == ["coder", "coder", "coder"]


async def test_two_agents_never_share_a_session_or_a_transcript(
    tmp_path, monkeypatch
):
    stream = Conversation(
        [f'{dispatch_call(key="a", label="A")}\n{dispatch_call(key="b", label="B")}',
         "all done"],
        ["answer one", "answer two"],
    )
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(stream))
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await run_primary(app, pilot)

        first = app._agents["a"]
        second = app._agents["b"]
        assert first.session is not second.session
        assert first.transcript is not second.transcript
        one = " ".join(m["content"] for m in first.session.messages)
        two = " ".join(m["content"] for m in second.session.messages)
        assert ("answer one" in one) != ("answer one" in two)
        assert set(agent_activity(app, "a")).isdisjoint(
            set(agent_activity(app, "b")) - {"do the work"}
        )


async def test_subagent_sessions_never_touch_the_filesystem(tmp_path, monkeypatch):
    stream = Conversation([dispatch_call(), "all done"], ["worker answer"])
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(stream))
    worker_history = tmp_path / "never" / "session.jsonl"
    monkeypatch.setattr(
        "jtech_cli.session.default_history_path", lambda: worker_history
    )
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await run_primary(app, pilot)
        managed = app._agents["coder"]
        assert managed.session.persist is False
        assert len(managed.session.messages) == 2
        assert not worker_history.exists()
        assert not worker_history.parent.exists()


async def test_a_failed_agent_stays_selectable_and_can_take_another_task(
    tmp_path, monkeypatch
):
    calls = {"n": 0, "worker": 0}

    def fake(profile, temperature, messages):
        if "You are a subagent" in messages[0]["content"]:
            calls["worker"] += 1
            if calls["worker"] == 1:
                raise RuntimeError("provider exploded")
            yield "recovered"
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
        await run_primary(app, pilot)

        summary = agent_summary(app, "coder")
        assert [(t.label, t.status) for t in summary.tasks] == [
            ("First", "failed"),
            ("Second", "completed"),
        ]
        assert summary.status == "completed"
        results = agent_results(app)
        assert [r["status"] for r in results] == ["failed", "completed"]
        assert "provider exploded" in results[0]["content"]
        assert any(
            "provider exploded" in line for line in agent_activity(app, "coder")
        )
        await select_agent(app, pilot, "coder")
        assert visible_activity(app) is workspace_of(app).activity_for("coder")


async def test_a_relaunch_restores_primary_history_only(tmp_path, monkeypatch):
    stream = Conversation([dispatch_call(), "all done"], ["worker answer"])
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(stream))
    session = Session(tmp_path / "s.jsonl")
    app = make_app(tmp_path, session=session)
    async with app.run_test() as pilot:
        await run_primary(app, pilot)
    assert agent_results(app)[0]["content"] == "worker answer"

    restored = Session(tmp_path / "s.jsonl")
    restored.load()
    relaunched = make_app(tmp_path, session=restored)
    async with relaunched.run_test() as pilot:
        await settle(pilot)
        # The result survives in Primary's context; the worker does not come back.
        assert agent_results(relaunched)[0]["content"] == "worker answer"
        assert relaunched._agents == {}
        assert len(workspace_of(relaunched).query(_AgentListItem)) == 1


async def test_dispatch_never_disturbs_the_composer_selection_or_queue(
    tmp_path, monkeypatch
):
    gate = threading.Event()
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        system = messages[0]["content"]
        if "You are a subagent" in system:
            gate.wait(5)
            yield "worker answer"
            return
        calls["n"] += 1
        yield dispatch_call() if calls["n"] == 1 else "all done"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: "coder" in app._agents)

        assert app.viewing_primary
        assert visible_activity(app) is chat_of(app)
        inp.value = "a draft"
        inp.value = "queued while busy"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: app._queue == ["queued while busy"])
        assert primary_summary(app).status == "waiting"

        gate.set()
        await wait_until(app, pilot, lambda: app._primary_turn_depth == 0)
        assert app._queue == []
        assert primary_summary(app).status == "idle"


async def test_a_result_is_recorded_once_with_its_exact_identity(
    tmp_path, monkeypatch
):
    stream = Conversation([dispatch_call(), "all done"], ["the worker answer"])
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(stream))
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await run_primary(app, pilot)

        results = agent_results(app)
        assert len(results) == 1
        result = results[0]
        assert result["agent_key"] == "coder"
        assert result["agent_label"] == "Coder"
        assert result["task_label"] == "Task"
        assert result["status"] == "completed"
        assert result["content"] == "the worker answer"
        assert result["task_id"] == agent_summary(app, "coder").tasks[0].task_id
        record = next(
            m for m in app.session.messages
            if m.get("_model_content", "").startswith("[JTECH agent result]")
        )
        assert record["role"] == "system"
        assert record["_model_role"] == "user"
        assert record["content"] == "Coder completed: Task"


async def test_a_primary_history_failure_is_visible_but_keeps_the_result(
    tmp_path, monkeypatch
):
    stream = Conversation([dispatch_call(), "all done"], ["worker answer"])
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(stream))
    app = make_app(tmp_path)
    original = app.session.add
    failures = {"n": 0}

    def failing(role, content, **kwargs):
        # The message still joins the in-memory conversation; only the append
        # to disk fails, exactly as a real OSError leaves it.
        original(role, content, **kwargs)
        if (kwargs.get("model_content") or "").startswith("[JTECH agent result]"):
            failures["n"] += 1
            raise OSError("disk full")

    monkeypatch.setattr(app.session, "add", failing)
    async with app.run_test() as pilot:
        await run_primary(app, pilot)

        assert failures["n"] == 1
        assert any("Could not save history: disk full" in b for b in bubbles(app))
        # The in-memory result still reached the next request.
        last_request = stream.sent_to("primary")[-1]
        assert any(
            m["content"].startswith("[JTECH agent result]") for m in last_request
        )


async def test_primary_reports_waiting_while_its_own_command_awaits_approval(
    tmp_path, monkeypatch
):
    """The requester is Primary here, so the same waiting rule applies to it."""
    fake, _ = cmd_stream(command_call("echo needs-approval"), "done")
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    app = make_app_with_cmd(tmp_path, CmdPolicy(mode="ask", allow=[]))
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await wait_until(
            app, pilot, lambda: isinstance(app.screen, CommandPrompt), tries=100
        )
        assert primary_summary(app).status == "waiting"
        title = str(app.screen.query_one(".dialog-title", Static).render())
        assert title == "Run command for Primary?"

        await pilot.press("y")
        await wait_until(app, pilot, lambda: app._primary_turn_depth == 0, tries=100)
        assert primary_summary(app).status == "idle"


async def test_a_late_runtime_notification_after_exit_is_not_an_error(tmp_path):
    """Every runtime is still unwinding its own `finally` while the app tears
    down, so those notifications arrive after the widgets are gone."""
    app = make_app(tmp_path)
    async with app.run_test():
        run = app._primary_run_state(
            ResolvedProfile(
                name="local",
                base_url="http://host:9000/v1",
                model="qwen3",
                api_key="none",
            )
        )
    assert not app.is_running
    run.generating = False
    run.running_proc = None
    app.runtime_changed(run)
