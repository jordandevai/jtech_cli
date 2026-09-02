"""Per-agent profile resolution and model discovery."""

import json
import threading

import pytest
from textual.widgets import Input, Static

from jtech_cli.config import Profile, Profiles, Settings
from jtech_cli.server_info import ServerInfo

from .support import (
    Conversation,
    LOCAL,
    agent_activity,
    agent_results,
    agent_summary,
    bubbles,
    dispatch_call,
    local_settings,
    make_app,
    run_primary,
    sync_stream,
    two_profile_settings,
    wait_until,
)


async def test_each_agent_uses_its_own_resolved_profile(tmp_path, monkeypatch):
    stream = Conversation(
        [
            (
                f'{dispatch_call(key="a", label="A", profile="local")}\n'
                f'{dispatch_call(key="b", label="B", profile="cloud")}'
            ),
            "all done",
        ],
        ["one", "two"],
    )
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(stream))
    monkeypatch.setenv("CLOUD_API_KEY", "sk-secret")
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test() as pilot:
        await run_primary(app, pilot)

        worker_profiles = [
            profile
            for profile, (role, _) in zip(
                stream.profiles, stream.requests, strict=True
            )
            if role == "worker"
        ]
        assert {p.name for p in worker_profiles} == {"local", "cloud"}
        by_name = {p.name: p for p in worker_profiles}
        assert by_name["cloud"].model == "cloud-model"
        assert by_name["cloud"].base_url == "https://api.example.com/v1"
        assert by_name["local"].api_key == "none"
        # The credential is never rendered, never returned, and never repr'd.
        assert "sk-secret" not in repr(by_name["cloud"])
        assert "sk-secret" not in "\n".join(bubbles(app))
        assert "sk-secret" not in "\n".join(agent_activity(app, "b"))
        assert "sk-secret" not in json.dumps(agent_results(app))
        assert "sk-secret" not in app.query_one("#status", Static).content


async def test_an_explicit_model_skips_discovery(tmp_path, monkeypatch):
    stream = Conversation([dispatch_call(), "all done"], ["ok"])
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(stream))
    probed: list[Profile] = []

    def discover(profile):
        probed.append(profile)
        raise AssertionError("discovery must not run for a configured model")

    app = make_app(tmp_path)
    app.fetch_server_info_fn = discover
    async with app.run_test() as pilot:
        await run_primary(app, pilot)
    assert probed == []
    assert agent_results(app)[0]["status"] == "completed"


async def test_an_empty_model_on_the_active_profile_uses_the_discovered_one(
    tmp_path, monkeypatch
):
    stream = Conversation([dispatch_call(), "all done"], ["ok"])
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(stream))
    settings = local_settings()
    settings.profiles = Profiles(
        items=(Profile(name="local", base_url="http://host:9000/v1"),),
        active_name="local",
    )
    app = make_app(tmp_path, settings=settings)
    app.fetch_server_info_fn = lambda profile: pytest.fail("no probe expected")
    async with app.run_test() as pilot:
        await run_primary(app, pilot)
    worker_profile = next(
        profile
        for profile, (role, _) in zip(stream.profiles, stream.requests, strict=True)
        if role == "worker"
    )
    assert worker_profile.model == "qwen3"


async def test_an_empty_model_elsewhere_is_discovered_without_touching_primary(
    tmp_path, monkeypatch
):
    stream = Conversation([dispatch_call(profile="cloud"), "all done"], ["ok"])
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(stream))
    monkeypatch.setenv("CLOUD_API_KEY", "sk-secret")
    cloud = Profile(
        name="cloud",
        base_url="https://api.example.com/v1",
        api_key_env="CLOUD_API_KEY",
    )
    settings = Settings(profiles=Profiles(items=(LOCAL, cloud), active_name="local"))
    threads: list[int] = []

    def discover(profile):
        threads.append(threading.get_ident())
        return ServerInfo(models=["discovered-cloud"], context_length=999)

    app = make_app(tmp_path, settings=settings)
    app.fetch_server_info_fn = discover
    async with app.run_test() as pilot:
        await run_primary(app, pilot)
        worker_profile = next(
            profile
            for profile, (role, _) in zip(
                stream.profiles, stream.requests, strict=True
            )
            if role == "worker"
        )
        assert worker_profile.model == "discovered-cloud"
        # Off the event loop, and the Primary footer keeps its own server.
        assert threads and threads[0] != threading.get_ident()
        assert app.server.models == ["qwen3"]
        assert app.server.context_length == 4096
        assert "discovered-cloud" not in app.query_one("#status", Static).content


@pytest.mark.parametrize(
    ("profile_name", "fragment"),
    [
        ("nope", "No profile named 'nope'"),
        ("cloud", "$CLOUD_API_KEY"),
    ],
)
async def test_a_profile_failure_fails_only_its_own_task(
    tmp_path, monkeypatch, profile_name, fragment
):
    monkeypatch.delenv("CLOUD_API_KEY", raising=False)
    stream = Conversation(
        [
            (
                f'{dispatch_call(key="bad", label="Bad", profile=profile_name)}\n'
                f'{dispatch_call(key="good", label="Good", profile="local")}'
            ),
            "all done",
        ],
        ["good answer"],
    )
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(stream))
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test() as pilot:
        await run_primary(app, pilot)

        results = agent_results(app)
        assert [r["agent_key"] for r in results] == ["bad", "good"]
        assert results[0]["status"] == "failed"
        assert fragment in results[0]["content"]
        assert results[1] == {
            **results[1],
            "status": "completed",
            "content": "good answer",
        }
        assert agent_summary(app, "bad").status == "failed"
        assert agent_summary(app, "good").status == "completed"
        assert any(fragment in line for line in agent_activity(app, "bad"))


async def test_an_unreachable_discovery_endpoint_fails_only_its_task(
    tmp_path, monkeypatch
):
    stream = Conversation([dispatch_call(profile="cloud"), "all done"])
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(stream))
    monkeypatch.setenv("CLOUD_API_KEY", "sk-secret")
    cloud = Profile(
        name="cloud",
        base_url="https://api.example.com/v1",
        api_key_env="CLOUD_API_KEY",
    )
    settings = Settings(profiles=Profiles(items=(LOCAL, cloud), active_name="local"))
    app = make_app(tmp_path, settings=settings)
    app.fetch_server_info_fn = lambda profile: ServerInfo(
        error="URLError: connection refused"
    )
    async with app.run_test() as pilot:
        await run_primary(app, pilot)
    content = agent_results(app)[0]["content"]
    assert "could not be reached" in content
    assert "connection refused" in content
    assert "sk-secret" not in content


async def test_a_cli_override_is_advertised_once_and_dispatchable(
    tmp_path, monkeypatch
):
    stream = Conversation([dispatch_call(profile="local"), "all done"], ["ok"])
    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(stream))
    settings = two_profile_settings()
    settings.profile_override = Profile(
        name="local", base_url="http://override:1/v1", model="override-model"
    )
    app = make_app(tmp_path, settings=settings)
    async with app.run_test() as pilot:
        await run_primary(app, pilot)

    system = stream.sent_to("primary")[0][0]["content"]
    listing = system[system.index("### Available profiles") :]
    assert listing.count("`local`") == 1
    assert listing.index("`local`") < listing.index("`cloud`")
    worker_profile = next(
        profile
        for profile, (role, _) in zip(stream.profiles, stream.requests, strict=True)
        if role == "worker"
    )
    assert worker_profile.base_url == "http://override:1/v1"
    assert worker_profile.model == "override-model"


async def test_each_continuation_re_resolves_the_named_profile(tmp_path, monkeypatch):
    """A key keeps its profile *name*, not a pinned resolution: the second task
    picks up whatever that name resolves to now."""
    inside = threading.Event()
    release = threading.Event()
    models: list[str] = []
    calls = {"n": 0, "worker": 0}

    def fake(profile, temperature, messages):
        if "You are a subagent" in messages[0]["content"]:
            models.append(profile.model)
            calls["worker"] += 1
            if calls["worker"] == 1:
                inside.set()
                release.wait(5)
            yield "worker answer"
            return
        calls["n"] += 1
        if calls["n"] == 1:
            yield dispatch_call(task_label="First")
        elif calls["n"] == 2:
            yield dispatch_call(task_label="Second")
        else:
            yield "all done"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    settings = local_settings()
    settings.profiles = Profiles(
        items=(Profile(name="local", base_url="http://host:9000/v1"),),
        active_name="local",
    )
    app = make_app(tmp_path, settings=settings)
    async with app.run_test() as pilot:
        inp = app.query_one("#input", Input)
        inp.value = "go"
        await pilot.press("enter")
        await wait_until(app, pilot, lambda: inside.is_set())
        # The endpoint's uniquely discovered model changes between the tasks.
        app.server.models = ["qwen4"]
        release.set()
        await wait_until(app, pilot, lambda: app._primary_turn_depth == 0)
        assert app.agents["coder"].profile_name == "local"

    assert models == ["qwen3", "qwen4"]
