"""The profiles modal, profile switching, and turn ownership."""

import asyncio
import threading

import pytest
from textual.widgets import Input, Static

from jtech_cli.cmd_tools import CmdPolicy
from jtech_cli.config import (
    Profile,
    ProfileError,
    Profiles,
    ResolvedProfile,
    Settings,
    build_settings,
)
from jtech_cli.server_info import ServerInfo
from jtech_cli.tui import ChatApp, ProfilesScreen
from jtech_cli.tui_runtime import AutonomousRuntime

from .support import (
    CLOUD,
    LOCAL,
    bubbles,
    command_call,
    make_app,
    make_app_with_cmd,
    notifications,
    send_and_drain,
    settle,
    sync_stream,
    two_profile_settings,
    wait_until,
)


def profiles_rows_text(app: ChatApp) -> str:
    return str(app.screen.query_one("#profiles-rows", Static).render())


def profiles_help_text(app: ChatApp) -> str:
    return str(app.screen.query_one("#profiles-help", Static).render())


def set_field(app: ChatApp, widget_id: str, value: str) -> None:
    app.screen.query_one(f"#{widget_id}", Input).value = value


async def open_profiles(app: ChatApp, pilot) -> None:
    app.query_one("#input", Input).value = "/profiles"
    await pilot.press("enter")
    await settle(pilot)


async def test_profiles_modal_lists_every_profile_and_the_active_marker(tmp_path):
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test(size=(80, 30)) as pilot:
        await open_profiles(app, pilot)

        assert isinstance(app.screen, ProfilesScreen)
        rows = profiles_rows_text(app)
        assert "local (active)" in rows
        assert "cloud" in rows and "cloud (active)" not in rows
        assert "https://api.example.com/v1" in rows
        assert "$CLOUD_API_KEY" in rows  # the variable name, never a key value
        assert ProfilesScreen.ADD_ROW in rows


async def test_profiles_modal_does_not_probe_any_endpoint(tmp_path, monkeypatch):
    """Connectivity is transient; a stopped local server is still editable."""
    probes = []
    monkeypatch.setattr(
        "jtech_cli.tui.fetch_server_info",
        lambda profile: probes.append(profile) or ServerInfo(),
    )
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test(size=(80, 30)) as pilot:
        await open_profiles(app, pilot)
        await pilot.press("down")
        await settle(pilot)
        assert probes == []


async def test_profiles_modal_activates_and_persists(tmp_path):
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test(size=(80, 30)) as pilot:
        await open_profiles(app, pilot)
        await pilot.press("down")  # cloud
        await pilot.press("enter")  # actions
        await settle(pilot)
        assert "Profile: cloud" in profiles_help_text(app)
        await pilot.press("enter")  # Activate
        await settle(pilot)

        assert app.settings.profiles.active_name == "cloud"
        assert 'active_profile = "cloud"' in (tmp_path / "config.toml").read_text()
        assert "profile: cloud" in app.query_one("#status", Static).content
        assert "cloud (active)" in profiles_rows_text(app)


async def test_profiles_modal_adds_a_profile_without_activating_it(tmp_path):
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test(size=(80, 30)) as pilot:
        await open_profiles(app, pilot)
        await pilot.press("down", "down")  # Add profile…
        await pilot.press("enter")
        await settle(pilot)

        set_field(app, "profile-name", "staging")
        set_field(app, "profile-url", "https://staging.example.com/v1")
        set_field(app, "profile-key", "STAGING_KEY")
        await pilot.press("enter")
        await settle(pilot)

        assert app.settings.profiles.names == ("local", "cloud", "staging")
        assert app.settings.profiles.active_name == "local"
        added = app.settings.profiles.get("staging")
        assert added.model == ""  # blank means auto-discover
        assert added.api_key_env == "STAGING_KEY"
        assert "[profiles.staging]" in (tmp_path / "config.toml").read_text()


async def test_profiles_modal_edits_and_renames_in_place(tmp_path):
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test(size=(80, 30)) as pilot:
        await open_profiles(app, pilot)
        await pilot.press("enter")  # local -> actions
        await pilot.press("down")  # Edit
        await pilot.press("enter")
        await settle(pilot)
        assert app.screen.query_one("#profile-url", Input).value == "http://host:9000/v1"

        set_field(app, "profile-name", "workstation")
        set_field(app, "profile-url", "http://renamed:1/v1")
        await pilot.press("enter")
        await settle(pilot)

        assert app.settings.profiles.names == ("workstation", "cloud")
        # a renamed profile that was active stays active
        assert app.settings.profiles.active_name == "workstation"
        assert app.settings.profiles.get("workstation").base_url == "http://renamed:1/v1"
        assert "profile: workstation" in app.query_one("#status", Static).content


async def test_profiles_modal_keeps_the_editor_open_on_invalid_input(tmp_path):
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test(size=(80, 30)) as pilot:
        before = app.settings.profiles
        await open_profiles(app, pilot)
        await pilot.press("enter", "down", "enter")  # edit local
        await settle(pilot)

        set_field(app, "profile-url", "not-a-url")
        await pilot.press("enter")
        await settle(pilot)

        assert app.screen.query_one("#profile-url", Input).value == "not-a-url"
        assert app.settings.profiles is before
        assert any("base_url" in message for message in notifications(app))

        set_field(app, "profile-url", "http://fixed:1/v1")
        await pilot.press("enter")
        await settle(pilot)
        assert app.settings.profiles.get("local").base_url == "http://fixed:1/v1"


async def test_profiles_modal_cancels_an_edit_without_saving(tmp_path):
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test(size=(80, 30)) as pilot:
        before = app.settings.profiles
        await open_profiles(app, pilot)
        await pilot.press("enter", "down", "enter")  # edit local
        await settle(pilot)

        set_field(app, "profile-url", "http://discarded/v1")
        await pilot.press("escape")
        await settle(pilot)

        assert app.settings.profiles is before
        assert not (tmp_path / "config.toml").exists()
        assert isinstance(app.screen, ProfilesScreen)  # back to the action list


async def test_profiles_modal_deletes_an_inactive_profile_after_confirming(tmp_path):
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test(size=(80, 30)) as pilot:
        await open_profiles(app, pilot)
        await pilot.press("down")  # cloud
        await pilot.press("enter")  # actions
        await pilot.press("down", "down")  # Delete
        await pilot.press("enter")  # confirm state
        await settle(pilot)
        assert "Delete profile cloud?" in profiles_help_text(app)

        await pilot.press("up")  # Confirm delete (the cursor defaults to Cancel)
        await pilot.press("enter")
        await settle(pilot)

        assert app.settings.profiles.names == ("local",)
        assert "[profiles.cloud]" not in (tmp_path / "config.toml").read_text()


async def test_profiles_modal_confirm_defaults_to_cancel(tmp_path):
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test(size=(80, 30)) as pilot:
        await open_profiles(app, pilot)
        await pilot.press("down", "enter", "down", "down", "enter")  # cloud -> Delete
        await settle(pilot)
        await pilot.press("enter")  # take the default choice
        await settle(pilot)

        assert app.settings.profiles.names == ("local", "cloud")
        assert not (tmp_path / "config.toml").exists()


async def test_profiles_modal_refuses_to_delete_the_active_profile(tmp_path):
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test(size=(80, 30)) as pilot:
        await open_profiles(app, pilot)
        await pilot.press("enter")  # local (active) -> actions
        await pilot.press("down", "down")  # Delete
        await pilot.press("enter")
        await pilot.press("up")  # Confirm delete
        await pilot.press("enter")
        await settle(pilot)

        assert app.settings.profiles.names == ("local", "cloud")
        assert any(
            "activate another profile" in message for message in notifications(app)
        )


async def test_a_failed_profile_save_keeps_the_modal_open_and_the_old_catalog(tmp_path):
    (tmp_path / "config.toml").mkdir()  # writing here raises OSError
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test(size=(80, 30)) as pilot:
        before = app.settings.profiles
        await open_profiles(app, pilot)
        await pilot.press("down", "enter", "enter")  # activate cloud
        await settle(pilot)

        assert app.settings.profiles is before
        assert app.settings.profiles.active_name == "local"
        assert notifications(app)
        # the modal is still open, still on the profile whose save failed
        assert isinstance(app.screen, ProfilesScreen)
        assert "Profile: cloud" in profiles_help_text(app)

        await pilot.press("escape")  # back to the list
        await settle(pilot)
        assert "local (active)" in profiles_rows_text(app)


async def test_switching_profiles_clears_stale_endpoint_state(tmp_path, monkeypatch):
    monkeypatch.setenv("CLOUD_API_KEY", "sk-secret")
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test() as pilot:
        app._prompt_tokens = 1234
        app._render_status()
        assert "ctx" in app.query_one("#status", Static).content

        app.query_one("#input", Input).value = "/profile cloud"
        await pilot.press("enter")
        await settle(pilot)

        assert app.settings.profiles.active_name == "cloud"
        assert app.server.models == []
        assert app.server.context_length is None
        assert app.server.error is None
        assert app._prompt_tokens == 0
        assert app.ctx.server is app.server  # cleared in place, not rebound
        status = app.query_one("#status", Static).content
        assert "profile: cloud" in status
        assert "https://api.example.com/v1" in status
        assert "ctx" not in status


async def test_a_stale_discovery_result_is_discarded(tmp_path, monkeypatch):
    """A slow probe of the previous endpoint must not describe the new one."""
    monkeypatch.setenv("CLOUD_API_KEY", "sk-secret")
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test() as pilot:
        await app._switch_profile("cloud")
        await settle(pilot)

        app._fetch_server_info_fn = lambda profile: ServerInfo(
            models=["stale"], context_length=999
        )
        await app._discover_server(LOCAL)  # a probe started before the switch

        assert app.server.models == []
        assert app.server.context_length is None
        assert not any("stale" in bubble for bubble in bubbles(app))


async def test_an_unknown_profile_name_reports_without_changing_state(tmp_path):
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test() as pilot:
        before = app.settings.profiles
        app.query_one("#input", Input).value = "/profile nope"
        await pilot.press("enter")
        await settle(pilot)

        assert app.settings.profiles is before
        assert app.settings.profiles.active_name == "local"
        assert not (tmp_path / "config.toml").exists()
        assert any("No profile named 'nope'" in bubble for bubble in bubbles(app))


async def test_switching_clears_a_cli_override_once_stored(tmp_path, monkeypatch):
    monkeypatch.setenv("CLOUD_API_KEY", "sk-secret")
    settings = two_profile_settings()
    settings.profile_override = Profile(
        name="local", base_url="http://override/v1", model="override-model"
    )
    app = make_app(tmp_path, settings=settings)
    async with app.run_test() as pilot:
        assert "(override)" in app.query_one("#status", Static).content

        await app._switch_profile("cloud")
        await settle(pilot)

        assert app.settings.profile_override is None
        assert app.settings.active_profile == CLOUD
        assert "(override)" not in app.query_one("#status", Static).content


async def test_a_failed_switch_keeps_the_override_and_the_catalog(tmp_path):
    (tmp_path / "config.toml").mkdir()  # writing here raises OSError
    settings = two_profile_settings()
    override = Profile(name="local", base_url="http://override/v1", model="override-model")
    settings.profile_override = override
    app = make_app(tmp_path, settings=settings)
    async with app.run_test() as pilot:
        before = app.settings.profiles
        await app._switch_profile("cloud")
        await settle(pilot)

        assert app.settings.profiles is before
        assert app.settings.profile_override is override
        assert any(
            "Could not save profile selection" in bubble for bubble in bubbles(app)
        )


async def test_a_profile_switch_is_refused_while_streaming(tmp_path, monkeypatch):
    app = make_app(tmp_path, settings=two_profile_settings())
    gate = threading.Event()

    def fake(profile, temperature, messages):
        yield "partial "
        gate.wait(5)
        yield "done"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        await send_and_drain(app, pilot, "go")
        await wait_until(app, pilot, lambda: any("partial" in b for b in bubbles(app)))
        assert app._generating

        app.query_one("#input", Input).value = "/profile cloud"
        await pilot.press("enter")
        await settle(pilot)
        assert app.settings.profiles.active_name == "local"
        assert any("Esc to stop it" in bubble for bubble in bubbles(app))

        app._open_profiles()
        await pilot.pause()
        assert not isinstance(app.screen, ProfilesScreen)

        gate.set()
        await wait_until(app, pilot, lambda: not app._generating)


def park_in_tool_round(app: ChatApp) -> None:
    """Install a Primary runtime parked between completions, as a batch is.

    The real state object, not a stand-in boolean: ``_busy()`` reads the run
    that owns the flag, so a test that fakes the flag would prove nothing.
    """
    state = app._primary_run_state(
        ResolvedProfile(
            name="local", base_url="http://host:9000/v1", model="qwen3", api_key="none"
        )
    )
    state.tool_rounds_active = True
    app._primary_runtime = AutonomousRuntime(
        state,
        host=app,
        stream_reply_fn=app._stream_reply_fn,
        cmd_policy=app.cmd,
        project_root=app._project_root,
    )


async def test_a_profile_change_is_refused_during_a_tool_round(tmp_path):
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test() as pilot:
        before = app.settings.profiles
        park_in_tool_round(app)
        try:
            await app._switch_profile("cloud")
            await pilot.pause()
            assert app.settings.profiles is before

            app._open_profiles()
            await pilot.pause()
            assert not isinstance(app.screen, ProfilesScreen)

            with pytest.raises(ProfileError):
                await app._commit_profiles(before.activate("cloud"))
            assert app.settings.profiles is before
            assert any("tool round" in bubble for bubble in bubbles(app))
        finally:
            app._primary_runtime = None


async def test_one_autonomous_turn_uses_one_resolved_profile(tmp_path, monkeypatch):
    """First reply, command continuation, and nudge share one immutable profile."""
    app = make_app_with_cmd(
        tmp_path, CmdPolicy(mode="auto", allow=["echo:*"]), settings=two_profile_settings()
    )
    seen: list[tuple[object, float]] = []
    calls = {"n": 0}

    def fake(profile, temperature, messages):
        calls["n"] += 1
        seen.append((profile, temperature))
        if calls["n"] == 1:
            yield command_call("echo turn-out")
        elif calls["n"] == 2:
            yield ""  # empty reply -> a nudge round
        else:
            yield "done"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        await send_and_drain(app, pilot, "go")
        await wait_until(app, pilot, lambda: calls["n"] >= 3, tries=150)
        await wait_until(app, pilot, lambda: not app._tool_rounds_active, tries=100)

    assert calls["n"] == 3
    used = [profile for profile, _ in seen]
    # the same object, not merely an equal one
    assert all(profile is used[0] for profile in used)
    assert used[0].base_url == "http://host:9000/v1"
    assert used[0].model == "qwen3"
    assert used[0].api_key == "none"
    assert {temperature for _, temperature in seen} == {0.7}


async def test_the_next_idle_turn_uses_the_newly_activated_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("CLOUD_API_KEY", "sk-secret")
    app = make_app(tmp_path, settings=two_profile_settings())
    seen = []

    def fake(profile, temperature, messages):
        seen.append(profile)
        yield "ok"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        await send_and_drain(app, pilot, "first")
        await wait_until(app, pilot, lambda: len(seen) == 1)

        app.query_one("#input", Input).value = "/profile cloud"
        await pilot.press("enter")
        await wait_until(
            app, pilot, lambda: app.settings.profiles.active_name == "cloud"
        )

        await send_and_drain(app, pilot, "second")
        await wait_until(app, pilot, lambda: len(seen) == 2)

    assert seen[0].base_url == "http://host:9000/v1"
    assert seen[0].api_key == "none"
    assert seen[1].base_url == "https://api.example.com/v1"
    assert seen[1].model == "cloud-model"
    assert seen[1].api_key == "sk-secret"
    # switching profiles does not clear or fork the conversation
    assert [m["content"] for m in app.session.messages] == [
        "first", "ok", "second", "ok",
    ]


async def test_a_missing_credential_stops_before_the_provider_thread(tmp_path, monkeypatch):
    monkeypatch.delenv("CLOUD_API_KEY", raising=False)
    settings = Settings(profiles=Profiles(items=(CLOUD,), active_name="cloud"))
    app = make_app(tmp_path, settings=settings)
    started = []

    def fake(profile, temperature, messages):
        started.append(profile)
        yield "never"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        await send_and_drain(app, pilot, "go")

        assert started == []
        assert any("CLOUD_API_KEY" in bubble for bubble in bubbles(app))
        assert any("unset or empty" in bubble for bubble in bubbles(app))
        assert app.session.messages == [{"role": "user", "content": "go"}]
        assert not app._generating


async def test_a_missing_model_stops_before_the_provider_thread(tmp_path, monkeypatch):
    """No configured model and no unique served model is an error, not a guess."""
    profile = Profile(name="local", base_url="http://host:9000/v1")
    settings = Settings(profiles=Profiles(items=(profile,), active_name="local"))
    app = make_app(
        tmp_path, settings=settings, server=ServerInfo(models=["one", "two"])
    )
    started = []

    def fake(profile, temperature, messages):
        started.append(profile)
        yield "never"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        await send_and_drain(app, pilot, "go")

        assert started == []
        assert any("no model configured" in bubble for bubble in bubbles(app))
        assert not app._generating


async def test_a_turn_without_a_profile_reports_instead_of_streaming(tmp_path, monkeypatch):
    app = make_app(tmp_path, settings=Settings())
    started = []

    def fake(profile, temperature, messages):
        started.append(profile)
        yield "never"

    monkeypatch.setattr("jtech_cli.tui.stream_reply", sync_stream(fake))
    async with app.run_test() as pilot:
        await send_and_drain(app, pilot, "go")

        assert started == []
        assert any("No API profile is configured" in b for b in bubbles(app))
        assert not app._generating


async def test_modal_activation_retires_a_cli_override(tmp_path):
    """Regression: the modal persisted a selection the override kept shadowing."""
    settings = two_profile_settings()
    settings.profile_override = Profile(
        name="local", base_url="http://override.example/v1", model="override-model"
    )
    app = make_app(tmp_path, settings=settings)
    async with app.run_test(size=(80, 30)) as pilot:
        assert "(override)" in app.query_one("#status", Static).content

        await open_profiles(app, pilot)
        await pilot.press("down")  # cloud
        await pilot.press("enter")  # actions
        await pilot.press("enter")  # Activate
        await settle(pilot)

        assert app.settings.profiles.active_name == "cloud"
        assert app.settings.profile_override is None
        assert app.settings.active_profile == CLOUD
        status = app.query_one("#status", Static).content
        assert "profile: cloud" in status
        assert "(override)" not in status
        assert "override.example" not in status


async def test_a_failed_modal_activation_keeps_the_override(tmp_path):
    (tmp_path / "config.toml").mkdir()  # writing here raises OSError
    settings = two_profile_settings()
    override = Profile(name="local", base_url="http://override.example/v1", model="ov")
    settings.profile_override = override
    app = make_app(tmp_path, settings=settings)
    async with app.run_test(size=(80, 30)) as pilot:
        await open_profiles(app, pilot)
        await pilot.press("down", "enter", "enter")  # activate cloud
        await settle(pilot)

        assert app.settings.profile_override is override
        assert app.settings.profiles.active_name == "local"


async def test_a_modal_edit_does_not_retire_a_cli_override(tmp_path):
    """Editing is not selecting: only Activate supersedes the CLI flag."""
    settings = two_profile_settings()
    override = Profile(name="local", base_url="http://override.example/v1", model="ov")
    settings.profile_override = override
    app = make_app(tmp_path, settings=settings)
    async with app.run_test(size=(80, 30)) as pilot:
        await open_profiles(app, pilot)
        await pilot.press("down", "enter", "down", "enter")  # edit cloud
        await settle(pilot)
        set_field(app, "profile-model", "edited-model")
        await pilot.press("enter")
        await settle(pilot)

        assert app.settings.profiles.get("cloud").model == "edited-model"
        assert app.settings.profile_override is override


async def test_adding_the_first_profile_persists_an_active_selection(tmp_path):
    """Regression: the modal could write a config that failed on next launch."""
    app = make_app(tmp_path, settings=Settings())
    async with app.run_test(size=(80, 30)) as pilot:
        await open_profiles(app, pilot)
        await pilot.press("enter")  # the only row is Add profile…
        await settle(pilot)

        set_field(app, "profile-name", "first")
        set_field(app, "profile-url", "http://first:1/v1")
        await pilot.press("enter")
        await settle(pilot)

        assert app.settings.profiles.active_name == "first"
        text = (tmp_path / "config.toml").read_text()
        assert 'active_profile = "first"' in text
        assert build_settings(config_path=tmp_path / "config.toml").profiles.active_name == "first"


async def test_a_stale_token_count_is_discarded(tmp_path, monkeypatch):
    """A count describes one tokenizer; a late one must not describe the new one."""
    monkeypatch.setenv("CLOUD_API_KEY", "sk-secret")
    app = make_app(
        tmp_path,
        settings=two_profile_settings(),
        fetch_token_count_fn=lambda profile, text: 7,
    )
    app.session.add("user", "hello world")
    entered = threading.Event()
    released = threading.Event()

    def slow_count(profile, text):
        entered.set()
        released.wait(5)
        return 42

    async with app.run_test() as pilot:
        await wait_until(app, pilot, lambda: app._prompt_tokens == 7)

        app._fetch_token_count_fn = slow_count
        counting = asyncio.ensure_future(app._init_token_count(LOCAL))
        await wait_until(app, pilot, entered.is_set, tries=100)

        await app._switch_profile("cloud")
        await settle(pilot)
        assert app._prompt_tokens == 0  # the switch cleared the old count

        released.set()
        await counting
        await settle(pilot)

        assert app._prompt_tokens == 0  # the old endpoint's 42 never landed
        assert "ctx" not in app.query_one("#status", Static).content


async def test_a_stale_credential_error_is_not_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("CLOUD_API_KEY", "sk-secret")
    app = make_app(
        tmp_path,
        settings=two_profile_settings(),
        fetch_token_count_fn=lambda profile, text: 7,
    )
    app.session.add("user", "hello world")

    def boom(profile, text):
        raise ProfileError("stale credential complaint")

    async with app.run_test() as pilot:
        await wait_until(app, pilot, lambda: app._prompt_tokens == 7)

        app._fetch_token_count_fn = boom
        await app._switch_profile("cloud")
        await settle(pilot)
        await app._init_token_count(LOCAL)  # a probe from before the switch
        await settle(pilot)

        assert not any("stale credential" in bubble for bubble in bubbles(app))


async def test_a_current_credential_error_is_still_reported(tmp_path):
    """The staleness guard silences late results, not live failures."""
    app = make_app(tmp_path, settings=two_profile_settings())
    app.session.add("user", "hello world")

    def boom(profile, text):
        raise ProfileError("live credential complaint")

    app._fetch_token_count_fn = boom
    async with app.run_test() as pilot:
        await settle(pilot)
        assert any("live credential complaint" in b for b in bubbles(app))


async def test_a_live_token_count_still_reaches_the_footer(tmp_path):
    """The staleness guard must not disable the normal path."""
    app = make_app(
        tmp_path,
        settings=two_profile_settings(),
        fetch_token_count_fn=lambda profile, text: 128,
    )
    app.session.add("user", "hello world")
    async with app.run_test() as pilot:
        await wait_until(app, pilot, lambda: app._prompt_tokens == 128)
        assert "ctx" in app.query_one("#status", Static).content


@pytest.mark.parametrize("url", ["http://[::1/v1", "https://host:0/v1"])
async def test_the_editor_reports_an_unparseable_url_instead_of_crashing(tmp_path, url):
    """urlparse raises bare ValueError on these; the modal only catches ProfileError."""
    app = make_app(tmp_path, settings=two_profile_settings())
    async with app.run_test(size=(80, 30)) as pilot:
        before = app.settings.profiles
        await open_profiles(app, pilot)
        await pilot.press("enter", "down", "enter")  # edit local
        await settle(pilot)

        set_field(app, "profile-url", url)
        await pilot.press("enter")
        await settle(pilot)

        assert app._exception is None
        assert isinstance(app.screen, ProfilesScreen)  # editor still open
        assert app.screen.query_one("#profile-url", Input).value == url
        assert app.settings.profiles is before
        assert notifications(app)
