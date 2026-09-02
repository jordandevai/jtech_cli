"""Endpoint selection and the state derived from it.

One object owns which profile is active, how a change to that is persisted,
and everything invalidated when the endpoint underneath changes: the
discovered models, the context length, the token count, and the theme and
settings the same modal writes. Keeping them together is what makes the
invalidation a single step rather than a rule every caller has to remember.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
from pathlib import Path
from typing import Any, Protocol

from jtech_cli.cmd_tools import CmdPolicy
from jtech_cli.config import (
    Profile,
    ProfileError,
    Profiles,
    ResolvedProfile,
    Settings,
    resolve_profile,
    save_settings,
)
from jtech_cli.server_info import FetchServerInfo, FetchTokenCount, ServerInfo
from jtech_cli.session import Session
from jtech_cli.theme import textual_theme_name
from jtech_cli.tui_runtime import AutonomousRuntime
from jtech_cli.tui_screens import ProfilesScreen, SettingsScreen
from jtech_cli.tui_status import StatusView

NO_PROFILE = "No API profile is configured — run /profiles to add one."
BUSY_GENERATING = "A reply is streaming — press Esc to stop it before changing profiles."
BUSY_TOOL_ROUND = "A tool round is running — wait for it to finish before changing profiles."


class ProfileHost(Protocol):
    """The app surface profile changes are applied and reported through."""

    settings: Settings
    session: Session
    server: ServerInfo
    cmd: CmdPolicy
    config_path: Path
    status: StatusView
    no_discover: bool
    fetch_server_info_fn: FetchServerInfo
    fetch_token_count_fn: FetchTokenCount
    primary_runtime: AutonomousRuntime | None
    theme: str

    def push_message(self, role: str, text: str) -> None:
        """Say something to the user in the Primary transcript."""
        ...

    def refresh_theme(self) -> None:
        """Rebuild rendered history for the newly registered theme."""
        ...

    def push_screen(self, screen: Any, callback: Any = None) -> Any:
        """Open a modal."""
        ...

    def call_later(self, callback: Any, *args: Any) -> Any:
        """Run ``callback`` on a later pass of the message pump."""
        ...


class ProfileManager:
    """The active endpoint, its persistence, and what a change invalidates."""

    def __init__(self, host: ProfileHost) -> None:
        self._host = host

    # ---------------------------------------------------------------- guard

    def busy(self) -> str | None:
        """Why a profile change is refused right now, or ``None`` when idle.

        The current turn must keep one endpoint/model/credential identity, so a
        switch that is visible in the footer but does not affect the running
        turn is refused outright rather than half-applied. A live agent batch
        is a tool round: Primary stays in it until every result is back.
        """
        runtime = self._host.primary_runtime
        if runtime is None:
            return None
        if runtime.state.generating:
            return BUSY_GENERATING
        return BUSY_TOOL_ROUND

    # ---------------------------------------------------------- persistence

    def save(self, settings: Settings) -> None:
        """Write ``settings`` with the complete live command policy."""
        host = self._host
        host.cmd.mode = host.settings.cmd_mode
        save_settings(settings, host.config_path, cmd=host.cmd)

    # ------------------------------------------------------------- catalog

    def open_modal(self) -> None:
        busy = self.busy()
        if busy:
            self._host.push_message("system", busy)
            return
        self._host.push_screen(
            ProfilesScreen(self._host.settings.profiles, self.commit)
        )

    def _adopt(
        self, candidate: Profiles, previous: Profile | None, *, activated: bool
    ) -> None:
        """Adopt a catalog that has just been persisted successfully.

        The single place the live catalog advances, so ``/profile`` and the
        modal's Activate action cannot drift apart.
        """
        self._host.settings.profiles = candidate
        if activated:
            # An explicit selection supersedes a --base-url/--model override,
            # but only once that selection is actually stored.
            self._host.settings.profile_override = None
        self._after_change(previous)

    async def commit(self, candidate: Profiles, *, activated: bool = False) -> None:
        """Persist ``candidate``, then adopt it as the live catalog.

        Persistence comes first, so a failed save needs no live-state rollback:
        the previous catalog is still the only one anything has seen.
        ``activated`` marks the modal's Activate action, which is an explicit
        selection and therefore retires a CLI endpoint override.

        Raises:
            ProfileError: if a turn is in progress.
            OSError: if the config file cannot be written.
        """
        busy = self.busy()
        if busy:
            self._host.push_message("system", busy)
            raise ProfileError(busy)
        previous = self._host.settings.active_profile
        replaced = dataclasses.replace(self._host.settings, profiles=candidate)
        if activated:
            replaced = dataclasses.replace(replaced, profile_override=None)
        self.save(replaced)
        self._adopt(candidate, previous, activated=activated)

    async def switch(self, name: str) -> None:
        """Activate ``name`` and persist the selection for the next launch."""
        host = self._host
        busy = self.busy()
        if busy:
            host.push_message("system", busy)
            return
        try:
            candidate = host.settings.profiles.activate(name)
        except ProfileError as error:
            host.push_message("system", str(error))
            return
        previous = host.settings.active_profile
        try:
            self.save(
                dataclasses.replace(
                    host.settings, profiles=candidate, profile_override=None
                )
            )
        except OSError as error:
            host.push_message("system", f"Could not save profile selection: {error}")
            return
        self._adopt(candidate, previous, activated=True)
        host.push_message("system", f"Profile: {name}")

    def _after_change(self, previous: Profile | None) -> None:
        """Invalidate endpoint-derived state when the selected endpoint changed."""
        host = self._host
        current = host.settings.active_profile
        if current == previous:
            host.status.render()
            return
        # CommandContext shares this ServerInfo, so clear it in place rather
        # than rebinding: /models and /stats read the same object.
        host.server.models = []
        host.server.context_length = None
        host.server.error = None
        host.status.prompt_tokens = 0
        host.status.render()
        if current is not None and not host.no_discover:
            host.call_later(self.discover, current)

    # ------------------------------------------------------------ discovery

    async def discover(self, profile: Profile) -> None:
        """Refresh metadata for ``profile`` without downgrading known values."""
        host = self._host
        try:
            info = await asyncio.to_thread(host.fetch_server_info_fn, profile)
        except ProfileError as error:
            host.push_message("system", str(error))
            return
        if host.settings.active_profile != profile:
            # The endpoint changed while this probe was in flight. A late answer
            # from the previous one must not describe the current one.
            return
        if not info.known:
            detail = f" ({info.error})" if info.error else ""
            host.push_message(
                "system",
                f"Could not reach {profile.base_url}{detail} — model and context "
                "info unavailable. Check the endpoint in /profiles.",
            )
            return
        host.server.models = info.models
        host.server.context_length = info.context_length
        host.server.error = None
        host.status.render()
        if host.session.messages and host.server.context_length:
            await self.count_tokens(profile)

    async def count_tokens(self, profile: Profile) -> None:
        """Count session tokens for ``profile`` so the footer is accurate.

        The count describes one endpoint's tokenizer, so a result that arrives
        after the user switched profiles is discarded rather than applied to the
        new one — the same staleness rule discovery uses.
        """
        host = self._host
        history = host.session.messages_with_system("")
        text = " ".join(message["content"] for message in history)
        if not text:
            return
        try:
            count = await asyncio.to_thread(host.fetch_token_count_fn, profile, text)
        except ProfileError as error:
            if host.settings.active_profile == profile:
                host.push_message("system", str(error))
            return
        if count and host.settings.active_profile == profile:
            host.status.prompt_tokens = count
            host.status.render()

    # -------------------------------------------------------------- pinning

    def resolve_turn_profile(self) -> ResolvedProfile:
        """Pin one endpoint, model, and credential for this whole user turn.

        Raises:
            ProfileError: if no profile is selected, no model resolves, or the
                credential is unavailable — before any provider thread starts.
        """
        profile = self._host.settings.active_profile
        if profile is None:
            raise ProfileError(NO_PROFILE)
        return resolve_profile(
            profile, discovered_model=self._host.server.model, environ=os.environ
        )

    # ------------------------------------------------------ settings / theme

    def open_settings(self) -> None:
        host = self._host
        host.push_screen(
            SettingsScreen(
                host.settings,
                host.config_path,
                self._on_settings_saved,
                cmd=host.cmd,
            )
        )

    def switch_theme(self) -> None:
        host = self._host
        name = textual_theme_name(host.settings.theme)
        if name != host.theme:
            host.theme = name
            # Live bubbles follow the theme through CSS; completed history is
            # rendered content, so it has to be rebuilt for the new colors —
            # in every registered stream, hidden ones included.
            host.refresh_theme()

    def _on_settings_saved(self) -> None:
        self.switch_theme()
        self._host.status.render()
