"""Command approval, serialized across every runtime in the app.

The lock is the whole point: one approval modal exists at a time, and the
policy is re-read inside it, so an allow rule saved by the agent ahead in the
queue decides the request behind it too.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Protocol

from jtech_cli.cmd_tools import CmdPolicy, allow_rule_for, decide
from jtech_cli.config import Settings
from jtech_cli.tui_profiles import ProfileManager
from jtech_cli.tui_runtime import AgentRunState, CommandAuthorization, RunPhase
from jtech_cli.tui_screens import CmdChoice, CommandPrompt
from jtech_cli.tui_widgets import TranscriptRecord


class ApprovalHost(Protocol):
    """The app surface one approval is decided and reported through."""

    cmd: CmdPolicy
    settings: Settings
    project_root: Path
    profile_manager: ProfileManager

    async def push_screen_wait(self, screen: Any) -> Any:
        """Open a modal and wait for its result."""
        ...

    def set_run_phase(self, run: AgentRunState, phase: RunPhase) -> None:
        """Move one run to ``phase`` from outside its own runtime."""
        ...


class CommandApprovals:
    """The app-wide command policy gate every runtime asks."""

    def __init__(self, host: ApprovalHost) -> None:
        self._host = host
        # One approval modal at a time, and the policy re-read inside the
        # lock: a rule another agent just saved must decide this request too.
        self._lock = asyncio.Lock()

    async def authorize(
        self, run: AgentRunState, command: str
    ) -> CommandAuthorization:
        """Decide one command for ``run`` under the live global policy.

        Serialized for every runtime: one approval modal exists at a time, and
        the decision is re-evaluated after the lock is acquired, because the
        agent ahead in the queue may have saved an allow rule that covers this
        command too.
        """
        host = self._host
        async with self._lock:
            decision = decide(command, host.cmd, host.project_root)
            if decision.action == "run":
                return CommandAuthorization("run")
            if decision.action == "blocked":
                return CommandAuthorization("blocked", decision.reason)
            host.set_run_phase(run, "waiting")
            try:
                choice = await host.push_screen_wait(
                    CommandPrompt(command, decision.reason, requester=run.agent_label)
                )
            finally:
                host.set_run_phase(run, "tool")
            if choice is CmdChoice.DECLINE:
                return CommandAuthorization("declined", "declined by the user")
            if choice is CmdChoice.ALWAYS:
                self._add_allow_rule(allow_rule_for(command, host.cmd.allow), run)
            return CommandAuthorization("run")

    def _add_allow_rule(self, rule: str | None, run: AgentRunState) -> None:
        """Persist one always-allow rule, reporting it to the run that earned it."""
        host = self._host
        if not rule or rule in host.cmd.allow:
            return
        host.cmd.allow.append(rule)
        try:
            host.profile_manager.save(host.settings)
            note = f"Always-allow saved: {rule}"
        except OSError as error:
            note = f"Could not save always-allow rule: {error}"
        run.transcript.append(TranscriptRecord(role="system", content=note))
