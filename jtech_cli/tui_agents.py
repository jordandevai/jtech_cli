"""Dispatched agents: their registry, their tasks, and one batch at a time.

Primary is deliberately not a member here — its session, transcript, and
composer belong to the app. What this owns is every agent the coordinator
created by dispatching to it: the conversation each one accumulates, the task
rows the sidebar shows, and the runtime a live task is executing in.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from jtech_cli.cmd_tools import AgentDispatch, CmdPolicy, duplicate_agent_keys
from jtech_cli.config import (
    Profile,
    ProfileError,
    ResolvedProfile,
    Settings,
    resolve_profile,
)
from jtech_cli.prompts import compose_worker_prompt
from jtech_cli.server_info import FetchServerInfo, ServerInfo
from jtech_cli.session import Session
from jtech_cli.tui_runtime import (
    AgentOutcome,
    AgentRunState,
    AutonomousRuntime,
    RunPhase,
    RuntimeHost,
    StreamReply,
)
from jtech_cli.tui_widgets import (
    AgentStatus,
    AgentSummary,
    AgentTaskSummary,
    Transcript,
    TranscriptRecord,
)

AGENT_STOPPED = "Agent stopped before completing the task."

#: Every non-terminal phase an agent row and its current task row can show.
#: ``stopped`` maps to ``failed``: an assignment that did not finish is not a
#: success, and Primary is told so.
_PHASE_STATUS: dict[RunPhase, AgentStatus] = {
    "starting": "running",
    "streaming": "running",
    "tool": "running",
    "command": "running",
    "waiting": "waiting",
    "completed": "completed",
    "failed": "failed",
    "stopped": "failed",
}


class DispatchRejected(Exception):
    """One dispatch call refused before any task or model request exists."""


@dataclass(slots=True)
class ManagedAgent:
    """One dispatched agent's identity, conversation, and current assignment.

    The profile *name* is retained, never a resolved credential-bearing value:
    each new task re-resolves that name and pins its current endpoint, model,
    and credential, exactly as a Primary turn does.
    """

    agent_key: str
    agent_label: str
    profile_name: str
    session: Session
    transcript: Transcript
    tasks: list[AgentTaskSummary]
    runtime: AutonomousRuntime | None = None
    active_task_id: str | None = None


class AgentHost(RuntimeHost, Protocol):
    """The app surface one dispatch batch runs against.

    Extends ``RuntimeHost`` because the coordinator hands the app itself to
    every subagent runtime it creates: approval, dispatch authority, and phase
    reporting stay the app's, exactly as they are for a Primary turn.
    """

    settings: Settings
    server: ServerInfo
    cmd: CmdPolicy
    project_root: Path
    stream_reply_fn: StreamReply
    fetch_server_info_fn: FetchServerInfo

    async def add_agent_view(
        self, summary: AgentSummary, records: Sequence[TranscriptRecord] = ()
    ) -> Transcript:
        """Register one agent's presentation and return its transcript."""
        ...

    def update_agent_view(self, summary: AgentSummary) -> None:
        """Repaint one registered agent's sidebar row."""
        ...

    def set_run_phase(self, run: AgentRunState, phase: RunPhase) -> None:
        """Move one run to ``phase`` from outside its own runtime."""
        ...


class AgentCoordinator(Mapping[str, ManagedAgent]):
    """Every dispatched agent, and the batches Primary runs against them.

    A ``Mapping`` because that is what it is: the registry keyed by agent key,
    insertion ordered by first dispatch, with the batch operations on it.
    """

    def __init__(self, host: AgentHost) -> None:
        self._host = host
        self._agents: dict[str, ManagedAgent] = {}

    def __getitem__(self, agent_key: str) -> ManagedAgent:
        return self._agents[agent_key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._agents)

    def __len__(self) -> int:
        return len(self._agents)

    def task_changed(self, run: AgentRunState) -> None:
        """Follow one subagent run's phase into the task row it owns.

        Raises:
            KeyError: if the agent is unknown or has no task in flight. That is
                an internal inconsistency, not a condition to skip past.
        """
        managed = self._agents[run.agent_key]
        task_id = managed.active_task_id
        if task_id is None:
            raise KeyError(
                f"Agent {run.agent_key!r} reported {run.phase!r} with no active task"
            )
        # The task the runtime is actually working on, never the last row of
        # the list: a continuation appends, and earlier rows keep their own
        # terminal status.
        self._set_task_status(run.agent_key, task_id, _PHASE_STATUS[run.phase])

    def request_stop_all(self) -> None:
        """Signal every agent currently executing a task."""
        for managed in self._agents.values():
            if managed.runtime is not None:
                managed.runtime.request_stop()

    def available_profiles(self) -> tuple[Profile, ...]:
        """Every profile an agent may be dispatched to, in advertised order.

        A session-only ``--base-url`` override comes first and hides a
        configured profile of the same name, so one name never advertises two
        endpoints.
        """
        configured = self._host.settings.profiles.items
        override = self._host.settings.profile_override
        if override is None:
            return tuple(configured)
        return (
            override,
            *(item for item in configured if item.name != override.name),
        )

    def _profile_for_dispatch(self, name: str) -> Profile:
        """The profile called ``name``, with the same override precedence.

        Raises:
            ProfileError: if no available profile has that name.
        """
        override = self._host.settings.profile_override
        if override is not None and override.name == name:
            return override
        return self._host.settings.profiles.get(name)

    async def _resolve_agent_profile(self, profile: Profile) -> ResolvedProfile:
        """Pin ``profile`` to one model and credential for one agent task.

        Discovery runs off the event loop and only when the profile configures
        no model. It never writes ``self._host.server`` or the Primary footer: a
        subagent's endpoint is not the one the status bar describes.

        Raises:
            ProfileError: if the credential is unavailable, the endpoint cannot
                be reached, or no single model resolves. There is no
                active-profile or model fallback.
        """
        if profile.model:
            return resolve_profile(profile, discovered_model=None, environ=os.environ)
        active = self._host.settings.active_profile
        if active is not None and profile == active and self._host.server.model:
            return resolve_profile(
                profile, discovered_model=self._host.server.model, environ=os.environ
            )
        info = await asyncio.to_thread(self._host.fetch_server_info_fn, profile)
        if not info.models and info.error:
            raise ProfileError(
                f"Profile {profile.name!r}: {profile.base_url} could not be "
                f"reached to discover its model ({info.error})"
            )
        return resolve_profile(
            profile, discovered_model=info.model, environ=os.environ
        )

    def _new_task_id(self) -> str:
        """A fresh opaque task id.

        A method so a test can make ids deterministic without an id factory in
        the public constructor. No user or model value ever enters it, and it
        is never used as a DOM id or a filesystem path.
        """
        return "task-" + uuid.uuid4().hex

    async def dispatch(
        self, run: AgentRunState, calls: tuple[AgentDispatch, ...]
    ) -> tuple[AgentOutcome, ...]:
        """Run one whole dispatch batch and return its results in call order.

        Distinct keys start together and run concurrently; one failure never
        cancels a sibling; and the results are appended to the coordinator in
        the order it wrote the calls, whatever order they finish in, so
        provider timing cannot make its context nondeterministic.

        Raises:
            ValueError: if a non-Primary run reaches here, or the batch repeats
                an agent key. The runtime refuses both first; this is the
                authority boundary saying so again, before anything is created.
        """
        if run.kind != "primary":
            raise ValueError("only Primary may dispatch agents")
        duplicates = duplicate_agent_keys(calls)
        if duplicates:
            raise ValueError(
                f"one response cannot dispatch {', '.join(duplicates)} twice"
            )

        outcomes: list[AgentOutcome | None] = [None] * len(calls)
        started: list[tuple[int, ManagedAgent, AgentDispatch, str]] = []
        for index, call in enumerate(calls):
            task_id = self._new_task_id()
            try:
                managed = await self._begin_agent_task(call, task_id)
            except DispatchRejected as rejection:
                outcomes[index] = self._setup_outcome(call, task_id, str(rejection))
                continue
            except Exception as error:  # noqa: BLE001 - one call, not the batch
                # Setting one agent up is that call's own work. An unexpected
                # failure here must not take its siblings down with it, so it
                # becomes that call's failed result like any other.
                outcomes[index] = self._setup_outcome(
                    call, task_id, f"{type(error).__name__}: {error}"
                )
                continue
            started.append((index, managed, call, task_id))

        self._host.set_run_phase(run, "waiting")
        try:
            results = await asyncio.gather(
                *(
                    self._dispatch_one(managed, call, task_id)
                    for _, managed, call, task_id in started
                ),
                return_exceptions=True,
            )
        finally:
            self._host.set_run_phase(run, "tool")

        for (index, managed, call, task_id), result in zip(
            started, results, strict=True
        ):
            if isinstance(result, asyncio.CancelledError):
                # Shutdown, not a task failure: it must unwind, not be reported
                # to the model as a completed batch.
                raise result
            if isinstance(result, BaseException):
                message = f"{type(result).__name__}: {result}"
                managed.transcript.append(
                    TranscriptRecord(role="system", content=message, error=True)
                )
                self._set_task_status(call.agent_key, task_id, "failed")
                outcomes[index] = AgentOutcome(
                    agent_key=call.agent_key,
                    agent_label=managed.agent_label,
                    task_id=task_id,
                    task_label=call.task_label,
                    status="failed",
                    content=message,
                )
                continue
            outcomes[index] = result
        # Every call has an outcome by construction: it was rejected, its setup
        # failed, it returned one, or its exception was converted above.
        # Checked, not filtered — a filter would silently hand the coordinator
        # a shorter batch than it dispatched — and checked at runtime rather
        # than asserted, because assertions vanish under `python -O`.
        settled: list[AgentOutcome] = []
        for index, outcome in enumerate(outcomes):
            if outcome is None:
                raise RuntimeError(
                    f"dispatch call {index} for agent {calls[index].agent_key!r} "
                    "produced no outcome"
                )
            settled.append(outcome)
        return tuple(settled)

    def _setup_outcome(
        self, call: AgentDispatch, task_id: str, content: str
    ) -> AgentOutcome:
        """Report one call that never reached a runtime, failing any row it made.

        A rejection creates nothing, so there is usually no row to correct. An
        unexpected setup failure can leave a task already committed to a live
        agent: that row is marked failed here rather than left running for the
        rest of the session.

        The result carries the identity the *call* asked for, so the model can
        match every outcome to the call it wrote. A label conflict is exactly
        the case where the existing agent's label differs, and answering a
        rejected ``Renamed`` call with ``Coder`` would hide which call failed.
        Where setup got far enough to touch an agent, the guards above have
        already proved the two labels equal.
        """
        managed = self._agents.get(call.agent_key)
        if managed is not None and any(
            task.task_id == task_id for task in managed.tasks
        ):
            self._set_task_status(call.agent_key, task_id, "failed")
        return AgentOutcome(
            agent_key=call.agent_key,
            agent_label=call.agent_label,
            task_id=task_id,
            task_label=call.task_label,
            status="failed",
            content=content,
        )

    async def _begin_agent_task(
        self, call: AgentDispatch, task_id: str
    ) -> ManagedAgent:
        """Create or continue one agent and append its new running task.

        A new key gets one in-memory session, one workspace view, and the task
        recorded exactly once — the transcript record seeded into the view is
        the presentation of that same message, not a second model message.

        Raises:
            DispatchRejected: if the key exists with a different label or
                profile, or is already running a task. Nothing is mutated.
        """
        task = AgentTaskSummary(task_id, call.task_label, "running")
        managed = self._agents.get(call.agent_key)
        if managed is None:
            session = Session(persist=False)
            transcript = await self._host.add_agent_view(
                AgentSummary(call.agent_key, call.agent_label, "running", (task,)),
                (TranscriptRecord.from_message(role="user", content=call.task),),
            )
            # Same rule as a continuation: the assignment joins the worker's
            # conversation only once its presentation exists. The seeded
            # transcript record is that message's presentation, not a second
            # model message.
            session.add("user", call.task)
            managed = ManagedAgent(
                agent_key=call.agent_key,
                agent_label=call.agent_label,
                profile_name=call.profile_name,
                session=session,
                transcript=transcript,
                tasks=[task],
            )
            self._agents[call.agent_key] = managed
            return managed
        if managed.agent_label != call.agent_label:
            raise DispatchRejected(
                f"Agent {call.agent_key!r} already exists with the label "
                f"{managed.agent_label!r}. An agent key keeps its label for the "
                "session; use a new key for a differently labelled agent."
            )
        if managed.profile_name != call.profile_name:
            raise DispatchRejected(
                f"Agent {call.agent_key!r} already exists on profile "
                f"{managed.profile_name!r}. An agent key keeps its profile for "
                "the session; use a new key to work on another profile."
            )
        if managed.runtime is not None:
            raise DispatchRejected(
                f"Agent {call.agent_key!r} is still working on its current task. "
                "Wait for its result before sending it another one."
            )
        # Presentation first, model context last. The sidebar and transcript
        # writes are the fallible ones, so the task row is committed only after
        # the sidebar accepted it and the assignment joins the worker's
        # conversation only after both succeeded: a setup that fails must never
        # leave an unanswered instruction in the context of the agent's next
        # task.
        self._host.update_agent_view(
            AgentSummary(
                call.agent_key,
                managed.agent_label,
                "running",
                (*managed.tasks, task),
            )
        )
        managed.tasks.append(task)
        managed.transcript.append(
            TranscriptRecord.from_message(role="user", content=call.task)
        )
        managed.session.add("user", call.task)
        return managed

    async def _dispatch_one(
        self, managed: ManagedAgent, call: AgentDispatch, task_id: str
    ) -> AgentOutcome:
        """Resolve one task's profile, run it, and map its typed outcome.

        A pre-stream failure is written into the worker's own transcript before
        the task is marked failed: a worker that never reached a provider still
        has to show why.
        """
        def finish(
            status: Literal["completed", "failed"], content: str
        ) -> AgentOutcome:
            """Mark this task terminal and describe it to the coordinator."""
            self._set_task_status(call.agent_key, task_id, status)
            return AgentOutcome(
                agent_key=call.agent_key,
                agent_label=managed.agent_label,
                task_id=task_id,
                task_label=call.task_label,
                status=status,
                content=content,
            )

        try:
            profile = self._profile_for_dispatch(call.profile_name)
            resolved = await self._resolve_agent_profile(profile)
        except ProfileError as error:
            managed.transcript.append(
                TranscriptRecord(role="system", content=str(error), error=True)
            )
            return finish("failed", str(error))

        runtime = AutonomousRuntime(
            AgentRunState(
                agent_key=call.agent_key,
                agent_label=managed.agent_label,
                kind="subagent",
                session=managed.session,
                transcript=managed.transcript,
                profile=resolved,
                temperature=self._host.settings.temperature,
                system_prompt=self._worker_system_prompt,
                reasoning_mode=lambda: self._host.settings.reasoning,
                debug_level=lambda: self._host.settings.debug_level,
            ),
            host=self._host,
            stream_reply_fn=self._host.stream_reply_fn,
            cmd_policy=self._host.cmd,
            project_root=self._host.project_root,
        )
        managed.runtime = runtime
        managed.active_task_id = task_id
        try:
            result = await runtime.run()
        finally:
            managed.runtime = None
            managed.active_task_id = None
        if result.status == "completed":
            return finish("completed", result.final_text)
        if result.status == "failed":
            return finish("failed", result.error)
        return finish("failed", AGENT_STOPPED)

    def _set_task_status(
        self, agent_key: str, task_id: str, status: AgentStatus
    ) -> None:
        """Repaint one agent row and exactly one of its task rows.

        Only one task of an agent runs at a time, so the agent row always shows
        the status of the task named here; earlier task rows keep the terminal
        status they finished with.

        Raises:
            KeyError: if the agent or the task is unknown. That is an internal
                inconsistency, not a condition to skip past.
        """
        managed = self._agents.get(agent_key)
        if managed is None:
            raise KeyError(f"Unknown agent key: {agent_key!r}")
        for index, task in enumerate(managed.tasks):
            if task.task_id == task_id:
                managed.tasks[index] = dataclasses.replace(task, status=status)
                break
        else:
            raise KeyError(f"Agent {agent_key!r} has no task {task_id!r}")
        self._host.update_agent_view(
            AgentSummary(
                agent_key, managed.agent_label, status, tuple(managed.tasks)
            )
        )

    def _worker_system_prompt(self) -> str:
        """The prompt every subagent completion carries."""
        return compose_worker_prompt(self._host.settings.effective_system_prompt())
