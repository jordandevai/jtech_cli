"""What the app says about the Primary conversation.

One object owns the three places a Primary turn becomes visible: the
transcript the user reads, the session file it is recorded into, and the
footer summarizing both. They are together because they change together —
clearing the chat drops the token count and repaints the footer in the same
step — and apart from the app because none of it is lifecycle or event
handling.
"""

from __future__ import annotations

from typing import Protocol

from textual.widgets import Static

from jtech_cli.config import Settings
from jtech_cli.server_info import ServerInfo
from jtech_cli.session import Session
from jtech_cli.tui_widgets import Transcript, TranscriptRecord


class StatusHost(Protocol):
    """The app surface one status view reads and writes through."""

    settings: Settings
    server: ServerInfo
    session: Session

    def primary_transcript(self) -> Transcript:
        """Primary's own activity stream."""
        ...

    def status_bar(self) -> Static:
        """The footer widget."""
        ...

    def queue_depth(self) -> int:
        """How many messages are waiting behind the accepted turn."""
        ...


class StatusView:
    """Primary's transcript, its stored record, and the footer over them."""

    def __init__(self, host: StatusHost) -> None:
        self._host = host
        self.prompt_tokens = 0

    def render(self, running: str | None = None) -> None:
        """Repaint the footer from live settings, server, and queue state."""
        host = self._host
        parts: list[str] = []
        if running:
            parts.append(running)
        profile = host.settings.active_profile
        if profile is not None:
            suffix = " (override)" if host.settings.profile_is_overridden else ""
            parts.append(f"profile: {profile.name}{suffix}")
            parts.append(profile.base_url)
            model = profile.model or host.server.model
            if model:
                parts.append(f"model: {model}")
        if host.server.context_length:
            if self.prompt_tokens:
                total = host.server.context_length
                remaining_pct = max(0, (1 - self.prompt_tokens / total) * 100)
                parts.append(
                    f"ctx {self.prompt_tokens // 1000}k"
                    f"/{total // 1000}k ({remaining_pct:.0f}% left)"
                )
            else:
                parts.append(f"ctx {host.server.context_length}")
        depth = host.queue_depth()
        if depth:
            parts.append(f"queue: {depth}")
        host.status_bar().update("  ·  ".join(parts))

    def push(self, role: str, text: str) -> None:
        """Add one already-complete message to the visible transcript."""
        self._host.primary_transcript().append(
            TranscriptRecord.from_message(role=role, content=text)
        )

    def record(
        self,
        role: str,
        content: str,
        *,
        include_in_context: bool = True,
        debug_only: bool = False,
        model_role: str | None = None,
        model_content: str | None = None,
    ) -> None:
        """Persist one message, surfacing an I/O failure in the transcript.

        Presenting the failure is UI work, so it lives here rather than in
        ``Session``. The message stays in memory when the append fails — a
        save failure is worth saying out loud, but it is not a reason to drop
        the live conversation — and the warning never joins model context.
        """
        try:
            self._host.session.add(
                role,
                content,
                include_in_context=include_in_context,
                debug_only=debug_only,
                model_role=model_role,
                model_content=model_content,
            )
        except OSError as error:
            self.push("system", f"Could not save history: {error}")

    def clear(self) -> None:
        """Drop the visible conversation and the count that described it."""
        self._host.primary_transcript().clear()
        self.prompt_tokens = 0
        self.render()
