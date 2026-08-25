"""Conversation history persistence as JSONL.

Every message is appended to ~/.mycli/session.jsonl as its own record the
moment it is added, so the CLI survives crashes, detaches from tmux, and
process kills without ever rewriting the history it already stored.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from jtech_cli.configuration.paths import home_dir

DEFAULT_DIR = home_dir()
MODEL_ROLES = frozenset({"system", "user", "assistant"})


def default_history_path() -> Path:
    return DEFAULT_DIR / "session.jsonl"


class Session:
    def __init__(self, path: Path | None = None, *, persist: bool = True) -> None:
        self.path = path or default_history_path()
        self.persist = persist
        self.messages: list[dict] = []

    def load(self) -> None:
        self.messages = []
        if not self.persist:
            return
        if not self.path.exists():
            return
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                self.messages.append(msg)

    def add(
        self,
        role: str,
        content: str,
        *,
        include_in_context: bool = True,
        debug_only: bool = False,
        model_role: str | None = None,
        model_content: str | None = None,
    ) -> None:
        """Append a message with optional model-facing role/content overrides.

        The stored role/content control CLI presentation and diagnostics. The
        optional model fields allow runtime events to remain visible as
        ``system`` messages in the UI while using a model-compatible role and
        framing in the next request. Context-excluded messages remain in
        history but are filtered before requests are sent to the model.

        Persisting sessions append this one message to the JSONL file; earlier
        records are never re-read or rewritten. ``OSError`` propagates so the
        caller can report the failure, and the message stays in memory either
        way: a failed write is not a reason to lose the live conversation.
        """
        if model_role is not None and model_role not in MODEL_ROLES:
            raise ValueError(f"unsupported model role: {model_role}")
        if model_role is not None and model_content is None:
            raise ValueError("model_role requires model_content")
        if model_content is not None and model_role is None:
            raise ValueError("model_content requires model_role")
        msg = {"role": role, "content": content}
        if not include_in_context:
            msg["_include_in_context"] = False
        if debug_only:
            msg["_debug_only"] = True
        if model_role is not None:
            msg["_model_role"] = model_role
            msg["_model_content"] = model_content
        self.messages.append(msg)
        if not self.persist:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = json.dumps(msg, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(record + "\n")

    @contextmanager
    def ephemeral(self, role: str, content: str) -> Iterator[None]:
        """Hold a message in history for the body of the block, then strip it.

        For prompts that must reach the model on one request without joining
        the conversation. Removal is by identity, not equality, and runs even
        if the body raises: an equal message may survive from a crashed run,
        and the list can be emptied concurrently (``/clear``), which would make
        ``list.remove`` raise.

        The held message bypasses ``add()``, so it is never written to disk: an
        ephemeral prompt joins one request and nothing else. Messages added
        normally inside the block are persisted normally.
        """
        msg = {"role": role, "content": content}
        self.messages.append(msg)
        try:
            yield
        finally:
            for i, held in enumerate(self.messages):
                if held is msg:
                    del self.messages[i]
                    break

    def clear(self) -> None:
        """Drop the in-memory history, and the file too when persisting.

        With ``persist=False`` the file was never loaded and is not ours to
        delete: --no-persist promises not to touch stored history, so /clear
        in a throwaway session must not destroy the real one.
        """
        self.messages = []
        if not self.persist:
            return
        if self.path.exists():
            self.path.unlink()

    def messages_with_system(self, system_prompt: str) -> list[dict]:
        """Return model-visible history prefixed with the system prompt."""
        history = [
            {
                "role": message.get("_model_role", message["role"]),
                "content": message.get("_model_content", message["content"]),
            }
            for message in self.messages
            if message.get("_include_in_context", True)
        ]
        if not system_prompt:
            return history
        return [{"role": "system", "content": system_prompt}, *history]

    def stats(self) -> dict:
        chars = sum(len(m["content"]) for m in self.messages)
        return {"messages": len(self.messages), "chars": chars}
