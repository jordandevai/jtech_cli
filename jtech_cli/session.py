"""Conversation history persistence as JSONL.

History is written to ~/.mycli/session.jsonl after every exchange so the CLI
survives crashes, detaches from tmux, and process kills.
"""

from __future__ import annotations

import json
from pathlib import Path

from jtech_cli.config import home_dir

DEFAULT_DIR = home_dir()


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

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def save(self) -> None:
        if not self.persist:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".jsonl.tmp")
        with tmp.open("w") as fh:
            for msg in self.messages:
                fh.write(json.dumps(msg, ensure_ascii=False) + "\n")
        tmp.replace(self.path)

    def clear(self) -> None:
        self.messages = []
        if self.path.exists():
            self.path.unlink()

    def messages_with_system(self, system_prompt: str) -> list[dict]:
        """Return history prefixed with the system prompt (if any)."""
        if not system_prompt:
            return list(self.messages)
        return [{"role": "system", "content": system_prompt}, *self.messages]

    def stats(self) -> dict:
        chars = sum(len(m["content"]) for m in self.messages)
        return {"messages": len(self.messages), "chars": chars}
