"""Slash-command registry and handlers.

Commands are registered by name so dispatch is a lookup, not a growing
if/elif chain. Handlers receive a CommandContext holding their dependencies
(session, settings, console) rather than reaching for globals. Handlers may be
sync or async: async handlers (e.g. /write, /stats) return a coroutine that the
caller runs — the TUI runs it as a worker so the event loop never blocks.
"""

from __future__ import annotations

import asyncio
import shlex
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

from jtech_cli import file_tools, server_info
from jtech_cli.cmd_tools import CmdPolicy
from jtech_cli.config import (
    CONFIG_PATH,
    SETTABLE_KEYS,
    ProfileError,
    Settings,
    load_cmd_policy,
    save_settings,
)
from jtech_cli.prompts import INSTRUCTIONS_HELP, PromptSourceError
from jtech_cli.session import Session
from jtech_cli.theme import VALID_THEMES

EnterMultiline = Callable[[str], Awaitable[str]]
SwitchProfile = Callable[[str], Awaitable[None]]
Handler = Callable[["CommandContext", str], None | Awaitable[None]]

WRITE_USAGE = "Usage: /write PATH  then paste content, end with a line containing only: END"
NO_PROFILE = "No API profile is configured — run /profiles to add one."


async def _no_multiline(_terminator: str) -> str:
    """Default multi-line reader: no editor available (standalone/test contexts)."""
    return ""


async def _no_profile_switch(_name: str) -> None:
    """Default profile switch: no running app to switch in (standalone/tests)."""


@dataclass
class CommandContext:
    session: Session
    settings: Settings
    console: Console
    enter_multiline: EnterMultiline = _no_multiline
    server: server_info.ServerInfo = field(default_factory=server_info.ServerInfo)
    cmd: CmdPolicy | None = None
    last_reply: str = ""
    config_path: Path = field(default_factory=lambda: CONFIG_PATH)
    refresh_footer: Callable[[], None] = field(default=lambda: None)
    open_settings: Callable[[], None] = field(default=lambda: None)
    clear_chat: Callable[[], None] = field(default=lambda: None)
    switch_theme: Callable[[], None] = field(default=lambda: None)
    open_profiles: Callable[[], None] = field(default=lambda: None)
    switch_profile: SwitchProfile = _no_profile_switch

    def persist_settings(self) -> None:
        try:
            cmd = self.cmd
            if cmd is None:
                # No policy in hand: carry the file's existing allow/timeout
                # through, so a settings write never drops the [cmd] table.
                cmd = load_cmd_policy(self.config_path)
            # The live setting is the source of truth for the mode either way.
            cmd.mode = self.settings.cmd_mode
            save_settings(self.settings, self.config_path, cmd=cmd)
            self.console.print(f"[dim]Saved settings to {self.config_path}[/dim]")
        except OSError as e:
            self.console.print(f"[yellow]Could not save settings:[/yellow] {e}")


def _parse_arg(raw: str) -> tuple[str, str]:
    """Split a raw slash line into (lowercase command, stripped arg)."""
    cmd, _, arg = raw.partition(" ")
    return cmd.strip().lstrip("/").lower(), arg.strip()


class CommandRegistry:
    def __init__(self, ctx: CommandContext) -> None:
        self._ctx = ctx
        self._commands: dict[str, tuple[Handler, str]] = {}

    def register(self, name: str, handler: Handler, help_text: str = "") -> None:
        self._commands[name] = (handler, help_text)

    def names(self) -> Iterator[str]:
        return iter(self._commands)

    def completions(self, raw: str) -> list[tuple[str, str]]:
        """(name, help) pairs, in name order, for every command the input starts."""
        if " " in raw:
            return []
        prefix = raw.lstrip("/").lower()
        return [
            (name, help_text)
            for name, (_, help_text) in sorted(self._commands.items())
            if name.startswith(prefix)
        ]

    def handle(self, raw: str) -> None | Awaitable[None]:
        """Dispatch a raw slash line; returns a coroutine for async handlers."""
        cmd, arg = _parse_arg(raw)
        entry = self._commands.get(cmd)
        if entry is None:
            self._ctx.console.print(f"[red]Unknown command:[/red] {cmd}  (try /help)")
            return None
        return entry[0](self._ctx, arg)


def build_registry(ctx: CommandContext) -> CommandRegistry:
    reg = CommandRegistry(ctx)
    c = ctx.console

    def cmd_exit(_: CommandContext, __: str) -> None:
        c.print("Bye.")
        raise SystemExit(0)

    def cmd_help(_: CommandContext, __: str) -> None:
        c.print(INSTRUCTIONS_HELP)

    def cmd_clear(_: CommandContext, __: str) -> None:
        ctx.session.clear()
        ctx.clear_chat()
        c.print("History cleared.")

    def cmd_read(_: CommandContext, arg: str) -> None:
        c.print(file_tools.cmd_read(arg))

    async def cmd_write(_: CommandContext, arg: str) -> None:
        if not arg:
            c.print(WRITE_USAGE)
            return
        c.print("Paste content, then end with a line containing only: END")
        content = await ctx.enter_multiline("END")
        c.print(file_tools.cmd_write(arg, content))

    def cmd_diff(_: CommandContext, arg: str) -> None:
        c.print(file_tools.cmd_diff(arg))

    def cmd_set(_: CommandContext, arg: str) -> None:
        parts = shlex.split(arg)
        if len(parts) < 2:
            c.print(
                "Usage: /set KEY VALUE  e.g. /set theme light   "
                f"(KEY: {', '.join(SETTABLE_KEYS)} — or run /settings for the menu)"
            )
            return
        key, value = parts[0], parts[1]
        try:
            ctx.settings.set(key, value)
        except ValueError as e:
            c.print(f"[red]{e}[/red]")
            return
        c.print(f"Set {key} = {value}")
        if key == "theme":
            ctx.switch_theme()
        if key in SETTABLE_KEYS:
            ctx.persist_settings()
            ctx.refresh_footer()

    def cmd_settings(_: CommandContext, __: str) -> None:
        ctx.open_settings()

    def cmd_profiles(_: CommandContext, __: str) -> None:
        ctx.open_profiles()

    async def cmd_profile(_: CommandContext, arg: str) -> None:
        """Activate one named profile, delegating the switch to the app.

        Dispatch stays free of settings, storage, network, and widget work: the
        app owns when a switch is allowed and what it invalidates.
        """
        name = arg.strip()
        if not name:
            active = ctx.settings.active_profile
            current = f"{active.name} ({active.base_url})" if active else "none"
            configured = ", ".join(ctx.settings.profiles.names) or "none"
            c.print(
                f"Usage: /profile NAME   ·   current: {current}   ·   "
                f"configured: {configured}"
            )
            return
        await ctx.switch_profile(name)

    def cmd_theme(_: CommandContext, arg: str) -> None:
        choice = arg.strip().lower()
        if choice not in VALID_THEMES:
            order = list(VALID_THEMES)
            cur = ctx.settings.theme if ctx.settings.theme in order else order[0]
            choice = order[(order.index(cur) + 1) % len(order)]
        try:
            ctx.settings.set("theme", choice)
        except ValueError as e:
            c.print(f"[red]{e}[/red]")
            return
        ctx.switch_theme()
        ctx.persist_settings()
        ctx.refresh_footer()
        c.print(f"Theme set to {choice}")

    def cmd_system(_: CommandContext, __: str) -> None:
        c.print(f"Prompt source: {ctx.settings.prompt_source}")
        if ctx.settings.prompt_notice:
            c.print(ctx.settings.prompt_notice)
        c.print(ctx.settings.effective_system_prompt())

    def cmd_prompt(_: CommandContext, arg: str) -> None:
        if not arg:
            c.print("Usage: /prompt FILE | /prompt reload | /prompt reset")
            return
        action = arg.strip().lower()
        try:
            if action == "reset":
                ctx.settings.reset_prompt()
                ctx.persist_settings()
                c.print("Reset to the bundled runtime prompt.")
            elif action == "reload":
                ctx.settings.reload_prompt()
                ctx.persist_settings()
                c.print(f"Reloaded prompt file {ctx.settings.prompt_file}")
            else:
                ctx.settings.set_prompt_file(Path(arg))
                ctx.persist_settings()
                c.print(f"Loaded prompt file {ctx.settings.prompt_file}")
        except PromptSourceError as error:
            c.print(f"[red]{error}[/red]")

    def cmd_models(_: CommandContext, __: str) -> None:
        if not ctx.server.models:
            c.print("[dim]No model info available (server unreachable or not discovered).[/dim]")
            return
        c.print("Models on server:")
        for m in ctx.server.models:
            c.print(f"  {m}")
        if ctx.server.context_length:
            c.print(f"context_length = {ctx.server.context_length}")

    async def cmd_stats(_: CommandContext, __: str) -> None:
        """History size, tokens, context usage (token fetch runs in a thread)."""
        s = ctx.session.stats()
        lines = [f"messages={s['messages']} chars={s['chars']}"]

        text = " ".join(
            m["content"] for m in ctx.session.messages_with_system("")
        )
        total: int | None = 0
        # A missing profile or credential is reported, never folded into an
        # empty token count that would read as "the server said zero".
        note: str | None = None
        if text:
            profile = ctx.settings.active_profile
            if profile is None:
                total, note = None, NO_PROFILE
            else:
                try:
                    total = await asyncio.to_thread(
                        server_info.fetch_token_count, profile, text
                    )
                except ProfileError as error:
                    total, note = None, str(error)
        if total is not None:
            lines.append(f"history_tokens={total}")
        if ctx.server.context_length:
            ctx_length = ctx.server.context_length
            lines.append(f"context_length={ctx_length}")
            if total is not None:
                lines.append(f"context_remaining={ctx_length - total}")

        c.print("  ".join(lines))
        if note:
            c.print(f"[yellow]{note}[/yellow]")

    def cmd_render(_: CommandContext, __: str) -> None:
        if not ctx.last_reply:
            c.print("[dim]No reply to render yet.[/dim]")
            return
        # Hand over the text, not a renderable: the console decides how to
        # present it, and the TUI's sink re-renders it as a Markdown bubble.
        c.print(ctx.last_reply)

    for name, handler, help_text in [
        ("exit", cmd_exit, "Quit"),
        ("help", cmd_help, "Show help"),
        ("clear", cmd_clear, "Clear history"),
        ("read", cmd_read, "Print a file with line numbers (PATH[:LINE])"),
        ("write", cmd_write, "Write pasted content to a file"),
        ("diff", cmd_diff, "Create a temp copy of a file to diff against"),
        ("set", cmd_set, f"Change {', '.join(SETTABLE_KEYS)}"),
        ("settings", cmd_settings, "Interactive menu to edit settings"),
        ("profiles", cmd_profiles, "Manage API profiles: add, edit, activate, delete"),
        ("profile", cmd_profile, "Activate a named API profile: /profile NAME"),
        ("theme", cmd_theme, f"Switch theme: {' / '.join(VALID_THEMES)}"),
        ("system", cmd_system, "Show current system prompt"),
        ("prompt", cmd_prompt, "Load a system prompt from a file"),
        ("models", cmd_models, "List models served by the endpoint"),
        ("stats", cmd_stats, "Show history size, tokens, context usage"),
        ("render", cmd_render, "Re-render last reply as Markdown"),
    ]:
        reg.register(name, handler, help_text)

    return reg
