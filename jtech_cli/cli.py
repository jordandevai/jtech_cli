"""Composition root: parse args, wire dependencies, run the TUI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from jtech_cli import __version__
from jtech_cli.config import Settings, apply_default_prompt, build_settings, load_cmd_policy
from jtech_cli.server_info import ServerInfo, fetch_server_info
from jtech_cli.session import Session
from jtech_cli.theme import VALID_THEMES
from jtech_cli.tui import ChatApp
from jtech_cli.wizard import run_setup


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="jtech-cli", description="Full-screen LLM chat TUI")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--base-url", help="OpenAI-compatible base URL (overrides config file)")
    p.add_argument("--model", help="Model name to use (overrides config file)")
    p.add_argument("--instructions", type=Path, help="Load system prompt from a file")
    p.add_argument("--no-persist", action="store_true", help="Do not load/save history")
    p.add_argument("--history", type=Path, help="Override history file path")
    p.add_argument("--no-discover", action="store_true", help="Skip server model/context discovery")
    p.add_argument("--setup", action="store_true", help="Force the setup wizard")
    p.add_argument(
        "--theme",
        choices=list(VALID_THEMES),
        default="auto",
        help="UI theme: auto (detect terminal), light, or dark (default: auto)",
    )
    return p.parse_args(argv)


def make_settings(args: argparse.Namespace) -> Settings:
    settings = build_settings(base_url=args.base_url, model=args.model)
    if args.instructions:
        settings.system_prompt = args.instructions.read_text()
    return settings


def resolve_settings(args: argparse.Namespace, console: Console) -> Settings:
    """Return Settings, running the setup wizard when no base_url resolves (or --setup)."""
    settings = make_settings(args)
    if args.theme != "auto":
        settings.theme = args.theme
    if args.setup or not settings.base_url:
        return run_setup(console, default_url=settings.base_url or None, theme=settings.theme)
    return settings


def make_session(args: argparse.Namespace) -> Session:
    session = Session(args.history, persist=not args.no_persist)
    session.load()
    return session


def make_app(args: argparse.Namespace) -> ChatApp:
    """Build a fully wired ChatApp from parsed args."""
    console = Console()

    settings = resolve_settings(args, console)
    settings = apply_default_prompt(settings)
    session = make_session(args)

    # The [cmd] policy (AI shell execution) is loaded after the wizard so a
    # freshly written config is picked up; its mode is the source of truth.
    cmd = load_cmd_policy()
    settings.cmd_mode = cmd.mode

    server = ServerInfo()
    if not args.no_discover and settings.base_url:
        server = fetch_server_info(settings)
    if server.model:
        settings.model = server.model

    return ChatApp(
        settings=settings,
        session=session,
        server=server,
        cmd=cmd,
    )


def main(argv: list[str] | None = None) -> None:
    # Parse first so --help/--version work without a TTY (e.g. piped output).
    args = parse_args(argv)
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print(
            "jtech-cli needs an interactive terminal (a TTY for both input and output) "
            "to run the TUI. If you launched it from an IDE or piped input, run it "
            "directly in a terminal instead.",
            file=sys.stderr,
        )
        sys.exit(1)
    make_app(args).run()
