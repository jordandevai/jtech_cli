"""First-run setup wizard: configure the LLM endpoint and save it.

Uses plain input()/print (via rich) so it is safe over SSH/tmux. The wizard
tests the connection by querying /v1/models; on failure it loops back to the
URL prompt. On success it writes the config file.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from rich.console import Console

from jtech_cli.config import (
    CONFIG_PATH,
    Settings,
    build_settings,
    load_cmd_policy,
    save_settings,
)
from jtech_cli.server_info import ServerInfo, fetch_server_info

Ask = Callable[[str], str]


def _prompt_url(console: Console, ask: Ask, default_url: str) -> tuple[str, ServerInfo]:
    prompt = f"Enter server URL [{default_url}]: " if default_url else "Enter server URL: "
    while True:
        url = ask(prompt).strip()
        if not url and default_url:
            url = default_url
        if not url:
            console.print("[red]A server URL is required.[/red]")
            continue
        console.print("Testing connection...")
        info = fetch_server_info(Settings(base_url=url))
        if info.models:
            console.print(f"[green]Success![/green] Model found: {info.models[0]}")
            if info.context_length:
                console.print(f"  context_length = {info.context_length}")
            return url, info
        console.print("[red]Connection failed. Check the URL and try again.[/red]")


def _pick_model(console: Console, ask: Ask, models: list[str], default: str) -> str:
    if len(models) == 1:
        return models[0]
    console.print("Multiple models served. Choose one:")
    for m in models:
        console.print(f"  {m}")
    return ask(f"Model [{default}]: ").strip() or default


def run_setup(
    console: Console,
    *,
    ask: Ask = input,
    config_path: Path = CONFIG_PATH,
    default_url: str | None = None,
    theme: str = "auto",
) -> Settings:
    """Walk through setup, test the endpoint, and save the config file.

    Starts from the existing config file (when present) so re-runs keep the
    user's temperature, prompt source, and reasoning, then override url/model/theme.
    """
    settings = build_settings(config_path=config_path)
    if default_url is None:
        default_url = settings.base_url
    console.print("Welcome to JTech CLI! Let's set up your LLM.")
    console.print("Provider: OpenAI-compatible API (e.g. llama-server) — the only supported option.")
    console.print("The URL should look like: http://127.0.0.1:8080/v1")

    url, info = _prompt_url(console, ask, default_url)
    settings.base_url = url
    settings.model = _pick_model(console, ask, info.models, info.model or info.models[0])
    settings.theme = theme

    # Carry the existing [cmd] policy through untouched: re-running setup
    # re-points the endpoint, it does not reset the user's shell allowlist.
    save_settings(settings, config_path, cmd=load_cmd_policy(config_path))
    console.print(f"[green]Configuration saved to {config_path}[/green]")
    console.print(f"  base_url = {settings.base_url}")
    console.print(f"  model    = {settings.model}")
    return settings
