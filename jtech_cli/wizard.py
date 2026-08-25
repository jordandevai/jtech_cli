"""First-run setup wizard: configure one LLM profile and save it.

Uses plain input()/print (via rich) so it is safe over SSH/tmux, and runs
before the Textual app exists — so it shares the profile domain with the TUI
modal, not its widgets. The wizard tests the connection by querying
/v1/models; a credential failure returns to the API-key field and a connection
failure returns to the URL field. Nothing is written until one complete,
reachable profile has been collected.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from rich.console import Console

from jtech_cli.config import (
    CONFIG_PATH,
    DEFAULT_PROFILE_NAME,
    Profile,
    ProfileError,
    Settings,
    build_settings,
    load_cmd_policy,
    save_settings,
)
from jtech_cli.server_info import ServerInfo, fetch_server_info

Ask = Callable[[str], str]


def _prompt_key_env(ask: Ask, default_key_env: str) -> str:
    """Ask which environment variable supplies the API key.

    The secret itself is never requested, echoed, or stored: only the variable
    name is kept, and it is read from the environment at request time. A blank
    answer keeps the current value, or means "no authentication" when there is
    none — the same convention the URL field uses.
    """
    if default_key_env:
        prompt = f"Environment variable holding the API key [{default_key_env}]: "
    else:
        prompt = (
            "Environment variable holding the API key "
            "(blank for a local server with no auth): "
        )
    return ask(prompt).strip() or default_key_env


def _prompt_url(console: Console, ask: Ask, default_url: str) -> str:
    """Ask for the endpoint URL, repeating until a non-empty value is given."""
    prompt = f"Enter server URL [{default_url}]: " if default_url else "Enter server URL: "
    while True:
        url = ask(prompt).strip()
        if url:
            return url
        if default_url:
            return default_url
        console.print("[red]A server URL is required.[/red]")


def _prompt_endpoint(
    console: Console,
    ask: Ask,
    name: str,
    default_url: str,
    default_key_env: str,
    environ: Mapping[str, str] | None,
) -> tuple[Profile, ServerInfo]:
    """Collect a credential source and URL, probing until the endpoint answers.

    The endpoint is validated in two steps so a failure can name the field that
    caused it: the URL alone first, then the credential variable. Both steps use
    ``Profile``'s own rules rather than re-implementing them here.
    """
    key_env = default_key_env
    # ``None`` until a URL has been collected, which is what lets a credential
    # retry go straight back to probing instead of re-asking the whole form.
    url: str | None = None
    # The field the next iteration collects. A failure sends the user back to
    # the field that caused it, not to the top of the form.
    field = "key_env"
    while True:
        if field == "key_env":
            key_env = _prompt_key_env(ask, key_env)
            field = "url" if url is None else "probe"
            continue
        if field == "url":
            url = _prompt_url(console, ask, default_url if url is None else url)
            field = "probe"
            continue
        try:
            Profile(name=name, base_url=url)
        except ProfileError as error:
            console.print(f"[red]{error}[/red]")
            # Do not re-offer a rejected value as the next default.
            url = None
            field = "url"
            continue
        try:
            profile = Profile(name=name, base_url=url, api_key_env=key_env)
        except ProfileError as error:
            console.print(f"[red]{error}[/red]")
            field = "key_env"
            continue
        console.print("Testing connection...")
        try:
            info = fetch_server_info(profile, environ=environ)
        except ProfileError as error:
            console.print(f"[red]{error}[/red]")
            field = "key_env"  # the named variable is unset or empty
            continue
        if info.models:
            console.print(f"[green]Success![/green] Model found: {info.models[0]}")
            if info.context_length:
                console.print(f"  context_length = {info.context_length}")
            return profile, info
        detail = f" ({info.error})" if info.error else ""
        console.print(
            f"[red]Connection failed{detail}. Check the URL and try again.[/red]"
        )
        # A reachable-looking URL that did not answer stays as the default: the
        # server may simply not be up yet.
        field = "url"


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
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """Walk through setup, test the endpoint, and save the config file.

    Starts from the existing config (when present) so re-runs keep the user's
    temperature, prompt source, reasoning, and the complete shell policy. The
    edited profile is the active one, or ``default`` on first run; it is
    replaced (or added) and activated in one save.

    Raises:
        ConfigurationError: if the existing config cannot be read.
        OSError: if the new config cannot be written.
    """
    settings = build_settings(config_path=config_path)
    target = settings.profiles.active
    name = target.name if target is not None else DEFAULT_PROFILE_NAME
    if default_url is None:
        default_url = target.base_url if target is not None else ""
    default_key_env = target.api_key_env if target is not None else ""

    console.print("Welcome to JTech CLI! Let's set up your LLM.")
    console.print("Provider: OpenAI-compatible API (e.g. llama-server) — the only supported option.")
    console.print("The URL should look like: http://127.0.0.1:8080/v1")
    console.print(f"Configuring profile: {name}")

    probed, info = _prompt_endpoint(
        console, ask, name, default_url, default_key_env, environ
    )
    model = _pick_model(console, ask, info.models, info.model or info.models[0])
    profile = Profile(
        name=name,
        base_url=probed.base_url,
        model=model,
        api_key_env=probed.api_key_env,
    )

    catalog = settings.profiles
    if name in catalog.names:
        catalog = catalog.replace(name, profile)
    else:
        catalog = catalog.add(profile)
    settings.profiles = catalog.activate(name)
    settings.theme = theme

    # Carry the existing [cmd] policy through untouched: re-running setup
    # re-points the endpoint, it does not reset the user's shell allowlist.
    save_settings(settings, config_path, cmd=load_cmd_policy(config_path))
    console.print(f"[green]Configuration saved to {config_path}[/green]")
    console.print(f"  profile  = {profile.name}")
    console.print(f"  base_url = {profile.base_url}")
    console.print(f"  model    = {profile.model}")
    if profile.api_key_env:
        console.print(f"  api_key  = ${profile.api_key_env}")
    return settings
