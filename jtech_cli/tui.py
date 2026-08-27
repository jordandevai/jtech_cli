"""Public TUI entry point and backwards-compatible dependency seam.

The implementation is split across widgets, modal screens, the app, and an
external stylesheet. This module intentionally keeps the historical imports
stable for the CLI, integrations, and tests.
"""

from __future__ import annotations

from pathlib import Path

from jtech_cli import server_info
from jtech_cli.cmd_tools import CmdPolicy
from jtech_cli.config import CONFIG_PATH, Profile, ResolvedProfile, Settings
from jtech_cli.llm_client import stream_reply
from jtech_cli.server_info import ServerInfo, fetch_server_info
from jtech_cli.session import Session
from jtech_cli.tui_app import (
    CONNECTION_ERROR,
    SPINNER_FRAMES,
    FetchServerInfo,
    FetchTokenCount,
    StreamReply,
)
from jtech_cli.tui_app import (
    ChatApp as _ChatApp,
)
from jtech_cli.tui_screens import (
    CmdChoice,
    CommandPrompt,
    ProfilesScreen,
    QuitScreen,
    SettingsScreen,
)
from jtech_cli.tui_widgets import (
    FieldCancel,
    FieldCommit,
    InputToMultiline,
    MultilineCancel,
    MultilineSubmit,
    OutputSink,
    render_menu_rows,
)


def _stream_reply_compat(
    profile: ResolvedProfile, temperature: float, messages: list[dict]
):
    """Late-bound stream seam retained for existing callers and tests."""
    yield from stream_reply(profile, temperature, messages)


def _fetch_server_info_compat(profile: Profile) -> ServerInfo:
    """Late-bound discovery seam retained for existing callers and tests."""
    return fetch_server_info(profile)


def _fetch_token_count_compat(profile: Profile, text: str) -> int | None:
    """Late-bound token-count seam retained for existing callers and tests."""
    return server_info.fetch_token_count(profile, text)


class ChatApp(_ChatApp):
    """Compatibility wrapper that injects this module's patchable boundaries."""

    def __init__(
        self,
        *,
        settings: Settings,
        session: Session,
        server: ServerInfo,
        config_path: Path = CONFIG_PATH,
        cmd: CmdPolicy | None = None,
        no_discover: bool = False,
        stream_reply_fn: StreamReply | None = None,
        fetch_server_info_fn: FetchServerInfo | None = None,
        fetch_token_count_fn: FetchTokenCount | None = None,
    ) -> None:
        super().__init__(
            settings=settings,
            session=session,
            server=server,
            config_path=config_path,
            cmd=cmd,
            no_discover=no_discover,
            stream_reply_fn=(
                _stream_reply_compat if stream_reply_fn is None else stream_reply_fn
            ),
            fetch_server_info_fn=(
                _fetch_server_info_compat
                if fetch_server_info_fn is None
                else fetch_server_info_fn
            ),
            fetch_token_count_fn=(
                _fetch_token_count_compat
                if fetch_token_count_fn is None
                else fetch_token_count_fn
            ),
        )


__all__ = [
    "CONNECTION_ERROR",
    "SPINNER_FRAMES",
    "ChatApp",
    "CmdChoice",
    "CommandPrompt",
    "FieldCancel",
    "FieldCommit",
    "InputToMultiline",
    "MultilineCancel",
    "MultilineSubmit",
    "OutputSink",
    "ProfilesScreen",
    "QuitScreen",
    "SettingsScreen",
    "fetch_server_info",
    "render_menu_rows",
    "server_info",
    "stream_reply",
]
