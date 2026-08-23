"""User-data paths shared by configuration and session persistence."""

from __future__ import annotations

import os
from pathlib import Path


def home_dir() -> Path:
    """Base directory for user data (~/.mycli, overridable via $MYCLI_HOME)."""
    return Path(os.environ.get("MYCLI_HOME", "~/.mycli")).expanduser()


CONFIG_PATH = home_dir() / "config.toml"
