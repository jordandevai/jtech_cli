"""Themes: light/dark terminal background detection and Textual theme mapping.

The app maps the user's ``auto/light/dark`` theme choice onto the built-in
``jtech-dark``/``jtech-light`` Textual themes. The base theme is auto-detected
from the terminal's light/dark background so the CLI blends in with the user's
system theme; it can be overridden at runtime via ``--theme`` or the
``/settings``/``/theme`` commands.
"""

from __future__ import annotations

import os

from textual.theme import Theme

VALID_THEMES = ("auto", "light", "dark")

JTECH_DARK = Theme(
    name="jtech-dark",
    primary="#7aa2f7",
    secondary="#565f89",
    warning="#e0af68",
    error="#f7768e",
    success="#9ece6a",
    accent="#7aa2f7",
    foreground="#d7dae0",
    background="#1c1e21",
    surface="#262a30",
    panel="#3a4048",
    dark=True,
)
"""Calm-blue dark theme: blue accents, neutral surfaces, red reserved for errors."""

JTECH_LIGHT = Theme(
    name="jtech-light",
    primary="#3b6fd4",
    secondary="#6b7a99",
    warning="#b07d2a",
    error="#c74e5e",
    success="#3f8f5f",
    accent="#3b6fd4",
    foreground="#3a4250",
    background="#f2f4f8",
    surface="#e9edf3",
    panel="#c3cbd8",
    dark=False,
)
"""Calm-blue light theme: deeper blue for contrast on the light gray-blue base."""


def detect_theme() -> str:
    """Detect the terminal's light/dark background from ``COLORFGBG``.

    ``COLORFGBG`` is ``fg;bg`` (bg may be a comma list). A background of ``0``
    (black) implies dark text-on-dark; ``15`` (white) implies a light background.
    Falls back to ``dark`` when the variable is absent or unparsable.
    """
    raw = os.environ.get("COLORFGBG", "")
    if not raw:
        return "dark"
    try:
        bg = raw.split(";")[-1].split(",")[-1]
        code = int(bg)
    except ValueError:
        return "dark"
    # 0..7 dark palette -> dark background; 8..15 light palette -> light background
    return "light" if code >= 8 else "dark"


def resolve_theme(choice: str) -> str:
    """Resolve a theme choice (auto/light/dark) to a concrete light/dark value."""
    choice = (choice or "auto").strip().lower()
    if choice == "auto":
        return detect_theme()
    if choice in ("light", "dark"):
        return choice
    raise ValueError(f"Unknown theme: {choice!r} (expected auto, light, or dark)")


def textual_theme_name(choice: str) -> str:
    """Map an auto/light/dark theme choice to a jtech Textual theme name."""
    resolved = resolve_theme(choice)
    return "jtech-light" if resolved == "light" else "jtech-dark"
