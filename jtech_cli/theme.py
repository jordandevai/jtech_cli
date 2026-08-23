"""Theme resources and terminal light/dark resolution."""

from __future__ import annotations

import os

from textual.theme import Theme

from jtech_cli.resource_loader import ResourceError, load_toml_resource

THEME_FIELDS = (
    "name",
    "primary",
    "secondary",
    "warning",
    "error",
    "success",
    "accent",
    "foreground",
    "background",
    "surface",
    "panel",
    "dark",
)


class ThemeResourceError(RuntimeError):
    """Raised when a bundled theme resource is missing or invalid."""


def load_theme(filename: str) -> Theme:
    """Load one validated Textual theme from the bundled theme resources."""
    try:
        values = load_toml_resource(f"themes/{filename}")
    except ResourceError as error:
        raise ThemeResourceError(
            f"Theme resource {filename!r} could not be loaded"
        ) from error

    missing = [field for field in THEME_FIELDS if field not in values]
    unknown = sorted(set(values) - set(THEME_FIELDS))
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown fields: {', '.join(unknown)}")
        raise ThemeResourceError(
            f"Theme resource {filename!r} has invalid fields ({'; '.join(details)})"
        )

    try:
        return Theme(**{field: values[field] for field in THEME_FIELDS})
    except (TypeError, ValueError) as error:
        raise ThemeResourceError(
            f"Theme resource {filename!r} has invalid field values"
        ) from error

VALID_THEMES = ("auto", "light", "dark")

JTECH_DARK = load_theme("dark.toml")
JTECH_LIGHT = load_theme("light.toml")


def detect_theme() -> str:
    """Detect the terminal background from ``COLORFGBG``."""
    raw = os.environ.get("COLORFGBG", "")
    if not raw:
        return "dark"
    try:
        bg = raw.split(";")[-1].split(",")[-1]
        code = int(bg)
    except ValueError:
        return "dark"
    return "light" if code >= 8 else "dark"


def resolve_theme(choice: str) -> str:
    """Resolve an auto/light/dark choice to a concrete theme."""
    choice = (choice or "auto").strip().lower()
    if choice == "auto":
        return detect_theme()
    if choice in ("light", "dark"):
        return choice
    raise ValueError(f"Unknown theme: {choice!r} (expected auto, light, or dark)")


def textual_theme_name(choice: str) -> str:
    """Map a user choice to the registered Textual theme name."""
    resolved = resolve_theme(choice)
    return "jtech-light" if resolved == "light" else "jtech-dark"

__all__ = [
    "JTECH_DARK",
    "JTECH_LIGHT",
    "VALID_THEMES",
    "ThemeResourceError",
    "detect_theme",
    "load_theme",
    "resolve_theme",
    "textual_theme_name",
]
