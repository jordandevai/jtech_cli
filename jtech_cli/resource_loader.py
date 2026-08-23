"""Validated access to static resources shipped with the application."""

from __future__ import annotations

import tomllib
from functools import cache
from importlib.resources import files
from pathlib import PurePosixPath


class ResourceError(RuntimeError):
    """Raised when a bundled resource is missing or cannot be parsed."""


def _resource(relative_path: str):
    path = PurePosixPath(relative_path)
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise ResourceError(f"Invalid bundled resource path: {relative_path!r}")
    return files("jtech_cli").joinpath("resources", *path.parts)


@cache
def load_text_resource(relative_path: str) -> str:
    """Read one non-empty-safe text resource by its resource-relative path."""
    resource = _resource(relative_path)
    try:
        return resource.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError) as error:
        raise ResourceError(
            f"Bundled resource {relative_path!r} could not be read"
        ) from error


@cache
def load_toml_resource(relative_path: str) -> dict[str, object]:
    """Read and parse one TOML resource, preserving parse errors as context."""
    source = load_text_resource(relative_path)
    try:
        values = tomllib.loads(source)
    except tomllib.TOMLDecodeError as error:
        raise ResourceError(
            f"Bundled resource {relative_path!r} contains invalid TOML"
        ) from error
    if not isinstance(values, dict):
        raise ResourceError(f"TOML resource {relative_path!r} must be a table")
    return values
