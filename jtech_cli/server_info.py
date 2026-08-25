"""Server introspection: query an OpenAI-compatible endpoint for real info.

Transport failures are best-effort: on a network, HTTP, or response-shape
failure the relevant field stays unset and a safe message lands in
``ServerInfo.error``, so the CLI shows valid info when available and nothing
when it is not. No mock/fake data.

Credential failures are not best-effort. A profile that names a missing
environment variable raises ``ProfileError`` rather than being reported as an
unreachable server, and no request is made.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.request import Request, urlopen

from jtech_cli.config import Profile, resolve_api_key

TIMEOUT = 5


@dataclass
class ServerInfo:
    models: list[str] = field(default_factory=list)
    context_length: int | None = None
    #: Safe one-line reason discovery failed; never carries credential data.
    error: str | None = None

    @property
    def model(self) -> str | None:
        """The model name, when the server serves exactly one."""
        return self.models[0] if len(self.models) == 1 else None

    @property
    def known(self) -> bool:
        return bool(self.models) or self.context_length is not None


def _safe_error(error: Exception) -> str:
    """A one-line failure description with no request headers or key values."""
    return f"{type(error).__name__}: {error}"


def _auth_header(profile: Profile, environ: Mapping[str, str] | None) -> dict[str, str]:
    """Bearer header for an authenticated profile; nothing for a local one.

    Raises:
        ProfileError: if the profile names a variable that is unset or empty.
            No request is made in that case.
    """
    # ``None`` means "read the real environment now", so a variable exported
    # after import time is still seen; tests inject a mapping instead.
    resolved = resolve_api_key(profile, os.environ if environ is None else environ)
    if not profile.api_key_env:
        return {}
    return {"Authorization": f"Bearer {resolved}"}


def _request(url: str, *, headers: dict[str, str], data: bytes | None = None) -> dict:
    req = Request(url, data=data, headers={"Content-Type": "application/json", **headers})
    with urlopen(req, timeout=TIMEOUT) as resp:
        return json.load(resp)


def _base(profile: Profile) -> str:
    return profile.base_url.rstrip("/")


def fetch_server_info(
    profile: Profile,
    *,
    environ: Mapping[str, str] | None = None,
) -> ServerInfo:
    """Discover served model(s) and context length from ``profile``'s endpoint.

    Discovery deliberately does not resolve a model: it is what makes an empty
    ``Profile.model`` usable in the first place.

    Raises:
        ProfileError: if the profile's credential cannot be resolved.
    """
    headers = _auth_header(profile, environ)
    info = ServerInfo()
    try:
        data = _request(f"{_base(profile)}/models", headers=headers)
        info.models = [m["id"] for m in data.get("data", []) if m.get("id")]
    except Exception as error:  # noqa: BLE001 - best-effort discovery must never crash
        info.error = _safe_error(error)
        return info

    if info.models:
        try:
            meta = data["data"][0].get("meta", {})
            info.context_length = meta.get("n_ctx") or meta.get("llama.context_length")
        except (IndexError, AttributeError):
            pass
    return info


def fetch_token_count(
    profile: Profile,
    text: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> int | None:
    """Tokenize ``text`` via the server /tokenize endpoint.

    Returns ``None`` when the endpoint is unreachable or does not implement
    /tokenize — that is documented, optional server capability, not a failure.

    Raises:
        ProfileError: if the profile's credential cannot be resolved.
    """
    headers = _auth_header(profile, environ)
    body = json.dumps({"content": text}).encode()
    try:
        data = _request(f"{_base(profile)}/tokenize", headers=headers, data=body)
        return len(data.get("tokens", []))
    except Exception:  # noqa: BLE001 - best-effort token count must never crash
        return None
