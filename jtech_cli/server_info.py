"""Server introspection: query an OpenAI-compatible endpoint for real info.

All calls are best-effort: on any failure the relevant field stays unset, so the
CLI shows valid info when available and nothing when it is not. No mock/fake data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from urllib.parse import quote
from urllib.request import Request, urlopen

from jtech_cli.config import Settings

TIMEOUT = 5


@dataclass
class ServerInfo:
    models: list[str] = field(default_factory=list)
    context_length: int | None = None

    @property
    def model(self) -> str | None:
        """The model name, when the server serves exactly one."""
        return self.models[0] if len(self.models) == 1 else None

    @property
    def known(self) -> bool:
        return bool(self.models) or self.context_length is not None


def _request(url: str, *, data: bytes | None = None) -> dict:
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=TIMEOUT) as resp:
        return json.load(resp)


def _base(settings: Settings) -> str:
    return settings.base_url.rstrip("/")


def fetch_server_info(settings: Settings) -> ServerInfo:
    """Discover served model(s) and context length from the endpoint."""
    info = ServerInfo()
    try:
        data = _request(f"{_base(settings)}/models")
        info.models = [m["id"] for m in data.get("data", []) if m.get("id")]
    except Exception:  # noqa: BLE001 - best-effort discovery must never crash
        return info

    if info.models:
        try:
            meta = _request(f"{_base(settings)}/models/{quote(info.models[0], safe='')}").get(
                "meta", {}
            )
            info.context_length = meta.get("llama.context_length")
        except Exception:  # noqa: BLE001, S110 - context length is optional
            pass
    return info


def fetch_token_count(settings: Settings, text: str) -> int | None:
    """Tokenize text via the server /tokenize endpoint. None on failure."""
    body = json.dumps({"content": text}).encode()
    try:
        data = _request(f"{_base(settings)}/tokenize", data=body)
        return len(data.get("tokens", []))
    except Exception:  # noqa: BLE001 - best-effort token count must never crash
        return None
