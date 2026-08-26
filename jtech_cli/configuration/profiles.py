"""Profile identity, collection invariants, and credential resolution.

A profile owns exactly the connection identity needed to reach one
OpenAI-compatible endpoint: ``name + base_url + model + api_key_env``. It is
deliberately separate from global UI settings and from TOML serialization —
those have different reasons to change, and only this module decides what a
valid profile is.

Nothing here reads the process environment or the filesystem: the environment
mapping is injected, so credential rules are testable without real secrets. A
resolved key is carried in :class:`ResolvedProfile` with ``repr=False`` and is
never placed in an error message.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from urllib.parse import ParseResult, urlparse

# The OpenAI SDK requires a non-empty key even for an unauthenticated local
# server; this is the placeholder the client has always sent to llama-server.
LOCAL_API_KEY = "none"

#: Name given to a migrated legacy endpoint and to the first profile setup creates.
DEFAULT_PROFILE_NAME = "default"
#: Name of the session-only profile built from --base-url with an empty catalog.
CLI_PROFILE_NAME = "cli"

_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")
_ENV_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

#: Ports that need no mention. An omitted port means the scheme's default, so a
#: credential's scope must not split on whether the user typed ":443".
_DEFAULT_PORTS = {"http": 80, "https": 443}


class ProfileError(ValueError):
    """A profile is invalid, unavailable, or cannot be resolved."""


def _check_str(field_name: str, value: object) -> str:
    if not isinstance(value, str):
        raise ProfileError(f"{field_name} must be a string, got {type(value).__name__}")
    return value


def _check_name(value: object) -> str:
    name = _check_str("name", value)
    if not _NAME_PATTERN.fullmatch(name):
        raise ProfileError(
            f"profile name {name!r} is invalid: use lowercase letters, digits, "
            "'-' or '_', starting with a letter or digit"
        )
    return name


def _parse_endpoint(url: str) -> ParseResult:
    """``urlparse`` the endpoint, typing its failures as ``ProfileError``.

    ``urlparse`` raises bare ``ValueError`` on inputs like ``http://[::1/v1``.
    Letting that escape would bypass the CLI and profile-editor handlers, which
    catch ``ProfileError``, and surface a traceback instead of a fixable message.

    Raises:
        ProfileError: if the URL cannot be parsed at all.
    """
    try:
        return urlparse(url)
    except ValueError as error:
        raise ProfileError(f"base_url is not a usable URL ({error}): {url!r}") from error


def _check_base_url(name: str, value: object) -> str:
    url = _check_str("base_url", value)
    if not url:
        raise ProfileError(f"profile {name!r}: base_url is required")
    if url != url.strip():
        raise ProfileError(
            f"profile {name!r}: base_url must not have leading or trailing whitespace"
        )
    try:
        parsed = _parse_endpoint(url)
        if parsed.username or parsed.password:
            # A userinfo URL would put a live secret in config.toml, the footer,
            # and the profile manager. api_key_env is the only way a credential
            # enters.
            raise ProfileError(
                "base_url must not embed credentials (user:password@host) — "
                "supply the key through api_key_env instead"
            )
        # Scheme, host, and port must all be usable — the same rule that decides
        # credential scope, so a URL can never be valid yet have no origin.
        endpoint_origin(url)
    except ProfileError as error:
        raise ProfileError(f"profile {name!r}: {error}") from error
    return url


def _check_model(name: str, value: object) -> str:
    model = _check_str("model", value)
    if model != model.strip():
        raise ProfileError(
            f"profile {name!r}: model must not have leading or trailing whitespace"
        )
    return model


def _check_api_key_env(name: str, value: object) -> str:
    env = _check_str("api_key_env", value)
    if env and not _ENV_PATTERN.fullmatch(env):
        raise ProfileError(
            f"profile {name!r}: api_key_env must be an environment variable name "
            f"([A-Za-z_][A-Za-z0-9_]*), got {env!r}"
        )
    return env


def endpoint_origin(url: str) -> tuple[str, str, int]:
    """Scheme, host, and effective port of ``url`` — a credential's scope.

    Two endpoints share an origin when they are the same server, so a key issued
    for one is meaningful for the other. The port is resolved to its effective
    value, so ``https://h/v1`` and ``https://h:443/v1`` are one origin rather
    than two — comparing raw host:port text would refuse a legitimate override.
    ``urlparse`` normalizes scheme and host to lowercase.

    Raises:
        ProfileError: if the URL has no usable scheme, host, or port.
    """
    parsed = _parse_endpoint(url)
    try:
        port = parsed.port
    except ValueError as error:
        raise ProfileError(f"base_url has an invalid port: {url!r}") from error
    if parsed.scheme not in _DEFAULT_PORTS or not parsed.hostname:
        raise ProfileError(
            "base_url must be an absolute http:// or https:// URL with a host, "
            f"got {url!r}"
        )
    if port == 0:
        # Port 0 cannot be connected to. Treating it as "unspecified" would fold
        # it into the default port and silently widen a credential's scope.
        raise ProfileError(f"base_url has an unusable port 0: {url!r}")
    # Explicitly against None: 0 is falsy, so `port or default` would map an
    # explicit port onto the default one.
    effective = port if port is not None else _DEFAULT_PORTS[parsed.scheme]
    return parsed.scheme, parsed.hostname, effective


@dataclass(frozen=True, slots=True)
class Profile:
    """One named OpenAI-compatible endpoint identity.

    ``model`` may be empty to keep today's single-model auto-discovery.
    ``api_key_env`` names the environment variable holding the bearer token and
    is empty for an unauthenticated local server — the key value itself is never
    stored on a profile.

    Raises:
        ProfileError: if any field violates the profile rules.
    """

    name: str
    base_url: str
    model: str = ""
    api_key_env: str = ""

    def __post_init__(self) -> None:
        # Name first: the other messages identify the profile by it.
        object.__setattr__(self, "name", _check_name(self.name))
        object.__setattr__(self, "base_url", _check_base_url(self.name, self.base_url))
        object.__setattr__(self, "model", _check_model(self.name, self.model))
        object.__setattr__(
            self, "api_key_env", _check_api_key_env(self.name, self.api_key_env)
        )


@dataclass(frozen=True, slots=True)
class ResolvedProfile:
    """One profile resolved for use: a concrete model and a usable API key.

    This is the value a whole user turn is pinned to. ``api_key`` is excluded
    from ``repr`` so it cannot reach a log, a traceback, or a debug dump.
    """

    name: str
    base_url: str
    model: str
    api_key: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class Profiles:
    """An immutable catalog of profiles plus the selected name.

    Every mutation returns a new value; the live catalog is replaced only after
    the candidate has been persisted, so a failed save needs no rollback.

    Raises:
        ProfileError: on duplicate names, a non-``Profile`` item, or an
            ``active_name`` that does not identify an item.
    """

    items: tuple[Profile, ...] = ()
    active_name: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.items, (str, bytes)) or not isinstance(self.items, Sequence):
            raise ProfileError("profiles must be a sequence of Profile values")
        # Normalize at the boundary: a list would quietly defeat immutability.
        items = tuple(self.items)
        for item in items:
            if not isinstance(item, Profile):
                raise ProfileError(
                    f"profiles must contain Profile values, got {type(item).__name__}"
                )
        names = [item.name for item in items]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ProfileError(f"duplicate profile name(s): {', '.join(duplicates)}")
        object.__setattr__(self, "items", items)
        if self.active_name is None:
            if names:
                # A stored catalog with no selection cannot be loaded back, so
                # it must never be constructible in the first place.
                raise ProfileError(
                    "a profile catalog with profiles must select an active one"
                )
            return
        active = _check_str("active_profile", self.active_name)
        if active not in names:
            raise ProfileError(f"active profile {active!r} is not a configured profile")

    @property
    def active(self) -> Profile | None:
        """The selected profile, or ``None`` when nothing is selected."""
        if self.active_name is None:
            return None
        return self.get(self.active_name)

    @property
    def names(self) -> tuple[str, ...]:
        """Configured names in stored order."""
        return tuple(item.name for item in self.items)

    def get(self, name: str) -> Profile:
        """Return the profile called ``name``.

        Raises:
            ProfileError: if no profile has that name.
        """
        for item in self.items:
            if item.name == name:
                return item
        raise ProfileError(f"No profile named {name!r}")

    def add(self, profile: Profile, *, activate: bool = False) -> Profiles:
        """Append ``profile``, selecting it when asked — or when it is the first.

        The first profile is always activated: leaving it unselected would write
        a catalog that fails to load on the next launch, and with one profile
        there is nothing to choose between.

        Raises:
            ProfileError: if the name is already configured.
        """
        if profile.name in self.names:
            raise ProfileError(f"A profile named {profile.name!r} already exists")
        select = activate or not self.items
        return Profiles(
            items=(*self.items, profile),
            active_name=profile.name if select else self.active_name,
        )

    def replace(self, old_name: str, profile: Profile) -> Profiles:
        """Replace ``old_name`` with ``profile``, supporting renames in place.

        The stored position is preserved, and a renamed profile that was active
        stays active under its new name.

        Raises:
            ProfileError: if ``old_name`` is unknown, or the new name collides
                with a different profile.
        """
        self.get(old_name)  # raises for an unknown target
        if profile.name != old_name and profile.name in self.names:
            raise ProfileError(f"A profile named {profile.name!r} already exists")
        items = tuple(
            profile if item.name == old_name else item for item in self.items
        )
        active_name = self.active_name
        if active_name == old_name:
            active_name = profile.name
        return Profiles(items=items, active_name=active_name)

    def activate(self, name: str) -> Profiles:
        """Select the profile called ``name``.

        Raises:
            ProfileError: if no profile has that name.
        """
        self.get(name)
        return Profiles(items=self.items, active_name=name)

    def delete(self, name: str) -> Profiles:
        """Remove the profile called ``name``.

        Deleting the active profile is refused rather than silently selecting a
        replacement: which endpoint takes over is the user's decision.

        Raises:
            ProfileError: if the name is unknown or currently active.
        """
        self.get(name)
        if name == self.active_name:
            raise ProfileError(
                f"Profile {name!r} is active — activate another profile before deleting it"
            )
        items = tuple(item for item in self.items if item.name != name)
        return Profiles(items=items, active_name=self.active_name)


def resolve_api_key(profile: Profile, environ: Mapping[str, str]) -> str:
    """Return the API key for ``profile`` from ``environ``.

    An unauthenticated profile (empty ``api_key_env``) resolves to the local
    placeholder the SDK requires. Otherwise the named variable must be present
    and non-empty.

    Raises:
        ProfileError: if the named variable is missing or empty. The message
            names the variable, never its value.
    """
    if not profile.api_key_env:
        return LOCAL_API_KEY
    value = environ.get(profile.api_key_env, "")
    if not value:
        raise ProfileError(
            f"Profile {profile.name!r} reads its API key from ${profile.api_key_env}, "
            "which is unset or empty in this environment"
        )
    return value


def resolve_profile(
    profile: Profile,
    *,
    discovered_model: str | None,
    environ: Mapping[str, str],
) -> ResolvedProfile:
    """Pin ``profile`` to one concrete model and credential for a whole turn.

    ``discovered_model`` is consulted only when the profile configures no model,
    which preserves auto-discovery for single-model servers. There is no default
    model and no credential fallback: an unresolvable profile raises before any
    request is made.

    Raises:
        ProfileError: if no model resolves, or the credential cannot be read.
    """
    model = profile.model
    if not model:
        model = (discovered_model or "").strip()
    if not model:
        raise ProfileError(
            f"Profile {profile.name!r} has no model configured, and {profile.base_url} "
            "did not report exactly one model. Set a model with /profiles."
        )
    return ResolvedProfile(
        name=profile.name,
        base_url=profile.base_url,
        model=model,
        api_key=resolve_api_key(profile, environ),
    )
