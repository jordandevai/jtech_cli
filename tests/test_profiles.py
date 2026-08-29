"""Unit tests for the profile domain: identity, catalog rules, and credentials."""

import pytest

from jtech_cli.cmd_tools import CmdPolicy
from jtech_cli.config import (
    ConfigurationError,
    Profile,
    ProfileError,
    Profiles,
    ResolvedProfile,
    build_settings,
    endpoint_origin,
    load_cmd_policy,
    resolve_api_key,
    resolve_profile,
    save_settings,
)

LOCAL = Profile(name="local", base_url="http://127.0.0.1:8080/v1", model="qwen3")
CLOUD = Profile(
    name="cloud",
    base_url="https://api.example.com/v1",
    model="cloud-model",
    api_key_env="CLOUD_API_KEY",
)


# --- profile construction --------------------------------------------------


def test_local_profile_needs_no_credential():
    assert LOCAL.api_key_env == ""
    assert LOCAL.base_url == "http://127.0.0.1:8080/v1"
    assert LOCAL.model == "qwen3"


def test_authenticated_profile_stores_only_the_variable_name():
    assert CLOUD.api_key_env == "CLOUD_API_KEY"
    assert "secret" not in repr(CLOUD)


def test_model_may_be_empty_for_discovery():
    profile = Profile(name="discover", base_url="http://127.0.0.1:8080/v1")
    assert profile.model == ""


@pytest.mark.parametrize(
    "name",
    ["", "Local", "_local", "-local", "my profile", "local!", "local.dev"],
)
def test_invalid_names_are_rejected(name):
    with pytest.raises(ProfileError, match="profile name"):
        Profile(name=name, base_url="http://127.0.0.1:8080/v1")


@pytest.mark.parametrize(
    "url",
    [
        "",
        "127.0.0.1:8080/v1",
        "ftp://host/v1",
        "http://",
        "/v1",
        "file:///tmp/v1",
        "http://:8080/v1",  # a port with no host is not an origin
    ],
)
def test_invalid_base_urls_are_rejected(url):
    with pytest.raises(ProfileError, match="base_url"):
        Profile(name="p", base_url=url)


@pytest.mark.parametrize("url", ["http://host:notaport/v1", "http://host:99999/v1"])
def test_an_unusable_port_is_rejected(url):
    """A URL with no resolvable port has no origin, so it cannot be scoped."""
    with pytest.raises(ProfileError, match="invalid port"):
        Profile(name="p", base_url=url)


@pytest.mark.parametrize("url", ["https://host:0/v1", "http://127.0.0.1:0/v1"])
def test_port_zero_is_rejected_rather_than_read_as_the_default(url):
    """0 is falsy: folding it into the default port would widen credential scope."""
    with pytest.raises(ProfileError, match="unusable port 0"):
        Profile(name="p", base_url=url)


@pytest.mark.parametrize("url", ["http://[::1/v1", "https://[not:an:ipv6/v1"])
def test_an_unparseable_url_raises_the_typed_error(url):
    """urlparse raises bare ValueError here; the CLI and modal catch ProfileError."""
    with pytest.raises(ProfileError, match="not a usable URL"):
        Profile(name="p", base_url=url)


@pytest.mark.parametrize(
    "url, origin",
    [
        ("http://[::1]:8080/v1", ("http", "::1", 8080)),
        ("https://[2001:db8::1]/v1", ("https", "2001:db8::1", 443)),
    ],
)
def test_well_formed_ipv6_endpoints_are_accepted(url, origin):
    assert Profile(name="p", base_url=url).base_url == url
    assert endpoint_origin(url) == origin


@pytest.mark.parametrize("url", [" http://x/v1", "http://x/v1 ", "http://x/v1\n"])
def test_base_url_whitespace_is_invalid_config(url):
    with pytest.raises(ProfileError, match="whitespace"):
        Profile(name="p", base_url=url)


@pytest.mark.parametrize("model", [" qwen3", "qwen3 ", "qwen3\t"])
def test_model_whitespace_is_rejected(model):
    with pytest.raises(ProfileError, match="model"):
        Profile(name="p", base_url="http://x/v1", model=model)


@pytest.mark.parametrize(
    "url",
    [
        "https://user:s3cret@example.com/v1",
        "https://token@example.com/v1",
        "http://user:pw@127.0.0.1:8080/v1",
    ],
)
def test_credentials_embedded_in_the_url_are_rejected(url):
    """A userinfo URL would put a live secret in TOML and on screen."""
    with pytest.raises(ProfileError, match="must not embed credentials"):
        Profile(name="p", base_url=url)


@pytest.mark.parametrize("env", ["1KEY", "MY KEY", "my-key", "$KEY", "KEY!"])
def test_invalid_api_key_env_identifiers_are_rejected(env):
    with pytest.raises(ProfileError, match="api_key_env"):
        Profile(name="p", base_url="http://x/v1", api_key_env=env)


@pytest.mark.parametrize("env", ["KEY", "_key", "My_Key9"])
def test_valid_api_key_env_identifiers_are_accepted(env):
    assert Profile(name="p", base_url="http://x/v1", api_key_env=env).api_key_env == env


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": 1, "base_url": "http://x/v1"},
        {"name": "p", "base_url": 2},
        {"name": "p", "base_url": "http://x/v1", "model": 3},
        {"name": "p", "base_url": "http://x/v1", "api_key_env": 4},
    ],
)
def test_non_string_fields_are_rejected(kwargs):
    with pytest.raises(ProfileError, match="must be a string"):
        Profile(**kwargs)


# --- catalog rules ---------------------------------------------------------


def test_empty_catalog_has_no_active_profile():
    assert Profiles().active is None
    assert Profiles().names == ()


def test_add_and_activate_preserve_order():
    catalog = Profiles().add(LOCAL, activate=True).add(CLOUD)
    assert catalog.names == ("local", "cloud")
    assert catalog.active == LOCAL
    assert catalog.activate("cloud").active == CLOUD
    assert catalog.activate("cloud").names == ("local", "cloud")


def test_the_first_profile_is_always_activated():
    """An unselected catalog cannot be loaded back, so it is never produced."""
    catalog = Profiles().add(LOCAL)  # activate not requested
    assert catalog.active_name == "local"
    # later additions still respect the caller's choice
    assert catalog.add(CLOUD).active_name == "local"


def test_a_catalog_with_profiles_must_select_one():
    with pytest.raises(ProfileError, match="must select an active one"):
        Profiles(items=(LOCAL,))


def test_add_rejects_a_duplicate_name():
    catalog = Profiles().add(LOCAL)
    with pytest.raises(ProfileError, match="already exists"):
        catalog.add(Profile(name="local", base_url="http://other/v1"))


def test_catalog_rejects_duplicates_at_construction():
    with pytest.raises(ProfileError, match="duplicate"):
        Profiles(items=(LOCAL, LOCAL))


def test_catalog_rejects_an_unknown_active_name():
    with pytest.raises(ProfileError, match="not a configured profile"):
        Profiles(items=(LOCAL,), active_name="missing")


def test_catalog_rejects_non_profile_items():
    with pytest.raises(ProfileError, match="Profile values"):
        Profiles(items=("local",))


def test_get_rejects_an_unknown_name():
    with pytest.raises(ProfileError, match="No profile named"):
        Profiles(items=(LOCAL,), active_name="local").get("nope")


def test_replace_edits_in_place_and_keeps_position():
    catalog = Profiles().add(LOCAL, activate=True).add(CLOUD)
    edited = catalog.replace("local", Profile(name="local", base_url="http://new/v1"))
    assert edited.names == ("local", "cloud")
    assert edited.get("local").base_url == "http://new/v1"
    assert edited.active_name == "local"


def test_replace_renames_and_carries_the_active_selection():
    catalog = Profiles().add(LOCAL, activate=True).add(CLOUD)
    renamed = catalog.replace("local", Profile(name="workstation", base_url="http://new/v1"))
    assert renamed.names == ("workstation", "cloud")
    assert renamed.active_name == "workstation"


def test_replace_rejects_an_unknown_target_and_a_name_collision():
    catalog = Profiles().add(LOCAL, activate=True).add(CLOUD)
    with pytest.raises(ProfileError, match="No profile named"):
        catalog.replace("nope", LOCAL)
    with pytest.raises(ProfileError, match="already exists"):
        catalog.replace("local", Profile(name="cloud", base_url="http://new/v1"))


def test_activate_rejects_an_unknown_name():
    with pytest.raises(ProfileError, match="No profile named"):
        Profiles().add(LOCAL).activate("nope")


def test_delete_refuses_the_active_profile_and_removes_an_inactive_one():
    catalog = Profiles().add(LOCAL, activate=True).add(CLOUD)
    with pytest.raises(ProfileError, match="activate another profile"):
        catalog.delete("local")
    remaining = catalog.delete("cloud")
    assert remaining.names == ("local",)
    assert remaining.active_name == "local"


def test_mutations_return_new_values_and_leave_the_original_alone():
    catalog = Profiles().add(LOCAL, activate=True)
    assert catalog.add(CLOUD) is not catalog
    assert catalog.names == ("local",)


# --- credential resolution -------------------------------------------------


def test_local_profile_resolves_to_the_sdk_placeholder():
    assert resolve_api_key(LOCAL, {}) == "none"


def test_present_environment_value_is_used():
    assert resolve_api_key(CLOUD, {"CLOUD_API_KEY": "sk-live-value"}) == "sk-live-value"


@pytest.mark.parametrize("environ", [{}, {"CLOUD_API_KEY": ""}])
def test_missing_or_empty_credential_fails_explicitly(environ):
    with pytest.raises(ProfileError) as excinfo:
        resolve_api_key(CLOUD, environ)
    message = str(excinfo.value)
    assert "CLOUD_API_KEY" in message
    assert "unset or empty" in message


def test_a_resolved_key_never_appears_in_repr_or_errors():
    resolved = resolve_profile(
        CLOUD, discovered_model=None, environ={"CLOUD_API_KEY": "sk-do-not-print"}
    )
    assert resolved.api_key == "sk-do-not-print"
    assert "sk-do-not-print" not in repr(resolved)

    with pytest.raises(ProfileError) as excinfo:
        resolve_profile(
            Profile(name="p", base_url="http://x/v1", api_key_env="CLOUD_API_KEY"),
            discovered_model=None,
            environ={"CLOUD_API_KEY": "sk-do-not-print"},
        )
    assert "sk-do-not-print" not in str(excinfo.value)


def test_configured_model_wins_over_discovery():
    resolved = resolve_profile(LOCAL, discovered_model="discovered", environ={})
    assert resolved == ResolvedProfile(
        name="local",
        base_url="http://127.0.0.1:8080/v1",
        model="qwen3",
        api_key="none",
    )


def test_a_uniquely_discovered_model_fills_an_empty_configured_model():
    profile = Profile(name="local", base_url="http://127.0.0.1:8080/v1")
    resolved = resolve_profile(profile, discovered_model="served-model", environ={})
    assert resolved.model == "served-model"


@pytest.mark.parametrize("discovered", [None, "", "   "])
def test_missing_model_fails_before_any_request(discovered):
    profile = Profile(name="local", base_url="http://127.0.0.1:8080/v1")
    with pytest.raises(ProfileError, match="no model configured"):
        resolve_profile(profile, discovered_model=discovered, environ={})


# --- persistence: migration, round trip, and failures ----------------------


def write(path, text):
    path.write_text(text)
    return path


def test_legacy_single_endpoint_migrates_in_memory(tmp_path):
    path = write(
        tmp_path / "config.toml",
        '[server]\nbase_url = "http://legacy:1/v1"\nmodel = "legacy-model"\n',
    )
    settings = build_settings(config_path=path)
    assert settings.profiles.names == ("default",)
    assert settings.profiles.active_name == "default"
    assert settings.base_url == "http://legacy:1/v1"
    assert settings.model == "legacy-model"
    # loading alone must not rewrite the file
    assert "[profiles." not in path.read_text()


def test_legacy_config_without_a_model_still_migrates(tmp_path):
    path = write(tmp_path / "config.toml", '[server]\nbase_url = "http://legacy:1/v1"\n')
    settings = build_settings(config_path=path)
    assert settings.profiles.get("default").model == ""


def test_the_next_save_writes_the_new_format(tmp_path):
    path = write(
        tmp_path / "config.toml",
        '[server]\nbase_url = "http://legacy:1/v1"\nmodel = "legacy-model"\n',
    )
    settings = build_settings(config_path=path)
    save_settings(settings, path, cmd=load_cmd_policy(path))

    text = path.read_text()
    assert 'active_profile = "default"' in text
    assert "[profiles.default]" in text
    assert 'base_url = "http://legacy:1/v1"' in text
    assert "\nbase_url" not in text.split("[profiles.default]")[0]  # not in [server]
    assert build_settings(config_path=path).base_url == "http://legacy:1/v1"


def test_no_config_file_yields_an_empty_catalog(tmp_path):
    settings = build_settings(config_path=tmp_path / "missing.toml")
    assert settings.profiles == Profiles()
    assert settings.base_url == ""
    assert settings.active_profile is None


def test_new_format_round_trip_with_several_profiles(tmp_path):
    path = tmp_path / "config.toml"
    settings = build_settings(config_path=path)
    settings.profiles = Profiles().add(LOCAL).add(CLOUD, activate=True)
    save_settings(settings, path, cmd=CmdPolicy())

    loaded = build_settings(config_path=path)
    assert loaded.profiles.names == ("local", "cloud")
    assert loaded.profiles.active == CLOUD
    assert loaded.profiles.get("cloud").api_key_env == "CLOUD_API_KEY"
    assert loaded.base_url == "https://api.example.com/v1"


def test_empty_optional_profile_fields_are_omitted(tmp_path):
    path = tmp_path / "config.toml"
    settings = build_settings(config_path=path)
    settings.profiles = Profiles().add(
        Profile(name="bare", base_url="http://bare/v1"), activate=True
    )
    save_settings(settings, path, cmd=CmdPolicy())

    text = path.read_text()
    assert "model =" not in text
    assert "api_key_env" not in text
    assert build_settings(config_path=path).profiles.get("bare").model == ""


def test_cli_overrides_are_session_only_and_never_saved(tmp_path):
    path = tmp_path / "config.toml"
    settings = build_settings(config_path=path)
    settings.profiles = Profiles().add(LOCAL, activate=True)
    save_settings(settings, path, cmd=CmdPolicy())

    overridden = build_settings(
        base_url="http://override:9/v1", model="override-model", config_path=path
    )
    assert overridden.profile_is_overridden
    assert overridden.base_url == "http://override:9/v1"
    assert overridden.model == "override-model"
    assert overridden.profiles.get("local").base_url == "http://127.0.0.1:8080/v1"

    save_settings(overridden, path, cmd=CmdPolicy())
    text = path.read_text()
    assert "override:9" not in text
    assert "override-model" not in text
    assert 'base_url = "http://127.0.0.1:8080/v1"' in text


def _saved_cloud(tmp_path):
    path = tmp_path / "config.toml"
    settings = build_settings(config_path=path)
    settings.profiles = Profiles().add(CLOUD, activate=True)
    save_settings(settings, path, cmd=CmdPolicy())
    return path


@pytest.mark.parametrize(
    "url",
    [
        "https://api.example.com/v2",  # different path, same server
        "https://api.example.com:443/v1",  # the default port, spelled out
        "https://API.Example.COM/v1",  # host case is not part of identity
    ],
)
def test_a_same_origin_url_override_keeps_the_credential_source(tmp_path, url):
    """A different path, an explicit default port, or different case is one server."""
    path = _saved_cloud(tmp_path)
    override = build_settings(base_url=url, config_path=path).active_profile
    assert override.name == "cloud"
    assert override.base_url == url
    assert override.model == "cloud-model"
    assert override.api_key_env == "CLOUD_API_KEY"


def test_an_explicit_default_port_in_the_profile_is_also_one_origin(tmp_path):
    """The equivalence holds whichever side spells the port out."""
    path = tmp_path / "config.toml"
    settings = build_settings(config_path=path)
    settings.profiles = Profiles().add(
        Profile(
            name="cloud",
            base_url="http://api.example.com:80/v1",
            model="m",
            api_key_env="CLOUD_API_KEY",
        ),
        activate=True,
    )
    save_settings(settings, path, cmd=CmdPolicy())

    override = build_settings(
        base_url="http://api.example.com/v1", config_path=path
    ).active_profile
    assert override.api_key_env == "CLOUD_API_KEY"


@pytest.mark.parametrize(
    "left, right",
    [
        ("https://h/v1", "https://h:443/v2"),
        ("http://h/v1", "http://h:80/v1"),
        ("https://H.example.com/v1", "https://h.example.com/v1"),
    ],
)
def test_equivalent_origins_compare_equal(left, right):
    assert endpoint_origin(left) == endpoint_origin(right)


@pytest.mark.parametrize(
    "left, right",
    [
        ("https://h/v1", "http://h/v1"),  # scheme
        ("https://h/v1", "https://other/v1"),  # host
        ("https://h/v1", "https://h:8443/v1"),  # explicit non-default port
        ("http://h:80/v1", "https://h:443/v1"),  # both defaults, different scheme
    ],
)
def test_distinct_origins_compare_unequal(left, right):
    assert endpoint_origin(left) != endpoint_origin(right)


@pytest.mark.parametrize("url", ["http://[::1/v1", "https://host:0/v1"])
def test_endpoint_origin_types_its_own_failures(url):
    """The helper is public: it may not leak a bare ValueError either."""
    with pytest.raises(ProfileError):
        endpoint_origin(url)


def test_an_origin_is_scheme_host_and_effective_port():
    assert endpoint_origin("https://API.Example.COM/v1") == (
        "https",
        "api.example.com",
        443,
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://staging.example.com/v1",  # different host
        "http://api.example.com/v1",  # downgraded scheme
        "https://api.example.com:8443/v1",  # explicit non-default port
    ],
)
def test_a_cross_origin_override_refuses_to_forward_the_credential(tmp_path, url):
    """--base-url must never hand another party a key it was not issued."""
    path = _saved_cloud(tmp_path)
    with pytest.raises(ProfileError) as excinfo:
        build_settings(base_url=url, config_path=path)
    message = str(excinfo.value)
    assert "different host" in message
    assert "CLOUD_API_KEY" in message


def test_a_cross_origin_override_is_fine_without_a_credential(tmp_path):
    """Only a credential is scoped; an unauthenticated profile may be re-pointed."""
    path = tmp_path / "config.toml"
    settings = build_settings(config_path=path)
    settings.profiles = Profiles().add(LOCAL, activate=True)
    save_settings(settings, path, cmd=CmdPolicy())

    override = build_settings(base_url="http://elsewhere/v1", config_path=path).active_profile
    assert override.base_url == "http://elsewhere/v1"
    assert override.api_key_env == ""


def test_a_malformed_override_reports_its_own_error(tmp_path):
    """A bad URL is a URL error, not a credential-scope error."""
    path = _saved_cloud(tmp_path)
    with pytest.raises(ProfileError, match="base_url"):
        build_settings(base_url="not-a-url", config_path=path)


def test_a_base_url_override_with_no_catalog_builds_a_cli_profile(tmp_path):
    settings = build_settings(base_url="http://oneshot/v1", config_path=tmp_path / "none.toml")
    assert settings.active_profile.name == "cli"
    assert settings.base_url == "http://oneshot/v1"
    assert settings.profiles == Profiles()


def test_an_invalid_cli_override_fails_explicitly(tmp_path):
    with pytest.raises(ProfileError, match="base_url"):
        build_settings(base_url="not-a-url", config_path=tmp_path / "none.toml")


def test_mixed_legacy_and_new_formats_are_refused(tmp_path):
    path = write(
        tmp_path / "config.toml",
        '[server]\nactive_profile = "local"\nbase_url = "http://legacy/v1"\n\n'
        '[profiles.local]\nbase_url = "http://127.0.0.1:8080/v1"\n',
    )
    with pytest.raises(ConfigurationError, match="cannot be combined"):
        build_settings(config_path=path)


def test_an_unknown_profile_field_is_refused(tmp_path):
    path = write(
        tmp_path / "config.toml",
        '[server]\nactive_profile = "local"\n\n[profiles.local]\n'
        'base_url = "http://127.0.0.1:8080/v1"\napi_key = "sk-inline"\n',
    )
    with pytest.raises(ConfigurationError, match="unknown field"):
        build_settings(config_path=path)


def test_a_missing_active_profile_key_is_refused(tmp_path):
    path = write(
        tmp_path / "config.toml",
        '[profiles.local]\nbase_url = "http://127.0.0.1:8080/v1"\n',
    )
    with pytest.raises(ConfigurationError, match="active_profile is required"):
        build_settings(config_path=path)


def test_an_active_profile_naming_nothing_is_refused(tmp_path):
    path = write(
        tmp_path / "config.toml",
        '[server]\nactive_profile = "gone"\n\n[profiles.local]\n'
        'base_url = "http://127.0.0.1:8080/v1"\n',
    )
    with pytest.raises(ConfigurationError, match="not a configured profile"):
        build_settings(config_path=path)


def test_an_invalid_profile_shape_is_refused_with_context(tmp_path):
    path = write(
        tmp_path / "config.toml",
        '[server]\nactive_profile = "local"\n\n[profiles.local]\nbase_url = "nope"\n',
    )
    with pytest.raises(ConfigurationError, match=r"\[profiles.local\]"):
        build_settings(config_path=path)


def test_a_profile_entry_that_is_not_a_table_is_refused(tmp_path):
    path = write(
        tmp_path / "config.toml",
        '[server]\nactive_profile = "local"\n\n[profiles]\nlocal = "http://x/v1"\n',
    )
    with pytest.raises(ConfigurationError, match="must be a table"):
        build_settings(config_path=path)


def test_malformed_toml_is_refused_with_the_path(tmp_path):
    path = write(tmp_path / "config.toml", "not = toml [[")
    with pytest.raises(ConfigurationError) as excinfo:
        build_settings(config_path=path)
    assert str(path) in str(excinfo.value)
    assert "not valid TOML" in str(excinfo.value)


def test_a_read_failure_is_refused_with_the_path(tmp_path):
    unreadable = tmp_path / "config.toml"
    unreadable.mkdir()  # exists, but opening it raises OSError
    with pytest.raises(ConfigurationError, match="could not be read"):
        build_settings(config_path=unreadable)


def test_every_profile_save_carries_the_command_policy_through(tmp_path):
    path = tmp_path / "config.toml"
    policy = CmdPolicy(mode="yolo", allow=["cargo build:*"], max_output=99)
    settings = build_settings(config_path=path)
    settings.profiles = Profiles().add(LOCAL, activate=True)
    save_settings(settings, path, cmd=policy)

    settings.profiles = settings.profiles.add(CLOUD, activate=True)
    save_settings(settings, path, cmd=load_cmd_policy(path))

    loaded = load_cmd_policy(path)
    assert loaded.mode == "yolo"
    assert loaded.allow == ["cargo build:*"]
    assert loaded.max_output == 99
    assert build_settings(config_path=path).profiles.active == CLOUD


def test_adding_a_first_profile_writes_a_config_that_loads_back(tmp_path):
    """Regression: an unselected catalog wrote TOML that failed on next launch."""
    path = tmp_path / "config.toml"
    settings = build_settings(config_path=path)
    settings.profiles = settings.profiles.add(Profile(name="first", base_url="http://a/v1"))
    save_settings(settings, path, cmd=CmdPolicy())

    assert 'active_profile = "first"' in path.read_text()
    assert build_settings(config_path=path).profiles.active_name == "first"


def test_a_userinfo_secret_can_never_reach_the_config_file(tmp_path):
    path = tmp_path / "config.toml"
    with pytest.raises(ProfileError):
        Profile(name="leaky", base_url="https://user:s3cret@example.com/v1")
    assert not path.exists()
