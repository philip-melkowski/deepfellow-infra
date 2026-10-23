# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Tests for server/dynamic_config.py."""

import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr, ValidationError

from server.config import AppSettings, ConfigError
from server.dynamic_config import (
    CURRENT_SCHEMA_VERSION,
    ConfigEnvelope,
    DynamicSettings,
    _build_legacy_defaults,  # pyright: ignore[reportPrivateUsage]
    _generate_container_name_suffix,  # pyright: ignore[reportPrivateUsage]
    _migrate,  # pyright: ignore[reportPrivateUsage]
    apply_dynamic_settings,
    get_config_json_path,
    load_or_init,
    persist,
    persist_settings,
    read_envelope,
    snapshot_dynamic_settings,
    write_envelope_atomic,
)
from server.utils.version import get_infra_version


def _make_config(storage_dir: Path | None = None, **overrides: object) -> MagicMock:
    mock = MagicMock(spec=AppSettings)
    if storage_dir is not None:
        mock.get_storage_dir.return_value = storage_dir
    mock.name = ""
    mock.infra_url = ""
    mock.container_name_prefix = ""
    mock.container_name_suffix = ""
    mock.mesh_key = SecretStr("")
    mock.infra_api_key = SecretStr("")
    mock.connect_to_mesh_url = ""
    mock.connect_to_mesh_key = SecretStr("")
    mock.share_models_downstream = True
    mock.hugging_face_token = SecretStr("")
    mock.civitai_token = SecretStr("")
    mock.adapter_registry_secret = SecretStr("")
    mock.log_payloads = ""
    mock.stop_containers_on_shutdown = ""
    mock.mcp_sse_session_ttl_seconds = 300
    mock.mcp_sse_max_sessions = 128
    mock.standard_proxy_timeout_seconds = 600
    mock.max_generated_tokens = 131072
    mock.max_context_tokens = 1048576
    mock.max_choices = 16
    mock.max_top_k = 1000
    mock.metrics_username = ""
    mock.metrics_password = SecretStr("")
    mock.otel_exporter_otlp_endpoint = "http://localhost:4317"
    mock.otel_tracing_enabled = False
    mock.otel_logging_enabled = False
    mock.sentry_enabled = False
    mock.sentry_dsn = SecretStr("")
    mock.sentry_environment = ""
    for key, value in overrides.items():
        setattr(mock, key, value)
    return mock


# --- get_config_json_path ---


def test_get_config_json_path_is_inside_storage_dir(tmp_path: Path) -> None:
    config = _make_config(storage_dir=tmp_path)

    assert get_config_json_path(config) == tmp_path / "config.json"


# --- _migrate ---


def test_migrate_mutates_and_returns_the_same_dict() -> None:
    raw = {"schema_version": 1, "foo": "bar"}

    assert _migrate(raw) is raw


def test_migrate_bumps_schema_version_to_current() -> None:
    raw = {"schema_version": 1, "settings": {}}

    migrated = _migrate(raw)

    assert migrated["schema_version"] == CURRENT_SCHEMA_VERSION


def test_migrate_drops_retired_custom_endpoint_read_timeout_seconds_key() -> None:
    raw = {"schema_version": 1, "settings": {"name": "node-a", "custom_endpoint_read_timeout_seconds": 900}}

    migrated = _migrate(raw)

    assert "custom_endpoint_read_timeout_seconds" not in migrated["settings"]
    assert migrated["settings"]["name"] == "node-a"


def test_migrate_drops_retired_adapter_registry_url_key() -> None:
    raw = {"schema_version": 2, "settings": {"name": "node-a", "adapter_registry_url": "http://localhost:8333"}}

    migrated = _migrate(raw)

    assert "adapter_registry_url" not in migrated["settings"]
    assert migrated["settings"]["name"] == "node-a"
    assert migrated["schema_version"] == CURRENT_SCHEMA_VERSION


def test_migrate_is_noop_when_already_at_current_schema_version() -> None:
    raw = {"schema_version": CURRENT_SCHEMA_VERSION, "settings": {"name": "node-a"}}

    migrated = _migrate(raw)

    assert migrated == {"schema_version": CURRENT_SCHEMA_VERSION, "settings": {"name": "node-a"}}


def test_migrate_raises_config_error_on_non_int_schema_version() -> None:
    raw = {"schema_version": "2", "settings": {"name": "node-a"}}

    with pytest.raises(ConfigError, match="schema_version"):
        _migrate(raw)


# --- read_envelope ---


def test_read_envelope_returns_none_when_file_missing(tmp_path: Path) -> None:
    config = _make_config(storage_dir=tmp_path)

    assert read_envelope(config) is None


def test_read_envelope_parses_existing_file(tmp_path: Path) -> None:
    data = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "infra_version": get_infra_version(),
        "settings": {"name": "node-a", "mesh_key": "secret-value"},
    }
    (tmp_path / "config.json").write_text(json.dumps(data))
    config = _make_config(storage_dir=tmp_path)

    envelope = read_envelope(config)

    assert envelope is not None
    assert envelope.settings.name == "node-a"
    assert envelope.settings.mesh_key.get_secret_value() == "secret-value"


def test_read_envelope_raises_config_error_on_malformed_json(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{not valid json")
    config = _make_config(storage_dir=tmp_path)

    with pytest.raises(ConfigError, match="not valid JSON"):
        read_envelope(config)


def test_read_envelope_raises_config_error_on_validation_failure(tmp_path: Path) -> None:
    data = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "infra_version": get_infra_version(),
        "settings": {"mcp_sse_max_sessions": "not-an-int"},
    }
    (tmp_path / "config.json").write_text(json.dumps(data))
    config = _make_config(storage_dir=tmp_path)

    with pytest.raises(ConfigError, match="failed validation"):
        read_envelope(config)


def test_read_envelope_migrates_legacy_config_with_retired_custom_endpoint_read_timeout_seconds(tmp_path: Path) -> None:
    data = {
        "schema_version": 1,
        "infra_version": get_infra_version(),
        "settings": {"name": "node-a", "custom_endpoint_read_timeout_seconds": 900},
    }
    (tmp_path / "config.json").write_text(json.dumps(data))
    config = _make_config(storage_dir=tmp_path)

    envelope = read_envelope(config)

    assert envelope is not None
    assert envelope.schema_version == CURRENT_SCHEMA_VERSION
    assert envelope.settings.name == "node-a"


def test_read_envelope_raises_config_error_on_non_int_schema_version(tmp_path: Path) -> None:
    data = {
        "schema_version": "2",
        "infra_version": get_infra_version(),
        "settings": {"name": "node-a"},
    }
    (tmp_path / "config.json").write_text(json.dumps(data))
    config = _make_config(storage_dir=tmp_path)

    with pytest.raises(ConfigError, match="schema_version"):
        read_envelope(config)


def test_read_envelope_raises_config_error_on_unknown_field(tmp_path: Path) -> None:
    data = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "infra_version": get_infra_version(),
        "settings": {"name": "node-a", "typo_field": "x"},
    }
    (tmp_path / "config.json").write_text(json.dumps(data))
    config = _make_config(storage_dir=tmp_path)

    with pytest.raises(ConfigError, match="failed validation"):
        read_envelope(config)


# --- write_envelope_atomic ---


def test_write_envelope_atomic_writes_unmasked_secrets(tmp_path: Path) -> None:
    config = _make_config(storage_dir=tmp_path)
    envelope = ConfigEnvelope(settings=DynamicSettings(mesh_key=SecretStr("real-secret")))

    write_envelope_atomic(config, envelope)

    written = json.loads((tmp_path / "config.json").read_text())
    assert written["settings"]["mesh_key"] == "real-secret"
    assert not (tmp_path / "config.json.tmp").exists()


def test_write_envelope_atomic_creates_storage_dir_if_missing(tmp_path: Path) -> None:
    storage_dir = tmp_path / "storage"
    config = _make_config(storage_dir=storage_dir)
    envelope = ConfigEnvelope(settings=DynamicSettings())

    write_envelope_atomic(config, envelope)

    assert (storage_dir / "config.json").exists()


# --- apply_dynamic_settings ---


def test_apply_dynamic_settings_copies_all_fields_onto_config() -> None:
    config = _make_config()
    settings = DynamicSettings(name="node-b", mcp_sse_max_sessions=999, otel_logging_enabled=True)

    apply_dynamic_settings(config, settings)

    assert config.name == "node-b"
    assert config.mcp_sse_max_sessions == 999
    assert config.otel_logging_enabled is True


# --- snapshot_dynamic_settings ---


def test_snapshot_dynamic_settings_builds_from_config() -> None:
    config = _make_config(name="node-c", metrics_username="admin")

    snapshot = snapshot_dynamic_settings(config)

    assert snapshot.name == "node-c"
    assert snapshot.metrics_username == "admin"


# --- _build_legacy_defaults ---


def test_build_legacy_defaults_returns_snapshot_of_config() -> None:
    config = _make_config(name="node-d")

    settings = _build_legacy_defaults(config)

    assert settings.name == "node-d"


@pytest.mark.parametrize("suffix", ["_Prod", "bad.name", "a" * 33])
def test_build_legacy_defaults_raises_config_error_naming_invalid_env_var(suffix: str) -> None:
    """Startup only reports a ConfigError's own message, so it must name the variable to fix."""
    config = _make_config(container_name_suffix=suffix)

    with pytest.raises(ConfigError, match="DF_CONTAINER_NAME_SUFFIX"):
        _build_legacy_defaults(config)


def test_build_legacy_defaults_warns_when_non_default(caplog: pytest.LogCaptureFixture) -> None:
    config = _make_config(name="node-e")

    with caplog.at_level("WARNING", logger="uvicorn"):
        _build_legacy_defaults(config)

    assert "Seeding config.json from legacy .env values" in caplog.text


def test_build_legacy_defaults_silent_when_all_default(caplog: pytest.LogCaptureFixture) -> None:
    config = _make_config()

    with caplog.at_level("WARNING", logger="uvicorn"):
        _build_legacy_defaults(config)

    assert "Seeding config.json" not in caplog.text


# --- load_or_init ---


def test_load_or_init_seeds_config_json_from_legacy_config_when_missing(tmp_path: Path) -> None:
    config = _make_config(storage_dir=tmp_path, name="node-f")

    load_or_init(config)

    written = json.loads((tmp_path / "config.json").read_text())
    assert written["settings"]["name"] == "node-f"
    assert config.name == "node-f"


def test_load_or_init_loads_existing_config_json_without_rewriting(tmp_path: Path) -> None:
    data = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "infra_version": get_infra_version(),
        "settings": {"name": "on-disk-name"},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(data))
    config = _make_config(storage_dir=tmp_path, name="ignored-legacy-name")

    load_or_init(config)

    assert config.name == "on-disk-name"
    assert json.loads(config_path.read_text())["settings"]["name"] == "on-disk-name"


def test_load_or_init_auto_generates_container_name_suffix_on_fresh_install(tmp_path: Path) -> None:
    config = _make_config(storage_dir=tmp_path)

    load_or_init(config)

    written = json.loads((tmp_path / "config.json").read_text())
    assert written["settings"]["container_name_suffix"]
    assert config.container_name_suffix == written["settings"]["container_name_suffix"]


def test_load_or_init_keeps_explicit_container_name_suffix_on_fresh_install(tmp_path: Path) -> None:
    config = _make_config(storage_dir=tmp_path, container_name_suffix="-pinned")

    load_or_init(config)

    written = json.loads((tmp_path / "config.json").read_text())
    assert written["settings"]["container_name_suffix"] == "-pinned"
    assert config.container_name_suffix == "-pinned"


def test_load_or_init_skips_suffix_auto_generation_for_existing_install(tmp_path: Path) -> None:
    """An upgrade from before config.json existed must not rename not-yet-pinned containers."""
    (tmp_path / "services.json").write_text("{}")
    config = _make_config(storage_dir=tmp_path)

    load_or_init(config)

    assert json.loads((tmp_path / "config.json").read_text())["settings"]["container_name_suffix"] == ""
    assert config.container_name_suffix == ""


def test_load_or_init_skips_suffix_auto_generation_while_legacy_prefix_set(tmp_path: Path) -> None:
    config = _make_config(storage_dir=tmp_path, container_name_prefix="df_")

    load_or_init(config)

    assert json.loads((tmp_path / "config.json").read_text())["settings"]["container_name_suffix"] == ""


def test_load_or_init_warns_when_suffix_is_ignored_due_to_legacy_prefix(tmp_path: Path) -> None:
    data = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "infra_version": get_infra_version(),
        "settings": {"container_name_suffix": "_mine"},
    }
    (tmp_path / "config.json").write_text(json.dumps(data))
    config = _make_config(storage_dir=tmp_path, container_name_prefix="df_")

    with patch("server.dynamic_config.logger") as logger_mock:
        load_or_init(config)

    logger_mock.warning.assert_called_once()
    assert "DF_CONTAINER_NAME_PREFIX" in logger_mock.warning.call_args.args[0]


def test_load_or_init_does_not_warn_without_legacy_prefix(tmp_path: Path) -> None:
    data = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "infra_version": get_infra_version(),
        "settings": {"container_name_suffix": "_mine"},
    }
    (tmp_path / "config.json").write_text(json.dumps(data))
    config = _make_config(storage_dir=tmp_path)

    with patch("server.dynamic_config.logger") as logger_mock:
        load_or_init(config)

    logger_mock.warning.assert_not_called()


def test_load_or_init_with_non_ascii_name_writes_config_json_that_loads_again(tmp_path: Path) -> None:
    """A non-ASCII DF_NAME must not brick the next startup's read_envelope(), with the generated suffix alongside it."""
    config = _make_config(storage_dir=tmp_path, name="Zażółć 日本", mesh_key=SecretStr("secret-key"))

    load_or_init(config)

    envelope = read_envelope(config)
    assert envelope is not None
    assert re.fullmatch(r"_[0-9a-f]{6}", envelope.settings.container_name_suffix)
    assert envelope.settings.name == "Zażółć 日本"
    assert envelope.settings.mesh_key.get_secret_value() == "secret-key"


def test_load_or_init_does_not_backfill_container_name_suffix_for_existing_config_json(tmp_path: Path) -> None:
    """Upgrading must not rename already-installed services' containers."""
    data = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "infra_version": get_infra_version(),
        "settings": {"name": "on-disk-name"},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(data))
    config = _make_config(storage_dir=tmp_path)

    load_or_init(config)

    assert config.container_name_suffix == ""
    assert "container_name_suffix" not in json.loads(config_path.read_text())["settings"]


def test_generate_container_name_suffix_is_separator_plus_six_random_hex_chars() -> None:
    """Kept short so that, with docker_subnet, the suffixed name still fits a 63-char hostname."""
    assert re.fullmatch(r"_[0-9a-f]{6}", _generate_container_name_suffix())


def test_generate_container_name_suffix_is_unique_across_calls() -> None:
    assert _generate_container_name_suffix() != _generate_container_name_suffix()


def test_generate_container_name_suffix_passes_suffix_validation() -> None:
    suffix = _generate_container_name_suffix()

    assert DynamicSettings(container_name_suffix=suffix).container_name_suffix == suffix


def test_container_name_suffix_rejects_characters_docker_disallows() -> None:
    with pytest.raises(ValidationError):
        DynamicSettings(container_name_suffix="bad/name")


def test_container_name_suffix_rejects_uppercase() -> None:
    """Case-only differences would still collide on hostname and compose project, both case-insensitive."""
    with pytest.raises(ValidationError):
        DynamicSettings(container_name_suffix="_Prod")


def test_container_name_suffix_accepts_blank_and_valid_values() -> None:
    assert DynamicSettings(container_name_suffix="").container_name_suffix == ""
    assert DynamicSettings(container_name_suffix="my-suffix_1").container_name_suffix == "my-suffix_1"


def test_container_name_suffix_rejects_values_over_32_chars() -> None:
    assert DynamicSettings(container_name_suffix="a" * 32).container_name_suffix == "a" * 32
    with pytest.raises(ValidationError):
        DynamicSettings(container_name_suffix="a" * 33)


# --- persist_settings / persist ---


def test_persist_settings_writes_given_settings(tmp_path: Path) -> None:
    config = _make_config(storage_dir=tmp_path)
    settings = DynamicSettings(name="persisted-name")

    persist_settings(config, settings)

    written = json.loads((tmp_path / "config.json").read_text())
    assert written["settings"]["name"] == "persisted-name"


def test_persist_writes_current_config_snapshot(tmp_path: Path) -> None:
    config = _make_config(storage_dir=tmp_path, name="live-name")

    persist(config)

    written = json.loads((tmp_path / "config.json").read_text())
    assert written["settings"]["name"] == "live-name"


# --- sentry settings ---


def test_load_or_init_does_not_seed_sentry_settings_from_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DF_INFRA_ADMIN_API_KEY", "admin-key")
    monkeypatch.setenv("DF_STORAGE_DIR", str(tmp_path))
    monkeypatch.setenv("DF_SENTRY_ENABLED", "true")
    monkeypatch.setenv("DF_SENTRY_DSN", "https://public@sentry.example.com/42")
    monkeypatch.setenv("DF_SENTRY_ENVIRONMENT", "staging")
    config = AppSettings()  # pyright: ignore[reportCallIssue]

    load_or_init(config)

    written = json.loads((tmp_path / "config.json").read_text())["settings"]
    assert written["sentry_enabled"] is False
    assert written["sentry_dsn"] == ""
    assert written["sentry_environment"] == ""


def test_write_envelope_atomic_round_trips_sentry_settings_with_unmasked_dsn(tmp_path: Path) -> None:
    config = _make_config(storage_dir=tmp_path)
    settings = DynamicSettings(
        sentry_enabled=True,
        sentry_dsn=SecretStr("https://public@sentry.example.com/42"),
        sentry_environment="staging",
    )

    write_envelope_atomic(config, ConfigEnvelope(settings=settings))
    envelope = read_envelope(config)

    written = json.loads((tmp_path / "config.json").read_text())["settings"]
    assert written["sentry_dsn"] == "https://public@sentry.example.com/42"
    assert envelope is not None
    assert envelope.settings.sentry_enabled is True
    assert envelope.settings.sentry_dsn.get_secret_value() == "https://public@sentry.example.com/42"
    assert envelope.settings.sentry_environment == "staging"
