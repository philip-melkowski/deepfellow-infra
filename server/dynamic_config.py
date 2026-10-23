# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Dynamic, hot-reloadable settings persisted to `config.json`.

`config.json` has exactly one writer: the `/admin/config` handler. There is no
file-watcher; the handler updates the shared `AppSettings` instance in-process
immediately after a successful write, so nothing else needs to poll or watch the
file to stay in sync with its own writes.
"""

import json
import secrets
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from server.config import AppSettings, ConfigError, get_services_json_path
from server.utils.logger import logger
from server.utils.version import get_infra_version

CURRENT_SCHEMA_VERSION = 3


class DynamicSettings(BaseModel):
    """The subset of `AppSettings` fields that live in `config.json`."""

    model_config = ConfigDict(extra="forbid")

    name: str = ""
    infra_url: str = ""
    # max_length: with docker_subnet a suffixed container name is the hostname DF Infra connects to, capped at 63
    # chars per DNS label, and base names (e.g. "vllm-<instance>-<model>") already take a good part of that - a
    # container whose full name doesn't fit is refused at install (DockerService.install_and_run_docker). Hence
    # auto-generated suffixes are kept short: "_" + 6 random hex chars.
    # Lowercase only: DNS names and compose project names are case-insensitive, so suffixes differing only in case
    # would give two instances' containers the same hostname and compose project.
    container_name_suffix: Annotated[str, Field(pattern=r"^[a-z0-9_-]*$", max_length=32)] = ""

    mesh_key: SecretStr = SecretStr("")
    infra_api_key: SecretStr = SecretStr("")

    connect_to_mesh_url: str = ""
    connect_to_mesh_key: SecretStr = SecretStr("")
    share_models_downstream: bool = True

    hugging_face_token: SecretStr = SecretStr("")
    civitai_token: SecretStr = SecretStr("")
    adapter_registry_secret: SecretStr = SecretStr("")
    log_payloads: str = ""
    stop_containers_on_shutdown: str = ""

    mcp_sse_session_ttl_seconds: int = 300
    mcp_sse_max_sessions: int = 128

    standard_proxy_timeout_seconds: int = 600

    max_generated_tokens: Annotated[int, Field(ge=1)] = 131072
    max_context_tokens: Annotated[int, Field(ge=1)] = 1048576
    max_choices: Annotated[int, Field(ge=1)] = 16
    max_top_k: Annotated[int, Field(ge=1)] = 1000

    metrics_username: str = ""
    metrics_password: SecretStr = SecretStr("")

    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    otel_tracing_enabled: bool = False
    otel_logging_enabled: bool = False

    sentry_enabled: bool = False
    sentry_dsn: SecretStr = SecretStr("")
    sentry_environment: str = ""

    def to_persistable_dict(self) -> dict[str, Any]:
        """Serialize with real secret values — for config.json and internal PUT-merge use only.

        Pydantic's `SecretStr` masks itself as `"**********"` under normal JSON serialization
        (`model_dump(mode="json")`/`model_dump_json()`) to prevent accidental leakage into logs
        or API responses. That's exactly wrong for our own on-disk persistence, which needs the
        real values back on the next load. Never use this for anything sent to a client.
        """
        data = self.model_dump(mode="json")
        for field_name in DynamicSettings.model_fields:
            value = getattr(self, field_name)
            if isinstance(value, SecretStr):
                data[field_name] = value.get_secret_value()
        return data


class ConfigEnvelope(BaseModel):
    """On-disk shape of `config.json`."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = CURRENT_SCHEMA_VERSION
    infra_version: str = Field(default_factory=get_infra_version)
    settings: DynamicSettings = DynamicSettings()


def get_config_json_path(config: AppSettings) -> Path:
    """Path to config.json, inside the storage directory so it survives container recreation.

    The storage directory is a mounted volume in Docker deployments; the app's own directory
    (where `.env` lives) is part of the container image layer and does not persist.
    """
    return config.get_storage_dir() / "config.json"


def _migrate(raw: dict[str, Any]) -> dict[str, Any]:
    """Apply schema migrations in order, from `raw["schema_version"]` up to CURRENT_SCHEMA_VERSION.

    v1 -> v2: drops `custom_endpoint_read_timeout_seconds`, an option retired (superseded by the
    per-service `proxy_timeout_seconds` install field) before it ever shipped in a release — but a
    config.json written by an unreleased build in the interim could still have it on disk, and
    `DynamicSettings`'s `extra="forbid"` would otherwise reject it outright.

    v2 -> v3: drops `adapter_registry_url`. The df-finetune adapter registry's address is now
    resolved live from the installed `deepfellow-finetune` Finetune-service container instead of a
    manually configured URL (which drifted from the container's actual, dynamically assigned host
    port) - same `extra="forbid"` concern as above for any config.json still carrying the old key.

    Raises `ConfigError` if `schema_version` is present but not an int, e.g. a hand-edited config.json.
    """
    schema_version = raw.get("schema_version", 1)
    if not isinstance(schema_version, int):
        message = f"config.json schema_version must be an int, got {type(schema_version).__name__}: {schema_version!r}"
        raise ConfigError(message)
    if schema_version < 2:
        settings = raw.get("settings")
        if isinstance(settings, dict):
            settings.pop("custom_endpoint_read_timeout_seconds", None)
        raw["schema_version"] = 2
    if schema_version < 3:
        settings = raw.get("settings")
        if isinstance(settings, dict):
            settings.pop("adapter_registry_url", None)
        raw["schema_version"] = 3
    return raw


def read_envelope(config: AppSettings) -> ConfigEnvelope | None:
    """Read and validate config.json, applying migrations. None if the file doesn't exist.

    Parses the raw JSON directly (not via `ConfigEnvelope.model_validate_json(...).model_dump(...)`)
    so secret fields never round-trip through Pydantic's masking serialization before being
    re-validated — doing so would silently replace real values with the literal `"**********"`.
    """
    path = get_config_json_path(config)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        message = f"config.json is not valid JSON: {e}"
        raise ConfigError(message) from e
    migrated = _migrate(raw)
    try:
        return ConfigEnvelope.model_validate(migrated)
    except ValidationError as e:
        message = f"config.json failed validation: {e}"
        raise ConfigError(message) from e


def write_envelope_atomic(config: AppSettings, envelope: ConfigEnvelope) -> None:
    """Write config.json atomically (temp file + rename), with real (unmasked) secret values."""
    path = get_config_json_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    data = envelope.model_dump(mode="json")
    data["settings"] = envelope.settings.to_persistable_dict()
    tmp_path.write_text(json.dumps(data, indent=2))
    tmp_path.replace(path)


def apply_dynamic_settings(config: AppSettings, settings: DynamicSettings) -> None:
    """Copy every DynamicSettings field onto the shared AppSettings instance in place."""
    for field_name in DynamicSettings.model_fields:
        setattr(config, field_name, getattr(settings, field_name))


def snapshot_dynamic_settings(config: AppSettings) -> DynamicSettings:
    """Build a DynamicSettings snapshot from the current values on the shared AppSettings instance."""
    return DynamicSettings(**{field_name: getattr(config, field_name) for field_name in DynamicSettings.model_fields})


def _generate_container_name_suffix() -> str:
    """Generate a default container-name suffix, disambiguating this instance from others on the same Docker host.

    Random rather than derived from `DF_NAME` (which may be shared/default), and short, since with docker_subnet
    the suffixed name must still fit a 63-char hostname.
    """
    return f"_{secrets.token_hex(3)}"


def _build_legacy_defaults(config: AppSettings) -> DynamicSettings:
    """Build initial DynamicSettings from `config`, for the release that introduces config.json.

    `config` was already loaded from `.env` via pydantic-settings using the *current* field
    declarations, which still include every dynamic field (with a default). So any legacy `.env`
    value for a now-dynamic field (e.g. `DF_HUGGING_FACE_TOKEN`) is already present on `config` —
    no separate manual env parsing is needed, just snapshot it before config.json exists.

    Raises ConfigError naming the offending variable(s) if a legacy value fails validation (e.g. an invalid
    `DF_CONTAINER_NAME_SUFFIX`) - otherwise startup would only report a generic config error.
    """
    try:
        settings = snapshot_dynamic_settings(config)
    except ValidationError as e:
        problems = "; ".join(f"DF_{'_'.join(str(part) for part in err['loc']).upper()}: {err['msg']}" for err in e.errors())
        msg = f"Invalid value in .env - {problems}"
        raise ConfigError(msg) from e
    if settings != DynamicSettings():
        logger.warning(
            "Seeding config.json from legacy .env values. These environment variables are no longer "
            "read once config.json exists — use the /admin/config API going forward."
        )
    return settings


def _should_auto_generate_container_name_suffix(config: AppSettings) -> bool:
    """Whether a new config.json should get an auto-derived container_name_suffix.

    Only for a genuinely fresh install. Skipped for an existing installation (services.json present, e.g.
    an upgrade from before config.json existed), whose not-yet-pinned container names would otherwise be
    re-derived with the new suffix and renamed; and while the legacy DF_CONTAINER_NAME_PREFIX is set,
    which takes priority over any suffix anyway.
    """
    return not config.container_name_prefix and not get_services_json_path(config).exists()


def load_or_init(config: AppSettings) -> None:
    """Load config.json into `config`, creating it (seeded from legacy env if present) if absent.

    On a genuinely fresh install with no `DF_CONTAINER_NAME_SUFFIX` set, auto-derives one (see
    `_should_auto_generate_container_name_suffix`). An upgrade - into an existing config.json or not -
    keeps the blank default so container names don't change.
    """
    envelope = read_envelope(config)
    if envelope is None:
        settings = _build_legacy_defaults(config)
        if not settings.container_name_suffix and _should_auto_generate_container_name_suffix(config):
            # model_validate, not model_copy(update=...) - the latter skips validation, so a bad suffix would
            # reach config.json and only fail on the next startup's read_envelope().
            suffix = _generate_container_name_suffix()
            settings = DynamicSettings.model_validate({**settings.model_dump(), "container_name_suffix": suffix})
        envelope = ConfigEnvelope(
            schema_version=CURRENT_SCHEMA_VERSION,
            infra_version=get_infra_version(),
            settings=settings,
        )
        write_envelope_atomic(config, envelope)
    apply_dynamic_settings(config, envelope.settings)
    if config.container_name_prefix and config.container_name_suffix:
        logger.warning(
            "container_name_suffix %r is ignored while the legacy DF_CONTAINER_NAME_PREFIX %r is set. Remove "
            "DF_CONTAINER_NAME_PREFIX from .env and restart to use the suffix - installed containers keep their names.",
            config.container_name_suffix,
            config.container_name_prefix,
        )


def persist_settings(config: AppSettings, settings: DynamicSettings) -> None:
    """Write the given dynamic settings to config.json atomically.

    Callers should write to disk (this function) before mutating the in-memory `AppSettings`
    instance, so disk remains the source of truth if the process dies mid-update.
    """
    envelope = ConfigEnvelope(
        schema_version=CURRENT_SCHEMA_VERSION,
        infra_version=get_infra_version(),
        settings=settings,
    )
    write_envelope_atomic(config, envelope)


def persist(config: AppSettings) -> None:
    """Write the current in-memory dynamic settings back to config.json."""
    persist_settings(config, snapshot_dynamic_settings(config))
