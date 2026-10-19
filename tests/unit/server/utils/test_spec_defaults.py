# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Tests for server/utils/spec_defaults.py."""

import json
from typing import Any

import pytest

from server.models.models import ModelField, OneOfOption
from server.utils.spec_defaults import fill_spec_defaults


def _field(type_: str, name: str = "f", default: str | None = None, values: list[OneOfOption | str] | None = None) -> ModelField:
    return ModelField(type=type_, name=name, description=name, default=default, values=values)


def test_fill_spec_defaults_adds_missing_text_default() -> None:
    fields = [_field("text", "infra_api_url", "http://infra:8086")]

    result = fill_spec_defaults(fields, {})

    assert result == {"infra_api_url": "http://infra:8086"}


def test_fill_spec_defaults_keeps_values_the_spec_already_has() -> None:
    fields = [_field("number", "max_concurrent", "2"), _field("text", "infra_api_key", "")]

    result = fill_spec_defaults(fields, {"max_concurrent": 1})

    assert result == {"max_concurrent": 1, "infra_api_key": ""}


def test_fill_spec_defaults_keeps_explicit_none() -> None:
    fields = [_field("text", "x", "d")]

    result = fill_spec_defaults(fields, {"x": None})

    assert result == {"x": None}


def test_fill_spec_defaults_skips_fields_without_default() -> None:
    fields = [_field("text", "rename"), _field("docker-tags", "image_version")]

    result = fill_spec_defaults(fields, None)

    assert result == {}


def test_fill_spec_defaults_skips_fields_with_required_keys() -> None:
    field = ModelField(
        type="map", name="headers", description="headers", default='{"Authorization": "Bearer "}', required_keys=["Authorization"]
    )

    result = fill_spec_defaults([field], {})

    assert result == {}


def test_fill_spec_defaults_keeps_keys_without_a_matching_field() -> None:
    result = fill_spec_defaults([], {"prefix": "p"})

    assert result == {"prefix": "p"}


def test_fill_spec_defaults_does_not_mutate_the_input_spec() -> None:
    spec: dict[str, Any] = {}

    fill_spec_defaults([_field("text", "x", "d")], spec)

    assert spec == {}


@pytest.mark.parametrize(
    ("default", "expected"),
    [
        ("true", True),
        ("false", False),
        (None, False),
    ],
)
def test_fill_spec_defaults_converts_bool_default(default: str | None, expected: bool) -> None:
    fields = [_field("bool", "b", default)]

    result = fill_spec_defaults(fields, {})

    assert result == {"b": expected}


@pytest.mark.parametrize(
    ("type_", "default", "expected"),
    [
        ("list", json.dumps(["a", "b"]), ["a", "b"]),
        ("list", None, []),
        ("list", "not json", []),
        ("list", json.dumps({"a": 1}), []),
        ("map", json.dumps({"A": ""}), {"A": ""}),
        ("map", None, {}),
        ("map", json.dumps([1]), {}),
    ],
)
def test_fill_spec_defaults_parses_list_and_map_defaults(type_: str, default: str | None, expected: Any) -> None:
    fields = [_field(type_, "x", default)]

    result = fill_spec_defaults(fields, {})

    assert result == {"x": expected}


def test_fill_spec_defaults_uses_oneof_default() -> None:
    fields = [_field("oneof", "hardware", "GPU", [OneOfOption(value="GPU", label="Default GPU"), "CPU"])]

    result = fill_spec_defaults(fields, {})

    assert result == {"hardware": "GPU"}


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([OneOfOption(value="GPU", label="Default GPU"), "CPU"], "GPU"),
        (["CPU", "GPU"], "CPU"),
        (None, ""),
    ],
)
def test_fill_spec_defaults_falls_back_to_first_oneof_value(values: list[OneOfOption | str] | None, expected: str) -> None:
    fields = [_field("oneof", "hardware", None, values)]

    result = fill_spec_defaults(fields, {})

    assert result == {"hardware": expected}
