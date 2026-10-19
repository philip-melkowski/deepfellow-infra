# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Fill a model install spec with the defaults of its fields."""

import json
from collections.abc import Mapping, Sequence
from typing import Any

from server.models.models import ModelField


def _field_default(field: ModelField) -> tuple[bool, Any]:
    """Return `(has_default, value)` for a field, converted the way the WebUI form initializes it."""
    default = field.default
    if field.type == "bool":
        return True, default == "true"
    if field.type in ("list", "map"):
        empty: list[Any] | dict[str, Any] = [] if field.type == "list" else {}
        if default is None:
            return True, empty
        try:
            parsed = json.loads(default)
        except json.JSONDecodeError:
            return True, empty
        return True, parsed if isinstance(parsed, type(empty)) else empty
    if field.type == "oneof":
        if default is not None:
            return True, default
        first = field.values[0] if field.values else None
        if first is None:
            return True, ""
        return True, first if isinstance(first, str) else first.value
    if default is not None:
        return True, default
    return False, None


def fill_spec_defaults(fields: Sequence[ModelField], spec: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return `spec` with the fields it leaves out filled in the way the WebUI install form fills them.

    Mirrors `initFormData` in `DynamicFormFields.tsx` for well-formed defaults, so a model installed
    through the API with a partial spec is stored with the same values the form would have sent.
    Keys present in `spec` always win. A missing `bool`, `list`, `map` or `oneof` field without a
    default gets its type's empty value (`false`, `[]`, `{}`, or the first option, else an empty string); any other field
    without a default is left out. A field with `required_keys` is left out too: its default is a template for the
    user to complete (e.g. `{"Authorization": "Bearer "}`), so filling it would get past the service's required-key check.
    """
    filled = dict(spec or {})
    for field in fields:
        if field.name in filled or field.required_keys:
            continue
        has_default, value = _field_default(field)
        if has_default:
            filled[field.name] = value
    return filled
