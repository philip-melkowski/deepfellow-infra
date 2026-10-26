# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Tests for server/utils/ollama.py."""

import json

import pytest
from fastapi import HTTPException

from server.utils.ollama import (
    ALREADY_QUANTIZED_MESSAGE,
    GGUF_QUANTIZATION_UNSUPPORTED_MESSAGE,
    MLX_RUNTIME_UNAVAILABLE_MESSAGE,
    UNSUPPORTED_MLX_ARCHITECTURE_MESSAGE,
    UNSUPPORTED_QUANTIZATION_TYPE_MESSAGE,
    decision_models_unsupported_message,
    get_ollama_create_error_message,
    raise_ollama_pull_error,
    supports_decision_models,
    supports_gguf_quantization,
)


def test_raise_ollama_pull_error_invalid_json_raises_400() -> None:
    with pytest.raises(HTTPException) as exc_info:
        raise_ollama_pull_error("not valid json {{{")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Model not available"


def test_raise_ollama_pull_error_known_error_fragment() -> None:
    raw = json.dumps({"error": "file does not exist at path"})

    with pytest.raises(HTTPException) as exc_info:
        raise_ollama_pull_error(raw)

    assert exc_info.value.status_code == 400
    assert "check if the model ID is correct" in exc_info.value.detail


def test_raise_ollama_pull_error_unknown_error_with_message() -> None:
    raw = json.dumps({"error": "something unexpected happened"})

    with pytest.raises(HTTPException) as exc_info:
        raise_ollama_pull_error(raw)

    assert exc_info.value.status_code == 400
    assert "something unexpected happened" in exc_info.value.detail


def test_raise_ollama_pull_error_empty_error_field() -> None:
    raw = json.dumps({"error": ""})

    with pytest.raises(HTTPException) as exc_info:
        raise_ollama_pull_error(raw)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Model not available"


def test_raise_ollama_pull_error_missing_error_field() -> None:
    raw = json.dumps({})

    with pytest.raises(HTTPException) as exc_info:
        raise_ollama_pull_error(raw)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Model not available"


def test_get_ollama_create_error_message_known_quantization_fragment() -> None:
    stderr = "Error: 500 Internal Server Error: quantization is only supported for F16 and F32 models"

    assert get_ollama_create_error_message(stderr) == ALREADY_QUANTIZED_MESSAGE


@pytest.mark.parametrize(
    ("stderr", "expected"),
    [
        (
            "Error: create-time quantization is only supported for safetensors imports; quantize GGUF models before importing",
            GGUF_QUANTIZATION_UNSUPPORTED_MESSAGE,
        ),
        (
            'Error: unsupported --quantize "q4_K_M": supported types are int4, int8, nvfp4, mxfp4, mxfp8',
            UNSUPPORTED_QUANTIZATION_TYPE_MESSAGE,
        ),
        (
            'Error: 400 Bad Request: unsupported quantize type "q4_K_M": supported types are int4, int8, nvfp4, mxfp4, mxfp8',
            UNSUPPORTED_QUANTIZATION_TYPE_MESSAGE,
        ),
        (
            "Error: MLX runtime is not available: failed to load MLX dynamic library (searched: [/usr/lib/ollama])",
            MLX_RUNTIME_UNAVAILABLE_MESSAGE,
        ),
        (
            'Error: unsupported MLX architecture: model "Gemma3ForCausalLM"',
            UNSUPPORTED_MLX_ARCHITECTURE_MESSAGE,
        ),
    ],
)
def test_get_ollama_create_error_message_new_ollama_quantization_fragments(stderr: str, expected: str) -> None:
    assert get_ollama_create_error_message(stderr) == expected


def test_get_ollama_create_error_message_unknown_fragment_returns_none() -> None:
    assert get_ollama_create_error_message("some other failure") is None


def test_get_ollama_create_error_message_empty_string_returns_none() -> None:
    assert get_ollama_create_error_message("") is None


@pytest.mark.parametrize(
    ("ollama_version", "expected"),
    [("0.32.6", True), ("0.34.0", True), ("0.34.1", False), ("0.35.0", False), ("0.40.0-rc0", False), ("not-a-version", None)],
)
def test_supports_gguf_quantization(ollama_version: str, expected: bool | None) -> None:
    assert supports_gguf_quantization(ollama_version) is expected


@pytest.mark.parametrize(
    ("ollama_version", "expected"),
    [("0.34.1", False), ("0.35.0-rc0", False), ("0.35.0", True), ("0.40.8", True), ("not-a-version", None)],
)
def test_supports_decision_models(ollama_version: str, expected: bool | None) -> None:
    assert supports_decision_models(ollama_version) is expected


def test_decision_models_unsupported_message_names_required_and_actual_version() -> None:
    assert decision_models_unsupported_message("0.34.1") == "Decision models need Ollama 0.35.0 or newer; this Ollama is 0.34.1."
