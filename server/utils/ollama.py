# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Common types for Ollama/Ollama External."""

import json

from fastapi import HTTPException
from packaging import version

ERROR_MAPPING = {
    "file does not exist": (
        400,
        "Model not found — check if the model ID is correct "
        "(for HuggingFace models use the hf.co/ prefix, "
        "e.g. hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF)",
    ),
    "not GGUF or is not compatible": (
        400,
        "This repository has no GGUF build compatible with llama.cpp — pick a GGUF version of the model",
    ),
    'realm host "huggingface.co" does not match': (
        400,
        "Cannot authenticate with HuggingFace — this may be a gated model. Try using a direct GGUF download URL instead.",
    ),
}


def raise_ollama_pull_error(raw_data: str) -> None:
    """Parse a failed Ollama /api/pull response and raise an appropriate HTTPException."""
    try:
        record = json.loads(raw_data)
    except json.JSONDecodeError:
        raise HTTPException(400, "Model not available") from None
    ollama_error: str = record.get("error", "")
    for error_fragment, (status_code, message) in ERROR_MAPPING.items():
        if error_fragment in ollama_error:
            raise HTTPException(status_code, message)
    raise HTTPException(400, f"Model not available: {ollama_error}" if ollama_error else "Model not available")


ALREADY_QUANTIZED_MESSAGE = (
    "This model is already quantized — quantization is only supported for F16/F32 (unquantized) source models. "
    "Remove the quantization option or pick an unquantized source model."
)
GGUF_QUANTIZATION_UNSUPPORTED_MESSAGE = (
    "This Ollama version can only quantize safetensors models, not GGUF files or models pulled by name. "
    "Remove the quantization option, or quantize the GGUF with llama.cpp tools before importing it."
)
UNSUPPORTED_QUANTIZATION_TYPE_MESSAGE = (
    "This Ollama version doesn't support this quantization type — safetensors models can only be quantized "
    "to int4, int8, nvfp4, mxfp4 or mxfp8."
)
MLX_RUNTIME_UNAVAILABLE_MESSAGE = (
    "Quantizing safetensors models needs Ollama's MLX runtime, which could not be loaded on this host "
    "(it requires a supported GPU). Remove the quantization option or install on a host with a supported GPU."
)
UNSUPPORTED_MLX_ARCHITECTURE_MESSAGE = (
    "Ollama can't quantize this model's architecture. Remove the quantization option or use a supported model."
)

# Ollama 0.34.1 dropped create-time quantization of GGUF sources (ollama/ollama#14969); only safetensors imports can be quantized.
_GGUF_QUANTIZATION_REMOVED_IN = version.parse("0.34.1")

CREATE_ERROR_MAPPING = {
    # Ollama <= 0.34.0
    "quantization is only supported for F16": ALREADY_QUANTIZED_MESSAGE,
    # Ollama >= 0.34.1
    "create-time quantization is only supported for safetensors imports": GGUF_QUANTIZATION_UNSUPPORTED_MESSAGE,
    "unsupported --quantize": UNSUPPORTED_QUANTIZATION_TYPE_MESSAGE,
    "unsupported quantize type": UNSUPPORTED_QUANTIZATION_TYPE_MESSAGE,
    "MLX runtime is not available": MLX_RUNTIME_UNAVAILABLE_MESSAGE,
    "unsupported MLX architecture": UNSUPPORTED_MLX_ARCHITECTURE_MESSAGE,
}


def get_ollama_create_error_message(stderr: str) -> str | None:
    """Map a failed `ollama create` command's stderr to a user-facing message, if recognized."""
    return next((message for fragment, message in CREATE_ERROR_MAPPING.items() if fragment in stderr), None)


def supports_gguf_quantization(ollama_version: str) -> bool | None:
    """Return whether this Ollama version can quantize GGUF sources at create time, or None if the version can't be parsed."""
    try:
        return version.parse(ollama_version) < _GGUF_QUANTIZATION_REMOVED_IN
    except version.InvalidVersion:
        return None


_DECISION_MODELS_ADDED_IN = version.parse("0.35.0")


def decision_models_unsupported_message(ollama_version: str) -> str:
    """Return the user-facing error for a decision model on an Ollama too old to serve `/v1/systemone`."""
    return f"Decision models need Ollama {_DECISION_MODELS_ADDED_IN} or newer; this Ollama is {ollama_version}."


def supports_decision_models(ollama_version: str) -> bool | None:
    """Return whether this Ollama version serves decision models on `/v1/systemone`, or None if the version can't be parsed."""
    try:
        return version.parse(ollama_version) >= _DECISION_MODELS_ADDED_IN
    except version.InvalidVersion:
        return None
