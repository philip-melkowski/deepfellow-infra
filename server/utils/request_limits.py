# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Upper bounds for client-supplied generation parameters, configurable via `/admin/config`."""

from fastapi import HTTPException
from pydantic import BaseModel

from server.config import AppSettings
from server.models.api import OllamaChatRequest, OllamaOptions


def _check(name: str, value: object, limit: int) -> None:
    if isinstance(value, int | float) and not isinstance(value, bool) and value > limit:
        raise HTTPException(status_code=422, detail=f"`{name}` must not exceed {limit}.")


def enforce_generation_limits(body: BaseModel, config: AppSettings) -> None:
    """Reject with 422 a request whose generation parameters exceed the configured limits.

    Covers the parameters of every LLM request type (chat, completions, responses, messages, Ollama chat) that
    the body defines, plus the Ollama `options` object.
    """
    for name in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
        _check(name, getattr(body, name, None), config.max_generated_tokens)
    _check("n", getattr(body, "n", None), config.max_choices)
    _check("top_k", getattr(body, "top_k", None), config.max_top_k)

    options = getattr(body, "options", None)
    if isinstance(options, BaseModel):
        _check("options.num_predict", getattr(options, "num_predict", None), config.max_generated_tokens)
        for name in ("num_ctx", "num_keep", "repeat_last_n"):
            _check(f"options.{name}", getattr(options, name, None), config.max_context_tokens)
        _check("options.top_k", getattr(options, "top_k", None), config.max_top_k)


def apply_default_generation_limits(body: BaseModel, config: AppSettings) -> None:
    """Cap generation that the client left unbounded, where the backend would otherwise pick its own limit.

    Ollama treats a missing or negative `num_predict` as 10 x `num_ctx`, which can exceed `max_generated_tokens`.
    """
    if not isinstance(body, OllamaChatRequest):
        return
    if body.options is None:
        body.options = OllamaOptions()
    if body.options.num_predict is None or body.options.num_predict <= 0:
        body.options.num_predict = config.max_generated_tokens
