# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

import pytest
from fastapi import HTTPException

from server.config import AppSettings
from server.models.api import ChatCompletionRequest, MessagesRequest, OllamaChatRequest, ResponsesRequest
from server.utils.request_limits import apply_default_generation_limits, enforce_generation_limits

_MESSAGES = [{"role": "user", "content": "hi"}]


def _config(**overrides: int) -> AppSettings:
    config = AppSettings()  # pyright: ignore[reportCallIssue]
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def _expect_rejected(body: object, config: AppSettings, field: str) -> None:
    with pytest.raises(HTTPException) as exc:
        enforce_generation_limits(body, config)  # pyright: ignore[reportArgumentType]

    assert exc.value.status_code == 422
    assert field in str(exc.value.detail)


def test_chat_completion_rejects_max_tokens_above_limit() -> None:
    body = ChatCompletionRequest.model_validate({"model": "m", "messages": _MESSAGES, "max_tokens": 101})

    _expect_rejected(body, _config(max_generated_tokens=100), "max_tokens")


def test_chat_completion_rejects_n_above_limit() -> None:
    body = ChatCompletionRequest.model_validate({"model": "m", "messages": _MESSAGES, "n": 5})

    _expect_rejected(body, _config(max_choices=4), "`n`")


def test_chat_completion_accepts_values_at_limit() -> None:
    body = ChatCompletionRequest.model_validate({"model": "m", "messages": _MESSAGES, "max_completion_tokens": 100, "n": 4})

    enforce_generation_limits(body, _config(max_generated_tokens=100, max_choices=4))


def test_responses_rejects_max_output_tokens_above_limit() -> None:
    body = ResponsesRequest.model_validate({"model": "m", "input": "hi", "max_output_tokens": 101})

    _expect_rejected(body, _config(max_generated_tokens=100), "max_output_tokens")


def test_messages_rejects_max_tokens_and_top_k_above_limit() -> None:
    config = _config(max_generated_tokens=100, max_top_k=10)

    _expect_rejected(MessagesRequest.model_validate({"model": "m", "messages": _MESSAGES, "max_tokens": 101}), config, "max_tokens")
    _expect_rejected(MessagesRequest.model_validate({"model": "m", "messages": _MESSAGES, "max_tokens": 50, "top_k": 11}), config, "top_k")


@pytest.mark.parametrize(
    ("options", "field"),
    [
        ({"num_ctx": 1001}, "options.num_ctx"),
        ({"num_predict": 101}, "options.num_predict"),
        ({"top_k": 11}, "options.top_k"),
    ],
)
def test_ollama_chat_rejects_options_above_limit(options: dict[str, int], field: str) -> None:
    body = OllamaChatRequest.model_validate({"model": "m", "messages": _MESSAGES, "options": options})

    _expect_rejected(body, _config(max_context_tokens=1000, max_generated_tokens=100, max_top_k=10), field)


def test_ollama_chat_without_options_is_accepted() -> None:
    body = OllamaChatRequest.model_validate({"model": "m", "messages": _MESSAGES})

    enforce_generation_limits(body, _config())


@pytest.mark.parametrize("options", [None, {}, {"num_predict": -1}, {"num_predict": -2}, {"num_predict": 0}])
def test_ollama_chat_unbounded_num_predict_is_capped(options: dict[str, int] | None) -> None:
    payload: dict[str, object] = {"model": "m", "messages": _MESSAGES}
    if options is not None:
        payload["options"] = options
    body = OllamaChatRequest.model_validate(payload)

    apply_default_generation_limits(body, _config(max_generated_tokens=100))

    assert body.options is not None
    assert body.options.num_predict == 100


def test_ollama_chat_explicit_num_predict_is_kept() -> None:
    body = OllamaChatRequest.model_validate({"model": "m", "messages": _MESSAGES, "options": {"num_predict": 50, "temperature": 0.1}})

    apply_default_generation_limits(body, _config(max_generated_tokens=100))

    assert body.options is not None
    assert body.options.num_predict == 50
    assert body.options.temperature == 0.1


def test_non_ollama_body_is_left_untouched() -> None:
    body = ChatCompletionRequest.model_validate({"model": "m", "messages": _MESSAGES})

    apply_default_generation_limits(body, _config(max_generated_tokens=100))

    assert body.model_dump(exclude_none=True) == {"model": "m", "messages": _MESSAGES}
