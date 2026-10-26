# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError
from starlette.datastructures import UploadFile

from server.models.api import (
    ChatCompletionReasoningConfig,
    ChatCompletionRequest,
    ComparisonFilter,
    CompletionLegacyRequest,
    CompoundFilter,
    CreateSpeechRequest,
    CreateTranscriptionRequest,
    EmbeddingRequest,
    FunctionToolCall,
    ImagesRequest,
    McpToolCall,
    Model,
    ModelProps,
    OllamaChatRequest,
    Reasoning,
    ReasoningConfig,
    ReasoningContentItem,
    ReasoningSummary,
    ResponsesRequest,
    SystemOneChoiceQuestion,
    SystemOneNoulQuestion,
    SystemOneRequest,
    SystemOneResponse,
    SystemOneScoreAnswer,
    SystemOneScoreQuestion,
)


def _make_model(**overrides: object) -> Model:
    props = ModelProps(private=False, type="llm", endpoints=["chat"])
    defaults: dict[str, object] = {"id": "rid", "name": "model", "type": "llm", "props": props, "usage": 0}
    defaults.update(overrides)
    return Model(**defaults)  # pyright: ignore[reportArgumentType]


def test_model_valid_capacity_forces_capacity_known_true() -> None:
    model = _make_model(capacity=5, capacity_known=False)
    assert model.capacity == 5
    assert model.capacity_known is True


def test_model_non_positive_capacity_coerced_to_unknown() -> None:
    model = _make_model(capacity=0, capacity_known=True)
    assert model.capacity is None
    assert model.capacity_known is False


def test_model_unbounded_capacity_left_untouched() -> None:
    model = _make_model(capacity=None, capacity_known=True)
    assert model.capacity is None
    assert model.capacity_known is True


def _make_filter(type_: str, key: str = "score", value: int = 10) -> ComparisonFilter:
    return ComparisonFilter(key=key, type=type_, value=value)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ("operator", "data", "expected"),
    [
        ("eq", {}, False),
        ("eq", {"score": 10}, True),
        ("eq", {"score": 5}, False),
        ("ne", {"score": 5}, True),
        ("ne", {"score": 10}, False),
        ("lt", {"score": 5}, True),
        ("lt", {"score": 15}, False),
        ("lte", {"score": 10}, True),
        ("lte", {"score": 9}, True),
        ("lte", {"score": 11}, False),
        ("gt", {"score": 15}, True),
        ("gt", {"score": 5}, False),
        ("gte", {"score": 10}, True),
        ("gte", {"score": 11}, True),
        ("gte", {"score": 9}, False),
    ],
    ids=[
        "missing_key",
        "eq_true",
        "eq_false",
        "ne_true",
        "ne_false",
        "lt_true",
        "lt_false",
        "lte_equal",
        "lte_less",
        "lte_false",
        "gt_true",
        "gt_false",
        "gte_equal",
        "gte_greater",
        "gte_false",
    ],
)
def test_comparison_filters(operator: str, data: dict[str, int], expected: bool) -> None:
    f = _make_filter(operator)
    assert f.matches(data) is expected


@pytest.mark.parametrize(
    ("filter_type", "filters", "data", "expected"),
    [
        ("and", [ComparisonFilter(key="a", type="eq", value=1), ComparisonFilter(key="b", type="eq", value=2)], {"a": 1, "b": 2}, True),
        ("and", [ComparisonFilter(key="a", type="eq", value=1), ComparisonFilter(key="b", type="eq", value=99)], {"a": 1, "b": 2}, False),
        ("or", [ComparisonFilter(key="a", type="eq", value=99), ComparisonFilter(key="b", type="eq", value=2)], {"a": 1, "b": 2}, True),
        ("or", [ComparisonFilter(key="a", type="eq", value=99), ComparisonFilter(key="b", type="eq", value=99)], {"a": 1, "b": 2}, False),
    ],
    ids=["and_all_match", "and_one_fails", "or_one_matches", "or_none_match"],
)
def test_compound_filter_logic(filter_type: str, filters: list[ComparisonFilter], data: dict[str, int], expected: bool):
    f = CompoundFilter(type=filter_type, filters=filters)  # pyright: ignore[reportArgumentType]
    assert f.matches(data) is expected


def _make_upload_file(content: bytes = b"audio") -> UploadFile:
    mock = MagicMock(spec=UploadFile)
    mock.read = AsyncMock(return_value=content)
    mock.filename = "audio.wav"
    mock.content_type = "audio/wav"
    return mock


@pytest.mark.asyncio
async def test_to_form_serializes_list_field() -> None:
    req = CreateTranscriptionRequest.model_construct(
        file=_make_upload_file(),
        model="whisper-1",
        include=["logprobs"],
        stream=None,
        temperature=None,
        timestamp_granularities=None,
        known_speaker_names=None,
        known_speaker_references=None,
        language=None,
        prompt=None,
        response_format=None,
        chunking_strategy=None,
    )

    form = await req.to_form(remove_model=False, rewrite_model_to=None)

    fields = {f[0]["name"]: f[2] for f in form._fields}  # type: ignore[attr-defined]
    assert "include[]" in fields


@pytest.mark.asyncio
async def test_to_form_serializes_bool_field() -> None:
    req = CreateTranscriptionRequest.model_construct(
        file=_make_upload_file(),
        model="whisper-1",
        include=None,
        stream=True,
        temperature=None,
        timestamp_granularities=None,
        known_speaker_names=None,
        known_speaker_references=None,
        language=None,
        prompt=None,
        response_format=None,
        chunking_strategy=None,
    )

    form = await req.to_form(remove_model=False, rewrite_model_to=None)

    fields = {f[0]["name"]: f[2] for f in form._fields}  # type: ignore[attr-defined]
    assert fields.get("stream") == "true"


@pytest.mark.asyncio
async def test_to_form_serializes_string_field() -> None:
    req = CreateTranscriptionRequest.model_construct(
        file=_make_upload_file(),
        model="whisper-1",
        include=None,
        stream=None,
        temperature=None,
        timestamp_granularities=None,
        known_speaker_names=None,
        known_speaker_references=None,
        language=None,
        prompt="transcribe this",
        response_format=None,
        chunking_strategy=None,
    )

    form = await req.to_form(remove_model=False, rewrite_model_to=None)

    fields = {f[0]["name"]: f[2] for f in form._fields}  # type: ignore[attr-defined]
    assert fields.get("prompt") == "transcribe this"


@pytest.mark.asyncio
async def test_to_form_serializes_float_field() -> None:
    req = CreateTranscriptionRequest.model_construct(
        file=_make_upload_file(),
        model="whisper-1",
        include=None,
        stream=None,
        temperature=0.5,
        timestamp_granularities=None,
        known_speaker_names=None,
        known_speaker_references=None,
        language=None,
        prompt=None,
        response_format=None,
        chunking_strategy=None,
    )

    form = await req.to_form(remove_model=False, rewrite_model_to=None)

    fields = {f[0]["name"]: f[2] for f in form._fields}  # type: ignore[attr-defined]
    assert fields.get("temperature") == "0.5"


@pytest.mark.asyncio
async def test_to_form_remove_model() -> None:
    model = "whisper-1"
    req = CreateTranscriptionRequest.model_construct(
        file=_make_upload_file(),
        model=model,
        include=None,
        stream=None,
        temperature=0.5,
        timestamp_granularities=None,
        known_speaker_names=None,
        known_speaker_references=None,
        language=None,
        prompt=None,
        response_format=None,
        chunking_strategy=None,
    )

    form = await req.to_form(remove_model=True, rewrite_model_to=None)

    fields = {f[0]["name"]: f[2] for f in form._fields}  # type: ignore[attr-defined]
    assert fields.get("model") is None


def _base_chat_req(**kwargs):  # type: ignore[no-untyped-def]
    return {"messages": [{"role": "user", "content": "hi"}], "model": "gpt-4", **kwargs}


@pytest.mark.parametrize("temperature", [-0.1, 2.1])
def test_chat_completion_temperature_out_of_range(temperature: float) -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest(**_base_chat_req(temperature=temperature))  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("temperature", [0.0, 1.0, 2.0])
def test_chat_completion_temperature_valid(temperature: float) -> None:
    req = ChatCompletionRequest(**_base_chat_req(temperature=temperature))  # pyright: ignore[reportArgumentType]

    assert req.temperature == temperature


@pytest.mark.parametrize("top_p", [-0.1, 1.1])
def test_chat_completion_top_p_out_of_range(top_p: float) -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest(**_base_chat_req(top_p=top_p))  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("n", [-1])
def test_chat_completion_n_out_of_range(n: int) -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest(**_base_chat_req(n=n))  # pyright: ignore[reportArgumentType]


def test_images_request_n_exceeds_limit() -> None:
    with pytest.raises(ValidationError):
        ImagesRequest(model="dall-e-3", prompt="cat", n=11)


@pytest.mark.parametrize("effort", ["none", "low", "medium", "high", "max"])
def test_chat_completion_reasoning_effort_valid(effort: str) -> None:
    req = ChatCompletionRequest(**_base_chat_req(reasoning_effort=effort))  # pyright: ignore[reportArgumentType]

    assert req.reasoning_effort == effort


def test_chat_completion_reasoning_effort_invalid() -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest(**_base_chat_req(reasoning_effort="extreme"))  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("effort", ["none", "low", "medium", "high", "max"])
def test_chat_completion_reasoning_nested_effort_valid(effort: str) -> None:
    req = ChatCompletionRequest(**_base_chat_req(reasoning={"effort": effort}))  # pyright: ignore[reportArgumentType]

    assert req.reasoning == ChatCompletionReasoningConfig(effort=effort)  # pyright: ignore[reportArgumentType]


def test_chat_completion_reasoning_nested_effort_invalid() -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest(**_base_chat_req(reasoning={"effort": "extreme"}))  # pyright: ignore[reportArgumentType]


def test_chat_completion_reasoning_fields_omitted_by_default() -> None:
    req = ChatCompletionRequest(**_base_chat_req())  # pyright: ignore[reportArgumentType]

    assert req.reasoning_effort is None
    assert req.reasoning is None
    assert "reasoning_effort" not in req.model_dump(exclude_none=True)
    assert "reasoning" not in req.model_dump(exclude_none=True)


def test_chat_completion_reasoning_both_fields_forwarded_without_disambiguation() -> None:
    req = ChatCompletionRequest(
        **_base_chat_req(reasoning_effort="low", reasoning={"effort": "high"})  # pyright: ignore[reportArgumentType]
    )

    raw = req.model_dump(exclude_none=True)
    assert raw["reasoning_effort"] == "low"
    assert raw["reasoning"] == {"effort": "high"}


_EXTRA_FIELDS_PAYLOAD: dict[str, object] = {
    "thinking_token_budget": 300,
    "chat_template_kwargs": {"enable_thinking": True},
    "repetition_detection": {"max_pattern_size": 20},
    "top_k": 40,
    "min_p": 0.05,
    "repetition_penalty": 1.1,
    "presence_penalty": 0.5,
    "frequency_penalty": -0.5,
    "logprobs": True,
    "parallel_tool_calls": False,
}


def test_chat_completion_extra_fields_accepted_and_dumped() -> None:
    req = ChatCompletionRequest(**_base_chat_req(**_EXTRA_FIELDS_PAYLOAD))  # pyright: ignore[reportArgumentType]

    raw = req.model_dump(exclude_none=True)
    assert {key: raw[key] for key in _EXTRA_FIELDS_PAYLOAD} == _EXTRA_FIELDS_PAYLOAD


def test_chat_completion_unlisted_extra_fields_are_dropped() -> None:
    req = ChatCompletionRequest(**_base_chat_req(guided_regex=".*", logits_processors=["os.system"]))  # pyright: ignore[reportArgumentType]

    raw = req.model_dump(exclude_none=True)
    assert "guided_regex" not in raw
    assert "logits_processors" not in raw


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("thinking_token_budget", -2),
        ("top_k", -2),
        ("top_k", 0),
        ("min_p", 1.5),
        ("repetition_penalty", 0),
        ("repetition_penalty", 2.5),
        ("presence_penalty", 2.5),
        ("frequency_penalty", -2.5),
        ("chat_template_kwargs", {"nested": {"a": 1}}),
        ("chat_template_kwargs", {"bad key": 1}),
        ("chat_template_kwargs", {"k": "x" * 1025}),
        ("chat_template_kwargs", {f"k{i}": i for i in range(17)}),
        ("repetition_detection", {f"k{i}": i for i in range(9)}),
    ],
)
def test_chat_completion_extra_fields_out_of_bounds(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest(**_base_chat_req(**{field: value}))  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("field", ["chat_template_kwargs", "repetition_detection"])
def test_chat_completion_flat_params_explicit_none_accepted(field: str) -> None:
    req = ChatCompletionRequest(**_base_chat_req(**{field: None}))  # pyright: ignore[reportArgumentType]

    assert getattr(req, field) is None


@pytest.mark.parametrize(("field", "value"), [("thinking_token_budget", -1), ("thinking_token_budget", 1_000_000), ("top_k", 50_000)])
def test_chat_completion_unbounded_extra_fields_accept_large_values(field: str, value: int) -> None:
    req = ChatCompletionRequest(**_base_chat_req(**{field: value}))  # pyright: ignore[reportArgumentType]

    assert req.model_dump(exclude_none=True)[field] == value


def test_completion_legacy_extra_fields_accepted() -> None:
    req = CompletionLegacyRequest(model="m", prompt="hi", thinking_token_budget=10, top_k=5, min_p=0.1, repetition_penalty=1.2)

    raw = req.model_dump(exclude_none=True)
    assert raw["thinking_token_budget"] == 10
    assert raw["top_k"] == 5


@pytest.mark.parametrize("value", [0, -2])
def test_completion_legacy_top_k_out_of_bounds(value: int) -> None:
    with pytest.raises(ValidationError):
        CompletionLegacyRequest(model="m", prompt="hi", top_k=value)


def test_chat_completion_response_format_json_object_without_schema() -> None:
    """`json_schema` is optional per the OpenAI spec — omitting it (the normal `json_object` case)
    must not raise, or forcing JSON mode on any downstream LLM call becomes impossible."""
    req = ChatCompletionRequest(**_base_chat_req(response_format={"type": "json_object"}))  # pyright: ignore[reportArgumentType]

    assert req.response_format is not None
    assert req.response_format.type == "json_object"
    assert req.response_format.json_schema is None


def test_images_request_n_at_limit() -> None:
    req = ImagesRequest(model="dall-e-3", prompt="cat", n=10)

    assert req.n == 10


def test_embedding_request_dimensions_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        EmbeddingRequest(input="text", model="ada", dimensions=0)


def test_create_speech_instructions_max_length() -> None:
    with pytest.raises(ValidationError):
        CreateSpeechRequest(input="hi", model="tts-1", instructions="x" * 4097)


def test_completion_legacy_best_of_constraint() -> None:
    with pytest.raises(ValidationError):
        CompletionLegacyRequest(prompt="hi", model="gpt-3.5", best_of=21)


def test_completion_legacy_max_tokens_zero_allowed() -> None:
    # ge=0 means 0 is valid now
    req = CompletionLegacyRequest(prompt="hi", model="gpt-3.5", max_tokens=0)

    assert req.max_tokens == 0


@pytest.mark.parametrize("effort", ["none", "minimal", "low", "medium", "high", "xhigh", "max"])
def test_reasoning_config_effort_valid(effort: str) -> None:
    config = ReasoningConfig(effort=effort)  # pyright: ignore[reportArgumentType]

    assert config.effort == effort


def test_reasoning_config_effort_invalid() -> None:
    with pytest.raises(ValidationError):
        ReasoningConfig(effort="extreme")  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("value", ["auto", "concise", "detailed"])
def test_reasoning_config_summary_and_generate_summary_valid(value: str) -> None:
    config = ReasoningConfig(summary=value, generate_summary=value)  # pyright: ignore[reportArgumentType]

    assert config.summary == value
    with pytest.deprecated_call():
        assert config.generate_summary == value


def test_reasoning_config_summary_invalid() -> None:
    with pytest.raises(ValidationError):
        ReasoningConfig(summary="verbose")  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("value", ["auto", "current_turn", "all_turns"])
def test_reasoning_config_context_valid(value: str) -> None:
    config = ReasoningConfig(context=value)  # pyright: ignore[reportArgumentType]

    assert config.context == value


def test_reasoning_config_context_invalid() -> None:
    with pytest.raises(ValidationError):
        ReasoningConfig(context="every_turn")  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("value", ["standard", "pro"])
def test_reasoning_config_mode_valid(value: str) -> None:
    config = ReasoningConfig(mode=value)  # pyright: ignore[reportArgumentType]

    assert config.mode == value


def test_reasoning_config_mode_invalid() -> None:
    with pytest.raises(ValidationError):
        ReasoningConfig(mode="turbo")  # pyright: ignore[reportArgumentType]


def test_responses_request_minimal_omits_optional_fields() -> None:
    req = ResponsesRequest(model="gpt-5.4-mini")

    raw = req.model_dump(exclude_none=True)
    for field in (
        "input",
        "instructions",
        "tool_choice",
        "tools",
        "temperature",
        "top_p",
        "stream",
        "max_tool_calls",
        "parallel_tool_calls",
        "include",
        "background",
        "metadata",
        "service_tier",
        "store",
        "truncation",
        "user",
    ):
        assert field not in raw, f"{field} should be omitted when not explicitly set"


def test_responses_request_explicit_values_are_forwarded() -> None:
    req = ResponsesRequest(
        model="gpt-5.4-mini",
        input="What is 2+2?",
        temperature=0.5,
        top_p=0.9,
        stream=True,
        store=False,
        truncation="auto",
        user="user-123",
    )

    raw = req.model_dump(exclude_none=True)
    assert raw["input"] == "What is 2+2?"
    assert raw["temperature"] == 0.5
    assert raw["top_p"] == 0.9
    assert raw["stream"] is True
    assert raw["store"] is False
    assert raw["truncation"] == "auto"
    assert raw["user"] == "user-123"


def test_responses_request_reasoning_omits_unset_fields() -> None:
    req = ResponsesRequest(model="gpt-5.4-mini", reasoning=ReasoningConfig(effort="high"))  # pyright: ignore[reportArgumentType]

    raw = req.model_dump(exclude_none=True)
    assert raw["reasoning"] == {"effort": "high"}


def test_reasoning_input_item_omits_id_status_encrypted_content() -> None:
    item = Reasoning(summary=[ReasoningSummary(text="thinking...")])

    raw = item.model_dump(exclude_none=True)
    assert "id" not in raw
    assert "status" not in raw
    assert "encrypted_content" not in raw
    assert raw["summary"] == [{"type": "summary_text", "text": "thinking..."}]


def test_reasoning_input_item_forwards_content() -> None:
    item = Reasoning(
        id="rs_abc123",
        summary=[ReasoningSummary(text="summary")],
        content=[ReasoningContentItem(text="raw reasoning text")],
    )

    raw = item.model_dump(exclude_none=True)
    assert raw["id"] == "rs_abc123"
    assert raw["content"] == [{"type": "reasoning_text", "text": "raw reasoning text"}]


def test_function_call_omits_id_and_call_id_when_not_supplied() -> None:
    item = FunctionToolCall(name="f", arguments="{}")

    raw = item.model_dump(exclude_none=True)
    assert "id" not in raw
    assert "call_id" not in raw


def test_mcp_tool_call_generates_unique_ids_per_instance() -> None:
    first = McpToolCall(server_label="s", name="f")
    second = McpToolCall(server_label="s", name="f")

    assert first.id != second.id


@pytest.mark.parametrize("value", [0, -1])
def test_model_props_rejects_non_positive_context_window(value: int) -> None:
    """0 is not a window a model can have, and consumers use these as a ceiling for their own sizing."""
    with pytest.raises(ValidationError):
        ModelProps(private=True, type="custom", endpoints=[], context_window=value)

    with pytest.raises(ValidationError):
        ModelProps(private=True, type="custom", endpoints=[], max_context_window=value)


def test_model_props_context_window_defaults_to_none() -> None:
    props = ModelProps(private=True, type="custom", endpoints=[])

    assert props.context_window is None
    assert props.max_context_window is None


SYSTEMONE_DOC_EXAMPLE = {
    "model": "nimble",
    "state": "Our checkout has returned 500 errors since 9am.",
    "questions": {
        "label": {
            "type": "choice",
            "instructions": "Which label fits this ticket?",
            "criteria": {"billing": "Payments and refunds", "bug": "Software errors", "account": "Login and account access"},
        }
    },
}


def test_systemone_request_parses_ollama_doc_example() -> None:
    request = SystemOneRequest.model_validate(SYSTEMONE_DOC_EXAMPLE)

    assert request.model == "nimble"
    assert isinstance(request.questions["label"], SystemOneChoiceQuestion)
    assert request.model_dump(exclude_none=True) == SYSTEMONE_DOC_EXAMPLE


def test_systemone_request_parses_every_question_type() -> None:
    request = SystemOneRequest.model_validate(
        {
            "model": "tev1:0.8b",
            "state": {"ticket": "500 errors"},
            "questions": {
                "urgent": {"type": "noul", "instructions": "Is this urgent?"},
                "severity": {"type": "score", "instructions": "How severe?", "criteria": ["trivial", "major"]},
            },
            "keep_alive": "5m",
        }
    )

    assert isinstance(request.questions["urgent"], SystemOneNoulQuestion)
    assert isinstance(request.questions["severity"], SystemOneScoreQuestion)
    assert request.state == {"ticket": "500 errors"}


def test_systemone_request_keeps_null_choice_criteria_when_dumped_without_none() -> None:
    request = SystemOneRequest.model_validate(
        {
            "model": "nimble",
            "state": "x",
            "questions": {"label": {"type": "choice", "instructions": "i", "criteria": {"bug": None, "billing": "Payments"}}},
        }
    )

    dumped = request.model_dump(exclude_none=True)

    assert dumped["questions"]["label"]["criteria"] == {"bug": None, "billing": "Payments"}


def test_systemone_request_omits_unset_optional_fields_when_dumped_without_none() -> None:
    request = SystemOneRequest.model_validate({"model": "nimble", "state": "x", "questions": {"u": {"type": "noul", "instructions": "i"}}})

    assert request.model_dump(exclude_none=True) == {
        "model": "nimble",
        "state": "x",
        "questions": {"u": {"type": "noul", "instructions": "i"}},
    }


def test_systemone_request_keeps_unknown_fields() -> None:
    request = SystemOneRequest.model_validate(
        {
            "model": "nimble",
            "state": "x",
            "questions": {"u": {"type": "noul", "instructions": "i", "future_option": 1}},
            "future_field": True,
        }
    )

    dumped = request.model_dump(exclude_none=True)

    assert dumped["future_field"] is True
    assert dumped["questions"]["u"]["future_option"] == 1


@pytest.mark.parametrize(
    "body",
    [
        {"model": "nimble", "state": "x"},
        {"model": "nimble", "state": "x", "questions": {"q": {"type": "rank", "instructions": "i"}}},
        {"model": "nimble", "state": "x", "questions": {"q": {"type": "choice", "instructions": "i"}}},
        {"state": "x", "questions": {"q": {"type": "noul", "instructions": "i"}}},
        {"model": "nimble", "questions": {"q": {"type": "noul", "instructions": "i"}}},
    ],
    ids=["missing-questions", "unknown-question-type", "choice-without-criteria", "missing-model", "missing-state"],
)
def test_systemone_request_rejects_invalid_body(body: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(body)


def test_systemone_request_accepts_questions_without_instructions() -> None:
    request = SystemOneRequest.model_validate(
        {
            "model": "Cloudflare/clef-omni",
            "state": "x",
            "questions": {
                "label": {"type": "choice", "criteria": {"bug": None}},
                "urgent": {"type": "noul"},
                "severity": {"type": "score", "criteria": ["low", "high"]},
            },
        }
    )

    dumped = request.model_dump(exclude_none=True)

    assert all("instructions" not in question for question in dumped["questions"].values())


def test_systemone_request_keeps_null_state_when_dumped_without_none() -> None:
    request = SystemOneRequest.model_validate(
        {"model": "LiquidAI/d1-3B", "state": None, "questions": {"u": {"type": "noul"}}, "chat_template_kwargs": {"enable_thinking": False}}
    )

    assert request.model_dump(exclude_none=True) == {
        "model": "LiquidAI/d1-3B",
        "state": None,
        "questions": {"u": {"type": "noul"}},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    assert json.loads(request.model_dump_json(exclude_none=True))["state"] is None


def test_systemone_response_parses_every_answer_type() -> None:
    response = SystemOneResponse.model_validate(
        {
            "model": "tev1:0.8b",
            "answers": {
                "label": {"type": "choice", "choice": "bug", "probabilities": {"bug": 0.9, "billing": 0.1}, "confidence": 0.8},
                "urgent": {"type": "noul", "noul": 0.71},
                "severity": {
                    "type": "score",
                    "score": 2.36,
                    "legend": {"0": "trivial", "1": "minor", "2": "major", "3": "critical"},
                    "probabilities": {"0": 0.01, "1": 0.02, "2": 0.58, "3": 0.39},
                    "confidence": 0.58,
                },
            },
            "usage": {"input_tokens": 406, "output_tokens": 0},
        }
    )

    assert isinstance(response.answers["severity"], SystemOneScoreAnswer)
    assert response.usage.input_tokens == 406


_CHAT_BASE: dict[str, object] = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


@pytest.mark.parametrize("overrides", [{"max_tokens": 0}, {"max_completion_tokens": 0}])
def test_chat_completion_request_rejects_non_positive_token_limits(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate({**_CHAT_BASE, **overrides})


@pytest.mark.parametrize("options", [{"num_predict": -3}, {"temperature": 3.0}])
def test_ollama_chat_request_rejects_invalid_options(options: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        OllamaChatRequest.model_validate({**_CHAT_BASE, "options": options})


def test_ollama_chat_request_drops_unsupported_options() -> None:
    options = {"temperature": 0.5, "num_gpu": 1, "num_thread": 4, "tfs_z": 1.0, "mirostat": 1, "unknown_option": 1}

    body = OllamaChatRequest.model_validate({**_CHAT_BASE, "options": options})

    assert body.model_dump(exclude_none=True)["options"] == {"temperature": 0.5}
