## Purpose

Lets clients control reasoning/thinking behavior (enable, disable, or tune effort)
for reasoning-capable models on `/v1/chat/completions`, by accepting and forwarding
`reasoning_effort`/`reasoning.effort` to the backend instead of silently dropping
them. Backends behind this endpoint include both Ollama and DeepFellow's OpenAI
proxy, so accepted values cover both vocabularies.

## Requirements

### Requirement: Chat completions request accepts reasoning effort fields
The `/v1/chat/completions` endpoint SHALL accept an optional `reasoning_effort` field
and an optional `reasoning` object field (with a nested `effort` field) on the request
body, each independently accepting one of `none`, `minimal`, `low`, `medium`, `high`,
`xhigh`, or `max` - the union of values documented by Ollama and OpenAI, since this
endpoint proxies to both. Requests omitting both fields SHALL behave exactly as before
this change.

#### Scenario: Client sends flat reasoning_effort field
- **WHEN** a client POSTs to `/v1/chat/completions` with
  `{"model": "deepseek-r1", "messages": [...], "reasoning_effort": "medium"}`
- **THEN** the request SHALL pass validation and `reasoning_effort` SHALL be included
  in the JSON body forwarded to the backend

#### Scenario: Client sends nested reasoning.effort field
- **WHEN** a client POSTs to `/v1/chat/completions` with
  `{"model": "deepseek-r1", "messages": [...], "reasoning": {"effort": "high"}}`
- **THEN** the request SHALL pass validation and `reasoning` SHALL be included in the
  JSON body forwarded to the backend as `{"effort": "high"}`

#### Scenario: Client sends an invalid effort value
- **WHEN** a client POSTs to `/v1/chat/completions` with
  `{"model": "deepseek-r1", "messages": [...], "reasoning_effort": "extreme"}`
- **THEN** the request SHALL fail validation with a 422 error, since `"extreme"` is
  not one of the accepted values

#### Scenario: Client omits both reasoning fields
- **WHEN** a client POSTs to `/v1/chat/completions` without `reasoning_effort` or
  `reasoning` in the body
- **THEN** the request SHALL be processed exactly as it was before this change, with
  neither field present in the JSON body forwarded to the backend

#### Scenario: Client sends both fields with different values
- **WHEN** a client POSTs to `/v1/chat/completions` with both
  `"reasoning_effort": "low"` and `"reasoning": {"effort": "high"}` set to different
  values
- **THEN** both fields SHALL be forwarded as-is in the JSON body sent to the backend,
  with no disambiguation or precedence applied by DeepFellow

### Requirement: Reasoning fields are forwarded without altering streaming behavior
Enabling `reasoning_effort` or `reasoning` SHALL NOT change how DeepFellow proxies or
streams the response. Any reasoning-related content returned by the backend in the
response body (streaming or non-streaming) SHALL reach the client unmodified, since
DeepFellow forwards backend responses as an unparsed byte stream.

#### Scenario: Backend returns reasoning content in a streamed response
- **WHEN** a client requests `stream: true` together with `reasoning_effort: "medium"`
  against a reasoning-capable model, and the backend emits reasoning content in its
  streamed chunks
- **THEN** DeepFellow SHALL relay those chunks to the client unmodified, without
  filtering, parsing, or stripping any reasoning-related fields
- **Verified** (2026-07-13, direct call to Ollama with `deepseek-r1:1.5b`, bypassing
  DeepFellow): streamed chunks carry `choices[0].delta.reasoning` (a plain string,
  distinct from `delta.content`) while the model is thinking, e.g.
  `{"delta":{"role":"assistant","content":"","reasoning":"First"}}`; `content` stays
  empty until reasoning is complete, then the final answer streams normally in
  `delta.content`. Since DeepFellow forwards bytes unmodified, this exact shape is
  what SHALL reach the client once `reasoning_effort` survives request validation.

### Requirement: Backend-specific sampling fields are accepted and forwarded only to backends that support them
The `/v1/chat/completions` endpoint SHALL accept the optional fields `thinking_token_budget`,
`chat_template_kwargs`, `repetition_detection`, `top_k`, `min_p`, `repetition_penalty`,
`presence_penalty`, `frequency_penalty`, `logprobs` and `parallel_tool_calls`, and the
`/v1/completions` endpoint SHALL accept `thinking_token_budget`, `repetition_detection`, `top_k`,
`min_p` and `repetition_penalty`. Every other undeclared field SHALL still be dropped. Values SHALL
be bounded: `thinking_token_budget` and `top_k` at least -1 with no upper bound (the ceiling is
the model's context window or vocabulary, unknown to DeepFellow), `top_k` of 0 rejected because
SGLang does not accept it (-1 disables `top_k` on every backend), `min_p` 0-1, `repetition_penalty`
above 0 up to 2, `presence_penalty` and `frequency_penalty` -2-2; `chat_template_kwargs`
(at most 16 keys) and `repetition_detection` (at most 8 keys) SHALL be flat dicts of scalars with
identifier-like keys and strings up to 1024 characters. Out-of-bounds values SHALL fail with 422.

These fields SHALL be forwarded to a backend only when its proxy sets `forward_extra_fields`:
enabled for vLLM, SGLang, llama.cpp, Ollama (local and external), Docker Model Runner and proxies
to other mesh nodes; disabled by default, so remote providers (OpenAI, Google AI, Claude, DeepSeek,
Kimi, Ollama Cloud) never receive them. `presence_penalty`, `frequency_penalty` and `logprobs` on
`/v1/completions` remain forwarded to all backends as before.

#### Scenario: Self-hosted backend receives the thinking budget
- **WHEN** a client sends `thinking_token_budget: 300` for a model served by vLLM
- **THEN** the JSON body sent to vLLM SHALL contain `thinking_token_budget: 300`

#### Scenario: Remote provider does not receive backend-specific fields
- **WHEN** a client sends `thinking_token_budget: 300` or `top_k: 40` for a model proxied to OpenAI
- **THEN** neither field SHALL be present in the JSON body sent to the provider

#### Scenario: Fields pass through a mesh hop
- **WHEN** a request with `thinking_token_budget` targets a model registered from a mesh peer
- **THEN** the field SHALL be forwarded to the peer, which applies its own backend's rule

#### Scenario: Undeclared field is dropped
- **WHEN** a client sends `guided_regex` or `logits_processors`
- **THEN** the field SHALL not be forwarded to any backend
