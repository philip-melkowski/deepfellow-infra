# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""RouterService wired to a real EndpointRegistry (no Docker, backends are in-process callbacks)."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from starlette.requests import Request

from server.endpointregistry import ChatCompletionEndpoint, EndpointRegistry, RegistrationOptions, SimpleEndpoint
from server.model_tester import ModelTester
from server.models.api import ChatCompletionRequest, EmbeddingRequest, ModelProps, UserMessage
from server.models.models import InstallModelIn
from server.models.services import InstallServiceIn
from server.services.router_service import RouterService

EMBEDDINGS = {"simple question": [1.0, 0.0], "hard question": [0.0, 1.0]}


def _registry() -> EndpointRegistry:
    metrics = MagicMock()
    metrics.requests_in_flight.labels.return_value = MagicMock()
    metrics.request_duration.labels.return_value = MagicMock()
    metrics.request_total.labels.return_value = MagicMock()
    metrics.request_errors.labels.return_value = MagicMock()
    config = MagicMock()
    config.is_log_payloads_enabled.return_value = False
    return EndpointRegistry(config=config, parent_infra=MagicMock(), model_tester=MagicMock(), metrics_registry=metrics)


def _props(kind: str) -> ModelProps:
    return ModelProps(private=False, type=kind, endpoints=["/v1/chat/completions"])


def _json(body: object) -> StreamingResponse:
    return StreamingResponse(iter([json.dumps(body).encode()]), media_type="application/json")


async def _read(response: Any) -> dict[str, Any]:
    return json.loads(b"".join([chunk async for chunk in response.body_iterator]))


def _register_backends(registry: EndpointRegistry, fail_small: bool = False, small_status: int = 200) -> None:
    async def embed(body: EmbeddingRequest, _request: Request | None) -> StreamingResponse:
        assert isinstance(body.input, str)
        return _json({"data": [{"embedding": EMBEDDINGS[body.input.removeprefix("classification: ")]}]})

    registry.register_embeddings("nomic-embed-text", _props("embedding"), SimpleEndpoint(embed), RegistrationOptions(origin="local"))

    def chat(name: str, fail: bool = False) -> ChatCompletionEndpoint:
        async def on_chat(body: ChatCompletionRequest, _request: Request | None) -> StreamingResponse:
            if fail:
                raise HTTPException(503, f"{name} unavailable")
            if name == "small" and small_status != 200:
                return StreamingResponse(iter([b'{"error": "gone"}']), status_code=small_status, media_type="application/json")
            return _json({"answered_by": name, "requested": body.model, "choices": [{"message": {"content": f"hello from {name}"}}]})

        endpoint = ChatCompletionEndpoint()
        endpoint.on_chat_completion = on_chat
        return endpoint

    registry.register_chat_completion("small", _props("llm"), chat("small", fail_small), RegistrationOptions(origin="local"))
    registry.register_chat_completion("large", _props("llm"), chat("large"), RegistrationOptions(origin="local"))


async def _router(registry: EndpointRegistry, tmp_path: Path) -> RouterService:
    path = tmp_path / "index.npz"
    np.savez(
        path,
        vectors=np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32),
        labels=np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
    )
    svc = RouterService(MagicMock(), registry, MagicMock(), MagicMock(), MagicMock(), MagicMock(gpus=[]), MagicMock())
    options = InstallServiceIn(spec={"small_model": "small", "large_model": "large", "index_path": str(path), "k": "2", "threshold": "0.5"})
    svc.instances_info["default"].installed = await (await svc._install_instance("default", options)).wait()  # pyright: ignore[reportPrivateUsage]
    await (await svc._install_model("default", "router", InstallModelIn(spec={}))).wait()  # pyright: ignore[reportPrivateUsage]
    return svc


def _request(text: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(model="router", messages=[UserMessage(content=text)])


@pytest.mark.asyncio
async def test_router_chooses_backend_through_real_registry(tmp_path: Path) -> None:
    registry = _registry()
    _register_backends(registry)
    await _router(registry, tmp_path)

    simple = await registry.execute_chat_completion(_request("simple question"))
    hard = await registry.execute_chat_completion(_request("hard question"))

    assert simple.headers["X-Router-Model"] == "small"
    assert hard.headers["X-Router-Model"] == "large"
    assert (await _read(simple))["answered_by"] == "small"
    assert (await _read(hard))["answered_by"] == "large"
    assert [m.name for m in registry.list_models() if m.name == "router"] == ["router"]


@pytest.mark.asyncio
async def test_router_falls_back_to_large_when_small_backend_fails(tmp_path: Path) -> None:
    registry = _registry()
    _register_backends(registry, fail_small=True)
    await _router(registry, tmp_path)

    response = await registry.execute_chat_completion(_request("simple question"))

    assert response.headers["X-Router-Model"] == "large"
    assert response.headers["X-Router-Layer"] == "fallback"
    assert (await _read(response))["answered_by"] == "large"


@pytest.mark.asyncio
async def test_router_falls_back_to_large_when_embedding_model_is_missing(tmp_path: Path) -> None:
    registry = _registry()
    _register_backends(registry)
    registry.embeddings_endpoints.models.clear()
    await _router(registry, tmp_path)

    response = await registry.execute_chat_completion(_request("simple question"))

    assert response.headers["X-Router-Model"] == "large"
    assert response.headers["X-Router-Layer"] == "fallback"


@pytest.mark.asyncio
async def test_unregistering_router_removes_the_model(tmp_path: Path) -> None:
    registry = _registry()
    _register_backends(registry)
    svc = await _router(registry, tmp_path)

    await svc._uninstall_model("default", "router", MagicMock(purge=False))  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(HTTPException) as e:
        await registry.execute_chat_completion(_request("simple question"))
    assert e.value.status_code == 404


@pytest.mark.asyncio
async def test_router_passes_the_model_health_check(tmp_path: Path) -> None:
    registry = _registry()
    _register_backends(registry)
    await _router(registry, tmp_path)
    entry = next(e for e in registry.registry.values() if e.model_id == "router")

    result = await ModelTester().test_model(entry)

    assert isinstance(result, dict)
    assert result["result"] == "ok"
    assert entry.registered_model.type == "llm-v2"


@pytest.mark.asyncio
async def test_router_falls_back_when_small_backend_returns_404_like_a_retired_model(tmp_path: Path) -> None:
    registry = _registry()
    _register_backends(registry, small_status=404)
    await _router(registry, tmp_path)

    response = await registry.execute_chat_completion(_request("simple question"))

    assert response.status_code == 200
    assert response.headers["X-Router-Model"] == "large"
    assert response.headers["X-Router-Layer"] == "fallback"
    assert (await _read(response))["answered_by"] == "large"
