# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.responses import Response, StreamingResponse

from server.models.api import ChatCompletionRequest, UserMessage
from server.models.models import InstallModelIn, ListModelsFilters
from server.models.services import InstallServiceIn, UninstallServiceIn
from server.services.router_service import ROUTER_MODEL_ID, RouterInstalledInfo, RouterService


@pytest.fixture
def index_path(tmp_path: Path) -> str:
    path = tmp_path / "index.npz"
    vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32)
    np.savez(path, vectors=vectors, labels=np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32))
    return str(path)


@pytest.fixture
def registry() -> MagicMock:
    registry = MagicMock()
    registry.register_chat_completion.return_value = "reg-1"
    return registry


@pytest.fixture
def svc(registry: MagicMock) -> RouterService:
    deps: dict[str, Any] = {
        "config": MagicMock(),
        "endpoint_registry": registry,
        "service_provider": MagicMock(),
        "model_downloader": MagicMock(),
        "docker_service": MagicMock(),
        "hardware": MagicMock(gpus=[]),
        "services_manager": MagicMock(),
    }
    return RouterService(**deps)


def _spec(index_path: str, **extra: Any) -> InstallServiceIn:
    return InstallServiceIn(
        spec={"small_model": "small", "large_model": "large", "index_path": index_path, "k": "2", "threshold": "0.5", **extra}
    )


def _json_response(body: object) -> StreamingResponse:
    return StreamingResponse(iter([json.dumps(body).encode()]), media_type="application/json")


async def _echo_model(body: ChatCompletionRequest, _request: object) -> StreamingResponse:
    return _json_response({"model": body.model})


def _embedding(vector: list[float]) -> StreamingResponse:
    return _json_response({"data": [{"embedding": vector}]})


async def _installed(svc: RouterService, index_path: str, **extra: Any) -> RouterInstalledInfo:
    promise = await svc._install_instance("default", _spec(index_path, **extra))  # pyright: ignore[reportPrivateUsage]
    info = await promise.wait()
    svc.instances_info["default"].installed = info
    return info


def _request(text: str = "hello") -> ChatCompletionRequest:
    return ChatCompletionRequest(model=ROUTER_MODEL_ID, messages=[UserMessage(content=text)])


@pytest.mark.asyncio
async def test_install_instance_parses_flat_options(svc: RouterService, index_path: str) -> None:
    info = await _installed(svc, index_path, force_large_keywords="prove, Derive", force_large_over_chars="500")
    assert info.parsed_options.k == 2
    assert info.parsed_options.threshold == 0.5
    assert info.parsed_options.rules.force_large_keywords == ["prove", "Derive"]
    assert info.parsed_options.rules.force_large_over_chars == 500


@pytest.mark.asyncio
async def test_install_instance_rejects_bad_options(svc: RouterService, index_path: str) -> None:
    with pytest.raises(HTTPException, match="Invalid router options") as e:
        await svc._install_instance("default", InstallServiceIn(spec={"small_model": "s"}))  # pyright: ignore[reportPrivateUsage]
    assert e.value.status_code == 400
    with pytest.raises(HTTPException, match="Cannot load the router index"):
        await svc._install_instance("default", _spec("/nonexistent/index.npz"))  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(HTTPException, match="must not be 'router'"):
        await svc._install_instance("default", _spec(index_path, small_model=ROUTER_MODEL_ID))  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_install_and_uninstall_model_registers_endpoint(svc: RouterService, registry: MagicMock, index_path: str) -> None:
    info = await _installed(svc, index_path)
    result = await (await svc._install_model("default", ROUTER_MODEL_ID, InstallModelIn(spec={}))).wait()  # pyright: ignore[reportPrivateUsage]
    assert result.status == "OK"
    assert info.models[ROUTER_MODEL_ID].registration_id == "reg-1"
    kwargs = registry.register_chat_completion.call_args.kwargs
    assert kwargs["model"] == ROUTER_MODEL_ID
    assert kwargs["props"].endpoints == ["/v1/chat/completions"]
    assert kwargs["endpoint"].on_chat_completion is not None

    await svc._uninstall_model("default", ROUTER_MODEL_ID, MagicMock(purge=False))  # pyright: ignore[reportPrivateUsage]
    registry.unregister_chat_completion.assert_called_once_with(ROUTER_MODEL_ID, "reg-1")
    assert info.models == {}


@pytest.mark.asyncio
async def test_install_model_unknown_id(svc: RouterService, index_path: str) -> None:
    await _installed(svc, index_path)
    with pytest.raises(HTTPException, match="Model not found"):
        await svc._install_model("default", "other", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]


async def _route(svc: RouterService, registry: MagicMock, index_path: str, vector: list[float]) -> StreamingResponse:
    await _installed(svc, index_path)
    await (await svc._install_model("default", ROUTER_MODEL_ID, InstallModelIn(spec={}))).wait()  # pyright: ignore[reportPrivateUsage]
    registry.execute_embeddings = AsyncMock(return_value=_embedding(vector))
    registry.execute_chat_completion = AsyncMock(side_effect=_echo_model)
    handler = registry.register_chat_completion.call_args.kwargs["endpoint"].on_chat_completion
    return await handler(_request(), None)


@pytest.mark.asyncio
async def test_handler_routes_simple_to_small(svc: RouterService, registry: MagicMock, index_path: str) -> None:
    response = await _route(svc, registry, index_path, [1.0, 0.0])
    assert registry.execute_chat_completion.call_args.args[0].model == "small"
    assert response.headers["X-Router-Model"] == "small"
    assert response.headers["X-Router-Layer"] == "knn"
    assert float(response.headers["X-Router-Score"]) == pytest.approx(0.0)
    assert registry.execute_embeddings.call_args.args[0].input == "classification: hello"


@pytest.mark.asyncio
async def test_handler_routes_complex_to_large(svc: RouterService, registry: MagicMock, index_path: str) -> None:
    response = await _route(svc, registry, index_path, [0.0, 1.0])
    assert registry.execute_chat_completion.call_args.args[0].model == "large"
    assert response.headers["X-Router-Model"] == "large"


@pytest.mark.asyncio
async def test_handler_uses_large_when_embedding_fails(svc: RouterService, registry: MagicMock, index_path: str) -> None:
    await _installed(svc, index_path)
    await (await svc._install_model("default", ROUTER_MODEL_ID, InstallModelIn(spec={}))).wait()  # pyright: ignore[reportPrivateUsage]
    registry.execute_embeddings = AsyncMock(side_effect=HTTPException(404, "Model not found"))
    registry.execute_chat_completion = AsyncMock(side_effect=_echo_model)
    handler = registry.register_chat_completion.call_args.kwargs["endpoint"].on_chat_completion
    response = await handler(_request(), None)
    assert registry.execute_chat_completion.call_args.args[0].model == "large"
    assert response.headers["X-Router-Layer"] == "fallback"
    assert "X-Router-Score" not in response.headers


@pytest.mark.asyncio
async def test_handler_retries_with_large_when_small_is_unavailable(svc: RouterService, registry: MagicMock, index_path: str) -> None:
    await _installed(svc, index_path)
    await (await svc._install_model("default", ROUTER_MODEL_ID, InstallModelIn(spec={}))).wait()  # pyright: ignore[reportPrivateUsage]
    registry.execute_embeddings = AsyncMock(return_value=_embedding([1.0, 0.0]))

    async def execute(body: ChatCompletionRequest, _request: object) -> StreamingResponse:
        if body.model == "small":
            raise HTTPException(404, "Model not found")
        return _json_response({"model": body.model})

    registry.execute_chat_completion = AsyncMock(side_effect=execute)
    handler = registry.register_chat_completion.call_args.kwargs["endpoint"].on_chat_completion
    response = await handler(_request(), None)
    assert response.headers["X-Router-Model"] == "large"
    assert response.headers["X-Router-Layer"] == "fallback"
    assert registry.execute_chat_completion.await_count == 2


@pytest.mark.asyncio
async def test_handler_does_not_retry_when_large_fails(svc: RouterService, registry: MagicMock, index_path: str) -> None:
    await _installed(svc, index_path)
    await (await svc._install_model("default", ROUTER_MODEL_ID, InstallModelIn(spec={}))).wait()  # pyright: ignore[reportPrivateUsage]
    registry.execute_embeddings = AsyncMock(return_value=_embedding([0.0, 1.0]))
    registry.execute_chat_completion = AsyncMock(side_effect=HTTPException(503, "down"))
    handler = registry.register_chat_completion.call_args.kwargs["endpoint"].on_chat_completion
    with pytest.raises(HTTPException, match="down"):
        await handler(_request(), None)
    assert registry.execute_chat_completion.await_count == 1


@pytest.mark.asyncio
async def test_list_and_get_model(svc: RouterService, index_path: str) -> None:
    assert (await svc.list_models(None, ListModelsFilters())).list == []
    await _installed(svc, index_path)
    models = (await svc.list_models(None, ListModelsFilters())).list
    assert [m.id for m in models] == [ROUTER_MODEL_ID]
    assert (await svc.get_model("default", ROUTER_MODEL_ID)).type == "llm"
    with pytest.raises(HTTPException, match="Model not found"):
        await svc.get_model("default", "other")


@pytest.mark.asyncio
async def test_uninstall_instance_unregisters_router(svc: RouterService, registry: MagicMock, index_path: str) -> None:
    await _installed(svc, index_path)
    await (await svc._install_model("default", ROUTER_MODEL_ID, InstallModelIn(spec={}))).wait()  # pyright: ignore[reportPrivateUsage]
    await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]
    registry.unregister_chat_completion.assert_called_once()
    assert svc.instances_info["default"].installed is None


def test_spec_exposes_all_options(svc: RouterService) -> None:
    names = {f.name for f in svc.get_spec().fields}
    assert {"small_model", "large_model", "index_path", "k", "threshold", "embedding_model"} <= names


@pytest.mark.asyncio
async def test_metadata_and_config_roundtrip(svc: RouterService, index_path: str) -> None:
    assert svc.get_type() == "router"
    assert svc.get_description()
    assert svc.get_size() == ""
    assert svc.get_model_spec().fields == []
    assert svc.get_custom_model_spec() is None
    assert svc._load_download_info({}).__class__.__name__ == "RouterDownloadedInfo"  # pyright: ignore[reportPrivateUsage]
    assert svc._generate_instance_config("default", None, None).models == []  # pyright: ignore[reportPrivateUsage]

    assert svc.get_installed_info("default") is False
    info = await _installed(svc, index_path)
    await (await svc._install_model("default", ROUTER_MODEL_ID, InstallModelIn(spec={"a": 1}))).wait()  # pyright: ignore[reportPrivateUsage]
    assert svc.get_installed_info("default") == info.options.spec
    config = svc._generate_instance_config("default", info, None)  # pyright: ignore[reportPrivateUsage]
    assert [m.model_id for m in config.models or []] == [ROUTER_MODEL_ID]
    assert info.models[ROUTER_MODEL_ID].get_info().spec == {"a": 1}
    await svc.stop_instance("default")


@pytest.mark.asyncio
async def test_install_model_twice_is_idempotent(svc: RouterService, registry: MagicMock, index_path: str) -> None:
    await _installed(svc, index_path)
    await (await svc._install_model("default", ROUTER_MODEL_ID, InstallModelIn(spec={}))).wait()  # pyright: ignore[reportPrivateUsage]
    again = await (await svc._install_model("default", ROUTER_MODEL_ID, InstallModelIn(spec={}))).wait()  # pyright: ignore[reportPrivateUsage]
    assert again.details == "Already installed"
    registry.register_chat_completion.assert_called_once()


@pytest.mark.asyncio
async def test_uninstall_missing_model_and_instance_are_noops(svc: RouterService, registry: MagicMock, index_path: str) -> None:
    await _installed(svc, index_path)
    await svc._uninstall_model("default", ROUTER_MODEL_ID, MagicMock(purge=False))  # pyright: ignore[reportPrivateUsage]
    registry.unregister_chat_completion.assert_not_called()
    svc.instances_info["default"].installed = None
    await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_uninstall_instance_with_purge_resets_download_flag(svc: RouterService, index_path: str) -> None:
    await _installed(svc, index_path)
    svc.service_downloaded = True
    await svc._uninstall_instance("default", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]
    assert svc.service_downloaded is False


@pytest.mark.asyncio
async def test_list_models_unknown_instance(svc: RouterService) -> None:
    with pytest.raises(HTTPException, match="doesn't exist"):
        await svc.list_models("nope", ListModelsFilters())


@pytest.mark.asyncio
async def test_embed_accepts_plain_response_and_rejects_errors(svc: RouterService, registry: MagicMock, index_path: str) -> None:
    info = await _installed(svc, index_path)
    registry.execute_embeddings = AsyncMock(
        return_value=Response(content=json.dumps({"data": [{"embedding": [0.5, 1]}]}), media_type="application/json")
    )
    assert await svc._embed(info.parsed_options, "x") == [0.5, 1.0]  # pyright: ignore[reportPrivateUsage]
    registry.execute_embeddings = AsyncMock(return_value=Response(content=b"{}", status_code=500))
    with pytest.raises(RuntimeError, match="HTTP 500"):
        await svc._embed(info.parsed_options, "x")  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_handler_rejects_non_streaming_response(svc: RouterService, registry: MagicMock, index_path: str) -> None:
    await _installed(svc, index_path)
    await (await svc._install_model("default", ROUTER_MODEL_ID, InstallModelIn(spec={}))).wait()  # pyright: ignore[reportPrivateUsage]
    registry.execute_embeddings = AsyncMock(return_value=_embedding([1.0, 0.0]))
    registry.execute_chat_completion = AsyncMock(return_value=Response(content=b"{}"))
    handler = registry.register_chat_completion.call_args.kwargs["endpoint"].on_chat_completion
    with pytest.raises(HTTPException, match="non-streaming"):
        await handler(_request(), None)


def _status_response(status: int) -> StreamingResponse:
    return StreamingResponse(iter([b'{"error": "upstream"}']), status_code=status, media_type="application/json")


async def _handler_with_small(svc: RouterService, registry: MagicMock, index_path: str, small: Any) -> Any:
    await _installed(svc, index_path)
    await (await svc._install_model("default", ROUTER_MODEL_ID, InstallModelIn(spec={}))).wait()  # pyright: ignore[reportPrivateUsage]
    registry.execute_embeddings = AsyncMock(return_value=_embedding([1.0, 0.0]))

    async def execute(body: ChatCompletionRequest, _request: object) -> Any:
        return small() if body.model == "small" else _json_response({"model": body.model})

    registry.execute_chat_completion = AsyncMock(side_effect=execute)
    return registry.register_chat_completion.call_args.kwargs["endpoint"].on_chat_completion


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [404, 408, 429, 500, 503])
async def test_handler_falls_back_when_small_answers_with_error_status(
    svc: RouterService, registry: MagicMock, index_path: str, status: int
) -> None:
    handler = await _handler_with_small(svc, registry, index_path, lambda: _status_response(status))
    response = await handler(_request(), None)
    assert response.status_code == 200
    assert response.headers["X-Router-Model"] == "large"
    assert response.headers["X-Router-Layer"] == "fallback"
    assert registry.execute_chat_completion.await_count == 2


@pytest.mark.asyncio
async def test_handler_falls_back_when_small_answers_with_plain_error_response(
    svc: RouterService, registry: MagicMock, index_path: str
) -> None:
    handler = await _handler_with_small(svc, registry, index_path, lambda: Response(content=b"", status_code=502))
    response = await handler(_request(), None)
    assert response.headers["X-Router-Model"] == "large"


@pytest.mark.asyncio
async def test_handler_returns_client_errors_of_small_model_unchanged(svc: RouterService, registry: MagicMock, index_path: str) -> None:
    handler = await _handler_with_small(svc, registry, index_path, lambda: _status_response(400))
    response = await handler(_request(), None)
    assert response.status_code == 400
    assert response.headers["X-Router-Model"] == "small"
    assert registry.execute_chat_completion.await_count == 1


def _heuristics_file(tmp_path: Path) -> str:
    path = tmp_path / "l1.json"
    path.write_text(json.dumps({"complex_phrases": ["prove"], "simple_phrases": ["what is"]}), encoding="utf-8")
    return str(path)


@pytest.mark.asyncio
async def test_install_instance_loads_heuristics(svc: RouterService, index_path: str, tmp_path: Path) -> None:
    info = await _installed(svc, index_path, heuristics_path=_heuristics_file(tmp_path), heuristic_threshold="1")
    assert info.parsed_options.heuristics_path is not None
    assert info.parsed_options.heuristic_threshold == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_install_instance_rejects_missing_heuristics(svc: RouterService, index_path: str, tmp_path: Path) -> None:
    with pytest.raises(HTTPException, match="Cannot load the router heuristics") as e:
        await svc._install_instance("default", _spec(index_path, heuristics_path=str(tmp_path / "nope.json")))  # pyright: ignore[reportPrivateUsage]
    assert e.value.status_code == 400


@pytest.mark.asyncio
async def test_handler_sends_prompt_over_heuristic_threshold_to_large_without_embedding(
    svc: RouterService, registry: MagicMock, index_path: str, tmp_path: Path
) -> None:
    await _installed(svc, index_path, heuristics_path=_heuristics_file(tmp_path), heuristic_threshold="1")
    await (await svc._install_model("default", ROUTER_MODEL_ID, InstallModelIn(spec={}))).wait()  # pyright: ignore[reportPrivateUsage]
    registry.execute_embeddings = AsyncMock()
    registry.execute_chat_completion = AsyncMock(side_effect=_echo_model)
    handler = registry.register_chat_completion.call_args.kwargs["endpoint"].on_chat_completion
    response = await handler(_request("prove it"), None)
    assert response.headers["X-Router-Layer"] == "heuristics"
    assert response.headers["X-Router-Model"] == "large"
    registry.execute_embeddings.assert_not_called()


def _decider_file(tmp_path: Path) -> str:
    path = tmp_path / "l3.json"
    params = {
        "model": "decider",
        "questions": {"hard": {"type": "noul", "instructions": "Is it hard?"}},
        "intercept": 0.0,
        "knn_weight": 1.0,
        "weights": {"hard": 1.0},
        "zone_low": 0.4,
        "zone_high": 0.6,
    }
    path.write_text(json.dumps(params), encoding="utf-8")
    return str(path)


@pytest.mark.asyncio
async def test_install_instance_rejects_bad_decider_params(svc: RouterService, index_path: str, tmp_path: Path) -> None:
    with pytest.raises(HTTPException, match="Cannot load the router decision model parameters") as e:
        await svc._install_instance("default", _spec(index_path, decider_path=str(tmp_path / "nope.json")))  # pyright: ignore[reportPrivateUsage]
    assert e.value.status_code == 400
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"model": "decider"}), encoding="utf-8")
    with pytest.raises(HTTPException, match="Cannot load the router decision model parameters"):
        await svc._install_instance("default", _spec(index_path, decider_path=str(bad)))  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_handler_asks_decision_model_in_the_uncertainty_zone(
    svc: RouterService, registry: MagicMock, index_path: str, tmp_path: Path
) -> None:
    await _installed(svc, index_path, decider_path=_decider_file(tmp_path))
    await (await svc._install_model("default", ROUTER_MODEL_ID, InstallModelIn(spec={}))).wait()  # pyright: ignore[reportPrivateUsage]
    registry.execute_embeddings = AsyncMock(return_value=_embedding([1.0, 1.0]))
    registry.execute_systemone = AsyncMock(return_value=_json_response({"answers": {"hard": {"type": "noul", "noul": 0.95}}}))
    registry.execute_chat_completion = AsyncMock(side_effect=_echo_model)
    handler = registry.register_chat_completion.call_args.kwargs["endpoint"].on_chat_completion
    response = await handler(_request("tricky"), None)
    assert response.headers["X-Router-Layer"] == "decider"
    assert response.headers["X-Router-Model"] == "large"
    assert float(response.headers["X-Router-Decider-Score"]) > 0.5
    body = registry.execute_systemone.call_args.args[0]
    assert (body.model, body.state, list(body.questions)) == ("decider", "tricky", ["hard"])


@pytest.mark.asyncio
async def test_handler_falls_back_to_large_when_decision_model_errors(
    svc: RouterService, registry: MagicMock, index_path: str, tmp_path: Path
) -> None:
    await _installed(svc, index_path, decider_path=_decider_file(tmp_path))
    await (await svc._install_model("default", ROUTER_MODEL_ID, InstallModelIn(spec={}))).wait()  # pyright: ignore[reportPrivateUsage]
    registry.execute_embeddings = AsyncMock(return_value=_embedding([1.0, 1.0]))
    registry.execute_systemone = AsyncMock(return_value=Response(status_code=503, content=b"{}"))
    registry.execute_chat_completion = AsyncMock(side_effect=_echo_model)
    handler = registry.register_chat_completion.call_args.kwargs["endpoint"].on_chat_completion
    response = await handler(_request("tricky"), None)
    assert response.headers["X-Router-Layer"] == "fallback"
    assert response.headers["X-Router-Model"] == "large"
