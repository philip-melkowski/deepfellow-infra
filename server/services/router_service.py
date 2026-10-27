# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Router service: the model `router` sends each chat request to a small or a large model."""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import ValidationError
from starlette.requests import Request
from starlette.responses import Response as StarletteResponse

from server.endpointregistry import ChatCompletionEndpoint, RegistrationOptions
from server.models.api import ChatCompletionRequest, EmbeddingRequest, ModelProps
from server.models.models import (
    CustomModelSpecification,
    InstallModelIn,
    InstallModelOut,
    ListModelsFilters,
    ListModelsOut,
    ModelInfo,
    ModelSpecification,
    RetrieveModelOut,
    UninstallModelIn,
)
from server.models.services import (
    InstallServiceIn,
    InstallServiceProgress,
    ServiceField,
    ServiceOptions,
    ServiceSize,
    ServiceSpecification,
    UninstallServiceIn,
)
from server.router.config import RouterOptions
from server.router.engine import RouterEngine
from server.router.heuristics import HeuristicScorer
from server.router.index import KnnIndex
from server.router.prompt import last_user_text
from server.services.base2_service import Base2Service, CustomModel, InstanceConfig, ModelConfig
from server.utils.core import PromiseWithProgress, Stream, StreamChunk, StreamChunkProgress

logger = logging.getLogger("uvicorn.error")

ROUTER_MODEL_ID = "router"
_RETRY_STATUSES = frozenset({404, 408, 429})
_SERVER_ERROR = 500


@dataclass
class RouterModelInfo:
    id: str
    registered_name: str
    options: InstallModelIn
    registration_id: str

    def get_info(self) -> ModelInfo:
        """Get info."""
        return ModelInfo(spec=self.options.spec, registration_id=self.registration_id)


@dataclass
class RouterInstalledInfo:
    models: dict[str, RouterModelInfo]
    options: InstallServiceIn
    parsed_options: RouterOptions
    engine: RouterEngine


@dataclass
class RouterDownloadedInfo:
    pass


class RouterService(Base2Service[RouterInstalledInfo, RouterDownloadedInfo]):
    """Chooses between a small and a large model for every `/v1/chat/completions` request sent to the model `router`."""

    _persists_model_definitions = False

    def get_type(self) -> str:
        """Return the service id."""
        return "router"

    def get_description(self) -> str:
        """Return the service description."""
        return "Sends each chat request to a small or a large model depending on how complex the request is."

    def get_size(self) -> ServiceSize:
        """Return the service size."""
        return ""

    def get_spec(self) -> ServiceSpecification:
        """Provide the options of the router as the install service form."""
        return ServiceSpecification(
            fields=[
                ServiceField(type="text", name="small_model", description="Model for simple requests"),
                ServiceField(type="text", name="large_model", description="Model for complex requests (also used when the router fails)"),
                ServiceField(type="text", name="index_path", description="Path to the .npz index (arrays: vectors, labels)"),
                ServiceField(
                    type="text", name="embedding_model", description="Embedding model", required=False, default="nomic-embed-text"
                ),
                ServiceField(
                    type="text",
                    name="embedding_prefix",
                    description="Prefix added before embedding",
                    required=False,
                    default="classification: ",
                ),
                ServiceField(type="number", name="k", description="Number of neighbours", required=False, default="20"),
                ServiceField(
                    type="number",
                    name="threshold",
                    description="Score (0-1) from which the large model is used",
                    required=False,
                    default="0.3",
                ),
                ServiceField(
                    type="text",
                    name="heuristics_path",
                    description="Path to the JSON parameters of the layer 1 heuristics (empty disables them)",
                    required=False,
                ),
                ServiceField(
                    type="number",
                    name="heuristic_threshold",
                    description="Heuristic points from which the large model is used without asking the embeddings",
                    required=False,
                    default="2",
                ),
                ServiceField(
                    type="number", name="force_large_over_chars", description="Rule: large model above this prompt length", required=False
                ),
                ServiceField(
                    type="text",
                    name="force_large_keywords",
                    description="Rule: large model for prompts containing any of these words (comma separated)",
                    required=False,
                ),
            ]
        )

    def get_model_spec(self) -> ModelSpecification:
        """Return the model specification."""
        return ModelSpecification(fields=[])

    def get_custom_model_spec(self) -> CustomModelSpecification | None:
        """Return None: custom models are not supported."""
        return None

    def get_installed_info(self, instance: str) -> bool | InstallServiceProgress | ServiceOptions:
        """Get service installed info."""
        installed = self.get_instance_info(instance).installed
        return self._get_service_installed_info(instance) if installed is None else installed.options.spec

    async def stop_instance(self, instance: str) -> None:
        """Nothing to stop: the router has no container."""

    def _load_download_info(self, data: dict[str, Any]) -> RouterDownloadedInfo:
        return RouterDownloadedInfo(**data)

    def _generate_instance_config(
        self,
        instance: str,  # noqa: ARG002
        info: RouterInstalledInfo | None,
        custom: list[CustomModel] | None,
    ) -> InstanceConfig:
        return InstanceConfig(
            options=info.options if info else None,
            models=[ModelConfig(model_id=x.id, options=x.options) for x in info.models.values()] if info else [],
            custom=custom,
        )

    async def _embed(self, options: RouterOptions, text: str) -> list[float]:
        response = await self.endpoint_registry.execute_embeddings(EmbeddingRequest(model=options.embedding_model, input=text))
        if isinstance(response, StreamingResponse):
            raw = b"".join([chunk.encode() if isinstance(chunk, str) else bytes(chunk) async for chunk in response.body_iterator])
        else:
            raw = bytes(response.body)
        if response.status_code != 200:
            msg = f"embedding request failed with HTTP {response.status_code}"
            raise RuntimeError(msg)
        return [float(x) for x in json.loads(raw)["data"][0]["embedding"]]

    async def _install_instance(
        self,
        instance: str,  # noqa: ARG002
        options: InstallServiceIn,
        container_name_override: str | None = None,  # noqa: ARG002
    ) -> PromiseWithProgress[RouterInstalledInfo, StreamChunk]:
        try:
            parsed = RouterOptions.from_spec(options.spec)
        except ValidationError as e:
            raise HTTPException(400, f"Invalid router options: {e}") from e
        if ROUTER_MODEL_ID in {parsed.small_model, parsed.large_model}:
            raise HTTPException(400, f"small_model and large_model must not be {ROUTER_MODEL_ID!r}")
        try:
            index = await asyncio.to_thread(KnnIndex.load, parsed.index_path)
        except (OSError, ValueError, KeyError) as e:
            raise HTTPException(400, f"Cannot load the router index {parsed.index_path!r}: {e}") from e

        heuristics: HeuristicScorer | None = None
        if parsed.heuristics_path:
            try:
                heuristics = await asyncio.to_thread(HeuristicScorer.load, parsed.heuristics_path)
            except (OSError, ValueError, KeyError, TypeError) as e:
                raise HTTPException(400, f"Cannot load the router heuristics {parsed.heuristics_path!r}: {e}") from e

        engine = RouterEngine(parsed, index, lambda text: self._embed(parsed, text), heuristics)

        async def func(stream: Stream[StreamChunk]) -> RouterInstalledInfo:  # noqa: ARG001
            self.service_downloaded = True
            return RouterInstalledInfo(models={}, options=options, parsed_options=parsed, engine=engine)

        return PromiseWithProgress(func=func)

    async def _uninstall_instance(self, instance: str, options: UninstallServiceIn) -> None:
        instance_info = self.get_instance_info(instance)
        if instance_info.installed:
            for model_id in list(instance_info.installed.models):
                await self._uninstall_model(instance, model_id, UninstallModelIn(purge=options.purge))
        instance_info.installed = None
        if options.purge and not any(i.installed for i in self.instances_info.values()):
            self.service_downloaded = False

    async def _try_small(self, body: ChatCompletionRequest, request: Request | None, model: str) -> tuple[StarletteResponse | None, str]:
        """Call the model; None (and the reason) when it is unavailable, so the caller can fall back to the large model."""
        try:
            response = await self.endpoint_registry.execute_chat_completion(body.model_copy(update={"model": model}), request)
        except HTTPException as e:
            logger.warning("router: model %r failed (%s)", model, e.detail)
            return None, f"HTTP {e.status_code}"
        if response.status_code in _RETRY_STATUSES or response.status_code >= _SERVER_ERROR:
            logger.warning("router: model %r answered with HTTP %d", model, response.status_code)
            aclose = getattr(getattr(response, "body_iterator", None), "aclose", None)
            if aclose is not None:
                await aclose()
            return None, f"HTTP {response.status_code}"
        return response, ""

    def _routed_handler(self, info: RouterInstalledInfo) -> Callable[[ChatCompletionRequest, Request | None], Awaitable[StreamingResponse]]:
        large = info.parsed_options.large_model

        async def on_chat_completion(body: ChatCompletionRequest, request: Request | None) -> StreamingResponse:
            decision = await info.engine.decide(last_user_text(body.messages))
            response: StarletteResponse | None = None
            if decision.model != large:
                response, reason = await self._try_small(body, request, decision.model)
                if response is None:
                    logger.warning("router: retrying with %r", large)
                    decision = decision.model_copy(update={"model": large, "layer": "fallback", "reason": reason})
            final = (
                response
                if response is not None
                else await self.endpoint_registry.execute_chat_completion(body.model_copy(update={"model": large}), request)
            )
            final.headers["X-Router-Model"] = decision.model
            final.headers["X-Router-Layer"] = decision.layer
            if decision.score is not None:
                final.headers["X-Router-Score"] = f"{decision.score:.4f}"
            if not isinstance(final, StreamingResponse):
                raise HTTPException(500, "Unexpected non-streaming response from the routed model")
            return final

        return on_chat_completion

    async def _install_model(
        self,
        instance: str,
        model_id: str,
        options: InstallModelIn,
        container_name_override: str | None = None,  # noqa: ARG002
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        if model_id != ROUTER_MODEL_ID:
            raise HTTPException(400, "Model not found")
        info = self.get_instance_installed_info(instance)
        if model_id in info.models:
            return PromiseWithProgress(value=InstallModelOut(status="OK", details="Already installed"))

        async def func(stream: Stream[StreamChunk]) -> InstallModelOut:
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=0, data={}))
            endpoint = ChatCompletionEndpoint()
            endpoint.on_chat_completion = self._routed_handler(info)
            registration_id = self.endpoint_registry.register_chat_completion(
                model=ROUTER_MODEL_ID,
                props=ModelProps(private=False, type="llm", endpoints=["/v1/chat/completions"]),
                endpoint=endpoint,
                registration_options=RegistrationOptions(origin="local", owned_by=self.get_type()),
            )
            info.models[model_id] = RouterModelInfo(
                id=model_id, registered_name=ROUTER_MODEL_ID, options=options, registration_id=registration_id
            )
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=1, data={}))
            return InstallModelOut(status="OK", details="Installed")

        return PromiseWithProgress(func=func)

    async def _uninstall_model(self, instance: str, model_id: str, options: UninstallModelIn) -> None:  # noqa: ARG002
        info = self.get_instance_installed_info(instance)
        model = info.models.pop(model_id, None)
        if model is not None:
            self.endpoint_registry.unregister_chat_completion(model.registered_name, model.registration_id)

    def _retrieve(self, instance: str) -> RetrieveModelOut:
        info = self.get_instance_installed_info(instance)
        installed = (
            info.models[ROUTER_MODEL_ID].get_info()
            if ROUTER_MODEL_ID in info.models
            else self._get_model_installed_info(instance, ROUTER_MODEL_ID)
        )
        return RetrieveModelOut(
            id=ROUTER_MODEL_ID,
            service=self.get_id(instance),
            type="llm",
            installed=installed,
            downloaded=False,
            size="",
            custom=None,
            spec=self.get_model_spec(),
            has_docker=False,
            capabilities_resolved=True,
            stale=False,
        )

    async def list_models(self, input_instance: str | list[str] | None, filters: ListModelsFilters) -> ListModelsOut:
        """List the single model `router` of every requested instance."""
        instances = [input_instance] if isinstance(input_instance, str) else input_instance if input_instance else list(self.instances_info)
        for instance in instances:
            if instance not in self.instances_info:
                raise HTTPException(404, f"Instance {instance} doesn't exist.")
        out = [self._retrieve(i) for i in instances if self.instances_info[i].installed is not None]
        return ListModelsOut(list=[m for m in out if filters.installed is None or filters.installed == bool(m.installed)])

    async def get_model(self, instance: str, model_id: str) -> RetrieveModelOut:
        """Get the model."""
        if model_id != ROUTER_MODEL_ID:
            raise HTTPException(400, "Model not found")
        return self._retrieve(instance)
