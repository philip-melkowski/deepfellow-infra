# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""OpenTelemetry tracing utilities for request and streaming response tracing."""

import asyncio
import contextlib
import functools
import inspect
import logging
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from opentelemetry import trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.metrics import Counter, Histogram
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Status, StatusCode, TracerProvider  # type: ignore
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, Tracer
from pydantic import BaseModel

from server.config import AppSettings
from server.utils.failed_span import remember_failed_span
from server.utils.logger import logger

_OTEL_SHUTDOWN_TIMEOUT_MILLIS = 5000

# "" is the root logger's name in logging.getLogger(). uvicorn/.error/.access have propagate=false
# in logging_config.yaml so they won't reach root and need the handler attached directly.
_OTLP_LOG_HANDLER_LOGGER_NAMES = ("", "uvicorn", "uvicorn.error", "uvicorn.access")


class OtlpLoggingManager:
    """Owns the OTLP log handler lifecycle for one app instance.

    Lives on `app.state.otlp_logging` (see `lifecycle.py`) rather than as module state, so each
    `FastAPI` app instance owns its own handler reference instead of sharing one across instances.
    """

    def __init__(self) -> None:
        self._handler: LoggingHandler | None = None
        self._provider: LoggerProvider | None = None
        # Serializes teardown()/reconfigure() so overlapping calls can't interleave across the
        # `await` on the provider shutdown. Not held by the sync `setup()`.
        self._lock = asyncio.Lock()
        # Set by `close()` at app exit; makes `setup()`/`reconfigure()` no-ops so nothing is rebuilt after.
        self._closed = False

        # Last confirmed-applied logging config, not `config` itself — `config` gets overwritten
        # regardless of success, which would hide a still-pending change from a retry.
        self.configured_endpoint: str | None = None
        self.configured_enabled: bool = False

    def setup(self, config: AppSettings) -> None:
        """Attach an OTLP log handler to the root Python logger and non-propagating uvicorn loggers.

        Idempotent: calling this again after `teardown()` rebuilds the exporter with the current
        `config.otel_exporter_otlp_endpoint`, so it can be re-applied when that value changes via
        `/admin/config` without a process restart. No-op after `close()`.
        """
        if self._closed:
            return
        resource = Resource(attributes={"service.name": "llm-audit"})
        provider = self._provider = LoggerProvider(resource=resource)
        processor = BatchLogRecordProcessor(OTLPLogExporter(endpoint=config.otel_exporter_otlp_endpoint, insecure=True))
        provider.add_log_record_processor(processor)
        set_logger_provider(provider)

        handler = self._handler = LoggingHandler(level=logging.DEBUG, logger_provider=provider)
        for name in _OTLP_LOG_HANDLER_LOGGER_NAMES:
            logging.getLogger(name).addHandler(handler)

        # Only record success once attached, so a prior failure still shows as pending on retry.
        self.configured_endpoint = config.otel_exporter_otlp_endpoint
        self.configured_enabled = True

    def _shutdown_provider(self, provider: LoggerProvider) -> None:
        """Shut down a `LoggerProvider` without blocking the caller for up to the default 30s.

        Like `TracerProvider.shutdown()`, `LoggerProvider.shutdown()` accepts no timeout parameter,
        so a stuck exporter (e.g. an unreachable OTLP endpoint) would otherwise stall whichever
        request triggered the reconfigure for up to 30s. Run it on a background thread and only wait
        up to our own bound — if the shutdown is still slow, it keeps finishing on its own thread
        instead of blocking this one.

        Called from `_teardown_locked()` (already off the event loop thread via `asyncio.to_thread()`).
        """
        thread = threading.Thread(target=self._run_provider_shutdown, args=(provider,), daemon=True)
        thread.start()
        thread.join(timeout=_OTEL_SHUTDOWN_TIMEOUT_MILLIS / 1000)

    def _run_provider_shutdown(self, provider: LoggerProvider) -> None:
        """Run `LoggerProvider.shutdown()`, logging (not propagating) any exception it raises.

        Never raising lets `_teardown_locked()` always reset `self._provider` afterwards (unless a newer one was
        installed meanwhile): a failed shutdown is logged and treated as done, not retried.
        """
        try:
            provider.shutdown()
        except Exception:
            logger.warning("Failed to shut down OTel log provider", exc_info=True)

    async def teardown(self) -> None:
        """Detach the OTLP log handler previously attached by `setup()`, if any."""
        async with self._lock:
            await self._teardown_locked()

    async def close(self) -> None:
        """Tear down for good at app exit: like `teardown()`, but nothing can be set up again afterwards.

        The flag is set before waiting for the lock, so a `reconfigure()` queued behind this call
        (or arriving later) can't rebuild a provider that nothing would shut down.
        """
        self._closed = True
        await self.teardown()

    async def _teardown_locked(self) -> None:
        """`teardown()` body; the caller must hold `self._lock`.

        Handler cleanup and provider cleanup are independent `if` blocks (not one `self._handler is
        None` guard covering both) — otherwise a provider left over from a previously failed shutdown
        would never be retried once the handler had already been cleared on that earlier attempt.

        `setup()` is sync and never takes the lock, so a direct call could still land during the await.
        In that case the new provider (and its `configured_*` state) is deliberately left running rather
        than torn down by this call. The in-app callers can't hit this: `lifecycle.py` calls `setup()`
        once at startup and `reconfigure()` calls it under the lock.
        """
        if self._handler is not None:
            for name in _OTLP_LOG_HANDLER_LOGGER_NAMES:
                logging.getLogger(name).removeHandler(self._handler)
            self._handler = None

        # Stop the old BatchLogRecordProcessor's background export thread — otherwise every
        # reconfigure() leaks one more thread that outlives its now-detached handler forever. Run off
        # the event loop thread since the shutdown itself may block; see `_shutdown_provider()`.
        provider = self._provider
        if provider is not None:
            await asyncio.to_thread(self._shutdown_provider, provider)
            # Only clear if still the provider we shut down; never drop one installed meanwhile.
            if self._provider is not provider:
                # A `setup()` ran during the await and already recorded its own `configured_*` state.
                return
            self._provider = None

        # _run_provider_shutdown() never raises, so this always reflects reality once reached.
        self.configured_enabled = False

    async def reconfigure(self, config: AppSettings) -> None:
        """Apply an `otel_logging_enabled`/`otel_exporter_otlp_endpoint` change without a restart."""
        async with self._lock:
            if self._closed:
                return
            await self._teardown_locked()
            if config.otel_logging_enabled:
                self.setup(config)


class FuncArgs:
    request: Request | None = None
    query: BaseModel | None = None
    model: BaseModel | None = None


@dataclass
class McpInstruments:
    """OTel metric instruments for MCP operations.

    Attribute values are identifiers/outcomes/sizes only — never tool arguments, tool results, or
    other MCP payload content.
    """

    healthcheck_count: Counter
    healthcheck_duration_ms: Histogram
    oauth_refresh_count: Counter


class InfraTracer:
    service_name: str
    config: AppSettings | None
    tracer: Tracer | None

    def __init__(self, service_name: str) -> None:
        self.service_name = service_name
        self.config: AppSettings | None = None
        self.tracer = None
        self._tracer_endpoint: str | None = None
        self._provider: TracerProvider | None = None
        self._meter_provider: MeterProvider | None = None
        self._meter_endpoint: str | None = None
        self._mcp_instruments: McpInstruments | None = None
        # asyncio.Lock binds to the running loop on first contention, and `tracer` is a module-level
        # singleton: contending shutdown() from a second event loop (e.g. across pytest loops) raises
        # RuntimeError. Fine under uvicorn's single loop per process.
        # Serializes shutdown() against itself. The lazy rebuilds in `_get_tracer()`/
        # `_get_mcp_instruments()` are sync and can't take it, so shutdown() additionally only clears
        # state whose provider is still the one it shut down.
        self._lock = asyncio.Lock()
        # Set by `close()` at app exit; makes `span()`/`mcp_instruments()`/`trace_request()` skip tracing
        # so no provider is rebuilt after the final `shutdown()`.
        self._closed = False

    def _get_config(self) -> AppSettings:
        if self.config is None:
            msg = "InfraTracer.config was not set. Ensure lifecycle.py assigns `tracer.config` at startup."
            raise RuntimeError(msg)
        return self.config

    @property
    def is_active(self) -> bool:
        """Whether shutdown() still has a live provider to tear down.

        Checks both independently — mcp_instruments() can build _meter_provider without span()
        ever building _provider.
        """
        return self._provider is not None or self._meter_provider is not None

    def _shutdown_span_provider(self, provider: TracerProvider) -> None:
        """Shut down a `TracerProvider` without blocking the caller for up to the default 30s.

        Unlike `MeterProvider.shutdown()`, `TracerProvider.shutdown()`/`BatchSpanProcessor.shutdown()`
        accept no timeout parameter, so a stuck exporter (e.g. an unreachable OTLP endpoint) would
        otherwise stall whichever request triggered the rebuild/disable for up to 30s. Run it on a
        background thread and only wait up to our own bound — if the shutdown is still slow, it
        keeps finishing on its own thread instead of blocking this one.

        Called from `shutdown()` (already off the event loop thread via `asyncio.to_thread()`), where
        this bounded wait caps thread-pool growth across repeated enable/disable cycles.
        """
        thread = threading.Thread(target=self._run_span_provider_shutdown, args=(provider,), daemon=True)
        thread.start()
        thread.join(timeout=_OTEL_SHUTDOWN_TIMEOUT_MILLIS / 1000)

    def _shutdown_span_provider_detached(self, provider: TracerProvider) -> None:
        """Fire-and-forget variant of `_shutdown_span_provider()` for the sync rebuild path.

        `_get_tracer()` runs synchronously on whatever thread calls it — including the event loop
        thread, via `span()`/`mcp_instruments()` used by MCP healthchecks and OAuth refresh. Nothing
        reads this old provider's shutdown result, so unlike `shutdown()` (an explicit, awaited
        disable/app-shutdown transition), there's no reason to block that thread waiting for it —
        just let the exporter finish flushing on its own daemon thread.
        """
        threading.Thread(target=self._run_span_provider_shutdown, args=(provider,), daemon=True).start()

    def _run_span_provider_shutdown(self, provider: TracerProvider) -> None:
        """Run `TracerProvider.shutdown()`, logging (not propagating) any exception it raises.

        Both callers above run this as a bare `threading.Thread` target, so an unhandled exception
        (e.g. an unreachable/broken OTLP collector at final flush) would otherwise only reach the
        default `threading.excepthook`, which prints to stderr and never reaches our structured
        logs. Mirrors `_shutdown_meter_provider()`'s try/except so a failed span-provider shutdown
        is just as visible as a failed meter-provider one.
        """
        try:
            provider.shutdown()
        except Exception:
            logger.warning("Failed to shut down OTel span provider", exc_info=True)

    def _shutdown_meter_provider(self, provider: MeterProvider) -> None:
        """Shut down a `MeterProvider`, swallowing any exception it raises.

        Unlike `TracerProvider.shutdown()` (run via a bare `threading.Thread`, whose exceptions
        never propagate to the caller), `MeterProvider.shutdown()` is invoked here via
        `asyncio.to_thread()` in `shutdown()`, and `asyncio.to_thread()` *does* propagate the
        thread's exception back to the awaiter. A slow/unreachable OTLP collector at final flush
        would then surface as a failure of whatever triggered the shutdown (e.g. `/admin/config`
        reporting "could not be applied live" even though the config itself applied fine) rather
        than a telemetry-only problem — so catch and log instead, matching the best-effort
        telemetry contract followed by `span()`/`mcp_instruments()`.
        """
        try:
            provider.shutdown(timeout_millis=_OTEL_SHUTDOWN_TIMEOUT_MILLIS)
        except Exception:
            logger.warning("Failed to shut down OTel meter provider", exc_info=True)

    def _get_tracer(self) -> Tracer:
        otlp_endpoint = self._get_config().otel_exporter_otlp_endpoint
        # Rebuild the provider if the endpoint changed via /admin/config, so the exporter target
        # updates without a process restart. Get the tracer directly from this provider instance
        # (not `trace.set_tracer_provider()` + `trace.get_tracer()`) — the OTel SDK's global
        # tracer provider is set-once; a second call is a silent no-op, which would leave every
        # rebuilt provider/exporter constructed-then-discarded and spans stuck on the old endpoint.
        if self.tracer is None or self._tracer_endpoint != otlp_endpoint:
            # Stop the old BatchSpanProcessor's background export thread before discarding its
            # provider — otherwise every endpoint change via /admin/config leaks one more thread.
            if self._provider is not None:
                self._shutdown_span_provider_detached(self._provider)

            resource = Resource(attributes={"service.name": self.service_name})
            provider = self._provider = TracerProvider(resource=resource)
            processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True))
            provider.add_span_processor(processor)

            self.tracer = provider.get_tracer(__name__)
            self._tracer_endpoint = otlp_endpoint
        return self.tracer

    def _get_mcp_instruments(self) -> McpInstruments:
        """Lazily build (and rebuild on endpoint change) the meter and MCP metric instruments.

        Mirrors `_get_tracer()`: a changed `otel_exporter_otlp_endpoint` requires a fresh
        `MeterProvider`, and instruments created on the old provider would keep exporting nowhere,
        so they're recreated alongside it rather than reused.
        """
        otlp_endpoint = self._get_config().otel_exporter_otlp_endpoint
        if self._mcp_instruments is None or self._meter_endpoint != otlp_endpoint:
            if self._meter_provider is not None:
                # Detached, like `_shutdown_span_provider_detached()`: this runs synchronously on
                # the request that triggered the rebuild (possibly the event loop thread, via
                # `mcp_instruments()`), and nothing reads the result, so don't block waiting for a
                # final flush against a possibly-unreachable old endpoint — let it finish on its own
                # daemon thread instead.
                threading.Thread(target=self._shutdown_meter_provider, args=(self._meter_provider,), daemon=True).start()

            resource = Resource(attributes={"service.name": self.service_name})
            reader = PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=otlp_endpoint, insecure=True))
            provider = self._meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
            meter = provider.get_meter(__name__)

            self._mcp_instruments = McpInstruments(
                healthcheck_count=meter.create_counter("mcp.healthcheck.count", description="MCP server health-check attempts, by outcome"),
                healthcheck_duration_ms=meter.create_histogram(
                    "mcp.healthcheck.duration", unit="ms", description="MCP server health-check duration"
                ),
                oauth_refresh_count=meter.create_counter(
                    "mcp.oauth.refresh.count", description="MCP OAuth access token refresh attempts, by outcome"
                ),
            )
            self._meter_endpoint = otlp_endpoint
        return self._mcp_instruments

    async def shutdown(self) -> None:
        """Stop the tracer's and MCP meter's background export threads.

        Call at app shutdown and whenever `otel_tracing_enabled` is turned off via `/admin/config` —
        otherwise the `BatchSpanProcessor`/`PeriodicExportingMetricReader` export threads built by
        `_get_tracer()`/`_get_mcp_instruments()` keep running (and exporting to the OTLP endpoint)
        even though nothing references them anymore. Resets provider state so a later call rebuilds
        fresh providers rather than reusing shut-down ones — except a provider rebuilt by a concurrent
        `_get_tracer()`/`_get_mcp_instruments()` while this awaits, which is kept running rather than
        orphaned. At app exit use `close()` instead, which also stops any further rebuilds.

        Both provider shutdowns block synchronously for up to `_OTEL_SHUTDOWN_TIMEOUT_MILLIS` against
        the OTLP endpoint, so they run via `asyncio.to_thread()` — otherwise a slow/unreachable
        endpoint would stall the event loop (and every other in-flight request) for up to that long.
        """
        async with self._lock:
            span_provider = self._provider
            if span_provider is not None:
                await asyncio.to_thread(self._shutdown_span_provider, span_provider)
            # A concurrent `_get_tracer()` may have installed a fresh provider during the await;
            # leave it (and its tracer/endpoint) alone rather than orphaning its export thread.
            if self._provider is span_provider:
                self._provider = None
                self.tracer = None
                self._tracer_endpoint = None

            meter_provider = self._meter_provider
            if meter_provider is not None:
                await asyncio.to_thread(self._shutdown_meter_provider, meter_provider)
            if self._meter_provider is meter_provider:
                self._meter_provider = None
                self._mcp_instruments = None
                self._meter_endpoint = None

    async def close(self) -> None:
        """Run `shutdown()` for good at app exit; tracing and metrics stay off afterwards.

        Background tasks (MCP health checks, OAuth refresh) can still call `span()`/`mcp_instruments()`
        while the app exits. Setting the flag first stops them from building a fresh provider whose
        export thread nothing would stop — only the OTel SDK's `atexit` hook, which can block interpreter
        exit for up to 30s against an unreachable endpoint.
        """
        self._closed = True
        await self.shutdown()

    def mcp_instruments(self) -> McpInstruments | None:
        """Return the MCP metric instruments, or `None` when disabled or on a build failure.

        Telemetry stays best-effort: a failure building the instruments never breaks callers.
        """
        if self._closed or self.config is None or not self.config.otel_tracing_enabled:
            return None
        try:
            return self._get_mcp_instruments()
        except Exception:
            logger.warning("Failed to initialize MCP OTel instruments; continuing without metrics", exc_info=True)
            return None

    @contextlib.contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span | None]:
        """Start a manual span for code outside `trace_request()`.

        Meant for background tasks and internal service calls that aren't FastAPI endpoints. Yields
        `None` when tracing is disabled (including when `self.config` was never set, e.g. in unit
        tests that call service methods without going through `lifecycle.py`) or when acquiring the
        tracer fails, so callers can guard extra `set_attribute` calls with `if span is not None:`.
        `attributes` should only carry identifiers/outcomes/sizes — never MCP tool arguments, tool
        results, or other payload content. A raised `HTTPException` with a 4xx status is recorded as
        `OK`, not `ERROR` — it's an expected client-input rejection, not a span-level failure.

        Uses `start_as_current_span` (not `start_span`) so the span is attached to the current OTel
        context — otherwise it would never nest under a caller's own `span()`/`trace_request()` span.
        """
        if self._closed or self.config is None or not self.config.otel_tracing_enabled:
            yield None
            return
        try:
            tracer_obj = self._get_tracer()
        except Exception:
            logger.warning("Failed to initialize OTel tracer for span %r; continuing without tracing", name, exc_info=True)
            yield None
            return
        with tracer_obj.start_as_current_span(
            name, attributes=attributes, record_exception=False, set_status_on_exception=False
        ) as span_obj:
            try:
                yield span_obj
            except HTTPException as e:
                # A 4xx is expected client-input rejection, not a span-level failure — recording it
                # as an OTel error would make routine bad input (e.g. an unknown model id) show up
                # in span-based error-rate alerting alongside genuine operational failures.
                if e.status_code < 500:
                    span_obj.set_status(Status(StatusCode.OK))
                else:
                    span_obj.record_exception(e)
                    span_obj.set_status(Status(StatusCode.ERROR))
                raise
            except Exception as e:
                span_obj.record_exception(e)
                span_obj.set_status(Status(StatusCode.ERROR))
                raise
            else:
                span_obj.set_status(Status(StatusCode.OK))

    def _set_success_attributes(self, span: Span, execution_time: float) -> None:
        """Set success attributes on the span."""
        attributes = {"execution.time_ms": round(execution_time * 1000, 2)}

        for key, value in attributes.items():
            span.set_attribute(key, value)
        span.set_status(Status(StatusCode.OK))

    def _bind_args(self, func: Callable[..., Any], args: Any, kwargs: Any) -> inspect.BoundArguments | None:  # noqa: ANN401
        try:
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            return bound_args  # noqa: TRY300
        except Exception:
            return None

    def _extract_arguments(self, bound_args: inspect.BoundArguments) -> FuncArgs:
        res = FuncArgs()
        if "request" in bound_args.arguments and isinstance(request := bound_args.arguments["request"], Request):
            res.request = request
        if "model" in bound_args.arguments and isinstance(model := bound_args.arguments["model"], BaseModel):
            res.model = model
        if "query" in bound_args.arguments and isinstance(query := bound_args.arguments["query"], BaseModel):
            res.query = query
        return res

    def _add_attributes_to_span(self, args: FuncArgs, span: Span) -> None:
        """Extract span attributes from function arguments."""
        try:
            if args.request:
                span.set_attribute("request.method", str(args.request.method))
                span.set_attribute("request.url", str(args.request.url))
                span.set_attribute("request.path", args.request.url.path)
                span.set_attribute("request.content_type", args.request.headers.get("content-type") or "<unknown>")
                span.set_attribute("request.content_length", args.request.headers.get("content-length") or "<unknown>")
                span.set_attribute("request.user-agent", args.request.headers.get("user-agent") or "<unknown>")
                span.set_attribute("request.accept", args.request.headers.get("accept") or "<unknown>")
                span.set_attribute("request.host", args.request.headers.get("host") or "<unknown>")

            if args.model:
                for key, value in args.model.model_dump().items():
                    span.set_attribute(f"model.{key}", value)
            if args.query:
                for key, value in args.query.model_dump().items():
                    span.set_attribute(f"query.{key}", value)

        except Exception:
            pass

    def trace_request(self) -> Callable[..., Any]:
        """Create a tracing decorator for async function."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
                if self._closed or not self._get_config().otel_tracing_enabled:
                    return await func(*args, **kwargs)
                span_name = "test"
                span = self._get_tracer().start_span(span_name)
                trace.set_span_in_context(span)
                start_time = time.time()
                # Extract and set attributes from function arguments
                bound_args = self._bind_args(func, args, kwargs)
                func_args = self._extract_arguments(bound_args) if bound_args else FuncArgs()
                self._add_attributes_to_span(func_args, span)

                def span_end() -> None:
                    execution_time = time.time() - start_time
                    self._set_success_attributes(span, execution_time)
                    span.end()

                try:
                    res = await func(*args, **kwargs)
                except Exception as e:
                    span.record_exception(e)
                    remember_failed_span(func_args.request, span.get_span_context())
                    span_end()
                    raise
                else:
                    if isinstance(res, StreamingResponse):
                        span.set_attribute("response.status", res.status_code)
                        span.set_attribute("response.content_type", res.media_type or "<unknown>")
                        span_end()
                    if isinstance(res, JSONResponse):
                        span.set_attribute("response.status", res.status_code)
                        span.set_attribute("response.content_type", "application/json")
                        span_end()
                    elif isinstance(res, BaseModel):
                        span.set_attribute("response.status", 200)
                        span.set_attribute("response.content_type", "application/json")
                        span_end()
                    else:
                        span.set_attribute("response.status", "<unknown>")
                        span.set_attribute("response.content_type", "<unknown>")
                        span_end()
                    return res

            return async_wrapper

        return decorator


tracer = InfraTracer(service_name="llm-audit")
