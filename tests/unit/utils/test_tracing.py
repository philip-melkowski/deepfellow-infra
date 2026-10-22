# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

import asyncio
import inspect
import logging
import threading
import time
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from opentelemetry.trace import SpanContext
from opentelemetry.trace.status import StatusCode
from pydantic import BaseModel

from server.utils.failed_span import get_failed_span_context
from server.utils.tracing import FuncArgs, InfraTracer, McpInstruments, OtlpLoggingManager


def _synchronous_daemon_threads() -> Any:
    """Patch `threading.Thread` in the tracing module to run its target immediately, in-line.

    The detached-shutdown paths (`_shutdown_span_provider_detached()`, and the equivalent in
    `_get_mcp_instruments()`) fire a daemon thread and deliberately don't join it — nothing waits
    for the old provider's shutdown to finish. Asserting on it right after `.start()` would
    otherwise race the real OS thread; this makes the target run synchronously so the assertion is
    deterministic without changing production behavior (which stays a real background thread).
    """

    class _ImmediateThread:
        def __init__(self, target: Any, args: Any = (), kwargs: Any = None, daemon: bool = False) -> None:
            self._target = target
            self._args = args
            self._kwargs = kwargs or {}

        def start(self) -> None:
            self._target(*self._args, **self._kwargs)

    return patch("server.utils.tracing.threading.Thread", side_effect=_ImmediateThread)


def _make_config(enabled: bool = True, endpoint: str = "http://localhost:4317") -> MagicMock:
    cfg = MagicMock()
    cfg.otel_tracing_enabled = enabled
    cfg.otel_exporter_otlp_endpoint = endpoint
    return cfg


def _make_span() -> MagicMock:
    span = MagicMock()
    span.set_attribute = MagicMock()
    span.set_status = MagicMock()
    span.record_exception = MagicMock()
    span.end = MagicMock()
    return span


def _make_request(
    method: str = "POST",
    path: str = "/path",
    headers: dict[str, str] | None = None,
) -> Request:
    if headers is None:
        headers = {
            "content-type": "application/json",
            "content-length": "42",
            "user-agent": "test-agent",
            "accept": "*/*",
            "host": "test.com",
        }
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": b"",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "server": ("test.com", 80),
    }
    return Request(scope)


def _make_tracer_with_config(enabled: bool) -> InfraTracer:
    t = InfraTracer("svc")
    t.config = _make_config(enabled=enabled)
    return t


def _bind(func: Any, *args: Any, **kwargs: Any) -> inspect.BoundArguments:
    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    return bound


def _patch_otel_context_managers():
    return [
        patch("server.utils.tracing.TracerProvider"),
        patch("server.utils.tracing.OTLPSpanExporter"),
        patch("server.utils.tracing.BatchSpanProcessor"),
    ]


def _patch_otel_metric_context_managers():
    return [
        patch("server.utils.tracing.MeterProvider"),
        patch("server.utils.tracing.OTLPMetricExporter"),
        patch("server.utils.tracing.PeriodicExportingMetricReader"),
    ]


def _patch_otlp_logging_context_managers():
    return [
        patch("server.utils.tracing.LoggerProvider"),
        patch("server.utils.tracing.OTLPLogExporter"),
        patch("server.utils.tracing.BatchLogRecordProcessor"),
        patch("server.utils.tracing.set_logger_provider"),
        patch("server.utils.tracing.LoggingHandler"),
    ]


def test_otlp_logging_manager_configured_state_defaults() -> None:
    manager = OtlpLoggingManager()
    assert manager.configured_endpoint is None
    assert manager.configured_enabled is False


def test_setup_otlp_logging_creates_provider_with_service_name() -> None:
    cfg = _make_config(endpoint="http://otel:4317")
    patches = _patch_otlp_logging_context_managers()

    with patches[0] as mock_provider_cls, patches[1], patches[2], patches[3], patches[4], patch("server.utils.tracing.logging.getLogger"):
        OtlpLoggingManager().setup(cfg)

    resource_arg = mock_provider_cls.call_args[1]["resource"]
    assert resource_arg.attributes["service.name"] == "llm-audit"


def test_setup_otlp_logging_uses_endpoint_from_config() -> None:
    cfg = _make_config(endpoint="http://otel:4317")
    patches = _patch_otlp_logging_context_managers()

    with patches[0], patches[1] as mock_exporter_cls, patches[2], patches[3], patches[4], patch("server.utils.tracing.logging.getLogger"):
        OtlpLoggingManager().setup(cfg)

    assert mock_exporter_cls.call_args == call(endpoint="http://otel:4317", insecure=True)


def test_setup_otlp_logging_registers_processor_and_sets_global_provider() -> None:
    cfg = _make_config(endpoint="http://otel:4317")
    patches = _patch_otlp_logging_context_managers()

    with (
        patches[0] as mock_provider_cls,
        patches[1],
        patches[2] as mock_processor_cls,
        patches[3] as mock_set,
        patches[4],
        patch("server.utils.tracing.logging.getLogger"),
    ):
        OtlpLoggingManager().setup(cfg)

    provider = mock_provider_cls.return_value
    assert provider.add_log_record_processor.call_args == call(mock_processor_cls.return_value)
    assert mock_set.call_args == call(provider)


def test_setup_otlp_logging_attaches_handler_to_root_and_uvicorn_loggers() -> None:
    cfg = _make_config(endpoint="http://otel:4317")
    patches = _patch_otlp_logging_context_managers()

    with (
        patches[0] as mock_provider_cls,
        patches[1],
        patches[2],
        patches[3],
        patches[4] as mock_handler_cls,
        patch("server.utils.tracing.logging.getLogger") as mock_get_logger,
    ):
        OtlpLoggingManager().setup(cfg)

    handler = mock_handler_cls.return_value
    assert mock_handler_cls.call_args == call(level=logging.DEBUG, logger_provider=mock_provider_cls.return_value)

    logger_names = [c[0][0] if c[0] else "" for c in mock_get_logger.call_args_list]
    assert "" in logger_names
    assert "uvicorn" in logger_names
    assert "uvicorn.error" in logger_names
    assert "uvicorn.access" in logger_names

    for logger_call in mock_get_logger.return_value.addHandler.call_args_list:
        assert logger_call == call(handler)


def test_setup_otlp_logging_records_configured_endpoint_and_enabled() -> None:
    cfg = _make_config(endpoint="http://otel:4317")
    patches = _patch_otlp_logging_context_managers()
    manager = OtlpLoggingManager()

    with patches[0], patches[1], patches[2], patches[3], patches[4], patch("server.utils.tracing.logging.getLogger"):
        manager.setup(cfg)

    assert manager.configured_endpoint == "http://otel:4317"
    assert manager.configured_enabled is True


@pytest.mark.asyncio
async def test_teardown_otlp_logging_noop_when_no_handler() -> None:
    manager = OtlpLoggingManager()

    with patch("server.utils.tracing.logging.getLogger") as mock_get_logger:
        await manager.teardown()

    assert mock_get_logger.call_count == 0


@pytest.mark.asyncio
async def test_teardown_otlp_logging_removes_handler_from_loggers() -> None:
    manager = OtlpLoggingManager()
    handler = MagicMock()
    manager._handler = handler  # pyright: ignore[reportPrivateUsage]

    with patch("server.utils.tracing.logging.getLogger") as mock_get_logger:
        await manager.teardown()

        logger_names = [c[0][0] if c[0] else "" for c in mock_get_logger.call_args_list]
        assert logger_names == ["", "uvicorn", "uvicorn.error", "uvicorn.access"]
        for logger_call in mock_get_logger.return_value.removeHandler.call_args_list:
            assert logger_call == call(handler)

        assert manager._handler is None  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_teardown_otlp_logging_shuts_down_old_provider() -> None:
    manager = OtlpLoggingManager()
    manager._handler = MagicMock()  # pyright: ignore[reportPrivateUsage]
    provider = MagicMock()
    manager._provider = provider  # pyright: ignore[reportPrivateUsage]

    with patch("server.utils.tracing.logging.getLogger"):
        await manager.teardown()

    assert provider.shutdown.call_count == 1
    assert manager._provider is None  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_teardown_otlp_logging_records_configured_enabled_false_on_success() -> None:
    manager = OtlpLoggingManager()
    manager._handler = MagicMock()  # pyright: ignore[reportPrivateUsage]
    manager._provider = MagicMock()  # pyright: ignore[reportPrivateUsage]
    manager.configured_enabled = True

    with patch("server.utils.tracing.logging.getLogger"):
        await manager.teardown()

    assert manager.configured_enabled is False


@pytest.mark.asyncio
async def test_teardown_otlp_logging_records_configured_enabled_false_even_when_provider_shutdown_fails() -> None:
    """DFINFRA-314 swallows the shutdown failure, so teardown() never raises here anymore."""
    manager = OtlpLoggingManager()
    manager._handler = MagicMock()  # pyright: ignore[reportPrivateUsage]
    provider = MagicMock()
    provider.shutdown.side_effect = RuntimeError("boom")
    manager._provider = provider  # pyright: ignore[reportPrivateUsage]
    manager.configured_enabled = True

    with patch("server.utils.tracing.logging.getLogger"):
        await manager.teardown()  # must not raise

    assert manager.configured_enabled is False


@pytest.mark.asyncio
async def test_reconfigure_otlp_logging_shuts_down_previous_provider_before_rebuilding() -> None:
    cfg = _make_config()
    cfg.otel_logging_enabled = True
    manager = OtlpLoggingManager()
    patches = _patch_otlp_logging_context_managers()

    with (
        patches[0] as mock_provider_cls,
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patch("server.utils.tracing.logging.getLogger"),
    ):
        first_provider = mock_provider_cls.return_value
        manager.setup(cfg)
        await manager.reconfigure(cfg)

    assert first_provider.shutdown.call_count == 1


@pytest.mark.asyncio
async def test_reconfigure_otlp_logging_sets_up_when_enabled() -> None:
    cfg = _make_config()
    cfg.otel_logging_enabled = True
    manager = OtlpLoggingManager()

    async def _noop_teardown() -> None:
        return None

    with (
        patch.object(manager, "_teardown_locked", side_effect=_noop_teardown) as mock_teardown,
        patch.object(manager, "setup") as mock_setup,
    ):
        await manager.reconfigure(cfg)

    assert mock_teardown.call_count == 1
    assert mock_setup.call_args == call(cfg)


@pytest.mark.asyncio
async def test_reconfigure_otlp_logging_skips_setup_when_disabled() -> None:
    cfg = _make_config()
    cfg.otel_logging_enabled = False
    manager = OtlpLoggingManager()

    async def _noop_teardown() -> None:
        return None

    with (
        patch.object(manager, "_teardown_locked", side_effect=_noop_teardown) as mock_teardown,
        patch.object(manager, "setup") as mock_setup,
    ):
        await manager.reconfigure(cfg)

    assert mock_teardown.call_count == 1
    assert mock_setup.call_count == 0


@pytest.mark.asyncio
async def test_teardown_otlp_logging_swallows_provider_shutdown_error(caplog: pytest.LogCaptureFixture) -> None:
    """A failed provider shutdown (e.g. an unreachable OTLP collector) must not raise, and must
    still reset `self._provider` so the manager isn't left holding a provider it can never retry.
    """
    manager = OtlpLoggingManager()
    manager._handler = MagicMock()  # pyright: ignore[reportPrivateUsage]
    provider = MagicMock()
    provider.shutdown.side_effect = RuntimeError("boom")
    manager._provider = provider  # pyright: ignore[reportPrivateUsage]

    with caplog.at_level(logging.WARNING):
        await manager.teardown()  # must not raise

    assert manager._provider is None  # pyright: ignore[reportPrivateUsage]
    assert "Failed to shut down OTel log provider" in caplog.text


@pytest.mark.asyncio
async def test_teardown_otlp_logging_retries_provider_left_over_from_prior_failure() -> None:
    """Regression test for DFINFRA-314.

    Before the fix, `teardown()` guarded both handler and provider cleanup behind one
    `if self._handler is None: return`. A provider orphaned by an earlier failed shutdown attempt
    always has its handler already cleared (that happens first), so the *next* `teardown()` call
    exited immediately on that guard, before ever reaching the provider cleanup line again — leaving
    the provider and its background export thread stuck forever, with no way to retry short of a
    process restart. Simulate that exact state directly: handler already `None`, provider still set.
    """
    manager = OtlpLoggingManager()
    manager._handler = None  # pyright: ignore[reportPrivateUsage]
    provider = MagicMock()
    manager._provider = provider  # pyright: ignore[reportPrivateUsage]

    with patch("server.utils.tracing.logging.getLogger"):
        await manager.teardown()

    assert provider.shutdown.call_count == 1
    assert manager._provider is None  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_teardown_otlp_logging_does_not_block_past_shutdown_timeout() -> None:
    """A stuck `provider.shutdown()` (e.g. an unreachable OTLP collector) must not stall the caller
    past `_OTEL_SHUTDOWN_TIMEOUT_MILLIS` — it keeps finishing on its own daemon thread instead.
    """
    manager = OtlpLoggingManager()
    manager._handler = MagicMock()  # pyright: ignore[reportPrivateUsage]
    provider = MagicMock()

    release = threading.Event()

    def _hanging_shutdown() -> None:
        release.wait(timeout=2)

    provider.shutdown.side_effect = _hanging_shutdown
    manager._provider = provider  # pyright: ignore[reportPrivateUsage]

    with patch("server.utils.tracing.logging.getLogger"), patch("server.utils.tracing._OTEL_SHUTDOWN_TIMEOUT_MILLIS", 50):
        start = time.monotonic()
        await manager.teardown()
        elapsed = time.monotonic() - start

    release.set()  # let the background thread finish so it doesn't outlive the test
    assert elapsed < 1.0


def test_func_args_defaults_are_none() -> None:
    args = FuncArgs()

    assert args.request is None
    assert args.query is None
    assert args.model is None


def test_infra_tracer_service_name_stored() -> None:
    t = InfraTracer("my-service")

    assert t.service_name == "my-service"


def test_infra_tracer_config_and_tracer_are_none() -> None:
    t = InfraTracer("svc")

    assert t.config is None
    assert t.tracer is None


def test_get_config_returns_config_when_set() -> None:
    t = InfraTracer("svc")
    cfg = _make_config()
    t.config = cfg

    assert t._get_config() is cfg  # pyright: ignore[reportPrivateUsage]


def test_is_active_false_when_no_provider() -> None:
    t = InfraTracer("svc")
    assert t.is_active is False


def test_is_active_true_when_provider_exists() -> None:
    t = InfraTracer("svc")
    t._provider = MagicMock()  # pyright: ignore[reportPrivateUsage]
    assert t.is_active is True


def test_is_active_true_when_only_meter_provider_exists() -> None:
    """mcp_instruments() can build _meter_provider without span() building _provider."""
    t = InfraTracer("svc")
    t._meter_provider = MagicMock()  # pyright: ignore[reportPrivateUsage]
    assert t.is_active is True


def test_get_config_raises_when_not_set() -> None:
    t = InfraTracer("svc")

    with pytest.raises(RuntimeError, match=r"InfraTracer\.config was not set"):
        t._get_config()  # pyright: ignore[reportPrivateUsage]


def test_get_tracer_returns_tracer() -> None:
    t = InfraTracer("svc")
    t.config = _make_config(endpoint="http://otel:4317")
    patches = _patch_otel_context_managers()

    with patches[0] as mock_provider_cls, patches[1], patches[2]:
        fake_tracer = mock_provider_cls.return_value.get_tracer.return_value
        result = t._get_tracer()  # pyright: ignore[reportPrivateUsage]

    assert result is fake_tracer


def test_get_tracer_caches_tracer() -> None:
    t = InfraTracer("svc")
    t.config = _make_config()
    patches = _patch_otel_context_managers()

    with patches[0] as mock_provider_cls, patches[1], patches[2]:
        fake_tracer = mock_provider_cls.return_value.get_tracer.return_value
        t._get_tracer()  # pyright: ignore[reportPrivateUsage]
        t._get_tracer()  # pyright: ignore[reportPrivateUsage]

    assert t.tracer is fake_tracer
    assert mock_provider_cls.return_value.get_tracer.call_count == 1


def test_get_tracer_rebuilds_from_new_providers_tracer_when_endpoint_changes() -> None:
    """The tracer must come from the freshly built provider, not a stale global one.

    `trace.set_tracer_provider()` is set-once in the OTel SDK — a second call is a silent no-op —
    so getting the tracer via `provider.get_tracer(...)` directly (bypassing the global registry)
    is the only way a changed endpoint actually takes effect without a restart.
    """
    t = InfraTracer("svc")
    t.config = _make_config(endpoint="http://old:4317")
    patches = _patch_otel_context_managers()

    with patches[0] as mock_provider_cls, patches[1], patches[2], _synchronous_daemon_threads():
        first_provider = MagicMock()
        second_provider = MagicMock()
        mock_provider_cls.side_effect = [first_provider, second_provider]

        first_tracer = t._get_tracer()  # pyright: ignore[reportPrivateUsage]
        t.config.otel_exporter_otlp_endpoint = "http://new:4317"
        second_tracer = t._get_tracer()  # pyright: ignore[reportPrivateUsage]

    assert first_tracer is first_provider.get_tracer.return_value
    assert second_tracer is second_provider.get_tracer.return_value
    assert first_tracer is not second_tracer
    assert first_provider.shutdown.call_count == 1
    assert second_provider.shutdown.call_count == 0


def test_set_success_attributes_sets_execution_time_and_ok_status() -> None:
    t = InfraTracer("svc")
    span = _make_span()

    t._set_success_attributes(span, 0.5)  # pyright: ignore[reportPrivateUsage]

    assert span.set_attribute.call_args == call("execution.time_ms", 500.0)
    assert span.set_status.call_count == 1
    status_arg = span.set_status.call_args[0][0]
    assert status_arg.status_code == StatusCode.OK


def test_set_success_attributes_rounds_execution_time() -> None:
    t = InfraTracer("svc")
    span = _make_span()

    t._set_success_attributes(span, 1.0 / 3)  # pyright: ignore[reportPrivateUsage]

    call = span.set_attribute.call_args_list[0]
    assert call[0][0] == "execution.time_ms"
    assert call[0][1] == round((1.0 / 3) * 1000, 2)


def test_bind_args_binds_correctly() -> None:
    t = InfraTracer("svc")

    def func(a: int, b: str = "x") -> None:
        pass

    bound = t._bind_args(func, (1,), {})  # pyright: ignore[reportPrivateUsage]

    assert bound is not None
    assert bound.arguments["a"] == 1
    assert bound.arguments["b"] == "x"


def test_bind_args_returns_none_on_binding_failure() -> None:
    t = InfraTracer("svc")

    def func(a: int) -> None:
        pass

    result = t._bind_args(func, (), {})  # pyright: ignore[reportPrivateUsage]

    assert result is None


def test_extract_arguments_extracts_request() -> None:
    t = InfraTracer("svc")
    req = _make_request()

    def func(request: Request) -> None:
        pass

    bound = _bind(func, req)

    result = t._extract_arguments(bound)  # pyright: ignore[reportPrivateUsage]

    assert result.request is req


def test_extract_arguments_extracts_model() -> None:
    t = InfraTracer("svc")

    class M(BaseModel):
        x: int

    def func(model: M) -> None:
        pass

    m = M(x=1)
    bound = _bind(func, m)

    result = t._extract_arguments(bound)  # pyright: ignore[reportPrivateUsage]

    assert result.model is m


def test_extract_arguments_extracts_query() -> None:
    t = InfraTracer("svc")

    class Q(BaseModel):
        q: str

    def func(query: Q) -> None:
        pass

    q = Q(q="hello")
    bound = _bind(func, q)

    result = t._extract_arguments(bound)  # pyright: ignore[reportPrivateUsage]

    assert result.query is q


def test_extract_arguments_ignores_non_request_type() -> None:
    t = InfraTracer("svc")

    def func(request: str) -> None:
        pass

    bound = _bind(func, "not-a-request")

    result = t._extract_arguments(bound)  # pyright: ignore[reportPrivateUsage]

    assert result.request is None


def test_extract_arguments_empty_when_no_matching_params() -> None:
    t = InfraTracer("svc")

    def func(x: int) -> None:
        pass

    bound = _bind(func, 42)

    result = t._extract_arguments(bound)  # pyright: ignore[reportPrivateUsage]

    assert result.request is None
    assert result.model is None
    assert result.query is None


def test_add_attributes_to_span_adds_request_attributes() -> None:
    t = InfraTracer("svc")
    span = _make_span()
    req = _make_request()
    args = FuncArgs()
    args.request = req

    t._add_attributes_to_span(args, span)  # pyright: ignore[reportPrivateUsage]

    keys = {c[0][0] for c in span.set_attribute.call_args_list}
    assert "request.method" in keys
    assert "request.url" in keys
    assert "request.path" in keys
    assert "request.content_type" in keys


def test_add_attributes_to_span_adds_model_attributes() -> None:
    t = InfraTracer("svc")
    span = _make_span()

    class M(BaseModel):
        value: int

    args = FuncArgs()
    args.model = M(value=7)

    t._add_attributes_to_span(args, span)  # pyright: ignore[reportPrivateUsage]

    keys = {c[0][0] for c in span.set_attribute.call_args_list}
    assert "model.value" in keys


def test_add_attributes_to_span_adds_query_attributes() -> None:
    t = InfraTracer("svc")
    span = _make_span()

    class Q(BaseModel):
        term: str

    args = FuncArgs()
    args.query = Q(term="hello")

    t._add_attributes_to_span(args, span)  # pyright: ignore[reportPrivateUsage]

    keys = {c[0][0] for c in span.set_attribute.call_args_list}
    assert "query.term" in keys


def test_add_attributes_to_span_silently_ignores_exceptions() -> None:
    t = InfraTracer("svc")
    span = _make_span()
    span.set_attribute.side_effect = Exception("unexpected")
    args = FuncArgs()
    req = _make_request()
    args.request = req

    t._add_attributes_to_span(args, span)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_trace_request_skips_tracing_when_disabled() -> None:
    t = _make_tracer_with_config(enabled=False)

    @t.trace_request()
    async def handler():
        return "result"

    result = await handler()

    assert result == "result"


@pytest.mark.asyncio
async def test_trace_request_returns_result_when_tracing_enabled() -> None:
    t = _make_tracer_with_config(enabled=True)
    span = _make_span()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = span
    t.tracer = fake_tracer
    t._tracer_endpoint = t.config.otel_exporter_otlp_endpoint  # pyright: ignore[reportPrivateUsage, reportOptionalMemberAccess]

    with patch("server.utils.tracing.trace.set_span_in_context"):

        @t.trace_request()
        async def handler():
            return "ok"

        result = await handler()

    assert result == "ok"
    assert span.end.call_count == 1


@pytest.mark.asyncio
async def test_trace_request_re_raises_exception_and_ends_span() -> None:
    t = _make_tracer_with_config(enabled=True)
    span = _make_span()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = span
    t.tracer = fake_tracer
    t._tracer_endpoint = t.config.otel_exporter_otlp_endpoint  # pyright: ignore[reportPrivateUsage, reportOptionalMemberAccess]

    with patch("server.utils.tracing.trace.set_span_in_context"):

        @t.trace_request()
        async def handler():
            raise ValueError("fail")

        with pytest.raises(ValueError, match="fail"):
            await handler()

    assert span.record_exception.call_count == 1
    assert span.end.call_count == 1


@pytest.mark.asyncio
@patch("server.utils.tracing.trace.set_span_in_context")
async def test_trace_request_remembers_failed_span_on_request(mock_set_span_in_context: MagicMock):
    t = _make_tracer_with_config(enabled=True)
    span = _make_span()
    span_context = SpanContext(trace_id=0x0AF7651916CD43DD8448EB211C80319C, span_id=0x00F067AA0BA902B7, is_remote=False)
    span.get_span_context.return_value = span_context
    _mock_tracer = MagicMock()
    _mock_tracer.start_span.return_value = span
    t.tracer = _mock_tracer
    t._tracer_endpoint = t.config.otel_exporter_otlp_endpoint  # pyright: ignore[reportPrivateUsage, reportOptionalMemberAccess]
    request = _make_request()

    @t.trace_request()
    async def handler(request: Request) -> None:
        raise ValueError("fail")

    with pytest.raises(ValueError, match="fail"):
        await handler(request)

    assert get_failed_span_context(request) is span_context


@pytest.mark.asyncio
async def test_trace_request_does_not_remember_span_when_tracing_disabled():
    t = _make_tracer_with_config(enabled=False)
    request = _make_request()

    @t.trace_request()
    async def handler(request: Request) -> None:
        raise ValueError("fail")

    with pytest.raises(ValueError, match="fail"):
        await handler(request)

    assert get_failed_span_context(request) is None


@pytest.mark.asyncio
async def test_trace_request_streaming_response_sets_attributes() -> None:
    t = _make_tracer_with_config(enabled=True)
    span = _make_span()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = span
    t.tracer = fake_tracer
    t._tracer_endpoint = t.config.otel_exporter_otlp_endpoint  # pyright: ignore[reportPrivateUsage, reportOptionalMemberAccess]

    async def body_gen():
        yield b"data"

    with patch("server.utils.tracing.trace.set_span_in_context"):

        @t.trace_request()
        async def handler():
            return StreamingResponse(body_gen(), media_type="text/plain")

        await handler()

    keys = {c[0][0] for c in span.set_attribute.call_args_list}
    assert "response.status" in keys
    assert "response.content_type" in keys


@pytest.mark.asyncio
async def test_trace_request_json_response_sets_attributes() -> None:
    t = _make_tracer_with_config(enabled=True)
    span = _make_span()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = span
    t.tracer = fake_tracer
    t._tracer_endpoint = t.config.otel_exporter_otlp_endpoint  # pyright: ignore[reportPrivateUsage, reportOptionalMemberAccess]

    with patch("server.utils.tracing.trace.set_span_in_context"):

        @t.trace_request()
        async def handler():
            return JSONResponse(content={"ok": True})

        await handler()

    attr_map = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
    assert attr_map.get("response.content_type") == "application/json"


@pytest.mark.asyncio
async def test_trace_request_base_model_response_sets_attributes() -> None:
    t = _make_tracer_with_config(enabled=True)
    span = _make_span()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = span
    t.tracer = fake_tracer
    t._tracer_endpoint = t.config.otel_exporter_otlp_endpoint  # pyright: ignore[reportPrivateUsage, reportOptionalMemberAccess]

    class Resp(BaseModel):
        value: int

    with patch("server.utils.tracing.trace.set_span_in_context"):

        @t.trace_request()
        async def handler():
            return Resp(value=1)

        await handler()

    attr_map = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
    assert attr_map.get("response.status") == 200
    assert attr_map.get("response.content_type") == "application/json"


@pytest.mark.asyncio
async def test_trace_request_unknown_response_type_sets_unknown_attributes() -> None:
    t = _make_tracer_with_config(enabled=True)
    span = _make_span()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = span
    t.tracer = fake_tracer
    t._tracer_endpoint = t.config.otel_exporter_otlp_endpoint  # pyright: ignore[reportPrivateUsage, reportOptionalMemberAccess]
    with patch("server.utils.tracing.trace.set_span_in_context"):

        @t.trace_request()
        async def handler():
            return 42

        await handler()

    attr_map = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
    assert attr_map.get("response.status") == "<unknown>"
    assert attr_map.get("response.content_type") == "<unknown>"


@pytest.mark.asyncio
async def test_trace_request_preserves_function_signature() -> None:
    t = _make_tracer_with_config(enabled=False)

    async def my_handler(x: int, y: str = "a") -> str:
        return f"{x}{y}"

    wrapped = t.trace_request()(my_handler)

    assert wrapped.__name__ == "my_handler"


def test_get_mcp_instruments_builds_provider_with_service_name_and_endpoint() -> None:
    t = InfraTracer("svc")
    t.config = _make_config(endpoint="http://otel:4317")
    patches = _patch_otel_metric_context_managers()

    with patches[0] as mock_provider_cls, patches[1] as mock_exporter_cls, patches[2] as mock_reader_cls:
        instruments = t._get_mcp_instruments()  # pyright: ignore[reportPrivateUsage]

    resource_arg = mock_provider_cls.call_args[1]["resource"]
    assert resource_arg.attributes["service.name"] == "svc"
    assert mock_exporter_cls.call_args == call(endpoint="http://otel:4317", insecure=True)
    assert mock_reader_cls.call_args == call(mock_exporter_cls.return_value)
    assert isinstance(instruments, McpInstruments)

    meter = mock_provider_cls.return_value.get_meter.return_value
    counter_names = [c.args[0] for c in meter.create_counter.call_args_list]
    assert counter_names == ["mcp.healthcheck.count", "mcp.oauth.refresh.count"]
    histogram_call = meter.create_histogram.call_args
    assert histogram_call.args[0] == "mcp.healthcheck.duration"
    assert histogram_call.kwargs["unit"] == "ms"


def test_get_mcp_instruments_caches_instruments() -> None:
    t = InfraTracer("svc")
    t.config = _make_config()
    patches = _patch_otel_metric_context_managers()

    with patches[0] as mock_provider_cls, patches[1], patches[2]:
        first = t._get_mcp_instruments()  # pyright: ignore[reportPrivateUsage]
        second = t._get_mcp_instruments()  # pyright: ignore[reportPrivateUsage]

    assert first is second
    assert mock_provider_cls.return_value.get_meter.call_count == 1


def test_get_mcp_instruments_rebuilds_and_shuts_down_old_provider_on_endpoint_change() -> None:
    t = InfraTracer("svc")
    t.config = _make_config(endpoint="http://old:4317")
    patches = _patch_otel_metric_context_managers()

    with patches[0] as mock_provider_cls, patches[1], patches[2], _synchronous_daemon_threads():
        first_provider = MagicMock()
        second_provider = MagicMock()
        mock_provider_cls.side_effect = [first_provider, second_provider]

        first = t._get_mcp_instruments()  # pyright: ignore[reportPrivateUsage]
        t.config.otel_exporter_otlp_endpoint = "http://new:4317"
        second = t._get_mcp_instruments()  # pyright: ignore[reportPrivateUsage]

    assert first is not second
    assert first_provider.shutdown.call_count == 1
    assert first_provider.shutdown.call_args == call(timeout_millis=5000)
    assert second_provider.shutdown.call_count == 0


def test_mcp_instruments_rebuild_skipped_while_disabled_even_if_endpoint_changed() -> None:
    t = InfraTracer("svc")
    t.config = _make_config(enabled=False, endpoint="http://old:4317")

    with patch.object(t, "_get_mcp_instruments") as mock_get:
        t.config.otel_exporter_otlp_endpoint = "http://new:4317"
        assert t.mcp_instruments() is None

    mock_get.assert_not_called()


def test_mcp_instruments_returns_cached_instruments_after_disable_then_reenable() -> None:
    t = InfraTracer("svc")
    t.config = _make_config(enabled=True)
    patches = _patch_otel_metric_context_managers()

    with patches[0] as mock_provider_cls, patches[1], patches[2]:
        first = t.mcp_instruments()
        t.config.otel_tracing_enabled = False
        assert t.mcp_instruments() is None
        t.config.otel_tracing_enabled = True
        second = t.mcp_instruments()

    assert first is second
    assert mock_provider_cls.return_value.get_meter.call_count == 1


def test_mcp_instruments_returns_none_when_config_not_set() -> None:
    t = InfraTracer("svc")

    assert t.mcp_instruments() is None


def test_mcp_instruments_returns_none_when_tracing_disabled() -> None:
    t = InfraTracer("svc")
    t.config = _make_config(enabled=False)

    assert t.mcp_instruments() is None


def test_mcp_instruments_returns_instruments_when_tracing_enabled() -> None:
    t = InfraTracer("svc")
    t.config = _make_config(enabled=True)
    patches = _patch_otel_metric_context_managers()

    with patches[0], patches[1], patches[2]:
        result = t.mcp_instruments()

    assert isinstance(result, McpInstruments)


def test_span_yields_none_when_config_not_set() -> None:
    t = InfraTracer("svc")

    with t.span("mcp.op") as span:
        assert span is None


def test_span_yields_none_when_tracing_disabled() -> None:
    t = _make_tracer_with_config(enabled=False)

    with t.span("mcp.op") as span:
        assert span is None


def test_span_starts_and_ends_span_with_ok_status_when_enabled() -> None:
    t = _make_tracer_with_config(enabled=True)
    fake_span = _make_span()
    fake_tracer = MagicMock()
    fake_tracer.start_as_current_span.return_value.__enter__.return_value = fake_span
    t.tracer = fake_tracer
    t._tracer_endpoint = t.config.otel_exporter_otlp_endpoint  # pyright: ignore[reportPrivateUsage, reportOptionalMemberAccess]

    with t.span("mcp.op", **{"mcp.instance": "default"}) as span:
        assert span is fake_span

    assert fake_tracer.start_as_current_span.call_args == call(
        "mcp.op", attributes={"mcp.instance": "default"}, record_exception=False, set_status_on_exception=False
    )
    assert fake_tracer.start_as_current_span.return_value.__exit__.call_count == 1
    status_arg = fake_span.set_status.call_args[0][0]
    assert status_arg.status_code == StatusCode.OK


def test_span_records_exception_and_reraises_when_body_raises() -> None:
    t = _make_tracer_with_config(enabled=True)
    fake_span = _make_span()
    fake_tracer = MagicMock()
    fake_tracer.start_as_current_span.return_value.__enter__.return_value = fake_span
    t.tracer = fake_tracer
    t._tracer_endpoint = t.config.otel_exporter_otlp_endpoint  # pyright: ignore[reportPrivateUsage, reportOptionalMemberAccess]

    with pytest.raises(ValueError, match="boom"), t.span("mcp.op"):
        raise ValueError("boom")

    assert fake_span.record_exception.call_count == 1
    status_arg = fake_span.set_status.call_args[0][0]
    assert status_arg.status_code == StatusCode.ERROR


def test_span_records_ok_status_and_no_exception_when_body_raises_client_http_exception() -> None:
    t = _make_tracer_with_config(enabled=True)
    fake_span = _make_span()
    fake_tracer = MagicMock()
    fake_tracer.start_as_current_span.return_value.__enter__.return_value = fake_span
    t.tracer = fake_tracer
    t._tracer_endpoint = t.config.otel_exporter_otlp_endpoint  # pyright: ignore[reportPrivateUsage, reportOptionalMemberAccess]

    with pytest.raises(HTTPException), t.span("mcp.op"):
        raise HTTPException(status_code=400, detail="bad input")

    assert fake_span.record_exception.call_count == 0
    status_arg = fake_span.set_status.call_args[0][0]
    assert status_arg.status_code == StatusCode.OK


def test_span_records_error_status_and_exception_when_body_raises_server_http_exception() -> None:
    t = _make_tracer_with_config(enabled=True)
    fake_span = _make_span()
    fake_tracer = MagicMock()
    fake_tracer.start_as_current_span.return_value.__enter__.return_value = fake_span
    t.tracer = fake_tracer
    t._tracer_endpoint = t.config.otel_exporter_otlp_endpoint  # pyright: ignore[reportPrivateUsage, reportOptionalMemberAccess]

    with pytest.raises(HTTPException), t.span("mcp.op"):
        raise HTTPException(status_code=502, detail="upstream failed")

    assert fake_span.record_exception.call_count == 1
    status_arg = fake_span.set_status.call_args[0][0]
    assert status_arg.status_code == StatusCode.ERROR


def test_span_yields_none_when_tracer_acquisition_fails() -> None:
    t = _make_tracer_with_config(enabled=True)

    with patch.object(t, "_get_tracer", side_effect=RuntimeError("boom")), t.span("mcp.op") as span:
        assert span is None


def test_mcp_instruments_returns_none_when_building_fails() -> None:
    t = InfraTracer("svc")
    t.config = _make_config(enabled=True)

    with patch.object(t, "_get_mcp_instruments", side_effect=RuntimeError("boom")):
        assert t.mcp_instruments() is None


@pytest.mark.asyncio
async def test_shutdown_stops_tracer_and_meter_providers_and_resets_state() -> None:
    t = InfraTracer("svc")
    provider = MagicMock()
    meter_provider = MagicMock()
    t._provider = provider  # pyright: ignore[reportPrivateUsage]
    t._meter_provider = meter_provider  # pyright: ignore[reportPrivateUsage]
    t.tracer = MagicMock()
    t._tracer_endpoint = "http://old:4317"  # pyright: ignore[reportPrivateUsage]
    t._mcp_instruments = MagicMock()  # pyright: ignore[reportPrivateUsage]
    t._meter_endpoint = "http://old:4317"  # pyright: ignore[reportPrivateUsage]

    await t.shutdown()

    assert provider.shutdown.call_count == 1
    assert meter_provider.shutdown.call_args == call(timeout_millis=5000)
    assert t._provider is None  # pyright: ignore[reportPrivateUsage]
    assert t._meter_provider is None  # pyright: ignore[reportPrivateUsage]
    assert t.tracer is None
    assert t._tracer_endpoint is None  # pyright: ignore[reportPrivateUsage]
    assert t._mcp_instruments is None  # pyright: ignore[reportPrivateUsage]
    assert t._meter_endpoint is None  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_shutdown_swallows_meter_provider_shutdown_error(caplog: pytest.LogCaptureFixture) -> None:
    """A failed final flush (e.g. an unreachable OTLP collector) must not fail the caller.

    `/admin/config` and `lifecycle.py`'s shutdown phase both call `InfraTracer.shutdown()`
    directly; if `MeterProvider.shutdown()` raised through it, an admin applying an unrelated
    config change would get a misleading "could not be applied live" error even though the
    config itself applied fine.
    """
    t = InfraTracer("svc")
    meter_provider = MagicMock()
    meter_provider.shutdown.side_effect = RuntimeError("boom")
    t._meter_provider = meter_provider  # pyright: ignore[reportPrivateUsage]

    with caplog.at_level(logging.WARNING):
        await t.shutdown()  # must not raise

    assert t._meter_provider is None  # pyright: ignore[reportPrivateUsage]
    assert "Failed to shut down OTel meter provider" in caplog.text


def test_run_span_provider_shutdown_logs_error(caplog: pytest.LogCaptureFixture) -> None:
    """Mirrors `test_shutdown_swallows_meter_provider_shutdown_error()` for the span-provider side.

    `_run_span_provider_shutdown()` runs as a bare `threading.Thread` target (from both
    `_shutdown_span_provider()` and `_shutdown_span_provider_detached()`), so an unhandled exception
    would otherwise only reach `threading.excepthook` and never our structured logs.
    """
    t = InfraTracer("svc")
    provider = MagicMock()
    provider.shutdown.side_effect = RuntimeError("boom")

    with caplog.at_level(logging.WARNING):
        t._run_span_provider_shutdown(provider)  # pyright: ignore[reportPrivateUsage]  # must not raise

    assert "Failed to shut down OTel span provider" in caplog.text


@pytest.mark.asyncio
async def test_shutdown_noop_when_nothing_built() -> None:
    t = InfraTracer("svc")

    await t.shutdown()  # must not raise


@pytest.mark.asyncio
async def test_teardown_otlp_logging_keeps_provider_installed_during_shutdown() -> None:
    """DFINFRA-321: a provider installed while teardown() awaits must not be dropped by its reset."""
    manager = OtlpLoggingManager()
    old_provider = MagicMock()
    new_provider = MagicMock()
    manager._provider = old_provider  # pyright: ignore[reportPrivateUsage]

    def _shutdown_and_swap(_provider: Any) -> None:
        manager._provider = new_provider  # pyright: ignore[reportPrivateUsage]

    with patch.object(manager, "_shutdown_provider", side_effect=_shutdown_and_swap):
        await manager.teardown()

    assert manager._provider is new_provider  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_teardown_otlp_logging_keeps_configured_state_set_during_shutdown() -> None:
    """DFINFRA-321: a setup() that lands during teardown()'s await keeps its configured_enabled=True."""
    manager = OtlpLoggingManager()
    manager._provider = MagicMock()  # pyright: ignore[reportPrivateUsage]
    manager.configured_enabled = True

    def _shutdown_and_setup(_provider: Any) -> None:
        manager._provider = MagicMock()  # pyright: ignore[reportPrivateUsage]
        manager.configured_enabled = True

    with patch.object(manager, "_shutdown_provider", side_effect=_shutdown_and_setup):
        await manager.teardown()

    assert manager.configured_enabled is True


@pytest.mark.asyncio
async def test_otlp_logging_teardown_calls_are_serialized() -> None:
    """DFINFRA-321: a second teardown() waits for the first instead of interleaving with it."""
    manager = OtlpLoggingManager()
    manager._provider = MagicMock()  # pyright: ignore[reportPrivateUsage]
    release = threading.Event()
    started = threading.Event()
    active = 0
    max_active = 0

    def _blocking_shutdown(_provider: Any) -> None:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        started.set()
        release.wait(timeout=2)
        active -= 1

    with patch.object(manager, "_shutdown_provider", side_effect=_blocking_shutdown) as mock_shutdown:
        first = asyncio.create_task(manager.teardown())
        await asyncio.to_thread(started.wait, 2)
        manager._provider = MagicMock()  # pyright: ignore[reportPrivateUsage]
        second = asyncio.create_task(manager.teardown())
        # Give an unserialized second shutdown time to reach its worker thread.
        await asyncio.sleep(0.2)
        assert not second.done()
        assert mock_shutdown.call_count == 1
        release.set()
        await asyncio.gather(first, second)

    assert max_active == 1


@pytest.mark.asyncio
async def test_shutdown_keeps_span_and_meter_providers_installed_during_shutdown() -> None:
    """DFINFRA-321: providers rebuilt while shutdown() awaits survive it, with their tracer/instruments."""
    t = InfraTracer(service_name="svc")
    t._provider = MagicMock()  # pyright: ignore[reportPrivateUsage]
    t._meter_provider = MagicMock()  # pyright: ignore[reportPrivateUsage]
    new_provider = MagicMock()
    new_meter_provider = MagicMock()
    new_tracer = MagicMock()
    new_instruments = MagicMock()

    def _swap_span(_provider: Any) -> None:
        t._provider = new_provider  # pyright: ignore[reportPrivateUsage]
        t.tracer = new_tracer
        t._tracer_endpoint = "new:4317"  # pyright: ignore[reportPrivateUsage]

    def _swap_meter(_provider: Any) -> None:
        t._meter_provider = new_meter_provider  # pyright: ignore[reportPrivateUsage]
        t._mcp_instruments = new_instruments  # pyright: ignore[reportPrivateUsage]
        t._meter_endpoint = "new:4317"  # pyright: ignore[reportPrivateUsage]

    with (
        patch.object(t, "_shutdown_span_provider", side_effect=_swap_span),
        patch.object(t, "_shutdown_meter_provider", side_effect=_swap_meter),
    ):
        await t.shutdown()

    assert t._provider is new_provider  # pyright: ignore[reportPrivateUsage]
    assert t.tracer is new_tracer
    assert t._tracer_endpoint == "new:4317"  # pyright: ignore[reportPrivateUsage]
    assert t._meter_provider is new_meter_provider  # pyright: ignore[reportPrivateUsage]
    assert t._mcp_instruments is new_instruments  # pyright: ignore[reportPrivateUsage]
    assert t._meter_endpoint == "new:4317"  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_otlp_logging_reconfigure_waits_for_in_flight_teardown() -> None:
    """DFINFRA-321: reconfigure() waits for an in-flight teardown() instead of re-shutting its provider."""
    cfg = _make_config()
    cfg.otel_logging_enabled = True
    manager = OtlpLoggingManager()
    manager._provider = MagicMock()  # pyright: ignore[reportPrivateUsage]
    release = threading.Event()
    started = threading.Event()

    def _blocking_shutdown(_provider: Any) -> None:
        started.set()
        release.wait(timeout=2)

    with (
        patch.object(manager, "_shutdown_provider", side_effect=_blocking_shutdown) as mock_shutdown,
        patch.object(manager, "setup") as mock_setup,
    ):
        teardown = asyncio.create_task(manager.teardown())
        await asyncio.to_thread(started.wait, 2)
        reconfigure = asyncio.create_task(manager.reconfigure(cfg))
        # Give an unserialized reconfigure() time to reach its own provider shutdown.
        await asyncio.sleep(0.2)
        assert not reconfigure.done()
        assert mock_shutdown.call_count == 1
        assert mock_setup.call_count == 0
        release.set()
        await asyncio.gather(teardown, reconfigure)

    assert mock_shutdown.call_count == 1
    assert mock_setup.call_args_list == [call(cfg)]


@pytest.mark.asyncio
async def test_shutdown_calls_are_serialized() -> None:
    """DFINFRA-321: a second InfraTracer.shutdown() waits for the first instead of re-shutting its provider."""
    t = InfraTracer(service_name="svc")
    t._provider = MagicMock()  # pyright: ignore[reportPrivateUsage]
    release = threading.Event()
    started = threading.Event()

    def _blocking_shutdown(_provider: Any) -> None:
        started.set()
        release.wait(timeout=2)

    with patch.object(t, "_shutdown_span_provider", side_effect=_blocking_shutdown) as mock_shutdown:
        first = asyncio.create_task(t.shutdown())
        await asyncio.to_thread(started.wait, 2)
        second = asyncio.create_task(t.shutdown())
        # Give an unserialized second shutdown time to reach its own provider shutdown.
        await asyncio.sleep(0.2)
        assert not second.done()
        assert mock_shutdown.call_count == 1
        release.set()
        await asyncio.gather(first, second)

    assert mock_shutdown.call_count == 1
    assert t._provider is None  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_close_stops_providers_and_disables_further_tracing() -> None:
    """DFINFRA-321: after close(), span()/mcp_instruments() don't rebuild providers that nothing would shut down."""
    t = _make_tracer_with_config(enabled=True)
    span_provider = MagicMock()
    meter_provider = MagicMock()
    t._provider = span_provider  # pyright: ignore[reportPrivateUsage]
    t._meter_provider = meter_provider  # pyright: ignore[reportPrivateUsage]

    await t.close()

    span_provider.shutdown.assert_called_once()
    meter_provider.shutdown.assert_called_once()
    with patch.object(t, "_get_tracer") as mock_get_tracer, patch.object(t, "_get_mcp_instruments") as mock_get_instruments:
        with t.span("mcp.op") as span:
            assert span is None
        assert t.mcp_instruments() is None
    mock_get_tracer.assert_not_called()
    mock_get_instruments.assert_not_called()
    assert t._provider is None  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_close_blocks_rebuild_started_during_shutdown() -> None:
    """DFINFRA-321: a span() during the final shutdown's await can't install a fresh provider."""
    t = _make_tracer_with_config(enabled=True)
    t._provider = MagicMock()  # pyright: ignore[reportPrivateUsage]
    release = threading.Event()
    started = threading.Event()

    def _blocking_shutdown(_provider: Any) -> None:
        started.set()
        release.wait(timeout=2)

    with (
        patch.object(t, "_shutdown_span_provider", side_effect=_blocking_shutdown),
        patch.object(t, "_get_tracer") as mock_get_tracer,
    ):
        closing = asyncio.create_task(t.close())
        await asyncio.to_thread(started.wait, 2)
        with t.span("mcp.op") as span:
            assert span is None
        release.set()
        await closing

    mock_get_tracer.assert_not_called()


@pytest.mark.asyncio
async def test_otlp_logging_close_tears_down_and_blocks_setup_and_reconfigure() -> None:
    """DFINFRA-321: nothing is set up after close(), so no provider outlives the final teardown."""
    cfg = _make_config()
    cfg.otel_logging_enabled = True
    manager = OtlpLoggingManager()
    provider = MagicMock()
    manager._provider = provider  # pyright: ignore[reportPrivateUsage]

    with patch.object(manager, "_shutdown_provider") as mock_shutdown:
        await manager.close()
    manager.setup(cfg)
    await manager.reconfigure(cfg)

    mock_shutdown.assert_called_once_with(provider)
    assert manager._provider is None  # pyright: ignore[reportPrivateUsage]
    assert manager.configured_enabled is False


@pytest.mark.asyncio
async def test_otlp_logging_reconfigure_queued_behind_close_does_not_set_up() -> None:
    """DFINFRA-321: a reconfigure() waiting on the lock while close() runs doesn't rebuild afterwards."""
    cfg = _make_config()
    cfg.otel_logging_enabled = True
    manager = OtlpLoggingManager()
    manager._provider = MagicMock()  # pyright: ignore[reportPrivateUsage]
    release = threading.Event()
    started = threading.Event()

    def _blocking_shutdown(_provider: Any) -> None:
        started.set()
        release.wait(timeout=2)

    with patch.object(manager, "_shutdown_provider", side_effect=_blocking_shutdown), patch.object(manager, "setup") as mock_setup:
        closing = asyncio.create_task(manager.close())
        await asyncio.to_thread(started.wait, 2)
        reconfigure = asyncio.create_task(manager.reconfigure(cfg))
        await asyncio.sleep(0.2)
        release.set()
        await asyncio.gather(closing, reconfigure)

    mock_setup.assert_not_called()
