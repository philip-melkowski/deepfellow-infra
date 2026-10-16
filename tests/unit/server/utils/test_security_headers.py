# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Tests for server/utils/security_headers.py."""

from unittest import mock

import pytest
from fastapi import FastAPI
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from server.utils.security_headers import (
    CACHE_CONTROL,
    CONTENT_SECURITY_POLICY,
    DOCS_CONTENT_SECURITY_POLICY,
    DOCS_PATHS,
    STRICT_TRANSPORT_SECURITY,
    SecurityHeadersMiddleware,
)


async def _index(_request: Request) -> PlainTextResponse:
    return PlainTextResponse("ok")


async def _custom_frame_options(_request: Request) -> PlainTextResponse:
    return PlainTextResponse("ok", headers={"X-Frame-Options": "SAMEORIGIN"})


async def _custom_cache_control(_request: Request) -> PlainTextResponse:
    return PlainTextResponse("ok", headers={"Cache-Control": "max-age=60"})


@pytest.fixture
def app() -> Starlette:
    routes = [
        Route("/", _index),
        Route("/docs", _index),
        Route("/redoc", _index),
        Route("/docs/oauth2-redirect", _index),
        Route("/framed", _custom_frame_options),
        Route("/cached", _custom_cache_control),
    ]
    app = Starlette(routes=routes)
    app.add_middleware(SecurityHeadersMiddleware)
    return app


@pytest.fixture
def client(app: Starlette) -> TestClient:
    return TestClient(app)


def test_security_headers_middleware_adds_headers_to_response(client: TestClient):
    resp = client.get("/")

    assert resp.headers["Content-Security-Policy"] == CONTENT_SECURITY_POLICY
    assert resp.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"


def test_security_headers_middleware_adds_anti_cache_headers_to_response(client: TestClient):
    resp = client.get("/")

    assert resp.headers["Cache-Control"] == CACHE_CONTROL
    assert resp.headers["Pragma"] == "no-cache"
    assert resp.headers["Expires"] == "0"


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/docs/oauth2-redirect"])
def test_security_headers_middleware_uses_docs_policy_on_docs_paths(client: TestClient, path: str):
    resp = client.get(path)

    assert resp.headers["Content-Security-Policy"] == DOCS_CONTENT_SECURITY_POLICY
    assert "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net" in resp.headers["Content-Security-Policy"]


def test_security_headers_middleware_forbids_inline_and_cdn_script_elsewhere(client: TestClient):
    resp = client.get("/")

    assert "script-src 'self';" in resp.headers["Content-Security-Policy"]
    assert "frame-ancestors 'none'" in resp.headers["Content-Security-Policy"]
    assert "cdn.jsdelivr.net" not in resp.headers["Content-Security-Policy"]


def test_content_security_policy_allows_data_uri_media():
    assert "media-src 'self' data: blob:" in CONTENT_SECURITY_POLICY


@pytest.mark.parametrize("base_url", ["http://testserver", "https://testserver"])
def test_security_headers_middleware_adds_hsts_regardless_of_scheme(app: Starlette, base_url: str):
    client = TestClient(app, base_url=base_url)

    resp = client.get("/")

    assert resp.headers["Strict-Transport-Security"] == STRICT_TRANSPORT_SECURITY


def test_security_headers_middleware_keeps_header_set_by_route(client: TestClient):
    resp = client.get("/framed")

    assert resp.headers["X-Frame-Options"] == "SAMEORIGIN"
    assert resp.headers["X-Content-Type-Options"] == "nosniff"


def test_security_headers_middleware_keeps_cache_control_set_by_route(client: TestClient):
    resp = client.get("/cached")

    assert resp.headers["Cache-Control"] == "max-age=60"
    assert "Pragma" not in resp.headers
    assert "Expires" not in resp.headers


@pytest.mark.asyncio
async def test_security_headers_middleware_passes_through_non_http_scope():
    _mock_app = mock.AsyncMock()
    scope = {"type": "websocket"}
    receive = mock.AsyncMock()
    send = mock.AsyncMock()

    await SecurityHeadersMiddleware(_mock_app)(scope, receive, send)

    assert _mock_app.call_count == 1
    assert _mock_app.call_args == mock.call(scope, receive, send)


# server/main.py keeps FastAPI's default docs URLs; importing it here would pull untested modules into coverage.
def test_docs_paths_match_fastapi_default_docs_urls():
    fastapi_app = FastAPI()

    assert {fastapi_app.docs_url, fastapi_app.redoc_url, fastapi_app.swagger_ui_oauth2_redirect_url} == DOCS_PATHS
