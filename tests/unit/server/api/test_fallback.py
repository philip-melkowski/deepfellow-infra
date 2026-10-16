# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Tests for server/api/fallback.py."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from starlette.exceptions import HTTPException
from starlette.testclient import TestClient

from server.api.fallback import STATIC_CACHE_CONTROL, StaticFilesHandler
from server.utils.security_headers import SecurityHeadersMiddleware


def _make_app(static_dir: str) -> FastAPI:
    app = FastAPI()
    app.mount("/", StaticFilesHandler(directory=static_dir, html=True))
    return app


def test_static_files_handler_serves_existing_file(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("world")
    (tmp_path / "index.html").write_text("<html/>")

    with TestClient(_make_app(str(tmp_path)), raise_server_exceptions=False) as client:
        resp = client.get("/hello.txt")

    assert resp.status_code == 200
    assert resp.text == "world"
    assert resp.headers["Cache-Control"] == STATIC_CACHE_CONTROL


def test_static_files_handler_keeps_its_cache_control_behind_security_headers_middleware(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<html/>")
    app = _make_app(str(tmp_path))
    app.add_middleware(SecurityHeadersMiddleware)

    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/")

    assert resp.headers["Cache-Control"] == STATIC_CACHE_CONTROL
    assert "Pragma" not in resp.headers
    assert "Expires" not in resp.headers
    assert resp.headers["X-Frame-Options"] == "DENY"


def test_static_files_handler_serves_index_html_for_unknown_path(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<html>spa</html>")

    with TestClient(_make_app(str(tmp_path)), raise_server_exceptions=False) as client:
        resp = client.get("/some/nested/route")

    # SPA routing: 404 is caught and index.html is returned with 200
    assert resp.status_code == 200
    assert resp.text == "<html>spa</html>"
    assert resp.headers["Cache-Control"] == STATIC_CACHE_CONTROL


def test_static_files_handler_serves_index_html_for_missing_file(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<html>app</html>")

    with TestClient(_make_app(str(tmp_path)), raise_server_exceptions=False) as client:
        resp = client.get("/does-not-exist.js")

    # SPA routing: missing asset 404 falls back to index.html
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_static_files_handler_reraises_non_404_http_exception(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<html/>")
    handler = StaticFilesHandler(directory=str(tmp_path))

    with (
        patch(
            "fastapi.staticfiles.StaticFiles.get_response",
            new=AsyncMock(side_effect=HTTPException(status_code=403)),
        ),
        pytest.raises(HTTPException) as exc_info,
    ):
        await handler.get_response("secret.txt", {})

    assert exc_info.value.status_code == 403
