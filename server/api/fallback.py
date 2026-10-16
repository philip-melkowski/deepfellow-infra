# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Custom static files handler for SPA routing support."""

from pathlib import Path

from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException
from starlette.responses import FileResponse, Response
from starlette.types import Scope

STATIC_CACHE_CONTROL = "no-cache"


class StaticFilesHandler(StaticFiles):
    """Custom static files handler that serves index.html for 404s (SPA routing)."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        """Return an HTTP response, given the incoming path, method and request headers."""
        try:
            response = await super().get_response(path, scope)
        except HTTPException as e:
            if e.status_code != 404 or self.directory is None:
                raise
            response = FileResponse(Path(self.directory) / "index.html")
        # Public files: browsers may keep them but must revalidate (ETag -> 304), unlike the API's no-store.
        response.headers["Cache-Control"] = STATIC_CACHE_CONTROL
        return response
