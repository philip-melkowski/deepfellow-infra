# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""ASGI middleware adding HTTP security headers to every response."""

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_CSP_DIRECTIVES: dict[str, str] = {
    "default-src": "'self'",
    "script-src": "'self'",
    "style-src": "'self' 'unsafe-inline'",
    "style-src-elem": "'self' 'unsafe-inline'",
    "style-src-attr": "'unsafe-inline'",
    "img-src": "'self' data: blob:",
    # The WebUI plays model test output (audio) from data: URIs.
    "media-src": "'self' data: blob:",
    "font-src": "'self' data:",
    "connect-src": "'self'",
    "worker-src": "'self' blob:",
    "object-src": "'none'",
    "base-uri": "'self'",
    "form-action": "'self'",
    "frame-ancestors": "'none'",
}

# FastAPI's default Swagger UI and ReDoc pages load their assets from jsDelivr (ReDoc also from Google Fonts)
# and bootstrap themselves with an inline <script>.
_DOCS_CSP_DIRECTIVES: dict[str, str] = {
    **_CSP_DIRECTIVES,
    "script-src": "'self' 'unsafe-inline' https://cdn.jsdelivr.net",
    "style-src": "'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com",
    "style-src-elem": "'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com",
    "img-src": "'self' data: blob: https://fastapi.tiangolo.com",
    "font-src": "'self' data: https://fonts.gstatic.com",
}


def _build_policy(directives: dict[str, str]) -> str:
    return "; ".join(f"{name} {value}" for name, value in directives.items())


CONTENT_SECURITY_POLICY = _build_policy(_CSP_DIRECTIVES)
DOCS_CONTENT_SECURITY_POLICY = _build_policy(_DOCS_CSP_DIRECTIVES)
# Must match FastAPI's docs_url, redoc_url and swagger_ui_oauth2_redirect_url.
DOCS_PATHS = frozenset({"/docs", "/redoc", "/docs/oauth2-redirect"})

STRICT_TRANSPORT_SECURITY = "max-age=31536000"

CACHE_CONTROL = "no-store, no-cache, must-revalidate"


def add_security_headers(headers: MutableHeaders, scope: Scope) -> None:
    """Add the security headers to response headers, keeping any header that is already set.

    Also used directly by the catch-all 500 handler, whose response is built outside the middleware stack.

    Args:
        headers: The response headers to extend.
        scope: The ASGI connection scope of the request.
    """
    docs = scope["path"] in DOCS_PATHS
    headers.setdefault("Content-Security-Policy", DOCS_CONTENT_SECURITY_POLICY if docs else CONTENT_SECURITY_POLICY)
    headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    headers.setdefault("X-Content-Type-Options", "nosniff")
    headers.setdefault("X-Frame-Options", "DENY")
    headers.setdefault("Strict-Transport-Security", STRICT_TRANSPORT_SECURITY)
    if "Cache-Control" not in headers:
        headers["Cache-Control"] = CACHE_CONTROL
        headers.setdefault("Pragma", "no-cache")
        headers.setdefault("Expires", "0")


class SecurityHeadersMiddleware:
    """Add security headers to HTTP responses without overriding headers a route already set.

    A pure ASGI middleware (not `BaseHTTPMiddleware`), so streamed responses (SSE, proxied completions) pass through untouched.
    `Strict-Transport-Security` is sent on every response: Infra usually sits behind a TLS-terminating proxy, so it cannot
    tell HTTPS requests apart, and browsers ignore the header when it arrives over plain HTTP.
    Anti-cache headers are added only when the response has no `Cache-Control` yet, so static files keep their own policy.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Wrap the given ASGI app."""
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle an ASGI call, adding the headers to the response start message."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_security_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                add_security_headers(MutableHeaders(scope=message), scope)
            await send(message)

        await self.app(scope, receive, send_with_security_headers)
