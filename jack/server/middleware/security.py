"""
Security Middleware - Request validation and protection.
"""

from __future__ import annotations
import time
import logging
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

logger = logging.getLogger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Security middleware for request validation.

    Features:
    - Request size limits
    - Content-Type validation
    - Security headers
    - Basic request logging
    """

    def __init__(
        self,
        app,
        max_content_length: int = 10 * 1024 * 1024,  # 10MB
        allowed_content_types: tuple = (
            "application/json",
            "text/plain",
            "multipart/form-data",
        ),
    ):
        super().__init__(app)
        self.max_content_length = max_content_length
        self.allowed_content_types = allowed_content_types

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with security checks."""

        # Check content length
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_content_length:
            return JSONResponse(
                {"detail": "Request too large"},
                status_code=413,
            )

        # Check content type for POST/PUT/PATCH
        if request.method in ("POST", "PUT", "PATCH"):
            content_type = request.headers.get("content-type", "")
            base_type = content_type.split(";")[0].strip()

            if base_type and base_type not in self.allowed_content_types:
                return JSONResponse(
                    {"detail": f"Unsupported content type: {base_type}"},
                    status_code=415,
                )

        # Process request
        response = await call_next(request)

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        return response
