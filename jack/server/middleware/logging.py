"""
Logging Middleware - Request/Response logging.
"""

from __future__ import annotations
import time
import logging
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Request/Response logging middleware.

    Logs:
    - Request method, path, and timing
    - Response status code
    - Client IP (for debugging)
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with logging."""

        start_time = time.time()

        # Get client IP
        client_ip = request.client.host if request.client else "unknown"

        # Log request
        logger.info(f"[{request.method}] {request.url.path} from {client_ip}")

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000

        # Log response
        log_level = logging.INFO if response.status_code < 400 else logging.WARNING
        logger.log(
            log_level,
            f"[{request.method}] {request.url.path} -> {response.status_code} ({duration_ms:.1f}ms)",
        )

        # Add timing header
        response.headers["X-Response-Time"] = f"{duration_ms:.1f}ms"

        return response
