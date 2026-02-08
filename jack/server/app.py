"""
Jack Server - FastAPI Application.

Main application factory with swappable LLM support and secure API.
"""

from __future__ import annotations
import os
import asyncio
import logging
from typing import Optional, Dict, Any, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Depends, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from jack.server.config import ServerConfig, LLMProvider
from jack.server.auth import (
    AuthManager, AuthError, TokenExpiredError,
    InvalidTokenError, InvalidAPIKeyError, RateLimitExceededError,
)

# Import core verifier for safety checks
from jack.core.verifier import Verifier

logger = logging.getLogger(__name__)

# Deep logging buffer for tracing agent decisions
_deep_log_buffer: list = []
_max_log_entries = 500


# =============================================================================
# Request/Response Models
# =============================================================================

class AgentQuery(BaseModel):
    """Request model for agent queries."""
    query: str = Field(..., min_length=1, max_length=10000)
    context: Optional[Dict[str, Any]] = None
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class AgentResponse(BaseModel):
    """Response model for agent queries."""
    response: str
    reasoning: Optional[str] = None
    confidence: float = 0.0
    actions_taken: list = Field(default_factory=list)
    tokens_used: Optional[int] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    llm_provider: str
    llm_model: str
    uptime_seconds: float


class TokenRequest(BaseModel):
    """Request for JWT token."""
    user_id: str
    scopes: list = Field(default_factory=list)


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class APIKeyRequest(BaseModel):
    """Request to create API key."""
    name: str
    scopes: list = Field(default_factory=list)
    expires_days: Optional[int] = None


class APIKeyResponse(BaseModel):
    """API key creation response."""
    key_id: str
    key: str  # Only returned once!
    name: str
    message: str = "Save this key - it cannot be recovered!"


class ExecuteRequest(BaseModel):
    """Request to execute a primitive action directly."""
    action: str = Field(..., description="Action type: shell, file_read, file_write, http")
    params: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")
    timeout: Optional[float] = Field(30.0, description="Timeout in seconds")


class ExecuteResponse(BaseModel):
    """Response from direct primitive execution."""
    success: bool
    outcome: str
    output: Optional[Any] = None
    error: Optional[str] = None
    blocked: bool = False
    block_reason: Optional[str] = None
    duration_ms: float = 0.0


# =============================================================================
# Application State
# =============================================================================

@dataclass
class AppState:
    """Application state container."""
    config: ServerConfig
    auth: AuthManager
    agent: Any = None  # Jack Loop instance
    reasoner: Any = None  # LLM Reasoner
    verified_executor: Any = None  # VerifiedExecutor for safe action execution
    verifier: Any = None  # Core Verifier instance
    start_time: float = 0.0

    @property
    def uptime(self) -> float:
        import time
        return time.time() - self.start_time


# =============================================================================
# Dependency Injection
# =============================================================================

async def get_state(request: Request) -> AppState:
    """Get application state from request."""
    return request.app.state.jack


async def verify_auth(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
) -> Dict[str, Any]:
    """
    Verify authentication from headers.

    Supports both:
    - Authorization: Bearer <jwt_token>
    - X-API-Key: <api_key>
    """
    state: AppState = request.app.state.jack

    try:
        if authorization and authorization.startswith("Bearer "):
            token = authorization[7:]
            payload = state.auth.verify_token(token)
            identifier = f"user:{payload.sub}"
            state.auth.check_rate_limit(identifier)
            return {"type": "jwt", "user_id": payload.sub, "scopes": payload.scopes}

        elif x_api_key:
            api_key = state.auth.verify_api_key(x_api_key)
            identifier = f"key:{api_key.key_id}"
            state.auth.check_rate_limit(identifier, api_key.rate_limit)
            return {"type": "api_key", "key_id": api_key.key_id, "scopes": api_key.scopes}

        else:
            raise HTTPException(
                status_code=401,
                detail="Authentication required. Use Authorization header or X-API-Key.",
            )

    except TokenExpiredError:
        raise HTTPException(status_code=401, detail="Token expired")
    except InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except InvalidAPIKeyError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except RateLimitExceededError as e:
        raise HTTPException(status_code=429, detail=str(e))


async def verify_admin(auth: Dict[str, Any] = Depends(verify_auth)) -> Dict[str, Any]:
    """Verify admin scope."""
    if "admin" not in auth.get("scopes", []):
        raise HTTPException(status_code=403, detail="Admin access required")
    return auth


# =============================================================================
# Application Factory
# =============================================================================

def create_app(config: Optional[ServerConfig] = None) -> FastAPI:
    """
    Create FastAPI application with Jack agent integration.

    Args:
        config: Server configuration (uses env vars if not provided)

    Returns:
        Configured FastAPI application
    """
    config = config or ServerConfig.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Application lifespan manager."""
        import time

        logger.info("Starting Jack Server...")

        # Initialize state
        state = AppState(
            config=config,
            auth=AuthManager(
                jwt_secret=config.auth.jwt_secret,
                jwt_expiry_hours=config.auth.jwt_expiry_hours,
                api_key_prefix=config.auth.api_key_prefix,
                rate_limit_requests=config.auth.rate_limit_requests,
                rate_limit_window=config.auth.rate_limit_window_seconds,
            ),
            start_time=time.time(),
        )

        # Initialize LLM reasoner based on provider
        try:
            from jack.foundation.llm import (
                create_local_reasoner,
                create_openai_reasoner,
                create_anthropic_reasoner,
            )

            if config.llm.provider == LLMProvider.LOCAL:
                state.reasoner = create_local_reasoner(
                    base_url=config.llm.base_url,
                    model=config.llm.model,
                )
                logger.info(f"Using local LLM: {config.llm.model} at {config.llm.base_url}")

            elif config.llm.provider == LLMProvider.OPENAI:
                state.reasoner = create_openai_reasoner(
                    api_key=config.llm.api_key,
                    model=config.llm.model,
                )
                logger.info(f"Using OpenAI: {config.llm.model}")

            elif config.llm.provider == LLMProvider.ANTHROPIC:
                state.reasoner = create_anthropic_reasoner(
                    api_key=config.llm.api_key,
                    model=config.llm.model,
                )
                logger.info(f"Using Anthropic: {config.llm.model}")

            else:
                # Custom/fallback
                state.reasoner = create_local_reasoner(
                    base_url=config.llm.base_url,
                    model=config.llm.model,
                )
                logger.info(f"Using custom LLM: {config.llm.model}")

        except ImportError as e:
            logger.warning(f"Could not import LLM module: {e}")
            state.reasoner = None

        # Initialize VerifiedExecutor (safe action execution)
        try:
            from jack.foundation.verified_executor import VerifiedExecutor, create_verified_executor
            from jack.foundation.action import Executor

            # Create verified executor with safety checks
            state.verifier = Verifier()
            state.verified_executor = create_verified_executor(
                working_dir="/tmp/jack",  # Safe working directory
            )
            logger.info("VerifiedExecutor initialized with safety checks")

        except ImportError as e:
            logger.warning(f"Could not import VerifiedExecutor: {e}")
            state.verified_executor = None
            state.verifier = Verifier()  # Still have verifier for manual checks

        # Initialize Jack Loop (agent)
        try:
            from jack.foundation.loop import Loop, LoopEventData

            if state.reasoner:
                # Pass verified executor to the Loop
                if state.verified_executor:
                    state.agent = Loop(
                        reasoner=state.reasoner,
                        executor=state.verified_executor,
                    )
                    logger.info("Jack Loop initialized with LLM reasoner and VerifiedExecutor")
                else:
                    state.agent = Loop(reasoner=state.reasoner)
                    logger.info("Jack Loop initialized with LLM reasoner (no executor)")
            else:
                state.agent = Loop()
                logger.info("Jack Loop initialized with default reasoner")

            # Add deep logging event listener
            def deep_log_event(event: LoopEventData):
                import time
                global _deep_log_buffer, _max_log_entries
                entry = {
                    "timestamp": time.time(),
                    "phase": event.phase.name,
                    "event": event.event.name,
                    "data": str(event.data)[:1000],  # Truncate large data
                }
                _deep_log_buffer.append(entry)
                if len(_deep_log_buffer) > _max_log_entries:
                    _deep_log_buffer = _deep_log_buffer[-_max_log_entries:]
                logger.info(f"[TRACE] [{event.phase.name}] {event.event.name}: {str(event.data)[:200]}")

            if state.agent:
                state.agent.add_listener(deep_log_event)
                logger.info("Deep logging event listener attached to Loop")

        except ImportError as e:
            logger.warning(f"Could not import Jack Foundation: {e}")
            state.agent = None

        app.state.jack = state
        logger.info(f"Jack Server ready on {config.host}:{config.port}")

        yield

        # Cleanup
        logger.info("Shutting down Jack Server...")

    # Create app
    app = FastAPI(
        title="Jack Agent API",
        description="Production API for Jack Foundation AI Agent",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.cors_origins),
        allow_credentials=config.cors_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ==========================================================================
    # Health & Info Routes
    # ==========================================================================

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check(state: AppState = Depends(get_state)) -> HealthResponse:
        """Check server health and get basic info."""
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            llm_provider=state.config.llm.provider.value,
            llm_model=state.config.llm.model,
            uptime_seconds=state.uptime,
        )

    @app.get("/", tags=["Health"])
    async def root() -> Dict[str, str]:
        """API root - basic info."""
        return {
            "name": "Jack Agent API",
            "version": "1.0.0",
            "docs": "/docs",
        }

    # ==========================================================================
    # Authentication Routes
    # ==========================================================================

    @app.post("/auth/token", response_model=TokenResponse, tags=["Auth"])
    async def create_token(
        request: TokenRequest,
        state: AppState = Depends(get_state),
    ) -> TokenResponse:
        """
        Create a JWT token.

        Note: In production, this should verify credentials first!
        """
        token = state.auth.create_token(
            user_id=request.user_id,
            scopes=tuple(request.scopes),
        )
        return TokenResponse(
            access_token=token,
            expires_in=state.config.auth.jwt_expiry_hours * 3600,
        )

    @app.post("/auth/refresh", tags=["Auth"])
    async def refresh_token(
        state: AppState = Depends(get_state),
        auth: Dict[str, Any] = Depends(verify_auth),
    ) -> TokenResponse:
        """Refresh an existing JWT token."""
        if auth["type"] != "jwt":
            raise HTTPException(400, "Can only refresh JWT tokens")

        new_token = state.auth.create_token(
            user_id=auth["user_id"],
            scopes=auth["scopes"],
        )
        return TokenResponse(
            access_token=new_token,
            expires_in=state.config.auth.jwt_expiry_hours * 3600,
        )

    @app.post("/auth/api-key", response_model=APIKeyResponse, tags=["Auth"])
    async def create_api_key(
        request: APIKeyRequest,
        state: AppState = Depends(get_state),
        auth: Dict[str, Any] = Depends(verify_admin),
    ) -> APIKeyResponse:
        """Create a new API key (admin only)."""
        key_id, key = state.auth.create_api_key(
            name=request.name,
            scopes=tuple(request.scopes),
            expires_days=request.expires_days,
        )
        return APIKeyResponse(
            key_id=key_id,
            key=key,
            name=request.name,
        )

    @app.get("/auth/api-keys", tags=["Auth"])
    async def list_api_keys(
        state: AppState = Depends(get_state),
        auth: Dict[str, Any] = Depends(verify_admin),
    ) -> list:
        """List all API keys (admin only)."""
        return state.auth.api_keys.list_keys()

    @app.delete("/auth/api-key/{key_id}", tags=["Auth"])
    async def revoke_api_key(
        key_id: str,
        state: AppState = Depends(get_state),
        auth: Dict[str, Any] = Depends(verify_admin),
    ) -> Dict[str, bool]:
        """Revoke an API key (admin only)."""
        success = state.auth.revoke_api_key(key_id)
        if not success:
            raise HTTPException(404, "API key not found")
        return {"revoked": True}

    # ==========================================================================
    # Agent Routes
    # ==========================================================================

    @app.post("/agent/query", response_model=AgentResponse, tags=["Agent"])
    async def query_agent(
        request: AgentQuery,
        state: AppState = Depends(get_state),
        auth: Dict[str, Any] = Depends(verify_auth),
    ) -> AgentResponse:
        """
        Query the Jack agent.

        This runs the full agent loop: perceive → reason → verify → act → observe.
        Actions are filtered through the Verifier for safety.
        """
        if not state.agent:
            raise HTTPException(503, "Agent not initialized")

        if request.stream:
            raise HTTPException(400, "Use /agent/stream for streaming responses")

        try:
            from jack.foundation.state import Goal, GoalType

            # Create goal from query
            goal = Goal(
                intent=request.query,
                goal_type=GoalType.QUERY,
                success_criteria=("provides_answer",),
                metadata=request.context or {},
            )

            # Run the full agent loop (synchronously in thread pool)
            def run_loop():
                return state.agent.run(goal)

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, run_loop)

            if result.is_ok():
                action_results = result.unwrap()
                # Collect action info
                actions_taken = []
                for ar in action_results:
                    actions_taken.append({
                        "type": ar.primitive.primitive_type.name,
                        "outcome": ar.outcome.name,
                        "output": str(ar.output)[:500] if ar.output else None,
                    })

                # Get final response from last successful action or reasoning
                final_output = ""
                for ar in reversed(action_results):
                    if ar.is_success and ar.output:
                        if isinstance(ar.output, dict):
                            final_output = ar.output.get("stdout", "") or ar.output.get("content", "") or str(ar.output)
                        else:
                            final_output = str(ar.output)
                        break

                return AgentResponse(
                    response=final_output or "Goal completed",
                    reasoning=state.agent.state.reasoning_trace if hasattr(state.agent.state, 'reasoning_trace') else None,
                    confidence=0.9,
                    actions_taken=actions_taken,
                )
            else:
                error = result.unwrap_err()
                return AgentResponse(
                    response=f"Agent failed: {error.message}",
                    confidence=0.0,
                    actions_taken=[],
                )

        except Exception as e:
            logger.exception("Agent query failed")
            raise HTTPException(500, f"Agent error: {str(e)}")

    @app.post("/agent/stream", tags=["Agent"])
    async def stream_agent(
        request: AgentQuery,
        state: AppState = Depends(get_state),
        auth: Dict[str, Any] = Depends(verify_auth),
    ) -> StreamingResponse:
        """
        Stream agent response.

        Returns Server-Sent Events (SSE) with incremental response.
        """
        if not state.reasoner:
            raise HTTPException(503, "Agent reasoner not initialized")

        async def generate():
            try:
                async for chunk in state.reasoner.reason_stream(request.query):
                    yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: [ERROR] {str(e)}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
        )

    @app.post("/agent/reason", tags=["Agent"])
    async def reason_only(
        request: AgentQuery,
        state: AppState = Depends(get_state),
        auth: Dict[str, Any] = Depends(verify_auth),
    ) -> Dict[str, Any]:
        """
        Direct LLM reasoning without full agent loop.

        Useful for simple queries that don't need planning/actions.
        """
        if not state.reasoner:
            raise HTTPException(503, "Reasoner not initialized")

        try:
            result = await state.reasoner.reason(request.query)
            if result.is_ok():
                return {"response": result.unwrap(), "success": True}
            else:
                return {"error": str(result.unwrap_err()), "success": False}
        except Exception as e:
            raise HTTPException(500, f"Reasoning error: {str(e)}")

    @app.post("/agent/reason/json", tags=["Agent"])
    async def reason_json(
        request: AgentQuery,
        state: AppState = Depends(get_state),
        auth: Dict[str, Any] = Depends(verify_auth),
    ) -> Dict[str, Any]:
        """
        Structured JSON reasoning.

        Returns parsed JSON from LLM response.
        """
        if not state.reasoner:
            raise HTTPException(503, "Reasoner not initialized")

        try:
            result = await state.reasoner.reason_json(request.query)
            if result.is_ok():
                return {"data": result.unwrap(), "success": True}
            else:
                return {"error": str(result.unwrap_err()), "success": False}
        except Exception as e:
            raise HTTPException(500, f"JSON reasoning error: {str(e)}")

    @app.post("/agent/execute", response_model=ExecuteResponse, tags=["Agent"])
    async def execute_primitive(
        request: ExecuteRequest,
        state: AppState = Depends(get_state),
        auth: Dict[str, Any] = Depends(verify_auth),
    ) -> ExecuteResponse:
        """
        Execute a primitive action directly (with safety verification).

        Supported actions:
        - shell: {"command": "ls -la"}
        - file_read: {"path": "/tmp/file.txt"}
        - file_write: {"path": "/tmp/file.txt", "content": "hello"}
        - http: {"method": "GET", "url": "https://api.example.com"}

        All actions pass through the Verifier - dangerous commands are BLOCKED.
        """
        if not state.verified_executor:
            raise HTTPException(503, "Executor not initialized")

        try:
            from jack.foundation.plan import Primitive, PrimitiveType
            from jack.foundation.action import OutcomeType

            # Map action string to primitive
            action_map = {
                "shell": lambda p: Primitive.shell(
                    p.get("command", ""),
                    timeout=request.timeout or 30.0
                ),
                "file_read": lambda p: Primitive.read_file(p.get("path", "")),
                "file_write": lambda p: Primitive.write_file(
                    p.get("path", ""),
                    p.get("content", "")
                ),
                "http": lambda p: Primitive.http(
                    p.get("method", "GET"),
                    p.get("url", ""),
                    body=p.get("body"),
                    headers=p.get("headers", {})
                ),
            }

            if request.action not in action_map:
                raise HTTPException(400, f"Unknown action: {request.action}. Use: {list(action_map.keys())}")

            # Create primitive
            primitive = action_map[request.action](request.params)

            # Execute through verified executor (runs in thread pool)
            def run_execute():
                return state.verified_executor.execute(primitive)

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, run_execute)

            # Build response
            if result.outcome == OutcomeType.BLOCKED:
                return ExecuteResponse(
                    success=False,
                    outcome="BLOCKED",
                    blocked=True,
                    block_reason=result.error.message if result.error else "Blocked by verifier",
                    duration_ms=result.duration_ms,
                )
            elif result.is_success:
                return ExecuteResponse(
                    success=True,
                    outcome="SUCCESS",
                    output=result.output,
                    duration_ms=result.duration_ms,
                )
            else:
                return ExecuteResponse(
                    success=False,
                    outcome=result.outcome.name,
                    error=result.error.message if result.error else "Unknown error",
                    duration_ms=result.duration_ms,
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Execute primitive failed")
            raise HTTPException(500, f"Execution error: {str(e)}")

    @app.get("/agent/executor/stats", tags=["Agent"])
    async def get_executor_stats(
        state: AppState = Depends(get_state),
        auth: Dict[str, Any] = Depends(verify_auth),
    ) -> Dict[str, Any]:
        """Get verified executor statistics (blocked/allowed counts)."""
        if not state.verified_executor:
            return {"error": "Executor not initialized"}

        return state.verified_executor.get_stats()

    # ==========================================================================
    # LLM Management Routes
    # ==========================================================================

    @app.get("/llm/status", tags=["LLM"])
    async def llm_status(
        state: AppState = Depends(get_state),
        auth: Dict[str, Any] = Depends(verify_auth),
    ) -> Dict[str, Any]:
        """Get current LLM provider status."""
        return {
            "provider": state.config.llm.provider.value,
            "model": state.config.llm.model,
            "base_url": state.config.llm.base_url,
            "available": state.reasoner is not None,
        }

    @app.post("/llm/test", tags=["LLM"])
    async def test_llm(
        state: AppState = Depends(get_state),
        auth: Dict[str, Any] = Depends(verify_auth),
    ) -> Dict[str, Any]:
        """Test LLM connectivity with a simple prompt."""
        if not state.reasoner:
            return {"success": False, "error": "Reasoner not initialized"}

        try:
            result = await state.reasoner.reason("Say 'Hello, I am working!' in exactly those words.")
            if result.is_ok():
                return {"success": True, "response": result.unwrap()}
            else:
                return {"success": False, "error": str(result.unwrap_err())}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ==========================================================================
    # Observability Routes (Deep Logging)
    # ==========================================================================

    @app.get("/agent/traces", tags=["Observability"])
    async def get_traces(
        limit: int = 50,
        state: AppState = Depends(get_state),
        auth: Dict[str, Any] = Depends(verify_auth),
    ) -> Dict[str, Any]:
        """
        Get recent agent decision traces.

        Returns the deep log buffer with phase/event/data for each step.
        """
        global _deep_log_buffer
        traces = _deep_log_buffer[-limit:] if limit < len(_deep_log_buffer) else _deep_log_buffer
        return {
            "total_traces": len(_deep_log_buffer),
            "returned": len(traces),
            "traces": traces,
        }

    @app.delete("/agent/traces", tags=["Observability"])
    async def clear_traces(
        state: AppState = Depends(get_state),
        auth: Dict[str, Any] = Depends(verify_admin),
    ) -> Dict[str, bool]:
        """Clear the trace buffer (admin only)."""
        global _deep_log_buffer
        _deep_log_buffer = []
        return {"cleared": True}

    @app.get("/agent/metrics", tags=["Observability"])
    async def get_agent_metrics(
        state: AppState = Depends(get_state),
        auth: Dict[str, Any] = Depends(verify_auth),
    ) -> Dict[str, Any]:
        """
        Get agent observability metrics.

        Returns metrics from the ObservabilityLayer if available.
        """
        if not state.agent:
            return {"error": "Agent not initialized"}

        # Get observability metrics from the Loop
        if hasattr(state.agent, 'observability') and state.agent.observability:
            return state.agent.observability.get_metrics()
        else:
            return {"message": "Observability layer not enabled"}

    @app.get("/agent/stats", tags=["Observability"])
    async def get_agent_stats(
        state: AppState = Depends(get_state),
        auth: Dict[str, Any] = Depends(verify_auth),
    ) -> Dict[str, Any]:
        """Get comprehensive agent statistics."""
        if not state.agent:
            return {"error": "Agent not initialized"}

        if hasattr(state.agent, 'get_stats'):
            return state.agent.get_stats()
        else:
            return {"message": "Stats not available"}

    return app


# =============================================================================
# Server Runner
# =============================================================================

class JackServer:
    """
    Jack Server wrapper for easy deployment.

    Usage:
        server = JackServer()
        server.run()

    Or for production:
        server = JackServer.production()
        server.run()
    """

    def __init__(self, config: Optional[ServerConfig] = None):
        self.config = config or ServerConfig.from_env()
        self.app = create_app(self.config)

    @classmethod
    def development(cls) -> JackServer:
        """Create development server."""
        return cls(ServerConfig.development())

    @classmethod
    def production(cls) -> JackServer:
        """Create production server."""
        return cls(ServerConfig.production())

    def run(self) -> None:
        """Run the server."""
        import uvicorn

        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level=self.config.log_level.lower(),
        )

    async def run_async(self) -> None:
        """Run server asynchronously."""
        import uvicorn

        config = uvicorn.Config(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level=self.config.log_level.lower(),
        )
        server = uvicorn.Server(config)
        await server.serve()


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys

    if "--dev" in sys.argv:
        server = JackServer.development()
    else:
        server = JackServer()

    server.run()
