"""
Jack Server - Production FastAPI Server for Jack Foundation Agent.

Provides a secure, swappable LLM-backed API for agent interactions.
Supports multiple authentication methods (JWT, API keys) and multiple
LLM backends (local llama.cpp, OpenAI, Anthropic).

Architecture:
    Client Request → Auth Layer → Rate Limiter → Agent Router → Jack Loop
                                                      ↓
                                               LLM Provider (swappable)
                                                      ↓
                                               Response Stream
"""

from jack.server.config import ServerConfig, LLMProviderConfig
from jack.server.auth import (
    AuthManager,
    JWTAuth,
    APIKeyAuth,
    create_api_key,
    verify_api_key,
)
from jack.server.app import create_app, JackServer

__all__ = [
    # Config
    "ServerConfig",
    "LLMProviderConfig",
    # Auth
    "AuthManager",
    "JWTAuth",
    "APIKeyAuth",
    "create_api_key",
    "verify_api_key",
    # App
    "create_app",
    "JackServer",
]
