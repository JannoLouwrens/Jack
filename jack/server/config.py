"""
Server Configuration - Swappable LLM and secure defaults.

Supports environment variables for all sensitive configuration.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any
from enum import Enum


class LLMProvider(Enum):
    """Supported LLM providers."""
    LOCAL = "local"          # llama.cpp via OpenAI-compatible API
    OPENAI = "openai"        # OpenAI API
    ANTHROPIC = "anthropic"  # Anthropic Claude API
    CUSTOM = "custom"        # Custom OpenAI-compatible endpoint


@dataclass
class LLMProviderConfig:
    """Configuration for a single LLM provider."""

    provider: LLMProvider
    base_url: str = "http://localhost:8080/v1"
    api_key: Optional[str] = None
    model: str = "deepseek-r1-14b"
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: float = 120.0

    # Provider-specific options
    extra_params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def local_llama(
        cls,
        base_url: str = "http://localhost:8080/v1",
        model: str = "deepseek-r1-14b"
    ) -> LLMProviderConfig:
        """Create config for local llama.cpp server."""
        return cls(
            provider=LLMProvider.LOCAL,
            base_url=base_url,
            model=model,
            api_key="not-needed",  # llama.cpp doesn't require auth
        )

    @classmethod
    def openai(
        cls,
        api_key: Optional[str] = None,
        model: str = "gpt-4o"
    ) -> LLMProviderConfig:
        """Create config for OpenAI API."""
        return cls(
            provider=LLMProvider.OPENAI,
            base_url="https://api.openai.com/v1",
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            model=model,
        )

    @classmethod
    def anthropic(
        cls,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514"
    ) -> LLMProviderConfig:
        """Create config for Anthropic Claude API."""
        return cls(
            provider=LLMProvider.ANTHROPIC,
            base_url="https://api.anthropic.com",
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            model=model,
        )

    @classmethod
    def from_env(cls) -> LLMProviderConfig:
        """Create config from environment variables."""
        provider_name = os.getenv("LLM_PROVIDER", "local").lower()

        if provider_name == "local":
            return cls.local_llama(
                base_url=os.getenv("LLM_BASE_URL", "http://localhost:8080/v1"),
                model=os.getenv("LLM_MODEL", "deepseek-r1-14b"),
            )
        elif provider_name == "openai":
            return cls.openai(model=os.getenv("LLM_MODEL", "gpt-4o"))
        elif provider_name == "anthropic":
            return cls.anthropic(model=os.getenv("LLM_MODEL", "claude-sonnet-4-20250514"))
        else:
            return cls(
                provider=LLMProvider.CUSTOM,
                base_url=os.getenv("LLM_BASE_URL", "http://localhost:8080/v1"),
                api_key=os.getenv("LLM_API_KEY"),
                model=os.getenv("LLM_MODEL", "custom-model"),
            )


@dataclass
class DatabaseConfig:
    """Database connection configuration."""

    driver: str = "ODBC Driver 17 for SQL Server"
    server: str = "localhost"
    database: str = "JackAgent"
    username: Optional[str] = None
    password: Optional[str] = None
    trusted_connection: bool = True

    @property
    def connection_string(self) -> str:
        """Generate MSSQL connection string."""
        if self.trusted_connection:
            return (
                f"DRIVER={{{self.driver}}};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                f"Trusted_Connection=yes;"
            )
        return (
            f"DRIVER={{{self.driver}}};"
            f"SERVER={self.server};"
            f"DATABASE={self.database};"
            f"UID={self.username};"
            f"PWD={self.password};"
        )

    @classmethod
    def from_env(cls) -> DatabaseConfig:
        """Create config from environment variables."""
        return cls(
            driver=os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server"),
            server=os.getenv("DB_SERVER", "localhost"),
            database=os.getenv("DB_DATABASE", "JackAgent"),
            username=os.getenv("DB_USERNAME"),
            password=os.getenv("DB_PASSWORD"),
            trusted_connection=os.getenv("DB_TRUSTED", "true").lower() == "true",
        )


@dataclass
class AuthConfig:
    """Authentication configuration."""

    # JWT settings
    jwt_secret: str = field(default_factory=lambda: os.getenv("JWT_SECRET", "change-me-in-production"))
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24

    # API key settings
    api_key_prefix: str = "jack_"
    api_key_hash_rounds: int = 12

    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60

    @classmethod
    def from_env(cls) -> AuthConfig:
        """Create config from environment variables."""
        return cls(
            jwt_secret=os.getenv("JWT_SECRET", "change-me-in-production"),
            jwt_algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
            jwt_expiry_hours=int(os.getenv("JWT_EXPIRY_HOURS", "24")),
            rate_limit_requests=int(os.getenv("RATE_LIMIT_REQUESTS", "100")),
            rate_limit_window_seconds=int(os.getenv("RATE_LIMIT_WINDOW", "60")),
        )


@dataclass
class SandboxConfig:
    """Sandbox execution configuration."""

    enabled: bool = True
    max_execution_time: float = 30.0
    max_memory_mb: int = 512
    allowed_modules: tuple = (
        "math", "json", "datetime", "re", "collections",
        "itertools", "functools", "typing", "dataclasses",
    )
    network_access: bool = False

    @classmethod
    def from_env(cls) -> SandboxConfig:
        """Create config from environment variables."""
        return cls(
            enabled=os.getenv("SANDBOX_ENABLED", "true").lower() == "true",
            max_execution_time=float(os.getenv("SANDBOX_TIMEOUT", "30")),
            max_memory_mb=int(os.getenv("SANDBOX_MEMORY_MB", "512")),
            network_access=os.getenv("SANDBOX_NETWORK", "false").lower() == "true",
        )


@dataclass
class ServerConfig:
    """Complete server configuration."""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    debug: bool = False

    # CORS settings
    cors_origins: tuple = ("*",)
    cors_credentials: bool = True

    # Component configs
    llm: LLMProviderConfig = field(default_factory=LLMProviderConfig.from_env)
    database: DatabaseConfig = field(default_factory=DatabaseConfig.from_env)
    auth: AuthConfig = field(default_factory=AuthConfig.from_env)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig.from_env)

    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    @classmethod
    def from_env(cls) -> ServerConfig:
        """Create complete config from environment variables."""
        return cls(
            host=os.getenv("SERVER_HOST", "0.0.0.0"),
            port=int(os.getenv("SERVER_PORT", "8000")),
            workers=int(os.getenv("SERVER_WORKERS", "1")),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )

    @classmethod
    def development(cls) -> ServerConfig:
        """Development configuration with sensible defaults."""
        return cls(
            debug=True,
            log_level="DEBUG",
            llm=LLMProviderConfig.local_llama(),
        )

    @classmethod
    def production(cls) -> ServerConfig:
        """Production configuration from environment."""
        config = cls.from_env()

        # Validate production settings
        if config.auth.jwt_secret == "change-me-in-production":
            raise ValueError("JWT_SECRET must be set in production!")

        return config
