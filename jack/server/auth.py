"""
Authentication Layer - JWT and API Key support.

Provides secure authentication for the Jack Server API.
Supports both stateless JWT tokens and persistent API keys.
"""

from __future__ import annotations
import os
import secrets
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from functools import wraps
import base64
import json
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================

class AuthError(Exception):
    """Base authentication error."""
    pass


class TokenExpiredError(AuthError):
    """JWT token has expired."""
    pass


class InvalidTokenError(AuthError):
    """JWT token is invalid."""
    pass


class InvalidAPIKeyError(AuthError):
    """API key is invalid."""
    pass


class RateLimitExceededError(AuthError):
    """Rate limit has been exceeded."""
    pass


# =============================================================================
# JWT Authentication
# =============================================================================

@dataclass
class JWTPayload:
    """JWT token payload."""
    sub: str  # Subject (user ID)
    exp: float  # Expiration timestamp
    iat: float  # Issued at timestamp
    scopes: tuple = ()
    extra: Dict[str, Any] = field(default_factory=dict)


class JWTAuth:
    """
    Simple JWT authentication without external dependencies.

    Uses HS256 (HMAC-SHA256) for signing.
    """

    def __init__(
        self,
        secret: str,
        algorithm: str = "HS256",
        expiry_hours: int = 24,
    ):
        self.secret = secret.encode() if isinstance(secret, str) else secret
        self.algorithm = algorithm
        self.expiry_hours = expiry_hours

    def _base64url_encode(self, data: bytes) -> str:
        """Base64 URL-safe encoding without padding."""
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    def _base64url_decode(self, data: str) -> bytes:
        """Base64 URL-safe decoding with padding restoration."""
        padding = 4 - len(data) % 4
        if padding != 4:
            data += "=" * padding
        return base64.urlsafe_b64decode(data)

    def _sign(self, message: str) -> str:
        """Create HMAC-SHA256 signature."""
        signature = hmac.new(
            self.secret,
            message.encode(),
            hashlib.sha256
        ).digest()
        return self._base64url_encode(signature)

    def create_token(
        self,
        user_id: str,
        scopes: tuple = (),
        extra: Optional[Dict[str, Any]] = None,
        expiry_hours: Optional[int] = None,
    ) -> str:
        """
        Create a JWT token.

        Args:
            user_id: Unique user identifier
            scopes: Permission scopes
            extra: Additional payload data
            expiry_hours: Override default expiry

        Returns:
            JWT token string
        """
        now = datetime.utcnow()
        exp = now + timedelta(hours=expiry_hours or self.expiry_hours)

        # Create header
        header = {"alg": self.algorithm, "typ": "JWT"}
        header_b64 = self._base64url_encode(json.dumps(header).encode())

        # Create payload
        payload = {
            "sub": user_id,
            "iat": now.timestamp(),
            "exp": exp.timestamp(),
            "scopes": list(scopes),
        }
        if extra:
            payload.update(extra)
        payload_b64 = self._base64url_encode(json.dumps(payload).encode())

        # Sign and return
        message = f"{header_b64}.{payload_b64}"
        signature = self._sign(message)

        return f"{message}.{signature}"

    def verify_token(self, token: str) -> JWTPayload:
        """
        Verify and decode a JWT token.

        Args:
            token: JWT token string

        Returns:
            Decoded payload

        Raises:
            InvalidTokenError: Token is malformed or signature invalid
            TokenExpiredError: Token has expired
        """
        try:
            parts = token.split(".")
            if len(parts) != 3:
                raise InvalidTokenError("Invalid token format")

            header_b64, payload_b64, signature = parts

            # Verify signature
            message = f"{header_b64}.{payload_b64}"
            expected_sig = self._sign(message)

            if not hmac.compare_digest(signature, expected_sig):
                raise InvalidTokenError("Invalid signature")

            # Decode payload
            payload_json = self._base64url_decode(payload_b64)
            payload = json.loads(payload_json)

            # Check expiration
            if payload.get("exp", 0) < datetime.utcnow().timestamp():
                raise TokenExpiredError("Token has expired")

            # Extract known fields
            return JWTPayload(
                sub=payload["sub"],
                exp=payload["exp"],
                iat=payload["iat"],
                scopes=tuple(payload.get("scopes", [])),
                extra={k: v for k, v in payload.items()
                       if k not in ("sub", "exp", "iat", "scopes")},
            )

        except (json.JSONDecodeError, KeyError) as e:
            raise InvalidTokenError(f"Malformed token: {e}")

    def refresh_token(self, token: str) -> str:
        """
        Refresh a token (issue new one with extended expiry).

        Args:
            token: Current valid token

        Returns:
            New token with fresh expiry
        """
        payload = self.verify_token(token)
        return self.create_token(
            user_id=payload.sub,
            scopes=payload.scopes,
            extra=payload.extra,
        )


# =============================================================================
# API Key Authentication
# =============================================================================

@dataclass
class APIKey:
    """API key record."""
    key_id: str
    key_hash: str
    name: str
    scopes: tuple
    created_at: datetime
    last_used: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    rate_limit: Optional[int] = None


class APIKeyAuth:
    """
    API key authentication manager.

    Uses secure hashing for key storage.
    """

    def __init__(
        self,
        prefix: str = "jack_",
        hash_rounds: int = 12,
    ):
        self.prefix = prefix
        self.hash_rounds = hash_rounds
        self._keys: Dict[str, APIKey] = {}  # key_id -> APIKey

    def _generate_key(self) -> tuple:
        """Generate a new API key and its ID."""
        # Generate 32 bytes of random data
        key_bytes = secrets.token_bytes(32)
        key_id = secrets.token_hex(8)

        # Create key string with prefix
        key_b64 = base64.urlsafe_b64encode(key_bytes).decode("ascii").rstrip("=")
        full_key = f"{self.prefix}{key_id}_{key_b64}"

        return key_id, full_key

    def _hash_key(self, key: str) -> str:
        """Hash an API key for storage."""
        # Use SHA-256 with secret salt
        salt = secrets.token_bytes(16)
        key_hash = hashlib.pbkdf2_hmac(
            "sha256",
            key.encode(),
            salt,
            iterations=100000,
        )
        # Store salt with hash
        return base64.b64encode(salt + key_hash).decode("ascii")

    def _verify_hash(self, key: str, stored_hash: str) -> bool:
        """Verify a key against its stored hash."""
        try:
            decoded = base64.b64decode(stored_hash)
            salt = decoded[:16]
            expected_hash = decoded[16:]

            actual_hash = hashlib.pbkdf2_hmac(
                "sha256",
                key.encode(),
                salt,
                iterations=100000,
            )

            return hmac.compare_digest(actual_hash, expected_hash)
        except Exception:
            return False

    def create_key(
        self,
        name: str,
        scopes: tuple = (),
        expires_days: Optional[int] = None,
        rate_limit: Optional[int] = None,
    ) -> tuple:
        """
        Create a new API key.

        Args:
            name: Human-readable name for the key
            scopes: Permission scopes
            expires_days: Days until expiration (None = never)
            rate_limit: Requests per minute (None = default)

        Returns:
            Tuple of (key_id, full_key)

        Note: The full_key is only returned once and cannot be recovered!
        """
        key_id, full_key = self._generate_key()
        key_hash = self._hash_key(full_key)

        expires_at = None
        if expires_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_days)

        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            scopes=scopes,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            rate_limit=rate_limit,
        )

        self._keys[key_id] = api_key
        logger.info(f"Created API key '{name}' with ID {key_id}")

        return key_id, full_key

    def verify_key(self, key: str) -> APIKey:
        """
        Verify an API key.

        Args:
            key: Full API key string

        Returns:
            APIKey record

        Raises:
            InvalidAPIKeyError: Key is invalid or expired
        """
        if not key.startswith(self.prefix):
            raise InvalidAPIKeyError("Invalid key format")

        try:
            # Extract key ID
            key_part = key[len(self.prefix):]
            key_id = key_part.split("_")[0]
        except (IndexError, ValueError):
            raise InvalidAPIKeyError("Invalid key format")

        if key_id not in self._keys:
            raise InvalidAPIKeyError("Unknown key")

        api_key = self._keys[key_id]

        # Check expiration
        if api_key.expires_at and api_key.expires_at < datetime.utcnow():
            raise InvalidAPIKeyError("Key has expired")

        # Verify hash
        if not self._verify_hash(key, api_key.key_hash):
            raise InvalidAPIKeyError("Invalid key")

        # Update last used
        api_key.last_used = datetime.utcnow()

        return api_key

    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        if key_id in self._keys:
            del self._keys[key_id]
            logger.info(f"Revoked API key {key_id}")
            return True
        return False

    def list_keys(self) -> list:
        """List all API keys (without hashes)."""
        return [
            {
                "key_id": k.key_id,
                "name": k.name,
                "scopes": k.scopes,
                "created_at": k.created_at.isoformat(),
                "last_used": k.last_used.isoformat() if k.last_used else None,
                "expires_at": k.expires_at.isoformat() if k.expires_at else None,
            }
            for k in self._keys.values()
        ]


# =============================================================================
# Rate Limiter
# =============================================================================

class RateLimiter:
    """
    Simple in-memory rate limiter using sliding window.

    For production, consider using Redis for distributed rate limiting.
    """

    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, list] = {}  # key -> list of timestamps

    def _clean_old_requests(self, key: str, now: float) -> None:
        """Remove requests outside the current window."""
        if key in self._requests:
            cutoff = now - self.window_seconds
            self._requests[key] = [
                ts for ts in self._requests[key] if ts > cutoff
            ]

    def check(self, key: str, limit: Optional[int] = None) -> bool:
        """
        Check if request is allowed under rate limit.

        Args:
            key: Identifier (user ID, API key ID, IP, etc.)
            limit: Override default limit

        Returns:
            True if allowed, False if rate limited
        """
        now = datetime.utcnow().timestamp()
        max_req = limit or self.max_requests

        self._clean_old_requests(key, now)

        if key not in self._requests:
            self._requests[key] = []

        if len(self._requests[key]) >= max_req:
            return False

        self._requests[key].append(now)
        return True

    def get_remaining(self, key: str, limit: Optional[int] = None) -> int:
        """Get remaining requests in current window."""
        now = datetime.utcnow().timestamp()
        max_req = limit or self.max_requests

        self._clean_old_requests(key, now)

        current = len(self._requests.get(key, []))
        return max(0, max_req - current)

    def get_reset_time(self, key: str) -> Optional[float]:
        """Get seconds until rate limit resets."""
        if key not in self._requests or not self._requests[key]:
            return None

        oldest = min(self._requests[key])
        reset_at = oldest + self.window_seconds
        now = datetime.utcnow().timestamp()

        return max(0, reset_at - now)


# =============================================================================
# Combined Auth Manager
# =============================================================================

class AuthManager:
    """
    Combined authentication manager supporting both JWT and API keys.

    Usage:
        auth = AuthManager(config)

        # JWT flow
        token = auth.create_token(user_id="user123")
        payload = auth.verify_token(token)

        # API key flow
        key_id, key = auth.create_api_key(name="My App")
        api_key = auth.verify_api_key(key)
    """

    def __init__(
        self,
        jwt_secret: str,
        jwt_expiry_hours: int = 24,
        api_key_prefix: str = "jack_",
        rate_limit_requests: int = 100,
        rate_limit_window: int = 60,
    ):
        self.jwt = JWTAuth(
            secret=jwt_secret,
            expiry_hours=jwt_expiry_hours,
        )
        self.api_keys = APIKeyAuth(prefix=api_key_prefix)
        self.rate_limiter = RateLimiter(
            max_requests=rate_limit_requests,
            window_seconds=rate_limit_window,
        )

    # JWT methods
    def create_token(self, user_id: str, scopes: tuple = (), **extra) -> str:
        """Create a JWT token."""
        return self.jwt.create_token(user_id, scopes, extra)

    def verify_token(self, token: str) -> JWTPayload:
        """Verify a JWT token."""
        return self.jwt.verify_token(token)

    def refresh_token(self, token: str) -> str:
        """Refresh a JWT token."""
        return self.jwt.refresh_token(token)

    # API key methods
    def create_api_key(
        self,
        name: str,
        scopes: tuple = (),
        expires_days: Optional[int] = None,
    ) -> tuple:
        """Create an API key."""
        return self.api_keys.create_key(name, scopes, expires_days)

    def verify_api_key(self, key: str) -> APIKey:
        """Verify an API key."""
        return self.api_keys.verify_key(key)

    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        return self.api_keys.revoke_key(key_id)

    # Rate limiting
    def check_rate_limit(
        self,
        identifier: str,
        custom_limit: Optional[int] = None,
    ) -> bool:
        """Check if request is within rate limit."""
        if not self.rate_limiter.check(identifier, custom_limit):
            raise RateLimitExceededError(
                f"Rate limit exceeded. Try again in "
                f"{self.rate_limiter.get_reset_time(identifier):.0f} seconds."
            )
        return True

    def get_rate_limit_headers(self, identifier: str) -> Dict[str, str]:
        """Get rate limit headers for response."""
        return {
            "X-RateLimit-Limit": str(self.rate_limiter.max_requests),
            "X-RateLimit-Remaining": str(self.rate_limiter.get_remaining(identifier)),
            "X-RateLimit-Reset": str(int(
                self.rate_limiter.get_reset_time(identifier) or 0
            )),
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def create_api_key(
    name: str,
    scopes: tuple = (),
    secret: Optional[str] = None,
) -> tuple:
    """
    Quick function to create an API key.

    Args:
        name: Key name
        scopes: Permission scopes
        secret: JWT secret (for AuthManager)

    Returns:
        Tuple of (key_id, full_key)
    """
    auth = AuthManager(jwt_secret=secret or os.getenv("JWT_SECRET", "dev-secret"))
    return auth.create_api_key(name, scopes)


def verify_api_key(key: str, prefix: str = "jack_") -> APIKey:
    """
    Quick function to verify an API key format.

    Note: This only validates format, not against a stored key database.
    """
    if not key.startswith(prefix):
        raise InvalidAPIKeyError("Invalid key format")

    parts = key[len(prefix):].split("_")
    if len(parts) != 2:
        raise InvalidAPIKeyError("Invalid key format")

    key_id, key_value = parts
    if len(key_id) != 16 or len(key_value) < 32:
        raise InvalidAPIKeyError("Invalid key format")

    return APIKey(
        key_id=key_id,
        key_hash="",  # Not verified
        name="",
        scopes=(),
        created_at=datetime.utcnow(),
    )
