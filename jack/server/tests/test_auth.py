"""
Authentication Tests.

Tests for JWT and API key authentication.
"""

import pytest
from datetime import datetime, timedelta

from jack.server.auth import (
    JWTAuth, APIKeyAuth, AuthManager, RateLimiter,
    InvalidTokenError, TokenExpiredError, InvalidAPIKeyError,
    RateLimitExceededError,
)


class TestJWTAuth:
    """Tests for JWT authentication."""

    def setup_method(self):
        """Set up test fixtures."""
        self.jwt = JWTAuth(secret="test-secret-key-12345", expiry_hours=1)

    def test_create_token(self):
        """Test token creation."""
        token = self.jwt.create_token(user_id="user123")
        assert token is not None
        assert len(token.split(".")) == 3  # JWT has 3 parts

    def test_verify_token(self):
        """Test token verification."""
        token = self.jwt.create_token(user_id="user123", scopes=("read", "write"))
        payload = self.jwt.verify_token(token)

        assert payload.sub == "user123"
        assert "read" in payload.scopes
        assert "write" in payload.scopes

    def test_token_with_extra_data(self):
        """Test token with extra payload data."""
        token = self.jwt.create_token(
            user_id="user123",
            extra={"role": "admin", "name": "Test User"}
        )
        payload = self.jwt.verify_token(token)

        assert payload.sub == "user123"
        assert payload.extra["role"] == "admin"
        assert payload.extra["name"] == "Test User"

    def test_invalid_token(self):
        """Test that invalid tokens are rejected."""
        with pytest.raises(InvalidTokenError):
            self.jwt.verify_token("invalid.token.here")

    def test_tampered_token(self):
        """Test that tampered tokens are rejected."""
        token = self.jwt.create_token(user_id="user123")
        # Tamper with the payload
        parts = token.split(".")
        parts[1] = "tampered"
        tampered_token = ".".join(parts)

        with pytest.raises(InvalidTokenError):
            self.jwt.verify_token(tampered_token)

    def test_expired_token(self):
        """Test that expired tokens are rejected."""
        jwt = JWTAuth(secret="test-secret", expiry_hours=-1)  # Already expired
        token = jwt.create_token(user_id="user123")

        with pytest.raises(TokenExpiredError):
            jwt.verify_token(token)

    def test_refresh_token(self):
        """Test token refresh."""
        original = self.jwt.create_token(user_id="user123", scopes=("admin",))
        refreshed = self.jwt.refresh_token(original)

        assert refreshed != original
        payload = self.jwt.verify_token(refreshed)
        assert payload.sub == "user123"
        assert "admin" in payload.scopes


class TestAPIKeyAuth:
    """Tests for API key authentication."""

    def setup_method(self):
        """Set up test fixtures."""
        self.api_keys = APIKeyAuth(prefix="test_")

    def test_create_key(self):
        """Test API key creation."""
        key_id, key = self.api_keys.create_key(name="Test Key")

        assert key_id is not None
        assert key.startswith("test_")
        assert len(key) > 40

    def test_verify_key(self):
        """Test API key verification."""
        key_id, key = self.api_keys.create_key(
            name="Test Key",
            scopes=("read", "write")
        )
        api_key = self.api_keys.verify_key(key)

        assert api_key.key_id == key_id
        assert api_key.name == "Test Key"
        assert "read" in api_key.scopes

    def test_invalid_key_format(self):
        """Test that invalid key formats are rejected."""
        with pytest.raises(InvalidAPIKeyError):
            self.api_keys.verify_key("invalid-key")

    def test_wrong_prefix(self):
        """Test that keys with wrong prefix are rejected."""
        with pytest.raises(InvalidAPIKeyError):
            self.api_keys.verify_key("wrong_prefix_key")

    def test_unknown_key(self):
        """Test that unknown keys are rejected."""
        with pytest.raises(InvalidAPIKeyError):
            # Valid format but not registered
            self.api_keys.verify_key("test_12345678_" + "a" * 43)

    def test_revoke_key(self):
        """Test API key revocation."""
        key_id, key = self.api_keys.create_key(name="Test Key")

        # Key should work initially
        self.api_keys.verify_key(key)

        # Revoke it
        assert self.api_keys.revoke_key(key_id) is True

        # Key should no longer work
        with pytest.raises(InvalidAPIKeyError):
            self.api_keys.verify_key(key)

    def test_list_keys(self):
        """Test listing API keys."""
        self.api_keys.create_key(name="Key 1")
        self.api_keys.create_key(name="Key 2")

        keys = self.api_keys.list_keys()
        assert len(keys) >= 2
        names = [k["name"] for k in keys]
        assert "Key 1" in names
        assert "Key 2" in names

    def test_key_expiration(self):
        """Test that expired keys are rejected."""
        key_id, key = self.api_keys.create_key(
            name="Expiring Key",
            expires_days=-1  # Already expired
        )

        with pytest.raises(InvalidAPIKeyError):
            self.api_keys.verify_key(key)


class TestRateLimiter:
    """Tests for rate limiting."""

    def test_within_limit(self):
        """Test requests within limit."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)

        for _ in range(5):
            assert limiter.check("user1") is True

    def test_exceeds_limit(self):
        """Test requests exceeding limit."""
        limiter = RateLimiter(max_requests=3, window_seconds=60)

        assert limiter.check("user1") is True
        assert limiter.check("user1") is True
        assert limiter.check("user1") is True
        assert limiter.check("user1") is False

    def test_different_users(self):
        """Test rate limiting is per-user."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)

        assert limiter.check("user1") is True
        assert limiter.check("user1") is True
        assert limiter.check("user1") is False

        # Different user should have their own limit
        assert limiter.check("user2") is True
        assert limiter.check("user2") is True

    def test_remaining_count(self):
        """Test getting remaining requests."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)

        assert limiter.get_remaining("user1") == 5
        limiter.check("user1")
        assert limiter.get_remaining("user1") == 4
        limiter.check("user1")
        assert limiter.get_remaining("user1") == 3


class TestAuthManager:
    """Tests for combined AuthManager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.auth = AuthManager(
            jwt_secret="test-secret",
            jwt_expiry_hours=1,
            api_key_prefix="jack_",
            rate_limit_requests=10,
            rate_limit_window=60,
        )

    def test_jwt_flow(self):
        """Test complete JWT flow."""
        token = self.auth.create_token(user_id="user123", scopes=("admin",))
        payload = self.auth.verify_token(token)

        assert payload.sub == "user123"
        assert "admin" in payload.scopes

    def test_api_key_flow(self):
        """Test complete API key flow."""
        key_id, key = self.auth.create_api_key(name="Test Key", scopes=("read",))
        api_key = self.auth.verify_api_key(key)

        assert api_key.key_id == key_id
        assert "read" in api_key.scopes

    def test_rate_limit_check(self):
        """Test rate limit checking."""
        for _ in range(10):
            self.auth.check_rate_limit("user1")

        with pytest.raises(RateLimitExceededError):
            self.auth.check_rate_limit("user1")

    def test_rate_limit_headers(self):
        """Test rate limit header generation."""
        headers = self.auth.get_rate_limit_headers("user1")

        assert "X-RateLimit-Limit" in headers
        assert "X-RateLimit-Remaining" in headers
        assert "X-RateLimit-Reset" in headers
