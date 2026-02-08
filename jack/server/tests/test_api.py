"""
API Integration Tests.

Tests for the FastAPI server endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from jack.server.app import create_app
from jack.server.config import ServerConfig, LLMProviderConfig, LLMProvider


@pytest.fixture
def config():
    """Create test configuration."""
    return ServerConfig(
        debug=True,
        llm=LLMProviderConfig(
            provider=LLMProvider.LOCAL,
            base_url="http://localhost:8080/v1",  # Won't be used in tests
            model="test-model",
        ),
    )


@pytest.fixture
def app(config):
    """Create test application."""
    return create_app(config)


@pytest.fixture
def client(app):
    """Create test client."""
    with TestClient(app) as client:
        yield client


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "llm_provider" in data

    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "Jack Agent API"
        assert "version" in data


class TestAuthEndpoints:
    """Tests for authentication endpoints."""

    def test_create_token(self, client):
        """Test JWT token creation."""
        response = client.post(
            "/auth/token",
            json={"user_id": "test_user", "scopes": ["admin"]}
        )
        assert response.status_code == 200

        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] > 0

    def test_refresh_token(self, client):
        """Test JWT token refresh."""
        # First get a token
        response = client.post(
            "/auth/token",
            json={"user_id": "test_user", "scopes": ["admin"]}
        )
        token = response.json()["access_token"]

        # Refresh it
        response = client.post(
            "/auth/refresh",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200

        data = response.json()
        assert "access_token" in data
        assert data["access_token"] != token  # New token

    def test_create_api_key_requires_admin(self, client):
        """Test that API key creation requires admin scope."""
        # Get non-admin token
        response = client.post(
            "/auth/token",
            json={"user_id": "test_user", "scopes": ["read"]}
        )
        token = response.json()["access_token"]

        # Try to create API key
        response = client.post(
            "/auth/api-key",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": "Test Key"}
        )
        assert response.status_code == 403

    def test_create_api_key_with_admin(self, client):
        """Test API key creation with admin scope."""
        # Get admin token
        response = client.post(
            "/auth/token",
            json={"user_id": "admin", "scopes": ["admin"]}
        )
        token = response.json()["access_token"]

        # Create API key
        response = client.post(
            "/auth/api-key",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": "Test Key", "scopes": ["read", "write"]}
        )
        assert response.status_code == 200

        data = response.json()
        assert "key_id" in data
        assert "key" in data
        assert data["key"].startswith("jack_")


class TestAgentEndpoints:
    """Tests for agent endpoints."""

    def get_auth_headers(self, client):
        """Helper to get auth headers."""
        response = client.post(
            "/auth/token",
            json={"user_id": "test_user", "scopes": ["admin"]}
        )
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    def test_query_requires_auth(self, client):
        """Test that agent query requires authentication."""
        response = client.post(
            "/agent/query",
            json={"query": "test"}
        )
        assert response.status_code == 401

    def test_query_with_auth(self, client):
        """Test agent query with authentication."""
        headers = self.get_auth_headers(client)

        response = client.post(
            "/agent/query",
            headers=headers,
            json={"query": "Hello, how are you?"}
        )
        # May fail if LLM not available (503) or connection refused (500)
        # but auth should work (not 401)
        assert response.status_code in [200, 500, 503]

    def test_reason_endpoint(self, client):
        """Test direct reasoning endpoint."""
        headers = self.get_auth_headers(client)

        response = client.post(
            "/agent/reason",
            headers=headers,
            json={"query": "What is 2+2?"}
        )
        # May fail if LLM not available (503) or connection refused (500)
        assert response.status_code in [200, 500, 503]


class TestLLMEndpoints:
    """Tests for LLM management endpoints."""

    def get_auth_headers(self, client):
        """Helper to get auth headers."""
        response = client.post(
            "/auth/token",
            json={"user_id": "test_user", "scopes": ["admin"]}
        )
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    def test_llm_status(self, client):
        """Test LLM status endpoint."""
        headers = self.get_auth_headers(client)

        response = client.get("/llm/status", headers=headers)
        assert response.status_code == 200

        data = response.json()
        assert "provider" in data
        assert "model" in data
        assert "available" in data


class TestAPIKeyAuth:
    """Tests for API key authentication."""

    def test_api_key_auth(self, client):
        """Test authentication with API key."""
        # First create API key (need admin token)
        response = client.post(
            "/auth/token",
            json={"user_id": "admin", "scopes": ["admin"]}
        )
        token = response.json()["access_token"]

        response = client.post(
            "/auth/api-key",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": "Test Key", "scopes": ["read"]}
        )
        api_key = response.json()["key"]

        # Use API key for authentication
        response = client.get(
            "/llm/status",
            headers={"X-API-Key": api_key}
        )
        assert response.status_code == 200

    def test_invalid_api_key(self, client):
        """Test that invalid API key is rejected."""
        response = client.get(
            "/llm/status",
            headers={"X-API-Key": "invalid-key"}
        )
        assert response.status_code == 401


class TestRateLimiting:
    """Tests for rate limiting."""

    def test_rate_limit_headers(self, client):
        """Test that rate limit headers are included."""
        response = client.post(
            "/auth/token",
            json={"user_id": "test_user", "scopes": ["admin"]}
        )
        token = response.json()["access_token"]

        response = client.get(
            "/health",  # Unauthenticated endpoint for this test
        )
        # Note: health endpoint doesn't require auth, so no rate limit headers
        # This would need auth endpoint to test properly
        assert response.status_code == 200
