"""
Integration tests for the FastAPI endpoints.

These test the API layer with a real FastAPI TestClient but mock
the Agent Framework agent (no actual LLM calls). The orchestrator
uses a human-in-the-loop flow for CPG creation.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# Mock the agent framework BEFORE importing the app
@pytest.fixture(autouse=True)
def mock_agent_framework():
    """Mock the orchestrator agent so no Azure calls are made."""
    mock_agent = MagicMock()
    mock_session = MagicMock()
    mock_agent.create_session.return_value = mock_session

    # Mock agent.run() to return an AgentResponse-like object
    mock_response = MagicMock()
    mock_response.text = "Here is your CPG for diabetes management."
    mock_agent.run = AsyncMock(return_value=mock_response)

    with patch("src.api.routes.create_orchestrator_agent", return_value=mock_agent):
        yield mock_agent


@pytest.fixture
def client():
    from src.main import create_app

    app = create_app()
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_health_shows_environment(self, client):
        response = client.get("/api/v1/health")
        data = response.json()
        assert "environment" in data


class TestChatEndpoint:
    def test_chat_new_session(self, client, mock_agent_framework):
        response = client.post(
            "/api/v1/chat",
            json={"message": "Create a CPG for Type 2 Diabetes"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "response" in data
        assert len(data["session_id"]) > 0

    def test_chat_with_existing_session(self, client, mock_agent_framework):
        # First request — create session
        r1 = client.post(
            "/api/v1/chat",
            json={"message": "Create a CPG for hypertension"},
        )
        session_id = r1.json()["session_id"]

        # Second request — same session
        r2 = client.post(
            "/api/v1/chat",
            json={"session_id": session_id, "message": "Change target population to elderly"},
        )
        assert r2.status_code == 200
        assert r2.json()["session_id"] == session_id

    def test_chat_empty_message_rejected(self, client):
        response = client.post(
            "/api/v1/chat",
            json={"message": ""},
        )
        assert response.status_code == 422  # validation error

    def test_chat_missing_message_rejected(self, client):
        response = client.post(
            "/api/v1/chat",
            json={},
        )
        assert response.status_code == 422


class TestCPGEndpoints:
    def test_get_cpg_not_found(self, client):
        response = client.get("/api/v1/cpg/nonexistent-id")
        assert response.status_code == 404

    def test_get_cpg_versions_empty(self, client):
        response = client.get("/api/v1/cpg/nonexistent-id/versions")
        assert response.status_code == 200
        data = response.json()
        assert data["versions"] == []


class TestIndexEndpoint:
    @patch("src.api.routes.index_documents", new_callable=AsyncMock)
    def test_index_documents(self, mock_index, client):
        mock_index.return_value = 1

        response = client.post(
            "/api/v1/index",
            json={
                "documents": [
                    {
                        "id": "doc-1",
                        "title": "Test Doc",
                        "content": "Test content",
                        "specialty": "General",
                        "publication_year": 2024,
                        "evidence_level": "High",
                        "source": "Test",
                    }
                ]
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["indexed_count"] == 1
        assert data["total_submitted"] == 1

    def test_index_empty_documents_rejected(self, client):
        response = client.post(
            "/api/v1/index",
            json={"documents": []},
        )
        assert response.status_code == 422
