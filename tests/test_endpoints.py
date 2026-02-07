"""
Integration tests for endpoint backwards compatibility.
Verifies all existing endpoints still work after lifespan migration and v2 addition.
"""
import re
import pytest
from fastapi.testclient import TestClient
from httpx import Response


@pytest.fixture(autouse=True)
def non_mocked_hosts():
    """Allow all hosts to be mocked - required for httpx_mock."""
    return []


@pytest.fixture
def client(httpx_mock, monkeypatch):
    """
    Create TestClient with mocked external dependencies.

    The lifespan runs on TestClient creation and makes Supabase calls
    (resume_stuck_jobs, check_incomplete_embeddings). Mock these.
    """
    # Allow responses to be used multiple times
    httpx_mock.can_send_already_matched_responses = True

    # Mock stuck jobs check (returns empty list)
    httpx_mock.add_response(
        method="GET",
        url=re.compile(r".*/rest/v1/processing_jobs.*"),
        json=[],
    )

    # Mock incomplete embeddings check (returns empty list)
    httpx_mock.add_response(
        method="GET",
        url=re.compile(r".*/rest/v1/user_profiles.*"),
        json=[],
    )

    # Mock conversation_chunks queries (used by some endpoints)
    httpx_mock.add_response(
        method="GET",
        url=re.compile(r".*/rest/v1/conversation_chunks.*"),
        json=[],
    )

    # Catch-all for any other Supabase GET requests
    httpx_mock.add_response(
        method="GET",
        url=re.compile(r".*/rest/v1/.*"),
        json=[],
    )

    # Catch-all for any Supabase POST requests (job creation, etc)
    httpx_mock.add_response(
        method="POST",
        url=re.compile(r".*/rest/v1/.*"),
        json=[{"id": "test-job-123"}],
        status_code=201,
    )

    # Import main.app after mocks are set up (triggers lifespan on first request)
    from main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_all_endpoints_registered():
    """
    Verify all 16 endpoints are registered (15 existing + 1 new v2).
    Uses route inspection to avoid TestClient lifespan complexity.
    """
    from main import app
    paths = {r.path for r in app.routes if hasattr(r, 'path')}

    expected = {
        "/health",
        "/health-deep",
        "/status",
        "/chat",
        "/query",
        "/analyze",
        "/create-soulprint",
        "/process-full",
        "/process-full-v2",  # NEW in Phase 3
        "/process-import",
        "/embed-chunks",
        "/test-embed",
        "/test-patch",
        "/generate-soulprint/{user_id}",
        "/embedding-status/{user_id}",
        "/complete-embeddings/{user_id}",
    }

    for ep in expected:
        assert ep in paths, f"Endpoint {ep} not registered"


def test_health_returns_processors_available(client):
    """GET /health returns 200 with processors_available=true."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "soulprint-rlm"
    assert data["processors_available"] is True
    assert "timestamp" in data


def test_health_deep_endpoint_exists(client):
    """
    GET /health-deep exists and doesn't return 404.
    May return unhealthy status due to mocked dependencies, but proves endpoint exists.
    """
    response = client.get("/health-deep")
    # Accept any non-404 status (200, 500, 503 all prove endpoint exists)
    assert response.status_code != 404


def test_status_endpoint(client):
    """GET /status returns 200."""
    response = client.get("/status")
    assert response.status_code == 200

    data = response.json()
    assert "rlm_available" in data
    assert "timestamp" in data


def test_process_full_v1_still_works(client):
    """POST /process-full with storage_path returns 200 (v1 pipeline)."""
    response = client.post(
        "/process-full",
        json={"user_id": "test-user", "storage_path": "user-exports/test/conv.json"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "processing"
    assert data["user_id"] == "test-user"
    # v1 doesn't have version field
    assert "version" not in data


def test_process_full_v2_works(client):
    """POST /process-full-v2 with storage_path returns 200 with version=v2."""
    response = client.post(
        "/process-full-v2",
        json={"user_id": "test-user", "storage_path": "user-exports/test/conv.json"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "processing"
    assert data["version"] == "v2"  # NEW: v2 endpoint includes version
    assert data["user_id"] == "test-user"
    assert data["job_id"] == "test-job-123"  # From catch-all mock
    assert "v2 pipeline started" in data["message"]


def test_process_full_v2_requires_storage_path(client):
    """POST /process-full-v2 without storage_path returns 400."""
    response = client.post(
        "/process-full-v2",
        json={"user_id": "test-user"}  # Missing storage_path
    )

    assert response.status_code == 400
    data = response.json()
    assert "storage_path required" in data["detail"]


def test_chat_endpoint_exists(client):
    """POST /chat exists (may fail validation but NOT 404)."""
    response = client.post("/chat", json={})
    # Validation error (422) or bad request (400) proves endpoint exists
    assert response.status_code in (400, 422)
    assert response.status_code != 404


def test_query_endpoint_exists(client):
    """POST /query exists (may fail validation but NOT 404)."""
    response = client.post("/query", json={})
    # Validation error proves endpoint exists
    assert response.status_code in (400, 422)
    assert response.status_code != 404


@pytest.mark.parametrize("endpoint", [
    "/health",
    "/health-deep",
    "/status",
    "/test-embed",
    "/test-patch",
])
def test_get_endpoints_not_404(client, endpoint):
    """Verify all GET endpoints are registered (not 404)."""
    response = client.get(endpoint)
    assert response.status_code != 404, f"Endpoint {endpoint} returned 404 (not registered)"


@pytest.mark.parametrize("endpoint,payload", [
    ("/chat", {"user_id": "test", "message": "hi"}),
    ("/query", {"user_id": "test", "question": "hi"}),
    ("/analyze", {"user_id": "test"}),
    ("/create-soulprint", {"user_id": "test"}),
    ("/process-import", {"user_id": "test"}),
    ("/embed-chunks", {"user_id": "test"}),
])
def test_post_endpoints_not_404(client, endpoint, payload):
    """
    Verify all POST endpoints are registered (not 404).
    They may return validation errors (400/422) or server errors (500),
    but NOT 404 (which would indicate missing endpoint).
    """
    response = client.post(endpoint, json=payload)
    assert response.status_code != 404, f"Endpoint {endpoint} returned 404 (not registered)"


def test_generate_soulprint_endpoint_exists(client):
    """GET /generate-soulprint/{user_id} exists (not 404)."""
    response = client.get("/generate-soulprint/test-user-123")
    # Any non-404 status proves endpoint exists
    assert response.status_code != 404


def test_embedding_status_endpoint_exists(client):
    """GET /embedding-status/{user_id} exists (not 404)."""
    response = client.get("/embedding-status/test-user-123")
    assert response.status_code != 404


def test_complete_embeddings_endpoint_exists(client):
    """POST /complete-embeddings/{user_id} exists (not 404)."""
    response = client.post("/complete-embeddings/test-user-123")
    assert response.status_code != 404
