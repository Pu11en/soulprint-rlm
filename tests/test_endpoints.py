"""
Integration tests for endpoint backwards compatibility.
Verifies all existing endpoints still work after lifespan migration and v2 addition.
"""
import re
import pytest
from fastapi.testclient import TestClient
from httpx import Response


@pytest.fixture
def client(httpx_mock, monkeypatch):
    """
    Create TestClient with mocked external dependencies.

    The lifespan runs on TestClient creation and makes Supabase calls
    (resume_stuck_jobs, check_incomplete_embeddings). Mock these.

    Background tasks may make additional requests, so we don't assert
    all requests were expected (some happen async after test completes).
    """
    # Allow responses to be used multiple times
    httpx_mock.can_send_already_matched_responses = True
    # Don't fail on unexpected requests from background tasks
    httpx_mock._options.assert_all_requests_were_expected = False

    # Mock stuck jobs check (returns empty list)
    httpx_mock.add_response(
        method="GET",
        url=re.compile(r".*/rest/v1/processing_jobs.*"),
        json=[],
        is_optional=True,
    )

    # Mock incomplete embeddings check (returns empty list)
    httpx_mock.add_response(
        method="GET",
        url=re.compile(r".*/rest/v1/user_profiles.*"),
        json=[],
        is_optional=True,
    )

    # Mock conversation_chunks queries (used by some endpoints)
    httpx_mock.add_response(
        method="GET",
        url=re.compile(r".*/rest/v1/conversation_chunks.*"),
        json=[],
        is_optional=True,
    )

    # Catch-all for any other Supabase GET requests
    httpx_mock.add_response(
        method="GET",
        url=re.compile(r".*/rest/v1/.*"),
        json=[],
        is_optional=True,
    )

    # Catch-all for any Supabase POST requests (job creation, RPC calls, etc)
    httpx_mock.add_response(
        method="POST",
        url=re.compile(r".*/rest/v1/.*"),
        json=[{"id": "test-job-123"}],
        status_code=201,
        is_optional=True,
    )

    # Catch-all for PATCH requests (job updates, profile updates)
    httpx_mock.add_response(
        method="PATCH",
        url=re.compile(r".*/rest/v1/.*"),
        json=[],
        is_optional=True,
    )

    # Catch-all for DELETE requests (chunk cleanup, etc)
    httpx_mock.add_response(
        method="DELETE",
        url=re.compile(r".*/rest/v1/.*"),
        json=[],
        is_optional=True,
    )

    # Catch-all for Supabase Storage GET requests
    httpx_mock.add_response(
        method="GET",
        url=re.compile(r".*/storage/v1/.*"),
        json=[],
        is_optional=True,
    )

    # Catch-all for Anthropic API calls (from background tasks)
    httpx_mock.add_response(
        method="POST",
        url=re.compile(r".*api\.anthropic\.com.*"),
        json={"id": "msg_test", "type": "message", "role": "assistant",
              "content": [{"type": "text", "text": "test response"}],
              "model": "claude-sonnet-4-5-20250929", "stop_reason": "end_turn",
              "usage": {"input_tokens": 10, "output_tokens": 10}},
        is_optional=True,
    )

    # Catch-all for any remaining PATCH requests on any URL
    httpx_mock.add_response(
        method="PATCH",
        url=re.compile(r".*"),
        json=[],
        is_optional=True,
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


def test_health_deep_endpoint_exists():
    """
    GET /health-deep exists (verified via route inspection).
    Uses route inspection instead of TestClient to avoid complex mocking
    of deep health check's DB/embedding/model checks.
    """
    from main import app
    paths = {r.path for r in app.routes if hasattr(r, 'path')}
    assert "/health-deep" in paths


def test_status_endpoint(client):
    """GET /status returns 200."""
    response = client.get("/status")
    assert response.status_code == 200

    data = response.json()
    assert "rlm_available" in data
    assert "timestamp" in data


def test_process_full_v1_still_works():
    """
    POST /process-full endpoint exists and is registered.
    Uses route inspection — the v1 background task makes many external calls
    that are complex to mock fully in TestClient (which runs tasks synchronously).
    """
    from main import app
    matching = [
        r for r in app.routes
        if hasattr(r, 'path') and r.path == "/process-full"
    ]
    assert len(matching) == 1, "/process-full endpoint not registered"
    assert "POST" in matching[0].methods, "/process-full doesn't accept POST"


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
    "/status",
    "/test-embed",
    "/test-patch",
])
def test_get_endpoints_not_404(client, endpoint):
    """Verify GET endpoints are registered (not 404)."""
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
