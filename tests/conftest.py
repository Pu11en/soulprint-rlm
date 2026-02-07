"""Shared test fixtures for soulprint-rlm tests."""
import pytest
from datetime import datetime, timedelta


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Mock environment variables for all tests."""
    monkeypatch.setenv("SUPABASE_URL", "https://swvljsixpvvcirjmflze.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "test-service-key-12345")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key-12345")


@pytest.fixture
def sample_conversations():
    """Sample conversation data for testing."""
    return [
        {
            "id": "conv-1",
            "title": "Test Conversation",
            "mapping": {
                "node-1": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["Hello"]},
                        "create_time": 1704067200,  # 2024-01-01
                    }
                }
            },
            "create_time": 1704067200,
        }
    ]


@pytest.fixture
def sample_chunks():
    """Sample chunk data for processor testing."""
    return [
        {
            "conversation_id": "conv-1",
            "title": "Test Chat",
            "content": "User: Hello\nAssistant: Hi there!",
            "chunk_index": 0,
            "total_chunks": 1,
            "chunk_tier": "medium",
            "message_count": 2,
            "created_at": "2024-06-01T00:00:00Z",
        }
    ]
