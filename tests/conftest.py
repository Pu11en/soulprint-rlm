"""Shared test fixtures for soulprint-rlm tests."""
import pytest
from datetime import datetime, timedelta


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Mock environment variables for all tests."""
    monkeypatch.setenv("SUPABASE_URL", "https://swvljsixpvvcirjmflze.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "test-service-key-12345")


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
