"""
Integration tests for full pass pipeline.

Tests concurrency configuration, status tracking, and pipeline step execution.
"""
import pytest
import os


# ============================================================================
# get_concurrency_limit tests
# ============================================================================

def test_get_concurrency_limit_default():
    """Default concurrency is 3 for Render Starter tier."""
    from processors.full_pass import get_concurrency_limit
    # conftest.py doesn't set FACT_EXTRACTION_CONCURRENCY, so default applies
    assert get_concurrency_limit() == 3


def test_get_concurrency_limit_from_env(monkeypatch):
    """Concurrency reads from FACT_EXTRACTION_CONCURRENCY env var."""
    from processors.full_pass import get_concurrency_limit
    monkeypatch.setenv("FACT_EXTRACTION_CONCURRENCY", "5")
    assert get_concurrency_limit() == 5


def test_get_concurrency_limit_invalid_returns_default(monkeypatch):
    """Invalid concurrency values fall back to default 3."""
    from processors.full_pass import get_concurrency_limit

    # Non-integer
    monkeypatch.setenv("FACT_EXTRACTION_CONCURRENCY", "abc")
    assert get_concurrency_limit() == 3

    # Too high
    monkeypatch.setenv("FACT_EXTRACTION_CONCURRENCY", "100")
    assert get_concurrency_limit() == 3

    # Zero
    monkeypatch.setenv("FACT_EXTRACTION_CONCURRENCY", "0")
    assert get_concurrency_limit() == 3

    # Negative
    monkeypatch.setenv("FACT_EXTRACTION_CONCURRENCY", "-1")
    assert get_concurrency_limit() == 3


def test_get_concurrency_limit_boundary_values(monkeypatch):
    """Concurrency accepts valid boundary values 1 and 50."""
    from processors.full_pass import get_concurrency_limit

    monkeypatch.setenv("FACT_EXTRACTION_CONCURRENCY", "1")
    assert get_concurrency_limit() == 1

    monkeypatch.setenv("FACT_EXTRACTION_CONCURRENCY", "50")
    assert get_concurrency_limit() == 50


# ============================================================================
# Pipeline logging verification
# ============================================================================

def test_pipeline_has_user_id_logging():
    """Pipeline source code includes user_id in step logging."""
    import inspect
    from processors.full_pass import run_full_pass_pipeline
    source = inspect.getsource(run_full_pass_pipeline)

    # Verify structured logging is present
    assert "user_id=" in source
    assert "step=" in source

    # Verify all 9 steps are logged
    assert "step=download_conversations" in source
    assert "step=chunk_conversations" in source
    assert "step=save_chunks" in source
    assert "step=extract_facts" in source
    assert "step=consolidate_facts" in source
    assert "step=hierarchical_reduce" in source
    assert "step=generate_memory" in source
    assert "step=save_memory" in source
    assert "step=v2_regeneration" in source


# ============================================================================
# Pipeline execution test with mocked dependencies
# ============================================================================

@pytest.mark.asyncio
async def test_pipeline_executes_all_steps(monkeypatch, capsys):
    """Pipeline executes all 9 steps with mocked dependencies."""
    from processors.full_pass import run_full_pass_pipeline

    # Mock adapter functions
    async def mock_download(path):
        return [
            {
                "id": "conv-1",
                "title": "Test Conversation",
                "messages": [
                    {"role": "user", "content": "What's the weather?"},
                    {"role": "assistant", "content": "I can help with that!"}
                ],
                "created_at": 1704067200
            }
        ]

    async def mock_update(user_id, updates):
        pass

    async def mock_save_chunks(user_id, chunks):
        pass

    async def mock_delete_chunks(user_id):
        pass

    monkeypatch.setattr("processors.full_pass.download_conversations", mock_download)
    monkeypatch.setattr("processors.full_pass.update_user_profile", mock_update)
    monkeypatch.setattr("processors.full_pass.save_chunks_batch", mock_save_chunks)
    monkeypatch.setattr("processors.full_pass.delete_user_chunks", mock_delete_chunks)

    # Mock Anthropic client responses
    class MockMessage:
        def __init__(self, text):
            self.text = text

    class MockResponse:
        def __init__(self, text):
            self.content = [MockMessage(text)]

    class MockMessages:
        async def create(self, **kwargs):
            # Get prompt content from messages
            messages = kwargs.get("messages", [{}])
            prompt_text = messages[0].get("content", "") if messages else ""

            # Fact extraction response
            if "Extract ONLY factual" in prompt_text:
                return MockResponse('{"preferences": ["test pref"], "projects": [], "dates": [], "beliefs": [], "decisions": []}')

            # Memory generation response
            elif "You are creating a MEMORY section" in prompt_text:
                return MockResponse("## Preferences\n- Test preference\n\n## Projects\nNo data yet.")

            # V2 regeneration response
            elif "personality profile" in prompt_text or "Generate EXACTLY" in prompt_text:
                return MockResponse('{"soul": {}, "identity": {}, "user": {}, "agents": {}, "tools": {}}')

            # Hierarchical reduce response (fallback)
            else:
                return MockResponse('{"preferences": [], "projects": [], "dates": [], "beliefs": [], "decisions": [], "total_count": 0}')

    class MockAnthropicClient:
        messages = MockMessages()

    # Patch Anthropic client
    monkeypatch.setattr("anthropic.AsyncAnthropic", lambda **kwargs: MockAnthropicClient())

    # Run pipeline
    memory_md = await run_full_pass_pipeline(
        user_id="test-user-123",
        storage_path="test-exports/test.json",
        conversation_count=1,
    )

    # Verify MEMORY was generated
    assert memory_md is not None
    assert len(memory_md) > 0
    assert "Preferences" in memory_md

    # Verify logging captured all steps
    captured = capsys.readouterr()
    assert "user_id=test-user-123" in captured.out

    # Verify all 9 steps were logged
    assert "step=download_conversations" in captured.out
    assert "step=chunk_conversations" in captured.out
    assert "step=save_chunks" in captured.out
    assert "step=extract_facts" in captured.out
    assert "step=consolidate_facts" in captured.out
    assert "step=hierarchical_reduce" in captured.out
    assert "step=generate_memory" in captured.out
    assert "step=save_memory" in captured.out
    assert "step=v2_regeneration" in captured.out
    assert "step=complete" in captured.out


# ============================================================================
# Concurrency configuration integration
# ============================================================================

@pytest.mark.asyncio
async def test_pipeline_uses_configured_concurrency(monkeypatch, capsys):
    """Pipeline respects FACT_EXTRACTION_CONCURRENCY environment variable."""
    from processors.full_pass import run_full_pass_pipeline

    # Set custom concurrency
    monkeypatch.setenv("FACT_EXTRACTION_CONCURRENCY", "7")

    # Mock dependencies (same as above)
    async def mock_download(path):
        return [{"id": "conv-1", "title": "Test", "messages": [{"role": "user", "content": "Hi"}], "created_at": 1704067200}]

    async def mock_update(user_id, updates):
        pass

    async def mock_save_chunks(user_id, chunks):
        pass

    async def mock_delete_chunks(user_id):
        pass

    monkeypatch.setattr("processors.full_pass.download_conversations", mock_download)
    monkeypatch.setattr("processors.full_pass.update_user_profile", mock_update)
    monkeypatch.setattr("processors.full_pass.save_chunks_batch", mock_save_chunks)
    monkeypatch.setattr("processors.full_pass.delete_user_chunks", mock_delete_chunks)

    # Mock Anthropic
    class MockMessage:
        def __init__(self, text):
            self.text = text

    class MockResponse:
        def __init__(self, text):
            self.content = [MockMessage(text)]

    class MockMessages:
        async def create(self, **kwargs):
            # Get prompt content from messages
            messages = kwargs.get("messages", [{}])
            prompt_text = messages[0].get("content", "") if messages else ""

            # Memory generation response
            if "You are creating a MEMORY section" in prompt_text:
                return MockResponse("## Preferences\n- Test preference\n\n## Projects\nNo data yet.")
            # Default to valid fact structure
            else:
                return MockResponse('{"preferences": [], "projects": [], "dates": [], "beliefs": [], "decisions": [], "total_count": 0}')

    class MockAnthropicClient:
        messages = MockMessages()

    monkeypatch.setattr("anthropic.AsyncAnthropic", lambda **kwargs: MockAnthropicClient())

    # Run pipeline
    await run_full_pass_pipeline(
        user_id="test-user-concurrency",
        storage_path="test.json",
        conversation_count=1,
    )

    # Verify concurrency was logged
    captured = capsys.readouterr()
    assert "concurrency=7" in captured.out
