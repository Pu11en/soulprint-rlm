"""Unit tests for Supabase adapter functions."""
import pytest
from datetime import datetime, timedelta
from adapters.supabase_adapter import (
    download_conversations,
    update_user_profile,
    save_chunks_batch,
)


# ============================================================================
# download_conversations tests
# ============================================================================


@pytest.mark.anyio
async def test_download_conversations_success(httpx_mock, sample_conversations):
    """Test successful download from Supabase Storage."""
    # Arrange
    storage_path = "user-exports/test-user/conversations.json"

    httpx_mock.add_response(
        url="https://swvljsixpvvcirjmflze.supabase.co/storage/v1/object/user-exports/test-user/conversations.json",
        json=sample_conversations,
        status_code=200,
    )

    # Act
    result = await download_conversations(storage_path)

    # Assert
    assert result == sample_conversations
    assert len(result) == 1
    assert result[0]["id"] == "conv-1"


@pytest.mark.anyio
async def test_download_conversations_failure_404(httpx_mock):
    """Test download failure handling - 404 not found."""
    # Arrange
    storage_path = "user-exports/missing-user/missing.json"

    httpx_mock.add_response(
        url="https://swvljsixpvvcirjmflze.supabase.co/storage/v1/object/user-exports/missing-user/missing.json",
        status_code=404,
        text="Not Found",
    )

    # Act & Assert
    with pytest.raises(Exception, match="Failed to download from storage: 404"):
        await download_conversations(storage_path)


@pytest.mark.anyio
async def test_download_conversations_url_parsing(httpx_mock):
    """Test that bucket/path split produces correct URL."""
    # Arrange
    storage_path = "user-exports/user-123/data/conversations.json"

    httpx_mock.add_response(
        url="https://swvljsixpvvcirjmflze.supabase.co/storage/v1/object/user-exports/user-123/data/conversations.json",
        json=[],
        status_code=200,
    )

    # Act
    result = await download_conversations(storage_path)

    # Assert
    assert result == []
    request = httpx_mock.get_request()
    assert "user-exports/user-123/data/conversations.json" in str(request.url)


# ============================================================================
# update_user_profile tests
# ============================================================================


@pytest.mark.anyio
async def test_update_user_profile_success_204(httpx_mock):
    """Test successful profile update with 204 No Content response."""
    # Arrange
    user_id = "00000000-0000-0000-0000-000000000000"
    updates = {"memory_md": "Test memory content"}

    httpx_mock.add_response(
        url="https://swvljsixpvvcirjmflze.supabase.co/rest/v1/user_profiles?user_id=eq.00000000-0000-0000-0000-000000000000",
        status_code=204,
    )

    # Act
    await update_user_profile(user_id, updates)

    # Assert - should not raise exception


@pytest.mark.anyio
async def test_update_user_profile_success_200(httpx_mock):
    """Test successful profile update with 200 OK response."""
    # Arrange
    user_id = "11111111-1111-1111-1111-111111111111"
    updates = {"import_status": "complete"}

    httpx_mock.add_response(
        url="https://swvljsixpvvcirjmflze.supabase.co/rest/v1/user_profiles?user_id=eq.11111111-1111-1111-1111-111111111111",
        status_code=200,
        json=[],
    )

    # Act
    await update_user_profile(user_id, updates)

    # Assert - should not raise exception


@pytest.mark.anyio
async def test_update_user_profile_failure_400(httpx_mock):
    """Test profile update failure handling - 400 bad request."""
    # Arrange
    user_id = "invalid-id"
    updates = {"invalid_field": "value"}

    httpx_mock.add_response(
        url="https://swvljsixpvvcirjmflze.supabase.co/rest/v1/user_profiles?user_id=eq.invalid-id",
        status_code=400,
        text="Bad Request",
    )

    # Act & Assert
    with pytest.raises(Exception, match="Failed to update user profile: 400"):
        await update_user_profile(user_id, updates)


@pytest.mark.anyio
async def test_update_user_profile_sends_patch(httpx_mock):
    """Test that update uses PATCH method and sends correct body."""
    # Arrange
    user_id = "test-user-id"
    updates = {"memory_md": "New memory", "import_status": "complete"}

    httpx_mock.add_response(
        url="https://swvljsixpvvcirjmflze.supabase.co/rest/v1/user_profiles?user_id=eq.test-user-id",
        status_code=204,
    )

    # Act
    await update_user_profile(user_id, updates)

    # Assert - verify request details
    request = httpx_mock.get_request()
    assert request.method == "PATCH"
    body = request.content.decode()
    assert "memory_md" in body
    assert "import_status" in body


# ============================================================================
# save_chunks_batch tests
# ============================================================================


@pytest.mark.anyio
async def test_save_chunks_batch_success(httpx_mock):
    """Test successful chunk batch save."""
    # Arrange
    user_id = "test-user-id"
    chunks = [
        {
            "conversation_id": "conv-1",
            "title": "Test",
            "content": "Content",
            "chunk_tier": "medium",
            "message_count": 2,
            "created_at": datetime.utcnow().isoformat(),
        }
    ]

    httpx_mock.add_response(
        url="https://swvljsixpvvcirjmflze.supabase.co/rest/v1/conversation_chunks",
        status_code=201,
    )

    # Act
    await save_chunks_batch(user_id, chunks)

    # Assert - should not raise exception


@pytest.mark.anyio
async def test_save_chunks_batch_adds_user_id(httpx_mock):
    """Test that save_chunks_batch adds user_id to each chunk."""
    # Arrange
    user_id = "test-user-123"
    chunks = [
        {"conversation_id": "conv-1", "title": "Test 1", "content": "Content 1"},
        {"conversation_id": "conv-2", "title": "Test 2", "content": "Content 2"},
    ]

    httpx_mock.add_response(
        url="https://swvljsixpvvcirjmflze.supabase.co/rest/v1/conversation_chunks",
        status_code=201,
    )

    # Act
    await save_chunks_batch(user_id, chunks)

    # Assert
    request = httpx_mock.get_request()
    body = request.content.decode()
    # Should have user_id in each chunk
    assert body.count('"user_id":"test-user-123"') == 2


@pytest.mark.anyio
async def test_save_chunks_batch_sets_is_recent_true(httpx_mock):
    """Test is_recent=True for chunks within 180 days."""
    # Arrange
    user_id = "test-user-id"
    recent_date = (datetime.utcnow() - timedelta(days=30)).isoformat()
    chunks = [
        {
            "conversation_id": "conv-recent",
            "title": "Recent",
            "content": "Recent content",
            "created_at": recent_date,
        }
    ]

    httpx_mock.add_response(
        url="https://swvljsixpvvcirjmflze.supabase.co/rest/v1/conversation_chunks",
        status_code=201,
    )

    # Act
    await save_chunks_batch(user_id, chunks)

    # Assert
    request = httpx_mock.get_request()
    body = request.content.decode()
    assert '"is_recent":true' in body


@pytest.mark.anyio
async def test_save_chunks_batch_sets_is_recent_false(httpx_mock):
    """Test is_recent=False for chunks older than 180 days."""
    # Arrange
    user_id = "test-user-id"
    old_date = (datetime.utcnow() - timedelta(days=250)).isoformat()
    chunks = [
        {
            "conversation_id": "conv-old",
            "title": "Old",
            "content": "Old content",
            "created_at": old_date,
        }
    ]

    httpx_mock.add_response(
        url="https://swvljsixpvvcirjmflze.supabase.co/rest/v1/conversation_chunks",
        status_code=201,
    )

    # Act
    await save_chunks_batch(user_id, chunks)

    # Assert
    request = httpx_mock.get_request()
    body = request.content.decode()
    assert '"is_recent":false' in body


@pytest.mark.anyio
async def test_save_chunks_batch_defaults_chunk_tier(httpx_mock):
    """Test chunk_tier defaults to 'medium' when not provided."""
    # Arrange
    user_id = "test-user-id"
    chunks = [
        {
            "conversation_id": "conv-1",
            "title": "No Tier",
            "content": "Content without tier",
            # No chunk_tier provided
        }
    ]

    httpx_mock.add_response(
        url="https://swvljsixpvvcirjmflze.supabase.co/rest/v1/conversation_chunks",
        status_code=201,
    )

    # Act
    await save_chunks_batch(user_id, chunks)

    # Assert
    request = httpx_mock.get_request()
    body = request.content.decode()
    assert '"chunk_tier":"medium"' in body


@pytest.mark.anyio
async def test_save_chunks_batch_defaults_message_count(httpx_mock):
    """Test message_count defaults to 0 when not provided."""
    # Arrange
    user_id = "test-user-id"
    chunks = [
        {
            "conversation_id": "conv-1",
            "title": "No Count",
            "content": "Content without message count",
            # No message_count provided
        }
    ]

    httpx_mock.add_response(
        url="https://swvljsixpvvcirjmflze.supabase.co/rest/v1/conversation_chunks",
        status_code=201,
    )

    # Act
    await save_chunks_batch(user_id, chunks)

    # Assert
    request = httpx_mock.get_request()
    body = request.content.decode()
    assert '"message_count":0' in body


@pytest.mark.anyio
async def test_save_chunks_batch_no_created_at(httpx_mock):
    """Test is_recent=False when created_at is not provided."""
    # Arrange
    user_id = "test-user-id"
    chunks = [
        {
            "conversation_id": "conv-1",
            "title": "No Date",
            "content": "Content without created_at",
            # No created_at provided
        }
    ]

    httpx_mock.add_response(
        url="https://swvljsixpvvcirjmflze.supabase.co/rest/v1/conversation_chunks",
        status_code=201,
    )

    # Act
    await save_chunks_batch(user_id, chunks)

    # Assert
    request = httpx_mock.get_request()
    body = request.content.decode()
    assert '"is_recent":false' in body


@pytest.mark.anyio
async def test_save_chunks_batch_failure_500(httpx_mock):
    """Test chunk save failure handling - 500 server error."""
    # Arrange
    user_id = "test-user-id"
    chunks = [{"conversation_id": "conv-1", "title": "Test", "content": "Content"}]

    httpx_mock.add_response(
        url="https://swvljsixpvvcirjmflze.supabase.co/rest/v1/conversation_chunks",
        status_code=500,
        text="Internal Server Error",
    )

    # Act & Assert
    with pytest.raises(Exception, match="Failed to save chunk batch: 500"):
        await save_chunks_batch(user_id, chunks)
