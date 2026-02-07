"""Supabase adapter functions for storage, database, and REST API operations."""
import os
import httpx
from typing import List
from datetime import datetime, timedelta


async def download_conversations(storage_path: str) -> List[dict]:
    """
    Download conversations.json from Supabase Storage.

    Args:
        storage_path: Path in format "bucket/path/to/file.json"
                     Example: "user-exports/user-123/conversations.json"

    Returns:
        List of conversation dicts parsed from JSON response

    Raises:
        Exception: If download fails with non-200 status code
    """
    # Read env vars inside function for testability with monkeypatch
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

    # Parse bucket and path from storage_path
    path_parts = storage_path.split("/", 1)
    bucket = path_parts[0]
    file_path = path_parts[1] if len(path_parts) > 1 else ""

    # Construct Supabase Storage URL
    download_url = f"{supabase_url}/storage/v1/object/{bucket}/{file_path}"

    # Use context manager for proper resource cleanup
    async with httpx.AsyncClient(timeout=60.0) as client:
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
        }

        response = await client.get(download_url, headers=headers)

        if response.status_code != 200:
            raise Exception(f"Failed to download from storage: {response.status_code}")

        return response.json()


async def update_user_profile(user_id: str, updates: dict) -> None:
    """
    Update user_profiles table fields via Supabase REST API.

    Args:
        user_id: User UUID
        updates: Dict of field: value pairs to update
                Example: {"memory_md": "content", "import_status": "complete"}

    Raises:
        Exception: If update fails with non-success status code
    """
    # Read env vars inside function for testability
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

    async with httpx.AsyncClient(timeout=30.0) as client:
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
        }

        response = await client.patch(
            f"{supabase_url}/rest/v1/user_profiles",
            params={"user_id": f"eq.{user_id}"},
            headers=headers,
            json=updates,
        )

        if response.status_code not in (200, 204):
            raise Exception(f"Failed to update user profile: {response.status_code}")


async def save_chunks_batch(user_id: str, chunks: List[dict]) -> None:
    """
    Save a batch of conversation chunks to the conversation_chunks table.

    This function enriches each chunk with:
    - user_id: Set from the user_id parameter
    - is_recent: True if created_at is within 180 days, False otherwise
    - chunk_tier: Defaults to "medium" if not provided
    - message_count: Defaults to 0 if not provided

    Valid chunk_tier values:
    - "micro": Ultra-precise chunks (~100 chars) for facts, names, dates
    - "medium": Context chunks (~500 chars) for topic understanding
    - "macro": Flow chunks (~2000 chars) for conversation context

    Args:
        user_id: User UUID
        chunks: List of chunk dicts with conversation_id, title, content, etc.

    Raises:
        Exception: If save fails with non-success status code
    """
    # Read env vars inside function for testability
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

    # Calculate date threshold for is_recent (180 days ago)
    six_months_ago = datetime.utcnow() - timedelta(days=180)

    # Enrich each chunk with required fields
    for chunk in chunks:
        # Add user_id
        chunk["user_id"] = user_id

        # Calculate is_recent based on created_at
        created_at = chunk.get("created_at")
        if created_at:
            try:
                # Parse ISO format datetime (handle both with and without timezone)
                created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                # Remove timezone info for comparison with naive datetime
                if created_dt.tzinfo:
                    created_dt = created_dt.replace(tzinfo=None)
                chunk["is_recent"] = created_dt > six_months_ago
            except Exception:
                # If parsing fails, default to not recent
                chunk["is_recent"] = False
        else:
            # No created_at means we can't determine recency
            chunk["is_recent"] = False

        # Default chunk_tier to "medium" if not provided
        if "chunk_tier" not in chunk:
            chunk["chunk_tier"] = "medium"

        # Default message_count to 0 if not provided
        if "message_count" not in chunk:
            chunk["message_count"] = 0

    async with httpx.AsyncClient(timeout=60.0) as client:
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }

        response = await client.post(
            f"{supabase_url}/rest/v1/conversation_chunks",
            headers=headers,
            json=chunks,
        )

        if response.status_code not in (200, 201):
            raise Exception(f"Failed to save chunk batch: {response.status_code}")
