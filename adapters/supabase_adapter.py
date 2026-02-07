"""Supabase adapter functions - stub implementations for TDD RED phase."""
from typing import List


async def download_conversations(storage_path: str) -> List[dict]:
    """Download conversations.json from Supabase Storage."""
    raise NotImplementedError


async def update_user_profile(user_id: str, updates: dict) -> None:
    """Update user_profiles table fields."""
    raise NotImplementedError


async def save_chunks_batch(user_id: str, chunks: List[dict]) -> None:
    """Save a batch of conversation chunks to the database."""
    raise NotImplementedError
