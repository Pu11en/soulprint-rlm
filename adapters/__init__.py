"""Adapter layer for Supabase operations."""
from .supabase_adapter import (
    download_conversations,
    update_user_profile,
    save_chunks_batch,
)

__all__ = [
    "download_conversations",
    "update_user_profile",
    "save_chunks_batch",
]
