"""
Full Pass Pipeline Orchestrator
Downloads conversations, chunks them, extracts facts, generates MEMORY section
"""
import os
import json
import httpx
import anthropic
from datetime import datetime, timedelta
from typing import List, Dict


# Supabase config from environment
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")


async def delete_user_chunks(user_id: str):
    """
    Delete all existing conversation chunks for a user (fresh start).

    Args:
        user_id: User ID to delete chunks for

    Best-effort: logs errors but doesn't throw
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{SUPABASE_URL}/rest/v1/conversation_chunks?user_id=eq.{user_id}",
                headers={
                    "apikey": SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                },
                timeout=30.0,
            )

            if response.status_code not in (200, 204):
                print(f"[FullPass] Warning: Failed to delete existing chunks: {response.text}")
            else:
                print(f"[FullPass] Deleted existing chunks for user {user_id}")

    except Exception as e:
        print(f"[FullPass] Error deleting chunks: {e}")


async def save_chunks_batch(user_id: str, chunks: List[dict]):
    """
    Save a batch of conversation chunks to the database.

    Args:
        user_id: User ID to associate chunks with
        chunks: List of chunk dicts (conversation_id, title, content, etc.)

    Best-effort: logs errors but doesn't throw
    """
    try:
        # Add user_id and calculate is_recent for each chunk
        six_months_ago = datetime.utcnow() - timedelta(days=180)

        # Fields that exist in the conversation_chunks table
        VALID_COLUMNS = {
            "user_id", "conversation_id", "title", "content",
            "token_count", "chunk_tier", "is_recent", "created_at",
            "message_count",
        }

        for chunk in chunks:
            chunk["user_id"] = user_id

            # Remove fields not in the DB schema (e.g. chunk_index, total_chunks)
            extra_keys = [k for k in chunk if k not in VALID_COLUMNS]
            for k in extra_keys:
                del chunk[k]

            # Set is_recent based on created_at
            created_at = chunk.get("created_at")
            if created_at:
                try:
                    # Parse ISO timestamp
                    created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    chunk["is_recent"] = created_dt > six_months_ago
                except Exception:
                    chunk["is_recent"] = False
            else:
                chunk["is_recent"] = False

            # Ensure chunk_tier is set
            if "chunk_tier" not in chunk:
                chunk["chunk_tier"] = "medium"

            # Set message_count from chunk (not used yet, but schema requires it)
            if "message_count" not in chunk:
                chunk["message_count"] = 0

        # POST batch to Supabase
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{SUPABASE_URL}/rest/v1/conversation_chunks",
                json=chunks,
                headers={
                    "apikey": SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal",
                },
                timeout=60.0,
            )

            if response.status_code not in (200, 201):
                print(f"[FullPass] Warning: Failed to save chunk batch: {response.text}")
            else:
                print(f"[FullPass] Saved batch of {len(chunks)} chunks")

    except Exception as e:
        print(f"[FullPass] Error saving chunk batch: {e}")


async def run_full_pass_pipeline(
    user_id: str,
    storage_path: str,
    conversation_count: int = 0,
    file_type: str = 'json',
) -> str:
    """
    Run the complete full pass pipeline.

    Steps:
    1. Download conversations from Supabase Storage
    2. Chunk conversations into ~2000 token segments
    3. Save chunks to database
    4. Extract facts in parallel via Haiku 4.5
    5. Consolidate and reduce facts if needed
    6. Generate MEMORY section from facts
    7. Save MEMORY to user_profiles.memory_md

    Args:
        user_id: User ID for the full pass
        storage_path: Path to conversations.json in Supabase Storage
        conversation_count: Number of conversations (for logging)

    Returns:
        Generated memory_md string (for v2 regeneration in Plan 02-03)
    """
    print(f"[FullPass] Starting pipeline for user {user_id}")
    print(f"[FullPass] Storage path: {storage_path}")
    print(f"[FullPass] Expected conversations: {conversation_count}")

    # Initialize Anthropic client
    client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Step 1: Download conversations
    from main import download_conversations
    conversations = await download_conversations(storage_path, file_type=file_type)
    print(f"[FullPass] Downloaded {len(conversations)} conversations")

    # Step 2: Chunk conversations
    from processors.conversation_chunker import chunk_conversations
    chunks = chunk_conversations(conversations, target_tokens=2000, overlap_tokens=200)
    print(f"[FullPass] Created {len(chunks)} chunks from {len(conversations)} conversations")

    # Free raw conversations — chunks and v2 regen will use sampled subset
    # Keep a lightweight copy for v2 regen (just id, title, first few messages)
    conversations_light = []
    for c in conversations:
        conversations_light.append({
            "id": c.get("id"),
            "title": c.get("title"),
            "messages": c.get("messages", [])[:20],  # First 20 messages only for v2
            "createdAt": c.get("createdAt"),
        })
    del conversations
    import gc; gc.collect()

    # Step 3: Save chunks to database (in batches to avoid request size limits)
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]

        # Delete existing chunks on first batch
        if i == 0:
            await delete_user_chunks(user_id)

        await save_chunks_batch(user_id, batch)

    print(f"[FullPass] Saved {len(chunks)} chunks to database")

    # Step 4: Extract facts in parallel
    from processors.fact_extractor import (
        extract_facts_parallel,
        consolidate_facts,
        hierarchical_reduce
    )

    all_facts = await extract_facts_parallel(chunks, client, concurrency=10)
    print(f"[FullPass] Extracted facts from {len(chunks)} chunks")

    # Step 5: Consolidate facts
    consolidated = consolidate_facts(all_facts)
    print(f"[FullPass] Consolidated {consolidated['total_count']} unique facts")

    # Step 6: Reduce if too large (over 200K tokens)
    reduced = await hierarchical_reduce(consolidated, client, max_tokens=200000)

    # Step 7: Generate MEMORY section
    from processors.memory_generator import generate_memory_section
    memory_md = await generate_memory_section(reduced, client)
    print(f"[FullPass] Generated MEMORY section ({len(memory_md)} chars)")

    # Step 8: Save MEMORY to database (early save so user benefits even if v2 regen fails)
    from main import update_user_profile
    await update_user_profile(user_id, {"memory_md": memory_md})
    print(f"[FullPass] Saved MEMORY section to database")

    # Free chunks and facts before v2 regen
    del chunks, all_facts, consolidated, reduced
    gc.collect()

    # Step 9: V2 Section Regeneration
    from processors.v2_regenerator import regenerate_sections_v2, sections_to_soulprint_text

    print(f"[FullPass] Starting v2 section regeneration for user {user_id}")
    v2_sections = await regenerate_sections_v2(conversations_light, memory_md, client)

    if v2_sections:
        # Build soulprint_text from v2 sections + MEMORY
        soulprint_text = sections_to_soulprint_text(v2_sections, memory_md)

        # Save all v2 sections + soulprint_text to database atomically
        await update_user_profile(user_id, {
            "soul_md": json.dumps(v2_sections["soul"]),
            "identity_md": json.dumps(v2_sections["identity"]),
            "user_md": json.dumps(v2_sections["user"]),
            "agents_md": json.dumps(v2_sections["agents"]),
            "tools_md": json.dumps(v2_sections["tools"]),
            "soulprint_text": soulprint_text,
        })
        print(f"[FullPass] V2 sections saved for user {user_id}")
    else:
        print(f"[FullPass] V2 regeneration failed -- keeping v1 sections for user {user_id}")
        # V1 sections stay, MEMORY already saved above

    print(f"[FullPass] Pipeline complete for user {user_id}")

    return memory_md
