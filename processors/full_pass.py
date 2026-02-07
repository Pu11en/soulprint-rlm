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

from adapters import download_conversations, update_user_profile, save_chunks_batch


async def delete_user_chunks(user_id: str):
    """
    Delete all existing conversation chunks for a user (fresh start).

    Args:
        user_id: User ID to delete chunks for

    Best-effort: logs errors but doesn't throw
    """
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_service_key = os.getenv("SUPABASE_SERVICE_KEY")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{supabase_url}/rest/v1/conversation_chunks?user_id=eq.{user_id}",
                headers={
                    "apikey": supabase_service_key,
                    "Authorization": f"Bearer {supabase_service_key}",
                },
                timeout=30.0,
            )

            if response.status_code not in (200, 204):
                print(f"[FullPass] Warning: Failed to delete existing chunks: {response.text}")
            else:
                print(f"[FullPass] Deleted existing chunks for user {user_id}")

    except Exception as e:
        print(f"[FullPass] Error deleting chunks: {e}")


async def run_full_pass_pipeline(
    user_id: str,
    storage_path: str,
    conversation_count: int = 0
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
    conversations = await download_conversations(storage_path)
    print(f"[FullPass] Downloaded {len(conversations)} conversations")

    # Step 2: Chunk conversations
    from processors.conversation_chunker import chunk_conversations
    chunks = chunk_conversations(conversations, target_tokens=2000, overlap_tokens=200)
    print(f"[FullPass] Created {len(chunks)} chunks from {len(conversations)} conversations")

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
    await update_user_profile(user_id, {"memory_md": memory_md})
    print(f"[FullPass] Saved MEMORY section to database")

    # Step 9: V2 Section Regeneration
    from processors.v2_regenerator import regenerate_sections_v2, sections_to_soulprint_text

    print(f"[FullPass] Starting v2 section regeneration for user {user_id}")
    v2_sections = await regenerate_sections_v2(conversations, memory_md, client)

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
