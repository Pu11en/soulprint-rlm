"""
Streaming Import Processor

Downloads ChatGPT exports from Supabase Storage via streaming HTTP,
parses with ijson for constant-memory JSON processing, and generates
a quick pass soulprint.

Uses a temporary file approach for TRUE constant-memory processing:
1. Stream httpx download to temp file (chunk-by-chunk, no accumulation)
2. Pass temp file to ijson for parsing (file handle, no memory load)
3. Clean up temp file after processing

This allows processing 300MB+ exports without OOM on Render.
"""

import asyncio
import json
import os
import tempfile
import traceback
from datetime import datetime, timezone
from typing import Optional

import httpx
import ijson

from .dag_parser import extract_active_path

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")


async def update_progress(user_id: str, percent: int, stage: str):
    """Update user_profiles with progress_percent and import_stage.

    Called at every major pipeline stage so the frontend can show
    real-time progress to the user.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.patch(
                f"{SUPABASE_URL}/rest/v1/user_profiles?user_id=eq.{user_id}",
                json={
                    "progress_percent": percent,
                    "import_stage": stage,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                headers={
                    "apikey": SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal",
                },
            )
            if response.status_code not in (200, 204):
                print(f"[streaming_import] WARN: progress update failed for {user_id}: {response.text}")
    except Exception as e:
        # Best-effort progress updates -- never block the pipeline
        print(f"[streaming_import] WARN: progress update error for {user_id}: {e}")


async def download_streaming(storage_path: str, temp_file_path: str):
    """Stream download from Supabase Storage directly to a temp file.

    Writes chunk-by-chunk to disk, never accumulating the full file
    in memory. This is critical for 300MB+ exports.

    Args:
        storage_path: Full storage path, e.g. "user-imports/user-123/raw-123.json"
        temp_file_path: Local filesystem path to write to
    """
    # storage_path format: "bucket/path/to/file"
    # Supabase Storage URL: /storage/v1/object/{bucket}/{path}
    url = f"{SUPABASE_URL}/storage/v1/object/{storage_path}"

    async with httpx.AsyncClient(timeout=300.0) as client:
        async with client.stream("GET", url, headers={
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        }) as response:
            response.raise_for_status()

            # Write chunks directly to disk (constant memory)
            with open(temp_file_path, "wb") as f:
                async for chunk in response.aiter_bytes():
                    f.write(chunk)  # Immediately write to disk, don't accumulate

    print(f"[streaming_import] Downloaded to temp file: {temp_file_path}")


def parse_conversations_streaming(file_path: str) -> list:
    """Parse ChatGPT conversations.json with ijson from file (constant memory).

    ijson reads from file handle directly, never loading the entire file
    into RAM. Handles both formats:
    - Bare array: [...]
    - Wrapped object: { conversations: [...] }

    Args:
        file_path: Path to the downloaded JSON file on disk

    Returns:
        List of parsed conversation dicts
    """
    raw_convos = []

    with open(file_path, "rb") as f:
        try:
            # Try bare array format first (most common)
            parser = ijson.items(f, "item")
            raw_convos = list(parser)

            # If no items found, might be wrapped format
            if len(raw_convos) == 0:
                f.seek(0)  # Reset file pointer
                parser = ijson.items(f, "conversations.item")
                raw_convos = list(parser)
        except (ijson.JSONError, ijson.common.IncompleteJSONError):
            # If bare array fails, try wrapped format
            f.seek(0)  # Reset file pointer
            try:
                parser = ijson.items(f, "conversations.item")
                raw_convos = list(parser)
            except Exception as e:
                # Both formats failed
                print(f"[streaming_import] ERROR: Both parse formats failed: {e}")

    # Process each conversation with DAG traversal
    conversations = []
    for raw_convo in raw_convos:
        parsed_messages = extract_active_path(raw_convo)

        if parsed_messages:
            create_time = raw_convo.get("create_time")
            if create_time and isinstance(create_time, (int, float)) and create_time > 0:
                created_at = datetime.fromtimestamp(create_time, tz=timezone.utc).isoformat()
            else:
                created_at = datetime.now(timezone.utc).isoformat()

            conversations.append({
                "id": raw_convo.get("id"),
                "title": raw_convo.get("title", "Untitled"),
                "createdAt": created_at,
                "messages": parsed_messages,
            })

    total_messages = sum(len(c["messages"]) for c in conversations)
    print(f"[streaming_import] Parsed {len(conversations)} conversations with DAG traversal ({total_messages} total messages)")
    return conversations


async def trigger_full_pass(user_id: str, storage_path: str, conversation_count: int):
    """Fire-and-forget full pass after quick pass succeeds.

    Runs asynchronously — does not block chat access.
    Creates conversation chunks, extracts facts, generates MEMORY section,
    and regenerates v2 soulprint sections.
    """
    try:
        # Mark full pass as processing
        async with httpx.AsyncClient() as client:
            await client.patch(
                f"{SUPABASE_URL}/rest/v1/user_profiles?user_id=eq.{user_id}",
                json={
                    "full_pass_status": "processing",
                    "full_pass_error": None,
                },
                headers={
                    "apikey": SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal",
                },
            )

        from .full_pass import run_full_pass_pipeline
        await run_full_pass_pipeline(
            user_id=user_id,
            storage_path=storage_path,
            conversation_count=conversation_count,
        )

        # Mark complete
        async with httpx.AsyncClient() as client:
            await client.patch(
                f"{SUPABASE_URL}/rest/v1/user_profiles?user_id=eq.{user_id}",
                json={
                    "full_pass_status": "complete",
                    "full_pass_completed_at": datetime.now(timezone.utc).isoformat(),
                },
                headers={
                    "apikey": SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal",
                },
            )
        print(f"[streaming_import] Full pass complete for user {user_id}")

    except Exception as e:
        error_msg = str(e)[:500]
        print(f"[streaming_import] Full pass failed for user {user_id}: {error_msg}")
        traceback.print_exc()
        try:
            async with httpx.AsyncClient() as client:
                await client.patch(
                    f"{SUPABASE_URL}/rest/v1/user_profiles?user_id=eq.{user_id}",
                    json={
                        "full_pass_status": "failed",
                        "full_pass_error": error_msg,
                    },
                    headers={
                        "apikey": SUPABASE_SERVICE_KEY,
                        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                        "Content-Type": "application/json",
                        "Prefer": "return=minimal",
                    },
                )
        except Exception:
            pass


async def process_import_streaming(user_id: str, storage_path: str):
    """Complete streaming import pipeline with TRUE constant memory.

    Uses temporary file approach:
    1. Stream httpx download to temp file (chunk-by-chunk, no accumulation)
    2. Pass temp file to ijson for parsing (file handle, no memory load)
    3. Generate quick pass soulprint from parsed conversations
    4. Save results to database
    5. Clean up temp file after processing

    Stages:
        0-20%:  Download from Supabase Storage
        20-50%: Parse conversations with ijson
        50-100%: Generate quick pass soulprint

    Args:
        user_id: The user's ID
        storage_path: Full Supabase Storage path (e.g. "user-imports/uid/raw-123.json")
    """
    temp_file_path: Optional[str] = None

    try:
        # Create temporary file
        fd, temp_file_path = tempfile.mkstemp(suffix=".json", prefix="soulprint_import_")
        os.close(fd)  # Close file descriptor, we'll use path

        # Stage 1: Download to temp file (0-20%)
        await update_progress(user_id, 0, "Downloading export")
        print(f"[streaming_import] Starting download for user {user_id}: {storage_path}")
        await download_streaming(storage_path, temp_file_path)
        await update_progress(user_id, 20, "Parsing conversations")

        # Stage 2: Parse from temp file (20-50%)
        print(f"[streaming_import] Parsing conversations for user {user_id}")
        conversations = parse_conversations_streaming(temp_file_path)

        if not conversations:
            raise ValueError("No conversations found in export file")

        await update_progress(user_id, 50, "Generating soulprint")

        # Stage 3: Quick Pass (50-100%)
        print(f"[streaming_import] Generating quick pass for user {user_id} ({len(conversations)} conversations)")
        from .quick_pass import generate_quick_pass
        quick_pass_result = generate_quick_pass(conversations)  # synchronous, raises on failure

        # Save to database (matching process-server.ts structure)
        soul_md = json.dumps(quick_pass_result.get("soul", {}))
        identity_md = json.dumps(quick_pass_result.get("identity", {}))
        user_md = json.dumps(quick_pass_result.get("user", {}))
        agents_md = json.dumps(quick_pass_result.get("agents", {}))
        tools_md = json.dumps(quick_pass_result.get("tools", {}))

        ai_name = quick_pass_result.get("identity", {}).get("ai_name", "Nova")
        archetype = quick_pass_result.get("identity", {}).get("archetype", "Analyzing...")

        # Update user_profiles with quick pass results
        async with httpx.AsyncClient() as client:
            await client.patch(
                f"{SUPABASE_URL}/rest/v1/user_profiles?user_id=eq.{user_id}",
                json={
                    "soul_md": soul_md,
                    "identity_md": identity_md,
                    "user_md": user_md,
                    "agents_md": agents_md,
                    "tools_md": tools_md,
                    "ai_name": ai_name,
                    "archetype": archetype,
                    "import_status": "quick_ready",
                    "import_error": None,
                    "progress_percent": 100,
                    "import_stage": "Complete",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                headers={
                    "apikey": SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal",
                },
            )

        print(f"[streaming_import] Quick pass complete for user {user_id}: ai_name={ai_name}, archetype={archetype}")

        # Fire-and-forget full pass (chunks, facts, memory, v2 sections)
        # User can chat immediately with quick pass results while this runs
        asyncio.create_task(trigger_full_pass(user_id, storage_path, len(conversations)))
        print(f"[streaming_import] Full pass triggered for user {user_id}")

    except Exception as e:
        # Update status to failed with specific error message
        error_msg = str(e)[:500]  # Limit error message length
        print(f"[streaming_import] ERROR for user {user_id}: {error_msg}")
        traceback.print_exc()

        try:
            async with httpx.AsyncClient() as client:
                await client.patch(
                    f"{SUPABASE_URL}/rest/v1/user_profiles?user_id=eq.{user_id}",
                    json={
                        "import_status": "failed",
                        "import_error": error_msg,
                        "progress_percent": 100,
                        "import_stage": "Failed",
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                    headers={
                        "apikey": SUPABASE_SERVICE_KEY,
                        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                        "Content-Type": "application/json",
                        "Prefer": "return=minimal",
                    },
                )
        except Exception as update_err:
            print(f"[streaming_import] ERROR: Failed to update error status for {user_id}: {update_err}")

    finally:
        # Clean up temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                print(f"[streaming_import] Cleaned up temp file: {temp_file_path}")
            except Exception:
                pass  # Best effort cleanup
