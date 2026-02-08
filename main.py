"""
SoulPrint RLM Service
TRUE Recursive Language Models - no fallback, memory is critical
Multi-tier precision chunking with vector similarity search
EXHAUSTIVE analysis - processes ALL conversations
+ NEW: /process-import for background processing from imported_chats
+ NEW: /chat with SoulPrint files (soul_md, identity_md, agents_md, user_md)
+ NEW: AWS Bedrock Titan v2 embeddings
"""
import os
import re
import json
import httpx
import time
import asyncio
import anthropic
from contextlib import asynccontextmanager
from datetime import datetime, date
from typing import Optional, List, Tuple
from fastapi import FastAPI, HTTPException, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Import RLM - REQUIRED, no fallback
try:
    from rlm import RLM
    RLM_AVAILABLE = True
except ImportError as e:
    RLM_AVAILABLE = False
    RLM_IMPORT_ERROR = str(e)

# AWS Bedrock for embeddings
try:
    import boto3
    from botocore.config import Config
    BEDROCK_AVAILABLE = True
except ImportError:
    BEDROCK_AVAILABLE = False

load_dotenv()

# Import prompt helpers for structured section formatting
from prompt_helpers import clean_section, format_section


async def startup_check_incomplete_embeddings_logic():
    """
    Check for any users with incomplete embeddings and queue them.
    This ensures embeddings always get completed even if the server restarts.

    Extracted from @app.on_event("startup") for lifespan migration.
    """
    print("[RLM] Checking for users with incomplete embeddings...")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = {
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            }

            # Find users with import_status=complete but embedding_status != complete
            resp = await client.get(
                f"{SUPABASE_URL}/rest/v1/user_profiles",
                params={
                    "import_status": "eq.complete",
                    "embedding_status": "neq.complete",
                    "select": "user_id",
                    "limit": "10",  # Process up to 10 users
                },
                headers=headers,
            )

            if resp.status_code != 200:
                print(f"[RLM] Failed to check incomplete embeddings: {resp.text[:200]}")
                return

            users = resp.json()
            if not users:
                print("[RLM] No users with incomplete embeddings")
                return

            print(f"[RLM] Found {len(users)} users with incomplete embeddings")

            for user in users:
                user_id = user.get("user_id")
                if user_id:
                    print(f"[RLM] Queuing embedding completion for {user_id}")
                    asyncio.create_task(complete_embeddings_background(user_id))

    except Exception as e:
        print(f"[RLM] Error checking incomplete embeddings: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle with processor validation."""
    # Validate all processor modules import correctly (fail fast)
    print("[Startup] Validating processor modules...")
    try:
        from processors.conversation_chunker import chunk_conversations
        from processors.fact_extractor import extract_facts_parallel, consolidate_facts, hierarchical_reduce
        from processors.memory_generator import generate_memory_section
        from processors.v2_regenerator import regenerate_sections_v2, sections_to_soulprint_text
        from processors.full_pass import run_full_pass_pipeline
        print("[Startup] All processor modules imported successfully")
    except ImportError as e:
        print(f"[FATAL] Processor import failed: {e}")
        raise  # Crash the app - Render will not route traffic

    # Migrate existing startup logic from @app.on_event("startup") handlers:

    # 1. From startup_event() at line ~171: Check for stuck jobs
    await asyncio.sleep(2)
    await resume_stuck_jobs()

    # 2. From startup() at line ~1710: Check RLM/Bedrock availability
    if not RLM_AVAILABLE:
        print(f"[RLM] Warning: RLM library not available: {RLM_IMPORT_ERROR}")
    else:
        print("[RLM] RLM library loaded successfully")

    if BEDROCK_AVAILABLE and AWS_ACCESS_KEY_ID:
        print("[RLM] AWS Bedrock configured - using Titan embeddings")
    elif COHERE_API_KEY:
        print("[RLM] Cohere API key configured - using Cohere embeddings")
    else:
        print("[RLM] Warning: No embedding service configured")

    # 3. From startup_check_incomplete_embeddings() at line ~3555: Resume incomplete embeddings
    await asyncio.sleep(5)
    await startup_check_incomplete_embeddings_logic()

    yield  # Application runs here

    # Shutdown
    print("[Shutdown] Cleanup complete")


app = FastAPI(title="SoulPrint RLM Service", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# JOB RECOVERY SYSTEM
# ============================================================

async def create_job(user_id: str, storage_path: str, conversation_count: int, message_count: int) -> str:
    """Create a job record and return job_id."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }
        resp = await client.post(
            f"{SUPABASE_URL}/rest/v1/processing_jobs",
            headers=headers,
            json={
                "user_id": user_id,
                "status": "pending",
                "storage_path": storage_path,
                "conversation_count": conversation_count,
                "message_count": message_count,
                "current_step": "queued",
                "progress": 0,
                "attempts": 1,
            },
        )
        if resp.status_code not in (200, 201):
            print(f"[RLM] Failed to create job: {resp.status_code} - {resp.text[:200]}")
            return None
        jobs = resp.json()
        return jobs[0]["id"] if jobs else None


async def update_job(job_id: str, **kwargs):
    """Update job status/progress."""
    if not job_id:
        return
    async with httpx.AsyncClient(timeout=30.0) as client:
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "Content-Type": "application/json",
        }
        await client.patch(
            f"{SUPABASE_URL}/rest/v1/processing_jobs",
            params={"id": f"eq.{job_id}"},
            headers=headers,
            json=kwargs,
        )


async def complete_job(job_id: str, success: bool, error_message: str = None):
    """Mark job as complete or failed."""
    if not job_id:
        return
    status = "complete" if success else "failed"
    update_data = {
        "status": status,
        "progress": 100 if success else None,
        "completed_at": datetime.utcnow().isoformat(),
    }
    if error_message:
        update_data["error_message"] = error_message
    await update_job(job_id, **update_data)


async def get_stuck_jobs() -> list:
    """Find jobs that were interrupted (status = processing or pending with old updated_at)."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        }
        # Find jobs stuck in processing (server died) or pending (never started)
        resp = await client.get(
            f"{SUPABASE_URL}/rest/v1/processing_jobs",
            params={
                "status": "in.(pending,processing)",
                "attempts": "lt.3",  # Max 3 retries
                "select": "*",
            },
            headers=headers,
        )
        if resp.status_code == 200:
            return resp.json()
        return []


async def resume_stuck_jobs():
    """Check for and resume any stuck jobs on startup."""
    print("[RLM] Checking for stuck jobs to resume...")
    stuck_jobs = await get_stuck_jobs()

    if not stuck_jobs:
        print("[RLM] No stuck jobs found")
        return

    print(f"[RLM] Found {len(stuck_jobs)} stuck job(s), resuming...")

    for job in stuck_jobs:
        job_id = job["id"]
        user_id = job["user_id"]
        storage_path = job["storage_path"]
        attempts = job.get("attempts", 0) + 1

        print(f"[RLM] Resuming job {job_id} for user {user_id} (attempt {attempts})")

        # Update attempt count
        await update_job(job_id, attempts=attempts, status="processing", current_step="resuming")

        # Start processing in background
        asyncio.create_task(
            process_full_background(user_id, storage_path, job_id=job_id)
        )


# Config
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
ALERT_TELEGRAM_BOT = os.getenv("ALERT_TELEGRAM_BOT")
ALERT_TELEGRAM_CHAT = os.getenv("ALERT_TELEGRAM_CHAT", "7414639817")

# AWS Config
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_EMBEDDING_MODEL_ID = os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")

# Vercel callback - hardcoded since it's not sensitive and simplifies deployment
VERCEL_API_URL = os.getenv("VERCEL_API_URL", "https://www.soulprintengine.ai")
RLM_API_KEY = os.getenv("RLM_API_KEY")  # Shared secret for auth

# Models - AWS Bedrock format
# Nova models have MUCH higher rate limits (~1000/min vs Claude's ~20/min)
NOVA_MICRO_MODEL = "amazon.nova-micro-v1:0"  # Fast batch processing
NOVA_LITE_MODEL = "amazon.nova-lite-v1:0"  # Chat, light tasks
NOVA_PRO_MODEL = "amazon.nova-pro-v1:0"  # Synthesis merging
NOVA_2_LITE_MODEL = "us.amazon.nova-2-lite-v1:0"  # Latest Nova with reasoning (1M context)

# Claude models - use for QUALITY tasks (final output users see)
HAIKU_MODEL = "us.anthropic.claude-3-5-haiku-20241022-v1:0"  # Legacy, keeping for fallback
SONNET_MODEL = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"  # Legacy, keeping for fallback
SONNET_4_5_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"  # Best quality for SoulPrint files
HAIKU_4_5_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"  # Fast quality alternative

USE_BEDROCK_CLAUDE = True  # Use AWS Bedrock instead of Anthropic API


class QueryRequest(BaseModel):
    user_id: str
    message: str
    soulprint_text: Optional[str] = None
    history: Optional[List[dict]] = []
    ai_name: Optional[str] = None
    sections: Optional[dict] = None
    web_search_context: Optional[str] = None


class QueryResponse(BaseModel):
    response: str
    chunks_used: int
    method: str
    latency_ms: int


class ChatRequest(BaseModel):
    user_id: str
    message: str
    history: Optional[List[dict]] = []


class ChatResponse(BaseModel):
    response: str
    memories_used: int
    latency_ms: int


class ProcessImportRequest(BaseModel):
    user_id: str
    user_email: Optional[str] = None
    conversations_json: Optional[str] = None  # Raw JSON from ChatGPT export


class AnalyzeRequest(BaseModel):
    user_id: str


class CreateSoulprintRequest(BaseModel):
    user_id: str
    conversations: List[dict]
    stats: Optional[dict] = None


# ============================================================================
# AWS BEDROCK EMBEDDINGS
# ============================================================================

def get_bedrock_client():
    """Get AWS Bedrock client"""
    if not BEDROCK_AVAILABLE:
        return None
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        return None

    config = Config(
        region_name=AWS_REGION,
        retries={'max_attempts': 3, 'mode': 'adaptive'}
    )

    return boto3.client(
        'bedrock-runtime',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        config=config
    )


async def embed_text_bedrock(text: str, bedrock_client=None) -> Optional[List[float]]:
    """Generate embedding using AWS Bedrock Titan v2"""
    if not bedrock_client:
        bedrock_client = get_bedrock_client()
    if not bedrock_client:
        print("[RLM] No bedrock client available")
        return None

    try:
        # Clean and validate text
        if not text or not isinstance(text, str):
            return None

        # Remove null bytes and other problematic characters
        text = text.replace('\x00', '').replace('\r', ' ').replace('\n', ' ').strip()

        # Must have actual content
        if len(text) < 3:
            return None

        # Truncate to max input size (Titan limit is ~8k tokens, ~32k chars safe)
        text = text[:8000]

        # Titan Embed v2 request format - use default 1024 dimensions
        request_body = json.dumps({
            "inputText": text,
            "normalize": True
        })

        print(f"[RLM] Calling Bedrock model: {BEDROCK_EMBEDDING_MODEL_ID}, text length: {len(text)}")

        response = bedrock_client.invoke_model(
            modelId=BEDROCK_EMBEDDING_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=request_body
        )

        result = json.loads(response['body'].read())
        embedding = result.get('embedding')
        if embedding:
            print(f"[RLM] Got embedding with {len(embedding)} dimensions")
        return embedding
    except Exception as e:
        error_str = str(e)
        print(f"[RLM] Bedrock embed error: {error_str[:200]}")
        # Log full error once
        if "first_error_logged" not in dir(embed_text_bedrock):
            embed_text_bedrock.first_error_logged = True
            print(f"[RLM] Full error: {error_str}")
        return None


async def embed_texts_batch(texts: List[str], batch_size: int = 10) -> List[Optional[List[float]]]:
    """Batch embed texts with rate limiting"""
    bedrock_client = get_bedrock_client()
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = []

        for text in batch:
            emb = await embed_text_bedrock(text, bedrock_client)
            batch_embeddings.append(emb)
            await asyncio.sleep(0.1)  # Rate limit

        embeddings.extend(batch_embeddings)

        if i + batch_size < len(texts):
            await asyncio.sleep(0.5)  # Batch delay

    return embeddings


# ============================================================================
# BEDROCK CLAUDE (Text Generation)
# ============================================================================

async def bedrock_claude_message(
    messages: List[dict],
    model: str = None,
    system: str = None,
    max_tokens: int = 4096,
    bedrock_client=None
) -> str:
    """
    Call Claude via AWS Bedrock Converse API.

    Args:
        messages: List of {"role": "user"|"assistant", "content": "text"}
        model: Bedrock model ID (defaults to SONNET_MODEL)
        system: Optional system prompt
        max_tokens: Max response tokens
        bedrock_client: Optional existing client

    Returns:
        Response text string
    """
    if not bedrock_client:
        bedrock_client = get_bedrock_client()
    if not bedrock_client:
        raise Exception("Bedrock client not available")

    model = model or SONNET_MODEL

    # Convert messages to Bedrock format
    bedrock_messages = []
    for msg in messages:
        bedrock_messages.append({
            "role": msg["role"],
            "content": [{"text": msg["content"]}]
        })

    # Build request
    request_body = {
        "modelId": model,
        "messages": bedrock_messages,
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": 0.7,
        }
    }

    if system:
        request_body["system"] = [{"text": system}]

    # Retry with exponential backoff for throttling
    max_retries = 5
    base_delay = 2

    for attempt in range(max_retries):
        try:
            # Use run_in_executor to call sync boto3 in async context
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: bedrock_client.converse(**request_body)
            )

            # Extract text from response
            output = response.get("output", {})
            message = output.get("message", {})
            content = message.get("content", [])

            text_parts = []
            for block in content:
                if "text" in block:
                    text_parts.append(block["text"])

            return "".join(text_parts)

        except Exception as e:
            error_str = str(e)
            is_throttle = "ThrottlingException" in error_str or "Too many requests" in error_str

            if is_throttle and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)  # 2, 4, 8, 16, 32 seconds
                print(f"[RLM] Bedrock throttled, waiting {delay}s before retry {attempt + 2}/{max_retries}")
                await asyncio.sleep(delay)
            else:
                print(f"[RLM] Bedrock Claude error: {e}")
                raise


# ============================================================================
# ALERTS
# ============================================================================

async def alert_drew(message: str):
    """Alert Drew on Telegram when something is wrong"""
    if not ALERT_TELEGRAM_BOT:
        print(f"[ALERT - NO BOT CONFIGURED] {message}")
        return

    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"https://api.telegram.org/bot{ALERT_TELEGRAM_BOT}/sendMessage",
                json={
                    "chat_id": ALERT_TELEGRAM_CHAT,
                    "text": f"🚨 SoulPrint RLM Alert\n\n{message}",
                    "parse_mode": "HTML",
                }
            )
    except Exception as e:
        print(f"Failed to alert Drew: {e}")


# ============================================================================
# HYBRID SEARCH (Vector + Keyword for precise memory retrieval)
# ============================================================================

def extract_keywords(query: str) -> List[str]:
    """Extract important keywords from query for keyword search"""
    # Remove common words, keep names/nouns/specifics
    stop_words = {
        'i', 'me', 'my', 'we', 'our', 'you', 'your', 'what', 'when', 'where',
        'who', 'how', 'why', 'the', 'a', 'an', 'is', 'are', 'was', 'were',
        'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
        'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in', 'for',
        'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'all', 'each', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but',
        'if', 'or', 'because', 'until', 'while', 'about', 'against', 'any',
        'both', 'that', 'this', 'these', 'those', 'am', 'tell', 'said', 'say',
        'talk', 'talked', 'about', 'remember', 'mentioned', 'told', 'asked',
    }

    words = re.findall(r'\b\w+\b', query.lower())
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    return keywords[:5]  # Max 5 keywords


async def keyword_search(user_id: str, keywords: List[str], limit: int = 20) -> List[dict]:
    """Search chunks by keyword matching (catches exact names, dates, etc.)"""
    if not keywords:
        return []

    try:
        async with httpx.AsyncClient() as client:
            headers = {
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            }

            # Build ILIKE query for each keyword
            # This catches exact matches that vector search might miss
            all_results = []
            seen_ids = set()

            for keyword in keywords:
                response = await client.get(
                    f"{SUPABASE_URL}/rest/v1/conversation_chunks",
                    params={
                        "user_id": f"eq.{user_id}",
                        "content": f"ilike.*{keyword}*",
                        "select": "id,content,title,chunk_tier,message_count,created_at",
                        "limit": str(limit // len(keywords) + 1),
                    },
                    headers=headers,
                    timeout=15.0,
                )

                if response.status_code == 200:
                    results = response.json()
                    for r in results:
                        if r['id'] not in seen_ids:
                            r['match_type'] = 'keyword'
                            r['matched_keyword'] = keyword
                            all_results.append(r)
                            seen_ids.add(r['id'])

            print(f"[RLM] Keyword search found {len(all_results)} matches for: {keywords}")
            return all_results[:limit]

    except Exception as e:
        print(f"[RLM] Keyword search error: {e}")
        return []


async def search_memories(user_id: str, query: str, limit: int = 50) -> List[dict]:
    """
    HYBRID SEARCH: Vector similarity + Keyword matching with RRF (Reciprocal Rank Fusion).

    Uses Supabase-recommended hybrid search pattern:
    1. Extract keywords from query (names, specific terms)
    2. Run keyword search (catches exact matches like names, dates)
    3. Run tier-aware vector search (catches semantic matches)
    4. Merge using RRF: score = 1/(k + rank) from each method
    5. Return deduplicated, ranked results

    RRF ensures both precise keyword matches AND semantically similar content surface.
    """

    # Extract keywords for exact matching
    keywords = extract_keywords(query)
    print(f"[RLM] Hybrid search - query: '{query[:50]}...', keywords: {keywords}")

    # Run keyword search first (catches exact names, dates, etc.)
    keyword_results = await keyword_search(user_id, keywords, limit=30)

    # Generate query embedding with Bedrock Titan
    query_embedding = await embed_text_bedrock(query)
    if not query_embedding:
        print("[RLM] Failed to generate query embedding, using keyword results only")
        return keyword_results if keyword_results else await get_recent_chunks(user_id, limit)

    vector_results = []

    try:
        async with httpx.AsyncClient() as client:
            headers = {
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type": "application/json",
            }

            # Tier-aware vector search
            # Macro: themes/relationships (few, high context)
            # Medium: conversation context (moderate)
            # Micro: precise facts (many, pinpoint accuracy)
            tier_limits = [
                ("macro", 15),   # Broad themes
                ("medium", 25),  # Conversation context
                ("micro", 25),   # Precise facts
            ]

            for tier, tier_limit in tier_limits:
                response = await client.post(
                    f"{SUPABASE_URL}/rest/v1/rpc/match_conversation_chunks_by_tier",
                    headers=headers,
                    json={
                        "query_embedding": query_embedding,
                        "match_user_id": user_id,
                        "match_tier": tier,
                        "match_count": tier_limit,
                        "match_threshold": 0.10,  # Very low threshold for better recall
                    },
                    timeout=30.0,
                )

                if response.status_code == 200:
                    tier_results = response.json()
                    for r in tier_results:
                        r["chunk_tier"] = tier
                        r["match_type"] = "vector"
                    vector_results.extend(tier_results)
                    print(f"[RLM] Vector tier '{tier}' returned {len(tier_results)} matches")
                else:
                    print(f"[RLM] Vector tier '{tier}' FAILED: {response.status_code} - {response.text}")

            # Fallback to non-tier search if no results
            if not vector_results:
                response = await client.post(
                    f"{SUPABASE_URL}/rest/v1/rpc/match_conversation_chunks",
                    headers=headers,
                    json={
                        "query_embedding": query_embedding,
                        "match_user_id": user_id,
                        "match_count": limit,
                        "match_threshold": 0.25,
                    },
                    timeout=30.0,
                )
                if response.status_code == 200:
                    vector_results = response.json()
                    for r in vector_results:
                        r["match_type"] = "vector"

            # ============================================================
            # RRF (Reciprocal Rank Fusion) - Supabase recommended approach
            # Formula: score = 1/(k + rank)
            # k=60 is smoothing constant (prevents top results from dominating)
            # ============================================================
            RRF_K = 60
            KEYWORD_WEIGHT = 1.5  # Boost keyword matches (exact is important)
            SEMANTIC_WEIGHT = 1.0

            # Build score map by chunk ID
            chunk_scores = {}  # id -> {score, data}

            # Score keyword results (rank 1 = best match)
            for rank, result in enumerate(keyword_results):
                chunk_id = result.get('id')
                if not chunk_id:
                    continue
                rrf_score = KEYWORD_WEIGHT * (1.0 / (RRF_K + rank + 1))
                if chunk_id in chunk_scores:
                    chunk_scores[chunk_id]['score'] += rrf_score
                    chunk_scores[chunk_id]['data']['match_type'] = 'hybrid'  # Found in both
                else:
                    chunk_scores[chunk_id] = {
                        'score': rrf_score,
                        'data': {**result, 'match_type': 'keyword'}
                    }

            # Score vector results
            for rank, result in enumerate(vector_results):
                chunk_id = result.get('id')
                if not chunk_id:
                    continue
                rrf_score = SEMANTIC_WEIGHT * (1.0 / (RRF_K + rank + 1))
                if chunk_id in chunk_scores:
                    chunk_scores[chunk_id]['score'] += rrf_score
                    # If already keyword, now it's hybrid
                    if chunk_scores[chunk_id]['data'].get('match_type') == 'keyword':
                        chunk_scores[chunk_id]['data']['match_type'] = 'hybrid'
                else:
                    chunk_scores[chunk_id] = {
                        'score': rrf_score,
                        'data': {**result}
                    }

            # Sort by RRF score (highest first) and return
            sorted_results = sorted(
                chunk_scores.values(),
                key=lambda x: x['score'],
                reverse=True
            )

            # Add RRF score to each result for debugging
            final_results = []
            for item in sorted_results[:limit]:
                result = item['data']
                result['rrf_score'] = round(item['score'], 4)
                final_results.append(result)

            # Log match type distribution
            hybrid_count = sum(1 for r in final_results if r.get('match_type') == 'hybrid')
            keyword_only = sum(1 for r in final_results if r.get('match_type') == 'keyword')
            vector_only = sum(1 for r in final_results if r.get('match_type') == 'vector')
            print(f"[RLM] Hybrid search results: {len(final_results)} total "
                  f"(hybrid={hybrid_count}, keyword={keyword_only}, vector={vector_only})")

            return final_results

    except Exception as e:
        print(f"[RLM] Vector search exception: {e}")
        # Fallback: return keyword results if vector failed
        if keyword_results:
            return keyword_results
        return await get_recent_chunks(user_id, limit)


async def get_recent_chunks(user_id: str, limit: int = 100) -> List[dict]:
    """Fallback: get recent chunks without vector search"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{SUPABASE_URL}/rest/v1/conversation_chunks",
                params={
                    "user_id": f"eq.{user_id}",
                    "select": "id,content,title,message_count,created_at",
                    "order": "created_at.desc",
                    "limit": str(limit),
                },
                headers={
                    "apikey": SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                },
                timeout=30.0,
            )
            return response.json() if response.status_code == 200 else []
    except Exception as e:
        print(f"[RLM] Recent chunks error: {e}")
        return []


async def get_soulprint(user_id: str) -> dict:
    """Get SoulPrint files from user_profiles table"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{SUPABASE_URL}/rest/v1/user_profiles",
                params={
                    "user_id": f"eq.{user_id}",
                    "select": "soul_md,identity_md,agents_md,user_md,archetype,soulprint,soulprint_text",
                },
                headers={
                    "apikey": SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                },
                timeout=30.0,
            )

            if response.status_code == 200:
                data = response.json()
                return data[0] if data else {}
            return {}
    except Exception as e:
        print(f"[RLM] Get soulprint error: {e}")
        return {}


# ============================================================================
# /chat - Memory-aware chat with SoulPrint
# ============================================================================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint with memory retrieval and SoulPrint personality.

    Flow:
    1. Embed user message (Bedrock Titan)
    2. Search imported_chats for similar past messages
    3. Load SoulPrint files
    4. Build prompt with memory context
    5. Generate response with Claude
    """
    start = time.time()

    try:
        # Get memories relevant to this message
        memories = await search_memories(request.user_id, request.message, limit=30)

        # Get SoulPrint
        soulprint = await get_soulprint(request.user_id)

        # Build memory context with hybrid search results
        memory_context = ""
        if memories:
            memory_context = "## Relevant Memories from Past Conversations\n\n"
            for i, mem in enumerate(memories[:25]):  # Use more memories
                title = mem.get('title', mem.get('conversation_title', 'Unknown'))
                content = mem.get('content', '')[:600]
                match_type = mem.get('match_type', 'unknown')
                rrf_score = mem.get('rrf_score', 0)
                tier = mem.get('chunk_tier', '')

                # Show match source for context
                match_indicator = ""
                if match_type == 'hybrid':
                    match_indicator = "🎯"  # Found by both keyword AND semantic
                elif match_type == 'keyword':
                    match_indicator = "📌"  # Exact keyword match
                else:
                    match_indicator = "💭"  # Semantic match

                memory_context += f"**{match_indicator} [{title}]** ({tier}, score: {rrf_score})\n{content}\n\n"

        # Build SoulPrint context
        soul_context = ""
        if soulprint:
            if soulprint.get('soul_md'):
                soul_context += f"## Soul Profile\n{soulprint['soul_md']}\n\n"
            if soulprint.get('identity_md'):
                soul_context += f"## Identity\n{soulprint['identity_md']}\n\n"
            if soulprint.get('agents_md'):
                soul_context += f"## Communication Guide\n{soulprint['agents_md']}\n\n"
            if soulprint.get('user_md'):
                soul_context += f"## User Context\n{soulprint['user_md']}\n\n"

        # Build conversation history
        history_context = ""
        if request.history:
            history_context = "## Current Conversation\n"
            for msg in request.history[-10:]:
                role = msg.get('role', 'user')
                content = msg.get('content', '')[:500]
                history_context += f"**{role}**: {content}\n"
            history_context += "\n"

        # Build the prompt
        system_prompt = f"""You are SoulPrint, a deeply personalized AI companion with perfect memory of the user's conversation history.

{soul_context}

{memory_context}

{history_context}

## Instructions
- You have access to the user's past conversations above. Reference them naturally when relevant.
- Embody the personality and communication style described in the Soul Profile.
- Be warm, helpful, and authentic.
- If the user asks about something from their past, the relevant memories are provided above.
- Keep responses concise but personal."""

        # Generate response using Amazon Nova Lite (higher rate limits than Claude)
        response_text = await bedrock_claude_message(
            messages=[{"role": "user", "content": request.message}],
            system=system_prompt[:15000],
            model=NOVA_LITE_MODEL,  # Amazon's model - ~1000 req/min vs Claude's 20/min
            max_tokens=2048,
        )

        latency = int((time.time() - start) * 1000)

        return ChatResponse(
            response=response_text,
            memories_used=len(memories),
            latency_ms=latency,
        )

    except Exception as e:
        error_msg = str(e)
        print(f"[RLM] Chat error: {error_msg}")
        await alert_drew(f"Chat Error\n\nUser: {request.user_id}\nError: {error_msg[:500]}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {error_msg}")


# ============================================================================
# /process-import - Background SoulPrint Generation
# ============================================================================

def parse_chatgpt_conversations(conversations_json: str) -> List[dict]:
    """
    Parse ChatGPT export JSON into flat message list.

    ChatGPT format:
    - Array of conversation objects
    - Each conversation has a 'mapping' dict with message nodes
    - Messages have 'author.role' and 'content.parts'
    """
    try:
        conversations = json.loads(conversations_json)
    except json.JSONDecodeError as e:
        print(f"[RLM] JSON parse error: {e}")
        return []

    messages = []

    for conv in conversations:
        title = conv.get('title', 'Untitled')
        mapping = conv.get('mapping', {})

        # Walk the message tree
        for node_id, node in mapping.items():
            msg = node.get('message')
            if not msg:
                continue

            author = msg.get('author', {})
            role = author.get('role', '')

            # Only get user messages for personality analysis
            if role != 'user':
                continue

            # Extract content
            content_obj = msg.get('content', {})
            parts = content_obj.get('parts', [])

            # Filter to text parts only
            text_parts = [p for p in parts if isinstance(p, str)]
            if not text_parts:
                continue

            content = '\n'.join(text_parts).strip()
            if not content or len(content) < 5:
                continue

            # Get timestamp
            create_time = msg.get('create_time')
            timestamp = None
            if create_time:
                try:
                    timestamp = datetime.fromtimestamp(create_time).isoformat()
                except:
                    pass

            messages.append({
                'content': content,
                'conversation_title': title,
                'original_timestamp': timestamp,
                'role': 'user',
            })

    # Sort by timestamp if available
    messages.sort(key=lambda m: m.get('original_timestamp') or '')

    return messages


async def process_import_background(user_id: str, user_email: Optional[str], conversations_json: Optional[str]):
    """
    Background job to process imported chats and generate SoulPrint.

    Flow:
    1. Parse conversations_json directly (no database table needed)
    2. Run recursive Haiku/Sonnet synthesis
    3. Generate SoulPrint files (soul_md, identity_md, agents_md, user_md)
    4. Save to Supabase user_profiles
    5. Call Vercel completion callback
    """
    start = time.time()
    print(f"[RLM] Starting import processing for user {user_id}")

    try:
        # 1. Parse the raw conversations JSON
        if not conversations_json:
            raise Exception("No conversations_json provided")

        messages = parse_chatgpt_conversations(conversations_json)
        total_messages = len(messages)
        print(f"[RLM] Parsed {total_messages} user messages from ChatGPT export")

        if total_messages == 0:
            raise Exception("No user messages found in export")

        async with httpx.AsyncClient(timeout=120.0) as client:
            headers = {
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type": "application/json",
            }

            # Upsert user_profile (create if doesn't exist)
            await client.post(
                f"{SUPABASE_URL}/rest/v1/user_profiles",
                headers={**headers, "Prefer": "resolution=merge-duplicates"},
                json={
                    "user_id": user_id,
                    "total_messages": total_messages,
                    "import_status": "processing",
                },
            )

            # Skip embeddings for now - we'll do synthesis directly
            # Embeddings can be generated later for memory search
            embedded_count = 0
            print(f"[RLM] Skipping embeddings for now, proceeding to synthesis...")

            # 2. Run recursive Haiku/Sonnet synthesis
            print("[RLM] Starting recursive synthesis...")
            soulprint_result = await recursive_synthesize(messages, user_id)

            # 3. Generate SoulPrint files
            print("[RLM] Generating SoulPrint files...")
            soul_files = await generate_soulprint_files(soulprint_result, messages, user_id)

            # 4. Save to user_profiles (simpler - no separate soulprints table needed)
            print("[RLM] Saving SoulPrint to user_profiles...")

            # Build soulprint_text for display
            soulprint_text = f"""# {soulprint_result.get('archetype', 'Your SoulPrint')}

{soulprint_result.get('core_essence', '')}

## Communication Style
{json.dumps(soulprint_result.get('voice', {}), indent=2)}

## Personality
{json.dumps(soulprint_result.get('mind', {}), indent=2)}
{json.dumps(soulprint_result.get('heart', {}), indent=2)}

## Your World
{json.dumps(soulprint_result.get('world', {}), indent=2)}
"""

            # Upsert user_profiles with SoulPrint data
            profile_payload = {
                "user_id": user_id,
                "import_status": "complete",
                "soulprint": soulprint_result,  # Full JSON
                "soulprint_text": soulprint_text,  # Readable version
                "archetype": soulprint_result.get("archetype", "Unique Individual"),
                "total_messages": total_messages,
                "soulprint_generated_at": datetime.utcnow().isoformat(),
            }

            await client.post(
                f"{SUPABASE_URL}/rest/v1/user_profiles",
                headers={**headers, "Prefer": "resolution=merge-duplicates"},
                json=profile_payload,
            )

            elapsed = time.time() - start
            print(f"[RLM] Import processing complete in {elapsed:.1f}s")

            # 6. Call Vercel completion callback
            if VERCEL_API_URL:
                try:
                    await client.post(
                        f"{VERCEL_API_URL}/api/import/complete",
                        headers={
                            "Content-Type": "application/json",
                            "X-RLM-API-Key": RLM_API_KEY or "",
                        },
                        json={
                            "user_id": user_id,
                            "email": user_email,
                            "success": True,
                            "messages_processed": total_messages,
                            "embeddings_generated": embedded_count,
                            "elapsed_seconds": round(elapsed, 1),
                        },
                        timeout=30.0,
                    )
                    print(f"[RLM] Notified Vercel of completion")
                except Exception as e:
                    print(f"[RLM] Failed to notify Vercel: {e}")

            # Alert success
            await alert_drew(
                f"✅ SoulPrint Generated!\n\n"
                f"User: {user_id}\n"
                f"Messages: {total_messages}\n"
                f"Embeddings: {embedded_count}\n"
                f"Time: {elapsed:.1f}s"
            )

    except Exception as e:
        error_msg = str(e)
        print(f"[RLM] Import processing failed: {error_msg}")

        # Update status to failed
        try:
            async with httpx.AsyncClient() as client:
                await client.patch(
                    f"{SUPABASE_URL}/rest/v1/user_profiles",
                    params={"user_id": f"eq.{user_id}"},
                    headers={
                        "apikey": SUPABASE_SERVICE_KEY,
                        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={"import_status": "failed"},
                )
        except:
            pass

        await alert_drew(f"❌ Import Failed\n\nUser: {user_id}\nError: {error_msg[:500]}")


async def recursive_synthesize(messages: List[dict], user_id: str, batch_size: int = 100) -> dict:
    """
    Recursive synthesis using Haiku for chunks, Sonnet for synthesis via Bedrock.

    Level 1: Haiku processes batches → summaries
    Level 2+: Sonnet merges summaries → until single result
    """
    if len(messages) <= batch_size:
        # Base case: small enough for single Sonnet call
        return await sonnet_generate_profile(messages)

    # Level 1: Process batches with Haiku (fast, cheap)
    print(f"[RLM] Level 1: Processing {len(messages)} messages in batches of {batch_size}")
    summaries = []

    batches = [messages[i:i+batch_size] for i in range(0, len(messages), batch_size)]

    for i, batch in enumerate(batches):
        print(f"[RLM] Haiku processing batch {i+1}/{len(batches)}")
        summary = await haiku_extract_patterns(batch, i, len(batches))
        summaries.append(summary)
        await asyncio.sleep(1.5)  # Rate limit - longer delay to avoid throttling

    # Level 2+: Recursive merge with Sonnet
    print(f"[RLM] Level 2: Merging {len(summaries)} summaries")
    return await recursive_merge_summaries(summaries)


async def haiku_extract_patterns(batch: List[dict], batch_num: int, total_batches: int) -> dict:
    """Level 1: Haiku extracts patterns from a batch of messages"""

    # Build message text
    batch_text = ""
    for msg in batch:
        content = msg.get('content', '')[:500]
        title = msg.get('conversation_title', '')
        if content.strip():
            batch_text += f"[{title}] {content}\n\n"

    batch_text = batch_text[:30000]  # Limit size

    prompt = f"""Analyze these user messages and extract personality patterns. Batch {batch_num + 1}/{total_batches}.

## MESSAGES
{batch_text}

## TASK
Extract SPECIFIC patterns with QUOTES from the text.

Return JSON:
{{
  "voice": {{
    "formality": "casual|mixed|formal",
    "tone_examples": ["direct quotes showing tone"],
    "humor_examples": ["jokes or sarcasm if any"],
    "emoji_usage": ["emojis they use"]
  }},
  "topics": {{
    "interests": ["topics they discuss"],
    "expertise": ["things they know well"],
    "questions": ["types of questions they ask"]
  }},
  "personality": {{
    "traits": ["observable traits"],
    "values": ["what matters to them"],
    "communication_style": "brief description"
  }},
  "facts": {{
    "people": ["names mentioned"],
    "places": ["locations mentioned"],
    "work": ["job/career info"],
    "life": ["life details"]
  }},
  "quotes": ["5-10 most characteristic quotes"]
}}

Be SPECIFIC. Quote actual text. Return valid JSON only."""

    try:
        # Use Nova Micro for batch processing (much higher rate limits than Claude)
        text = await bedrock_claude_message(
            messages=[{"role": "user", "content": prompt}],
            model=NOVA_MICRO_MODEL,  # Nova has ~1000 req/min vs Claude's ~20/min
            max_tokens=2048,
        )

        # Parse JSON
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        return json.loads(text.strip())
    except Exception as e:
        print(f"[RLM] Nova batch {batch_num} error: {e}")
        return {"error": str(e), "batch": batch_num}


async def recursive_merge_summaries(summaries: List[dict], merge_size: int = 10) -> dict:
    """Recursively merge summaries until we have a single profile"""

    if len(summaries) <= merge_size:
        # Final merge
        return await sonnet_final_synthesis(summaries)

    # Merge in groups
    merged = []
    for i in range(0, len(summaries), merge_size):
        group = summaries[i:i+merge_size]
        partial = await sonnet_merge_partial(group)
        merged.append(partial)
        await asyncio.sleep(0.5)

    # Recurse
    return await recursive_merge_summaries(merged, merge_size)


async def sonnet_merge_partial(summaries: List[dict]) -> dict:
    """Sonnet merges a group of summaries into one"""

    prompt = f"""Merge these personality summaries into one consolidated summary.

## SUMMARIES TO MERGE
{json.dumps(summaries, indent=2)[:40000]}

## TASK
Combine all patterns, keeping the most specific quotes and examples.
Deduplicate but preserve unique insights.

Return JSON with same structure as input summaries, but merged."""

    try:
        # Use Nova Pro for merging (higher rate limits than Claude)
        text = await bedrock_claude_message(
            messages=[{"role": "user", "content": prompt}],
            model=NOVA_PRO_MODEL,  # Nova Pro for synthesis merging
            max_tokens=4096,
        )

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        return json.loads(text.strip())
    except Exception as e:
        print(f"[RLM] Merge error: {e}")
        return {"merged": summaries, "error": str(e)}


async def sonnet_final_synthesis(summaries: List[dict]) -> dict:
    """Sonnet creates final personality profile from all summaries"""

    prompt = f"""Create a comprehensive personality profile from these analyzed patterns.

## ALL PATTERNS
{json.dumps(summaries, indent=2)[:50000]}

## TASK
Synthesize everything into a deep, personalized profile.

Return JSON:
{{
  "archetype": "Creative 2-4 word title (e.g., 'The Relentless Builder')",
  "core_essence": "2-3 sentences capturing who they are",
  "voice": {{
    "formality": "casual|mixed|formal",
    "humor": "dry|playful|sarcastic|warm|none",
    "emoji_style": "heavy|moderate|minimal|none",
    "signature_phrases": ["their catchphrases"],
    "tone": "overall communication tone"
  }},
  "mind": {{
    "thinking_style": "how they process ideas",
    "interests": ["core interests"],
    "expertise": ["areas of knowledge"],
    "curiosity": "what drives their questions"
  }},
  "heart": {{
    "emotional_expression": "how they show feelings",
    "values": ["what matters to them"],
    "connection_style": "how they relate to others"
  }},
  "world": {{
    "people": ["important people in their life"],
    "places": ["relevant locations"],
    "work": "career/professional context",
    "life_context": "relevant life details"
  }},
  "best_quotes": ["10-15 most characteristic quotes"],
  "communication_guide": {{
    "do": ["how to communicate as them"],
    "avoid": ["what to avoid"]
  }}
}}

Make it DEEPLY PERSONAL and SPECIFIC."""

    try:
        # Use Nova Pro for final synthesis (higher rate limits)
        text = await bedrock_claude_message(
            messages=[{"role": "user", "content": prompt}],
            model=NOVA_PRO_MODEL,  # Nova Pro for synthesis
            max_tokens=8192,
        )

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        return json.loads(text.strip())
    except Exception as e:
        print(f"[RLM] Final synthesis error: {e}")
        return {"archetype": "Unique Individual", "error": str(e)}


async def sonnet_generate_profile(messages: List[dict]) -> dict:
    """Generate profile from small message set (base case)"""

    msg_text = ""
    for msg in messages:
        content = msg.get('content', '')[:300]
        if content.strip():
            msg_text += f"{content}\n\n"

    msg_text = msg_text[:40000]

    prompt = f"""Analyze these messages and create a personality profile.

## MESSAGES
{msg_text}

Return JSON with: archetype, core_essence, voice, mind, heart, world, best_quotes, communication_guide.
Be specific and quote actual text."""

    try:
        # Use Nova Pro for internal processing
        text = await bedrock_claude_message(
            messages=[{"role": "user", "content": prompt}],
            model=NOVA_PRO_MODEL,  # Higher rate limits for processing
            max_tokens=4096,
        )

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        return json.loads(text.strip())
    except Exception as e:
        return {"archetype": "Unique Individual", "error": str(e)}


async def generate_soulprint_files(profile: dict, messages: List[dict], user_id: str) -> dict:
    """Generate the four SoulPrint markdown files using Claude Sonnet 4.5 (best quality for user-facing content)"""

    print("[RLM] Generating SoulPrint files with Claude Sonnet 4.5...")

    # SOUL.md - Core personality
    soul_prompt = f"""Based on this personality profile, create SOUL.md - a comprehensive guide to who this person IS.

## PROFILE
{json.dumps(profile, indent=2)[:20000]}

## TASK
Write SOUL.md in markdown format covering:
- Core personality traits
- Communication style and vibe
- Vocabulary patterns and phrases they use
- Emoji usage patterns
- Emotional expression patterns
- Boundaries and sensitivities

Write it as a reference guide for an AI to understand and embody this person's communication style.
Be specific with examples and quotes.
Keep it under 2000 words."""

    soul_md = await bedrock_claude_message(
        messages=[{"role": "user", "content": soul_prompt}],
        model=SONNET_4_5_MODEL,  # Best quality for user-facing content
        max_tokens=4096,
    )

    # IDENTITY.md - AI persona
    identity_prompt = f"""Based on this profile, create IDENTITY.md - defining the AI persona.

## PROFILE
{json.dumps(profile, indent=2)[:10000]}

## TASK
Write IDENTITY.md covering:
- Persona name (derived from their style)
- Signature emoji that represents them
- One-sentence vibe description
- How to maintain persona consistency
- What makes this persona unique

Keep it under 500 words. Be creative but grounded in their actual patterns."""

    identity_md = await bedrock_claude_message(
        messages=[{"role": "user", "content": identity_prompt}],
        model=SONNET_4_5_MODEL,  # Best quality for user-facing content
        max_tokens=1024,
    )

    # AGENTS.md - Behavior rules
    agents_prompt = f"""Based on this profile, create AGENTS.md - operational rules for the AI.

## PROFILE
{json.dumps(profile, indent=2)[:10000]}

## TASK
Write AGENTS.md covering:
- Response pacing and rhythm
- Context adaptation rules (formal vs casual situations)
- Memory protocol (how to reference past conversations)
- Emotional intelligence guidelines
- Consistency rules

Keep it under 1000 words. Make rules actionable and specific."""

    agents_md = await bedrock_claude_message(
        messages=[{"role": "user", "content": agents_prompt}],
        model=SONNET_4_5_MODEL,  # Best quality for user-facing content
        max_tokens=2048,
    )

    # USER.md - Facts about the user
    user_prompt = f"""Based on this profile, create USER.md - factual information about the user.

## PROFILE
{json.dumps(profile, indent=2)[:10000]}

## TASK
Write USER.md covering:
- Personal info (name if known, location hints)
- Key relationships (people mentioned)
- Interests and hobbies
- Work/career context
- Recurring topics they care about
- Current life context

Keep it under 1000 words. Stick to facts from the analysis."""

    user_md = await bedrock_claude_message(
        messages=[{"role": "user", "content": user_prompt}],
        model=SONNET_4_5_MODEL,  # Best quality for user-facing content
        max_tokens=2048,
    )

    return {
        "soul_md": soul_md,
        "identity_md": identity_md,
        "agents_md": agents_md,
        "user_md": user_md,
    }


async def generate_memory_log(messages: List[dict], profile: dict, user_id: str) -> str:
    """
    Generate a memory log summarizing the user's conversation history.
    This creates a daily summary style log using Bedrock Claude.
    """

    # Group messages by date
    messages_by_date = {}
    for msg in messages:
        timestamp = msg.get("original_timestamp")
        if timestamp:
            try:
                if isinstance(timestamp, str):
                    # Parse ISO format
                    msg_date = timestamp[:10]  # YYYY-MM-DD
                else:
                    msg_date = str(date.today())
            except:
                msg_date = str(date.today())
        else:
            msg_date = str(date.today())

        if msg_date not in messages_by_date:
            messages_by_date[msg_date] = []
        messages_by_date[msg_date].append(msg)

    # Get recent dates (last 7 days worth of data or less)
    sorted_dates = sorted(messages_by_date.keys(), reverse=True)[:7]

    if not sorted_dates:
        return f"# Memory Log - {date.today()}\n\nNo conversations to summarize yet."

    # Build summary text
    summary_input = ""
    for d in sorted_dates:
        day_messages = messages_by_date[d][:20]  # Limit per day
        summary_input += f"\n## {d}\n"
        for msg in day_messages:
            title = msg.get("conversation_title", "")
            content = msg.get("content", "")[:200]
            summary_input += f"- [{title}] {content}\n"

    prompt = f"""Based on this user's recent conversations, create a memory log summary.

## USER PROFILE
Archetype: {profile.get('archetype', 'Unknown')}
Core: {profile.get('core_essence', '')}

## RECENT CONVERSATIONS
{summary_input[:15000]}

## TASK
Create a memory log in markdown format:

# Memory Log - {date.today()}

## Summary
[2-3 sentences summarizing their recent activity and interests]

## Key Topics
- [Topic 1]
- [Topic 2]
- [Topic 3]

## Notable Details
- [Any important facts, projects, or events mentioned]

## Current Focus
[What they seem to be working on or thinking about]

## Emotional State
[General mood/energy based on conversations]

Keep it concise but informative. This will help the AI remember context."""

    try:
        # Use Nova Micro for memory log (fast, high rate limit)
        return await bedrock_claude_message(
            messages=[{"role": "user", "content": prompt}],
            model=NOVA_MICRO_MODEL,  # Fast batch processing
            max_tokens=1024,
        )
    except Exception as e:
        print(f"[RLM] Memory log error: {e}")
        return f"# Memory Log - {date.today()}\n\nFailed to generate: {e}"


@app.post("/process-import")
async def process_import(request: ProcessImportRequest, background_tasks: BackgroundTasks):
    """
    Start background processing of ChatGPT export.

    Receives raw conversations_json and processes in background:
    1. Parse ChatGPT JSON format
    2. Run recursive Haiku/Sonnet synthesis
    3. Generate SoulPrint files
    4. Save to Supabase user_profiles
    5. Notify Vercel when complete (triggers email)
    """
    print(f"[RLM] Received process-import request for user {request.user_id}")

    if not request.conversations_json:
        raise HTTPException(status_code=400, detail="conversations_json is required")

    # Quick validation
    try:
        convs = json.loads(request.conversations_json)
        conv_count = len(convs) if isinstance(convs, list) else 0
        print(f"[RLM] Validated JSON: {conv_count} conversations")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    # Start background processing
    background_tasks.add_task(
        process_import_background,
        request.user_id,
        request.user_email,
        request.conversations_json
    )

    return {
        "status": "processing",
        "message": "SoulPrint generation started. You'll be notified when complete.",
        "user_id": request.user_id,
        "conversations_received": conv_count,
    }


# ============================================================================
# EXISTING ENDPOINTS (kept for compatibility)
# ============================================================================

async def embed_query(query: str) -> Optional[List[float]]:
    """Embed a query - prefer Bedrock, fallback to Cohere"""
    # Try Bedrock first
    embedding = await embed_text_bedrock(query)
    if embedding:
        return embedding

    # Fallback to Cohere
    if not COHERE_API_KEY:
        return None

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.cohere.ai/v1/embed",
                headers={
                    "Authorization": f"Bearer {COHERE_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "texts": [query],
                    "model": "embed-english-v3.0",
                    "input_type": "search_query",
                },
                timeout=30.0,
            )
            if response.status_code == 200:
                data = response.json()
                return data["embeddings"][0]
    except Exception as e:
        print(f"[RLM] Cohere embed exception: {e}")

    return None


async def vector_search_chunks(user_id: str, query_embedding: List[float], limit: int = 50) -> List[dict]:
    """Search chunks by vector similarity using Supabase RPC"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{SUPABASE_URL}/rest/v1/rpc/match_conversation_chunks",
                headers={
                    "apikey": SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "query_embedding": query_embedding,
                    "match_user_id": user_id,
                    "match_count": limit,
                    "match_threshold": 0.3,
                },
                timeout=30.0,
            )
            if response.status_code == 200:
                chunks = response.json()
                print(f"[RLM] Vector search returned {len(chunks)} chunks")
                return chunks
            else:
                print(f"[RLM] Vector search error: {response.status_code} - {response.text}")
                return []
    except Exception as e:
        print(f"[RLM] Vector search exception: {e}")
        return []


async def get_user_data(user_id: str, query: Optional[str] = None) -> tuple[List[dict], Optional[str]]:
    """Fetch conversation chunks and soulprint from Supabase"""
    async with httpx.AsyncClient() as client:
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        }

        chunks = []

        if query:
            query_embedding = await embed_query(query)
            if query_embedding:
                chunks = await vector_search_chunks(user_id, query_embedding, limit=100)

        if not chunks:
            chunks_response = await client.get(
                f"{SUPABASE_URL}/rest/v1/conversation_chunks",
                params={
                    "user_id": f"eq.{user_id}",
                    "select": "conversation_id,title,content,message_count,created_at",
                    "order": "created_at.desc",
                    "limit": "200",
                },
                headers=headers,
            )
            chunks = chunks_response.json() if chunks_response.status_code == 200 else []

        profile_response = await client.get(
            f"{SUPABASE_URL}/rest/v1/user_profiles",
            params={
                "user_id": f"eq.{user_id}",
                "select": "soulprint_text",
            },
            headers=headers,
        )

        profile = profile_response.json()
        soulprint_text = profile[0]["soulprint_text"] if profile else None

        return chunks, soulprint_text


def build_context(chunks: List[dict], soulprint_text: Optional[str], history: List[dict]) -> str:
    """Build the context string for RLM"""
    context_parts = []

    if soulprint_text:
        context_parts.append(f"## User Profile\n{soulprint_text}")

    if chunks:
        context_parts.append("## Relevant Conversation History (by similarity)")
        for i, chunk in enumerate(chunks[:50]):
            similarity = chunk.get('similarity', 'N/A')
            content = chunk.get('content', '')[:2000]
            title = chunk.get('title', 'Untitled')
            context_parts.append(f"\n### [{i+1}] {title} (sim: {similarity})\n{content}")

    if history:
        recent_history = json.dumps(history[-10:], indent=2)
        context_parts.append(f"## Current Conversation\n{recent_history}")

    return "\n\n".join(context_parts)


@app.get("/health")
async def health():
    """Health check with processor validation for Render auto-restart."""
    health_status = {
        "status": "ok",
        "service": "soulprint-rlm",
        "rlm_available": RLM_AVAILABLE,
        "bedrock_available": BEDROCK_AVAILABLE and bool(AWS_ACCESS_KEY_ID),
        "timestamp": datetime.utcnow().isoformat(),
    }

    # Validate processor imports (lightweight check)
    try:
        from processors.full_pass import run_full_pass_pipeline
        health_status["processors_available"] = True
    except ImportError as e:
        health_status["processors_available"] = False
        health_status["processor_error"] = str(e)
        # Return 503 to trigger Render auto-restart
        raise HTTPException(status_code=503, detail=f"Processor modules unavailable: {e}")

    return health_status


@app.get("/health-deep")
async def health_deep():
    """
    Deep health check - verifies:
    1. Supabase DB connection
    2. Vector search functions exist (match_conversation_chunks, match_conversation_chunks_by_tier)
    3. Bedrock embeddings work
    4. Nova Lite chat model works

    Use this to verify the system is fully operational.
    """
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {},
        "overall": "healthy",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "Content-Type": "application/json",
        }

        # 1. Check DB connection
        try:
            db_resp = await client.get(
                f"{SUPABASE_URL}/rest/v1/user_profiles",
                params={"select": "user_id", "limit": "1"},
                headers=headers,
            )
            results["checks"]["db_connection"] = {
                "status": "ok" if db_resp.status_code == 200 else "error",
                "http_status": db_resp.status_code,
            }
        except Exception as e:
            results["checks"]["db_connection"] = {"status": "error", "error": str(e)[:100]}
            results["overall"] = "unhealthy"

        # 2. Check match_conversation_chunks function
        try:
            # Call with dummy embedding (all zeros) - should return empty but not error
            dummy_embedding = [0.0] * 1024
            func_resp = await client.post(
                f"{SUPABASE_URL}/rest/v1/rpc/match_conversation_chunks",
                headers=headers,
                json={
                    "query_embedding": dummy_embedding,
                    "match_user_id": "00000000-0000-0000-0000-000000000000",
                    "match_count": 1,
                    "match_threshold": 0.99,
                },
            )
            if func_resp.status_code == 200:
                results["checks"]["match_conversation_chunks"] = {"status": "ok"}
            else:
                results["checks"]["match_conversation_chunks"] = {
                    "status": "error",
                    "http_status": func_resp.status_code,
                    "error": func_resp.text[:200],
                }
                results["overall"] = "unhealthy"
        except Exception as e:
            results["checks"]["match_conversation_chunks"] = {"status": "error", "error": str(e)[:100]}
            results["overall"] = "unhealthy"

        # 3. Check match_conversation_chunks_by_tier function
        try:
            tier_resp = await client.post(
                f"{SUPABASE_URL}/rest/v1/rpc/match_conversation_chunks_by_tier",
                headers=headers,
                json={
                    "query_embedding": dummy_embedding,
                    "match_user_id": "00000000-0000-0000-0000-000000000000",
                    "match_tier": "macro",
                    "match_count": 1,
                    "match_threshold": 0.99,
                },
            )
            if tier_resp.status_code == 200:
                results["checks"]["match_conversation_chunks_by_tier"] = {"status": "ok"}
            else:
                results["checks"]["match_conversation_chunks_by_tier"] = {
                    "status": "error",
                    "http_status": tier_resp.status_code,
                    "error": tier_resp.text[:200],
                }
                results["overall"] = "unhealthy"
        except Exception as e:
            results["checks"]["match_conversation_chunks_by_tier"] = {"status": "error", "error": str(e)[:100]}
            results["overall"] = "unhealthy"

    # 4. Check Bedrock embeddings
    try:
        test_embedding = await embed_text_bedrock("health check test")
        if test_embedding and len(test_embedding) == 1024:
            results["checks"]["bedrock_embeddings"] = {
                "status": "ok",
                "dimensions": len(test_embedding),
            }
        else:
            results["checks"]["bedrock_embeddings"] = {
                "status": "error",
                "error": "No embedding returned or wrong dimensions",
            }
            results["overall"] = "unhealthy"
    except Exception as e:
        results["checks"]["bedrock_embeddings"] = {"status": "error", "error": str(e)[:100]}
        results["overall"] = "unhealthy"

    # 5. Check Nova Lite model (quick test)
    try:
        test_resp = await bedrock_claude_message(
            messages=[{"role": "user", "content": "Say 'ok' and nothing else."}],
            model=NOVA_LITE_MODEL,
            max_tokens=10,
        )
        if test_resp and "ok" in test_resp.lower():
            results["checks"]["nova_lite_model"] = {"status": "ok"}
        else:
            results["checks"]["nova_lite_model"] = {
                "status": "warning",
                "response": test_resp[:50] if test_resp else "empty",
            }
    except Exception as e:
        results["checks"]["nova_lite_model"] = {"status": "error", "error": str(e)[:100]}
        # Don't mark as unhealthy for chat model - embeddings are more critical

    return results


@app.get("/test-embed")
async def test_embed():
    """Test Bedrock embedding with a simple string"""
    test_text = "Hello, this is a test message for embedding."

    try:
        embedding = await embed_text_bedrock(test_text)
        if embedding:
            return {
                "success": True,
                "model": BEDROCK_EMBEDDING_MODEL_ID,
                "dimensions": len(embedding),
                "sample": embedding[:5],
            }
        else:
            return {
                "success": False,
                "error": "No embedding returned",
                "model": BEDROCK_EMBEDDING_MODEL_ID,
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "model": BEDROCK_EMBEDDING_MODEL_ID,
        }


@app.get("/test-patch")
async def test_patch():
    """
    Test the full embedding PATCH flow:
    1. Insert a test chunk
    2. Generate embedding with Bedrock
    3. PATCH embedding to Supabase
    4. Verify it was saved
    5. Clean up
    """
    test_user_id = "00000000-0000-0000-0000-000000000000"  # Dummy user
    test_chunk_id = None

    async with httpx.AsyncClient(timeout=60.0) as client:
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }

        try:
            # Step 1: Insert test chunk
            insert_resp = await client.post(
                f"{SUPABASE_URL}/rest/v1/conversation_chunks",
                headers=headers,
                json={
                    "user_id": test_user_id,
                    "conversation_id": "test-conv",
                    "title": "Test Chunk",
                    "content": "This is a test chunk for verifying the embedding PATCH flow works correctly.",
                    "chunk_tier": "micro",
                    "created_at": datetime.now().isoformat(),
                },
            )
            if insert_resp.status_code not in (200, 201):
                return {"success": False, "step": "insert", "error": insert_resp.text[:200]}

            inserted = insert_resp.json()
            test_chunk_id = inserted[0]["id"] if inserted else None
            if not test_chunk_id:
                return {"success": False, "step": "insert", "error": "No chunk ID returned"}

            # Step 2: Generate embedding
            embedding = await embed_text_bedrock("This is a test chunk for verifying the embedding PATCH flow works correctly.")
            if not embedding:
                return {"success": False, "step": "embed", "error": "Bedrock returned no embedding"}

            # Step 3: PATCH embedding
            patch_resp = await client.patch(
                f"{SUPABASE_URL}/rest/v1/conversation_chunks",
                params={"id": f"eq.{test_chunk_id}"},
                headers=headers,
                json={"embedding": embedding},
            )

            if patch_resp.status_code not in (200, 204):
                return {"success": False, "step": "patch", "error": f"Status {patch_resp.status_code}: {patch_resp.text[:200]}"}

            patch_result = patch_resp.json() if patch_resp.text else []
            if not patch_result:
                return {"success": False, "step": "patch", "error": "PATCH returned empty - row not updated"}

            # Step 4: Verify embedding was saved
            verify_resp = await client.get(
                f"{SUPABASE_URL}/rest/v1/conversation_chunks",
                params={"id": f"eq.{test_chunk_id}", "select": "id,embedding"},
                headers=headers,
            )
            verify_result = verify_resp.json() if verify_resp.status_code == 200 else []

            if not verify_result or not verify_result[0].get("embedding"):
                return {"success": False, "step": "verify", "error": "Embedding not found after PATCH"}

            saved_dims = len(verify_result[0]["embedding"])

            return {
                "success": True,
                "message": "Full PATCH flow works!",
                "chunk_id": test_chunk_id,
                "embedding_dims": len(embedding),
                "saved_dims": saved_dims,
            }

        finally:
            # Step 5: Clean up test chunk
            if test_chunk_id:
                await client.delete(
                    f"{SUPABASE_URL}/rest/v1/conversation_chunks",
                    params={"id": f"eq.{test_chunk_id}"},
                    headers=headers,
                )


def detect_query_intent(message: str, history: list = None) -> Tuple[str, Optional[str]]:
    """
    Smart routing - detect what kind of response the user wants:
    - 'memory': User explicitly asking about their past/history
    - 'memory_accept': User accepting a memory offer from previous message
    - 'realtime': User asking about current events, prices, news
    - 'normal': General question - answer directly, offer memory if relevant

    Returns: (intent, topic_override) - topic_override is set when accepting memory offer
    """
    msg_lower = message.lower()
    history = history or []

    # Check if user is ACCEPTING a memory offer
    # Look for short affirmative responses that indicate "yes show me"
    accept_triggers = [
        'yes', 'yeah', 'yep', 'sure', 'ok', 'okay', 'please', 'pls',
        'show me', 'tell me', 'go ahead', 'do it', 'yes please', 'yes pls',
        'yea', 'ya', 'yup', 'uh huh', 'absolutely', 'definitely',
    ]

    # Only check for acceptance if message is short (likely a response)
    if len(msg_lower.split()) <= 5:
        is_acceptance = any(trigger in msg_lower for trigger in accept_triggers)

        if is_acceptance and history:
            # Find last assistant message and check for memory offer
            for msg in reversed(history):
                if msg.get('role') == 'assistant':
                    content = msg.get('content', '')
                    # Check if it had the memory offer format
                    if '💭' in content and 'past conversations about' in content:
                        # Extract the topic from the memory offer
                        # Format: "I found X past conversations about TOPIC - want me..."
                        match = re.search(r'past conversations about ([^-]+)', content)
                        if match:
                            topic = match.group(1).strip()
                            print(f"[RLM] Memory offer accepted! Topic: '{topic}'")
                            return ('memory_accept', topic)
                    break

    # MEMORY MODE - user explicitly asking about their past
    memory_triggers = [
        'what did i say', 'what have i said', 'did i mention', 'did i talk about',
        'in the past', 'in my history', 'remember when', 'do you remember',
        'what i said about', 'my thoughts on', 'my opinion on',
        'what do i think about', 'what did i think', 'have i ever',
        'tell me about me', 'what do you know about me',
        'my past conversations', 'from our conversations',
        'based on what i said', 'according to my history',
    ]
    if any(trigger in msg_lower for trigger in memory_triggers):
        return ('memory', None)

    # REALTIME MODE - user asking about current/live data
    realtime_triggers = [
        'price of', 'stock price', 'crypto price', 'bitcoin price',
        'weather', 'forecast', 'temperature',
        'latest news', 'current news', 'today\'s', 'right now',
        'happening now', 'live', 'real-time', 'realtime',
        'score', 'game score', 'who won',
    ]
    if any(trigger in msg_lower for trigger in realtime_triggers):
        return ('realtime', None)

    # Default to normal mode
    return ('normal', None)


def build_rlm_system_prompt(
    ai_name: str,
    sections: Optional[dict],
    soulprint_text: Optional[str],
    conversation_context: str,
    web_search_context: Optional[str] = None,
) -> str:
    """Build a high-quality system prompt from structured sections."""
    now = datetime.utcnow()
    date_str = now.strftime("%A, %B %d, %Y")
    time_str = now.strftime("%I:%M %p UTC")

    prompt = f"""# {ai_name}

You have memories of this person — things they've said, how they think, what they care about. Use them naturally. Don't announce that you have memories. Don't offer to "show" or "look up" memories. Just know them like a friend would.

Be direct. Have opinions. Push back when you disagree. Don't hedge everything. If you don't know something, say so.

NEVER start responses with greetings like "Hey", "Hi", "Hello", "Hey there", "Great question", or any pleasantries. Jump straight into substance. Talk like a person, not a chatbot.

Today is {date_str}, {time_str}."""

    # Add structured sections if available — these define who this AI is and who the user is
    if sections:
        soul = clean_section(sections.get("soul"))
        identity_raw = sections.get("identity")
        # Remove ai_name from identity before cleaning (preserving existing behavior)
        if isinstance(identity_raw, dict):
            identity_raw = {k: v for k, v in identity_raw.items() if k != "ai_name"}
        identity = clean_section(identity_raw)
        user_info = clean_section(sections.get("user"))
        agents = clean_section(sections.get("agents"))
        tools = clean_section(sections.get("tools"))
        memory = sections.get("memory")

        has_sections = any([soul, identity, user_info, agents, tools])

        if has_sections:
            soul_md = format_section("SOUL", soul)
            identity_md = format_section("IDENTITY", identity)
            user_md = format_section("USER", user_info)
            agents_md = format_section("AGENTS", agents)
            tools_md = format_section("TOOLS", tools)

            if soul_md:
                prompt += f"\n\n{soul_md}"
            if identity_md:
                prompt += f"\n\n{identity_md}"
            if user_md:
                prompt += f"\n\n{user_md}"
            if agents_md:
                prompt += f"\n\n{agents_md}"
            if tools_md:
                prompt += f"\n\n{tools_md}"

            if memory and isinstance(memory, str) and memory.strip():
                prompt += f"\n\n## MEMORY\n{memory}"
        elif soulprint_text:
            prompt += f"\n\n## ABOUT THIS PERSON\n{soulprint_text}"
    elif soulprint_text:
        prompt += f"\n\n## ABOUT THIS PERSON\n{soulprint_text}"

    if conversation_context:
        prompt += f"\n\n## CONTEXT\n{conversation_context}"

    if web_search_context:
        prompt += f"\n\n## WEB SEARCH RESULTS\n{web_search_context}"

    return prompt


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Main query endpoint with SMART ROUTING"""
    start = time.time()

    if not RLM_AVAILABLE:
        await alert_drew(f"Query failed - RLM not available\nUser: {request.user_id}")
        raise HTTPException(
            status_code=503,
            detail="Memory service unavailable. Please try again later."
        )

    try:
        # SMART ROUTING - detect what the user wants (now context-aware)
        intent, topic_override = detect_query_intent(request.message, request.history)
        print(f"[RLM] Query intent: {intent} | Topic override: {topic_override} | Message: '{request.message[:50]}...'")

        # Extract AI name for prompt builder
        ai_name = request.ai_name or "SoulPrint"

        # Get soulprint for personality context (always useful)
        soulprint_data = await get_soulprint(request.user_id)
        soulprint = request.soulprint_text or (soulprint_data.get('soulprint_text') if soulprint_data else None)

        chunks = []

        if intent == 'memory' or intent == 'memory_accept':
            # MEMORY MODE - Deep dive into their history
            # Use topic_override if accepting an offer, otherwise use the message
            search_query = topic_override if topic_override else request.message
            chunks = await search_memories(request.user_id, search_query, limit=50)
            context = build_context(chunks, soulprint, request.history or [])
            limited_context = context[:25000] if len(context) > 25000 else context

            # Build base prompt with personality sections
            base_prompt = build_rlm_system_prompt(
                ai_name=ai_name,
                sections=request.sections,
                soulprint_text=soulprint,
                conversation_context="",  # We'll add memory-specific context below
                web_search_context=None,
            )

            # Customize the prompt based on whether it's an offer acceptance
            intro = f"The user accepted your offer to show their past conversations about '{topic_override}'." if intent == 'memory_accept' else "The user wants to know what they said/discussed before."

            # Append memory-specific instructions to base prompt
            system_prompt = f"""{base_prompt}

## CONVERSATION HISTORY
The following are REAL conversations from the user's ChatGPT history. Use ONLY this data:

{limited_context}

## YOUR TASK
{intro}

Look through the conversation history above and tell them what you found. You MUST:
- Quote their EXACT words from the history above (copy-paste, don't paraphrase)
- Include the actual dates/timestamps shown in the history
- If multiple conversations mention the topic, summarize the pattern

IMPORTANT:
- ONLY use information from the conversation history above
- If the history above doesn't contain relevant conversations, say "I didn't find any conversations about this in your history"
- DO NOT make up quotes or dates
- DO NOT output template placeholders like [date] or [topic]"""

        elif intent == 'realtime':
            # REALTIME MODE - Current data, minimal memory
            # Build base prompt with personality sections and web search context
            system_prompt = build_rlm_system_prompt(
                ai_name=ai_name,
                sections=request.sections,
                soulprint_text=soulprint,
                conversation_context="",
                web_search_context=request.web_search_context,
            )

            # Append realtime-specific instructions
            if request.web_search_context:
                system_prompt += f"""

## REALTIME MODE
The user is asking about current/live information.
- Use the web search results provided above and cite sources
- Keep response focused on their specific question"""
            else:
                system_prompt += f"""

## REALTIME MODE
The user is asking about current/live information.
- No real-time data is available
- Answer based on your knowledge but mention your knowledge cutoff (January 2025)
- Keep response focused on their specific question"""

        else:
            # NORMAL MODE - Answer directly, then offer memory check
            # Do a quick memory search to see if there's relevant history
            chunks = await search_memories(request.user_id, request.message, limit=10)

            # Build base prompt with personality sections
            system_prompt = build_rlm_system_prompt(
                ai_name=ai_name,
                sections=request.sections,
                soulprint_text=soulprint,
                conversation_context="",
                web_search_context=None,
            )

            # If relevant memories found, include them as context
            if chunks:
                memory_context = "\n".join([
                    f"- {c.get('title', 'Untitled')}: {c.get('content', '')[:200]}"
                    for c in chunks[:5]
                ])
                system_prompt += f"\n\n## RELEVANT MEMORIES\n{memory_context}"

        # Generate response
        response_text = await bedrock_claude_message(
            messages=[{"role": "user", "content": request.message}],
            system=system_prompt,
            model=NOVA_LITE_MODEL,
            max_tokens=2048,
        )

        latency = int((time.time() - start) * 1000)
        print(f"[RLM] Response generated in {latency}ms | Intent: {intent} | Chunks: {len(chunks)}")

        return QueryResponse(
            response=response_text,
            chunks_used=len(chunks),
            method=f"rlm-{intent}",
            latency_ms=latency,
        )

    except Exception as e:
        error_msg = str(e)
        await alert_drew(f"RLM Query Error\n\nUser: {request.user_id}\nError: {error_msg[:500]}")
        raise HTTPException(status_code=500, detail=f"Memory query failed: {error_msg}")


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    """Deep personality analysis using RLM on conversation history"""
    start = time.time()

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            headers = {
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            }

            response = await client.get(
                f"{SUPABASE_URL}/rest/v1/raw_conversations",
                params={
                    "user_id": f"eq.{request.user_id}",
                    "select": "title,messages,message_count,created_at",
                    "order": "created_at.desc",
                    "limit": "500",
                },
                headers=headers,
            )

            if response.status_code != 200:
                raise HTTPException(status_code=500, detail="Failed to fetch conversations")

            conversations = response.json()

            if not conversations:
                return {"error": "No conversations found", "profile": None}

            recent = conversations[:50]
            oldest = conversations[-50:] if len(conversations) > 50 else []
            by_length = sorted(conversations, key=lambda c: c.get("message_count", 0), reverse=True)
            longest = by_length[:100]

            seen = set()
            sampled = []
            for conv in recent + oldest + longest:
                conv_id = conv.get("title", "") + str(conv.get("created_at", ""))
                if conv_id not in seen:
                    seen.add(conv_id)
                    sampled.append(conv)

            sample_text = ""
            for conv in sampled[:100]:
                title = conv.get("title", "Untitled")
                messages = conv.get("messages", [])[:10]
                msg_text = "\n".join([f"- {m.get('role', 'unknown')}: {m.get('content', '')[:200]}" for m in messages])
                sample_text += f"\n\n### {title}\n{msg_text}"

            analysis_prompt = f"""Analyze this user's conversation history and create a detailed personality profile.

{sample_text[:30000]}

Based on these conversations, provide a JSON analysis with:
1. archetype: Their primary personality archetype
2. tone: How they communicate
3. humor: Their sense of humor
4. interests: Top 10 interests/topics
5. communication_style: How they prefer to receive information
6. key_traits: 5 defining personality traits
7. avoid: Things the AI should avoid

Return ONLY valid JSON, no markdown."""

            # Use Nova Pro for internal analysis
            analysis_text = await bedrock_claude_message(
                messages=[{"role": "user", "content": analysis_prompt}],
                model=NOVA_PRO_MODEL,  # Higher rate limits
                max_tokens=2048,
            )

            try:
                json_match = re.search(r'\{[\s\S]*\}', analysis_text)
                if json_match:
                    profile = json.loads(json_match.group())
                else:
                    profile = {"raw": analysis_text}
            except json.JSONDecodeError:
                profile = {"raw": analysis_text}

            latency = int((time.time() - start) * 1000)

            return {
                "success": True,
                "profile": profile,
                "conversations_analyzed": len(sampled),
                "latency_ms": latency,
            }

    except Exception as e:
        error_msg = str(e)
        print(f"[RLM] Analyze error: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {error_msg}")


@app.post("/create-soulprint")
async def create_soulprint(request: CreateSoulprintRequest):
    """
    EXHAUSTIVE soulprint generation with full SoulPrint files.

    Returns:
    - soulprint: JSON profile with archetype, core_essence, voice, mind, heart, world
    - soul_md: SOUL.md content
    - identity_md: IDENTITY.md content
    - agents_md: AGENTS.md content
    - user_md: USER.md content
    - memory_log: Today's memory log
    """
    try:
        conversations = request.conversations
        stats = request.stats or {}
        user_id = request.user_id

        if not conversations:
            return {"error": "No conversations provided", "soulprint": None, "archetype": None}

        print(f"[RLM] EXHAUSTIVE soulprint for user {user_id} from {len(conversations)} conversations")

        # Convert conversations to message format
        messages = []
        for conv in conversations:
            title = conv.get("title", "Untitled")
            created_at = conv.get("createdAt") or conv.get("created_at")
            conv_messages = conv.get("messages", [])
            for m in conv_messages:
                if isinstance(m, dict) and m.get("role") == "user":
                    messages.append({
                        "content": m.get("content", ""),
                        "conversation_title": title,
                        "original_timestamp": created_at,
                    })

        # Run recursive synthesis to get personality profile
        print(f"[RLM] Running recursive synthesis on {len(messages)} messages...")
        profile = await recursive_synthesize(messages[:5000], user_id)
        archetype = profile.get("archetype", "Unique Individual")

        # Generate the 4 SoulPrint markdown files
        print("[RLM] Generating SoulPrint files (SOUL.md, IDENTITY.md, AGENTS.md, USER.md)...")
        soul_files = await generate_soulprint_files(profile, messages, user_id)

        # Generate today's memory log
        print("[RLM] Generating memory log...")
        memory_log = await generate_memory_log(messages, profile, user_id)

        # Save to Supabase if configured
        if SUPABASE_URL and SUPABASE_SERVICE_KEY:
            print("[RLM] Saving SoulPrint to Supabase...")
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    headers = {
                        "apikey": SUPABASE_SERVICE_KEY,
                        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                        "Content-Type": "application/json",
                        "Prefer": "resolution=merge-duplicates",
                    }

                    # Update user_profiles with full soulprint data
                    await client.post(
                        f"{SUPABASE_URL}/rest/v1/user_profiles",
                        headers=headers,
                        json={
                            "user_id": user_id,
                            "soulprint": profile,
                            "soulprint_text": profile.get("core_essence", archetype),
                            "archetype": archetype,
                            "soul_md": soul_files.get("soul_md"),
                            "identity_md": soul_files.get("identity_md"),
                            "agents_md": soul_files.get("agents_md"),
                            "user_md": soul_files.get("user_md"),
                            "soulprint_generated_at": datetime.utcnow().isoformat(),
                            "updated_at": datetime.utcnow().isoformat(),
                        },
                    )
                    print(f"[RLM] SoulPrint saved for user {user_id}")
            except Exception as e:
                print(f"[RLM] Failed to save to Supabase: {e}")

        return {
            "soulprint": profile,
            "archetype": archetype,
            "soul_md": soul_files.get("soul_md"),
            "identity_md": soul_files.get("identity_md"),
            "agents_md": soul_files.get("agents_md"),
            "user_md": soul_files.get("user_md"),
            "memory_log": memory_log,
            "conversations_analyzed": len(conversations),
        }

    except Exception as e:
        error_msg = str(e)
        print(f"[RLM] Create soulprint error: {error_msg}")
        await alert_drew(f"Create Soulprint Error\n\nUser: {request.user_id}\nError: {error_msg[:500]}")
        raise HTTPException(status_code=500, detail=f"Soulprint creation failed: {error_msg}")


class ProcessFullRequest(BaseModel):
    user_id: str
    storage_path: Optional[str] = None  # Path to parsed JSON in Supabase Storage
    conversation_count: Optional[int] = None
    message_count: Optional[int] = None
    conversations: Optional[List[dict]] = None  # Legacy: direct conversations (for small imports)


@app.post("/process-full")
async def process_full(request: ProcessFullRequest, background_tasks: BackgroundTasks, response: Response):
    """
    DEPRECATED: Full processing pipeline: Create chunks → Embed → Generate SoulPrint.

    This endpoint is deprecated. Use /process-full-v2 instead.
    Sunset date: March 1, 2026.

    Called by Vercel after parsing ZIP. Runs in background - no timeout.

    Supports two modes:
    1. storage_path: RLM downloads parsed JSON from Supabase Storage (scalable, 10k+ convos)
    2. conversations: Legacy direct passing (for small imports only)

    Jobs are tracked in processing_jobs table for recovery after server restarts.
    """
    # Deprecation headers (RFC 8594)
    response.headers["Deprecation"] = "true"
    response.headers["Sunset"] = "Sat, 01 Mar 2026 00:00:00 GMT"
    response.headers["Link"] = '</process-full-v2>; rel="alternate"'

    # Log deprecation usage
    print(f"[DEPRECATED] /process-full called by user {request.user_id}")

    print(f"[RLM] Received process-full request for user {request.user_id}")

    job_id = None
    if request.storage_path:
        print(f"[RLM] Storage path: {request.storage_path} ({request.conversation_count} convos, {request.message_count} msgs)")
        # Create job record for recovery
        job_id = await create_job(
            request.user_id,
            request.storage_path,
            request.conversation_count or 0,
            request.message_count or 0,
        )
        if job_id:
            print(f"[RLM] Created job {job_id}")
    elif request.conversations:
        print(f"[RLM] Direct conversations: {len(request.conversations)}")
    else:
        print(f"[RLM] No conversations provided")

    # Start background processing
    background_tasks.add_task(
        process_full_background,
        request.user_id,
        request.storage_path,
        request.conversations,
        job_id,
    )

    return {
        "status": "processing",
        "message": "Full processing started: chunking → embedding → soulprint.",
        "user_id": request.user_id,
        "conversation_count": request.conversation_count,
        "job_id": job_id,
        "deprecation_notice": "This endpoint is deprecated. Use /process-full-v2 instead. Sunset: March 1, 2026.",
    }


async def run_full_pass_v2_background(
    user_id: str,
    storage_path: str,
    job_id: Optional[str] = None,
):
    """Background task wrapper for v2 pipeline (processors from Phase 2)."""
    from adapters.supabase_adapter import update_user_profile as update_profile_status

    try:
        # Mark pipeline as processing
        await update_profile_status(user_id, {
            "full_pass_status": "processing",
            "full_pass_started_at": datetime.utcnow().isoformat(),
            "full_pass_error": None,
        })

        from processors.full_pass import run_full_pass_pipeline

        print(f"[v2] Starting pipeline for user {user_id}")
        print(f"[v2] Storage path: {storage_path}")

        memory_md = await run_full_pass_pipeline(
            user_id=user_id,
            storage_path=storage_path,
        )

        print(f"[v2] Pipeline complete for user {user_id}")

        # Mark pipeline as complete
        await update_profile_status(user_id, {
            "full_pass_status": "complete",
            "full_pass_completed_at": datetime.utcnow().isoformat(),
        })

        if job_id:
            await complete_job(job_id, success=True)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[v2] user_id={user_id} step=pipeline status=failed error={error_msg}")
        import traceback
        traceback.print_exc()

        # Mark pipeline as failed with error context
        await update_profile_status(user_id, {
            "full_pass_status": "failed",
            "full_pass_error": f"Pipeline failed: {error_msg[:500]}",
        })

        if job_id:
            await complete_job(job_id, success=False, error_message=str(e)[:500])


@app.post("/process-full-v2")
async def process_full_v2(request: ProcessFullRequest, background_tasks: BackgroundTasks):
    """
    V2 full processing pipeline using modular processors from v1.2.

    Pipeline: chunk conversations -> extract facts (parallel) -> consolidate ->
    generate MEMORY section -> regenerate v2 sections (SOUL, IDENTITY, USER, AGENTS, TOOLS)

    Uses processors/ modules (Phase 2) instead of inline main.py logic.
    Runs alongside v1 /process-full for gradual migration.
    """
    print(f"[v2] Received process-full-v2 request for user {request.user_id}")

    if not request.storage_path:
        raise HTTPException(
            status_code=400,
            detail="storage_path required for v2 pipeline (direct conversations not supported)"
        )

    # Create job record for recovery (reuses existing job system)
    job_id = await create_job(
        request.user_id,
        request.storage_path,
        request.conversation_count or 0,
        request.message_count or 0,
    )

    if job_id:
        print(f"[v2] Created job {job_id}")

    # Dispatch to background
    background_tasks.add_task(
        run_full_pass_v2_background,
        request.user_id,
        request.storage_path,
        job_id,
    )

    return {
        "status": "processing",
        "version": "v2",
        "message": "v2 pipeline started: chunk → facts → MEMORY → v2 sections",
        "user_id": request.user_id,
        "conversation_count": request.conversation_count,
        "job_id": job_id,
    }


class EmbedChunksRequest(BaseModel):
    user_id: str


@app.post("/embed-chunks")
async def embed_chunks(request: EmbedChunksRequest, background_tasks: BackgroundTasks):
    """
    Embed existing chunks for a user using Bedrock Titan.
    Use /process-full for full pipeline instead.
    """
    print(f"[RLM] Received embed-chunks request for user {request.user_id}")

    background_tasks.add_task(
        embed_chunks_background,
        request.user_id
    )

    return {
        "status": "processing",
        "message": "Embedding started. SoulPrint will be generated after.",
        "user_id": request.user_id,
    }


async def process_full_background(user_id: str, storage_path: Optional[str], conversations: Optional[List[dict]] = None, job_id: Optional[str] = None):
    """
    Full background pipeline: Create chunks → Embed → Generate SoulPrint.

    Multi-tier chunking:
    - micro: 200 chars (precise facts, names, dates)
    - medium: 2000 chars (conversation context)
    - macro: 5000 chars (themes, relationships)

    Supports:
    - storage_path: Download parsed JSON from Supabase Storage (scalable)
    - conversations: Direct list (legacy, for small imports)
    - job_id: For progress tracking and recovery
    """
    start = time.time()
    print(f"[RLM] Starting full processing for user {user_id}" + (f" (job {job_id})" if job_id else ""))

    try:
        # Mark job as processing
        if job_id:
            await update_job(job_id, status="processing", current_step="starting", progress=0)

        async with httpx.AsyncClient(timeout=600.0) as client:
            headers = {
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type": "application/json",
            }

            # Update status
            await client.patch(
                f"{SUPABASE_URL}/rest/v1/user_profiles",
                params={"user_id": f"eq.{user_id}"},
                headers=headers,
                json={"import_status": "processing", "embedding_status": "pending"},
            )

            # ============================================================
            # STEP 0: Get conversations (from storage or direct)
            # ============================================================
            if storage_path:
                if job_id:
                    await update_job(job_id, current_step="downloading", progress=5)
                print(f"[RLM] Downloading parsed JSON from storage: {storage_path}")
                # Parse bucket and path
                path_parts = storage_path.split("/", 1)
                bucket = path_parts[0]
                file_path = path_parts[1] if len(path_parts) > 1 else ""

                # Download from Supabase Storage
                download_url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{file_path}"
                download_resp = await client.get(download_url, headers=headers)

                if download_resp.status_code != 200:
                    print(f"[RLM] Storage download failed: {download_resp.status_code} - {download_resp.text[:200]}")
                    raise Exception(f"Failed to download from storage: {download_resp.status_code}")

                conversations = download_resp.json()
                print(f"[RLM] Downloaded {len(conversations)} conversations from storage")
                # Keep raw JSON in storage for future memory upgrades

            if not conversations:
                print(f"[RLM] No conversations provided, checking existing chunks...")
                await embed_chunks_background(user_id)
                return

            # ============================================================
            # STEP 1: Parse raw ChatGPT format and create multi-tier chunks
            # ============================================================
            if job_id:
                await update_job(job_id, current_step="chunking", progress=15)
            print(f"[RLM] Parsing {len(conversations)} raw ChatGPT conversations...")

            all_chunks = []
            total_messages = 0

            def parse_chatgpt_conversation(conv: dict) -> tuple:
                """
                Parse raw ChatGPT format with mapping structure.
                Returns (messages_with_timestamps, conv_created_at)

                Raw format has:
                - mapping: { node_id: { message: { author, content, create_time } } }
                - create_time: Unix timestamp for conversation
                """
                messages = []

                # Get conversation-level timestamp
                conv_create_time = conv.get("create_time")
                if conv_create_time:
                    conv_created_at = datetime.fromtimestamp(conv_create_time).isoformat()
                else:
                    conv_created_at = datetime.utcnow().isoformat()

                # Parse mapping structure (ChatGPT's nested format)
                mapping = conv.get("mapping", {})
                for node_id, node in mapping.items():
                    if not isinstance(node, dict):
                        continue

                    msg = node.get("message")
                    if not msg or not isinstance(msg, dict):
                        continue

                    # Get message content
                    content_obj = msg.get("content", {})
                    if not isinstance(content_obj, dict):
                        continue

                    parts = content_obj.get("parts", [])
                    if not parts:
                        continue

                    # Only text content (skip images, files)
                    text_content = ""
                    for part in parts:
                        if isinstance(part, str):
                            text_content += part

                    if not text_content.strip():
                        continue

                    # Get role
                    author = msg.get("author", {})
                    role = author.get("role", "user") if isinstance(author, dict) else "user"

                    # Skip system messages
                    if role == "system":
                        continue

                    # Get message timestamp
                    msg_create_time = msg.get("create_time")
                    if msg_create_time:
                        msg_timestamp = datetime.fromtimestamp(msg_create_time).isoformat()
                    else:
                        msg_timestamp = conv_created_at

                    messages.append({
                        "role": role,
                        "content": text_content,
                        "timestamp": msg_timestamp,
                        "create_time": msg_create_time or 0,  # For sorting
                    })

                # Sort messages by timestamp (mapping is unordered)
                messages.sort(key=lambda m: m["create_time"])

                return messages, conv_created_at

            for idx, conv in enumerate(conversations):  # Process ALL conversations
                title = conv.get("title", "Untitled")
                conv_id = conv.get("id") or conv.get("conversation_id") or f"conv_{idx}"

                # Parse raw ChatGPT format
                messages, conv_created_at = parse_chatgpt_conversation(conv)

                if not messages:
                    continue

                # Build full content WITH timestamps
                full_content = ""
                for m in messages[:30]:  # Max 30 messages per conversation
                    role = m["role"]
                    content = m["content"]
                    timestamp = m["timestamp"]
                    # Include timestamp in content for searchability
                    full_content += f"[{timestamp}] {role}: {content}\n"
                    total_messages += 1

                if not full_content.strip():
                    continue

                is_recent = idx < 100
                message_count = len(messages)

                # MICRO chunks (200 chars) - precise facts
                MICRO_SIZE = 200
                for i in range(0, min(len(full_content), 2000), MICRO_SIZE):
                    chunk_text = full_content[i:i + MICRO_SIZE].strip()
                    if len(chunk_text) > 50:
                        all_chunks.append({
                            "user_id": user_id,
                            "conversation_id": conv_id,
                            "title": title,
                            "content": chunk_text,
                            "chunk_tier": "micro",
                            "message_count": message_count,
                            "created_at": conv_created_at,
                            "is_recent": is_recent,
                        })

                # MEDIUM chunk (2000 chars) - conversation context
                medium_content = full_content[:2000].strip()
                if len(medium_content) > 100:
                    all_chunks.append({
                        "user_id": user_id,
                        "conversation_id": conv_id,
                        "title": title,
                        "content": medium_content,
                        "chunk_tier": "medium",
                        "message_count": message_count,
                        "created_at": conv_created_at,
                        "is_recent": is_recent,
                    })

                # MACRO chunk (5000 chars) - themes, relationships
                macro_content = full_content[:5000].strip()
                if len(macro_content) > 500:
                    all_chunks.append({
                        "user_id": user_id,
                        "conversation_id": conv_id,
                        "title": title,
                        "content": macro_content,
                        "chunk_tier": "macro",
                        "message_count": message_count,
                        "created_at": conv_created_at,
                        "is_recent": is_recent,
                    })

            # Limit chunks: prioritize macro + medium + recent micro
            macro_chunks = [c for c in all_chunks if c["chunk_tier"] == "macro"]
            medium_chunks = [c for c in all_chunks if c["chunk_tier"] == "medium"]
            micro_chunks = [c for c in all_chunks if c["chunk_tier"] == "micro" and c.get("is_recent")][:300]

            chunks_to_insert = macro_chunks + medium_chunks + micro_chunks

            tier_counts = {
                "macro": len(macro_chunks),
                "medium": len(medium_chunks),
                "micro": len(micro_chunks),
            }
            print(f"[RLM] Created {len(chunks_to_insert)} chunks: {tier_counts}")

            # ============================================================
            # STEP 2: Insert chunks into Supabase
            # ============================================================
            print(f"[RLM] Inserting chunks into Supabase...")

            # Clear existing chunks for this user first
            await client.delete(
                f"{SUPABASE_URL}/rest/v1/conversation_chunks",
                params={"user_id": f"eq.{user_id}"},
                headers=headers,
            )

            # Insert in batches
            BATCH_SIZE = 100
            inserted_count = 0
            for i in range(0, len(chunks_to_insert), BATCH_SIZE):
                batch = chunks_to_insert[i:i + BATCH_SIZE]
                # Remove is_recent before inserting (not a DB column)
                for c in batch:
                    c.pop("is_recent", None)

                resp = await client.post(
                    f"{SUPABASE_URL}/rest/v1/conversation_chunks",
                    headers=headers,
                    json=batch,
                )
                if resp.status_code in [200, 201]:
                    inserted_count += len(batch)
                else:
                    print(f"[RLM] Chunk insert error: {resp.text[:200]}")

            print(f"[RLM] Inserted {inserted_count} chunks")

            # Update profile with chunk counts
            await client.patch(
                f"{SUPABASE_URL}/rest/v1/user_profiles",
                params={"user_id": f"eq.{user_id}"},
                headers=headers,
                json={
                    "total_chunks": inserted_count,
                    "total_messages": total_messages,
                    "embedding_status": "pending",
                    "memory_status": "building",  # New field for progressive UX
                },
            )

            # ============================================================
            # STEP 3: Generate SoulPrint FIRST (enables chat immediately)
            # ============================================================
            # SoulPrint uses raw chunks, not embeddings - so we do this first
            # This way user can start chatting while embeddings run in background
            if job_id:
                await update_job(job_id, current_step="synthesizing", progress=40)

            print(f"[RLM] Generating SoulPrint FIRST (for immediate chat access)...")
            soulprint_success = False
            for attempt in range(3):
                try:
                    async with httpx.AsyncClient(timeout=300.0) as soulprint_client:
                        soulprint_headers = {
                            "apikey": SUPABASE_SERVICE_KEY,
                            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                            "Content-Type": "application/json",
                        }
                        await generate_soulprint_from_chunks(user_id, soulprint_client, soulprint_headers)
                    soulprint_success = True
                    print(f"[RLM] SoulPrint ready on attempt {attempt + 1}")
                    break
                except Exception as sp_error:
                    print(f"[RLM] SoulPrint attempt {attempt + 1} failed: {sp_error}")
                    if attempt < 2:
                        delay = 10 * (attempt + 1)  # 10s, 20s
                        print(f"[RLM] Waiting {delay}s before retry...")
                        await asyncio.sleep(delay)

            if not soulprint_success:
                raise Exception("SoulPrint generation failed after 3 attempts")

            # ============================================================
            # STEP 3.5: EARLY NOTIFICATION - User can chat NOW!
            # ============================================================
            # Mark import as complete so user can start chatting
            # Embeddings will continue in background to improve memory search
            soulprint_time = time.time() - start
            completion_patch_resp = await client.patch(
                f"{SUPABASE_URL}/rest/v1/user_profiles",
                params={"user_id": f"eq.{user_id}"},
                headers=headers,
                json={
                    "import_status": "complete",  # User can chat!
                    "embedding_status": "processing",
                    "memory_status": "building",  # Memory still improving
                },
            )

            if completion_patch_resp.status_code not in (200, 204):
                error_msg = f"Failed to mark import complete: {completion_patch_resp.status_code} - {completion_patch_resp.text[:500]}"
                print(f"[RLM] ❌ {error_msg}")
                raise Exception(error_msg)

            # Send email NOW - user can start chatting
            vercel_url = VERCEL_API_URL
            print(f"[RLM] 📧 Sending email notification via {vercel_url}/api/import/complete")
            try:
                callback_resp = await client.post(
                    f"{vercel_url}/api/import/complete",
                    json={
                        "user_id": user_id,
                        "soulprint_ready": True,
                        "memory_building": True,  # Indicate memory is still building
                        "processing_time": soulprint_time,
                    },
                    headers={"Content-Type": "application/json"},
                    timeout=30.0,
                )
                if callback_resp.status_code == 200:
                    resp_data = callback_resp.json()
                    print(f"[RLM] ✅ Email callback success! email_sent={resp_data.get('email_sent')}, push_sent={resp_data.get('push_sent')}")
                else:
                    print(f"[RLM] ⚠️ Vercel callback returned {callback_resp.status_code}: {callback_resp.text[:200]}")
            except Exception as e:
                print(f"[RLM] ❌ Vercel callback failed: {e}")

            await alert_drew(
                f"🚀 User Can Chat Now!\n\n"
                f"User: {user_id}\n"
                f"SoulPrint: ✅ Ready\n"
                f"Time to chat: {soulprint_time:.1f}s\n"
                f"Embeddings: Starting in background..."
            )

            # ============================================================
            # STEP 4: Embed all chunks in BACKGROUND (improves memory over time)
            # ============================================================
            if job_id:
                await update_job(job_id, current_step="embedding", progress=35)
            print(f"[RLM] Embedding {inserted_count} chunks with Bedrock Titan (parallel)...")

            # Fetch chunks we just inserted
            resp = await client.get(
                f"{SUPABASE_URL}/rest/v1/conversation_chunks",
                params={
                    "user_id": f"eq.{user_id}",
                    "select": "id,content",
                    "limit": "10000",  # Support 10k+ conversations
                },
                headers=headers,
            )
            chunks_to_embed = resp.json() if resp.status_code == 200 else []

            bedrock_client = get_bedrock_client()
            embedded_count = 0
            PARALLEL_BATCH_SIZE = 5  # Balanced: 5 concurrent embeddings

            consecutive_failures = 0
            MAX_CONSECUTIVE_FAILURES = 10  # Abort if 10 in a row fail

            async def embed_and_store(chunk: dict) -> bool:
                """Embed a single chunk and store to Supabase."""
                try:
                    embedding = await embed_text_bedrock(chunk["content"], bedrock_client)
                    if not embedding:
                        print(f"[RLM] No embedding returned for chunk {chunk['id']}")
                        return False

                    # Create dedicated client for this PATCH to avoid connection pooling issues
                    async with httpx.AsyncClient(timeout=30.0) as patch_client:
                        # PATCH with return=representation to verify update happened
                        patch_headers = {
                            "apikey": SUPABASE_SERVICE_KEY,
                            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                            "Content-Type": "application/json",
                            "Prefer": "return=representation",
                        }
                        patch_resp = await patch_client.patch(
                            f"{SUPABASE_URL}/rest/v1/conversation_chunks",
                            params={"id": f"eq.{chunk['id']}"},
                            headers=patch_headers,
                            json={"embedding": embedding},  # Send as array, not string
                        )

                        # Check status AND that a row was actually returned
                        if patch_resp.status_code not in (200, 204):
                            print(f"[RLM] PATCH failed for {chunk['id']}: {patch_resp.status_code} - {patch_resp.text[:200]}")
                            return False

                        # Verify row was actually updated (return=representation gives us the row)
                        result = patch_resp.json() if patch_resp.text else []
                        if not result:
                            print(f"[RLM] PATCH returned empty - row {chunk['id']} may not exist or wasn't updated")
                            print(f"[RLM] Response status: {patch_resp.status_code}, body: {patch_resp.text[:500] if patch_resp.text else 'empty'}")
                            return False

                        return True
                except Exception as e:
                    print(f"[RLM] Embed error for {chunk['id']}: {e}")
                return False

            # Log first few chunk IDs for debugging
            if chunks_to_embed:
                first_ids = [c.get("id", "NO_ID") for c in chunks_to_embed[:3]]
                print(f"[RLM] First chunk IDs to embed: {first_ids}")

            # Process in parallel batches
            for batch_start in range(0, len(chunks_to_embed), PARALLEL_BATCH_SIZE):
                batch = chunks_to_embed[batch_start:batch_start + PARALLEL_BATCH_SIZE]
                results = await asyncio.gather(*[embed_and_store(c) for c in batch])

                # Count successes and failures
                batch_successes = sum(results)
                batch_failures = len(results) - batch_successes
                embedded_count += batch_successes

                # Log first batch results for debugging
                if batch_start == 0:
                    print(f"[RLM] First batch results: {results} ({batch_successes}/{len(batch)} succeeded)")

                # Track consecutive failures
                if batch_failures == len(results):
                    consecutive_failures += len(results)
                else:
                    consecutive_failures = 0

                # Abort if too many consecutive failures
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    error_msg = f"Aborting: {consecutive_failures} consecutive embedding failures"
                    print(f"[RLM] {error_msg}")
                    raise Exception(error_msg)

                # Progress update every 50 chunks
                if batch_start % 50 < PARALLEL_BATCH_SIZE:
                    progress = int((batch_start + len(batch)) / len(chunks_to_embed) * 100)
                    await client.patch(
                        f"{SUPABASE_URL}/rest/v1/user_profiles",
                        params={"user_id": f"eq.{user_id}"},
                        headers=headers,
                        json={"embedding_progress": progress, "processed_chunks": embedded_count},
                    )
                    print(f"[RLM] Embedding progress: {progress}% ({embedded_count}/{len(chunks_to_embed)})")

                await asyncio.sleep(0.1)  # Brief pause between batches

            # Verify embeddings were actually saved
            verify_resp = await client.get(
                f"{SUPABASE_URL}/rest/v1/conversation_chunks",
                params={
                    "user_id": f"eq.{user_id}",
                    "embedding": "not.is.null",
                    "select": "id",
                    "limit": "1",
                },
                headers=headers,
            )
            verified_chunks = verify_resp.json() if verify_resp.status_code == 200 else []
            if not verified_chunks:
                raise Exception("Embeddings not saved to database - verification failed")

            print(f"[RLM] Embedded {embedded_count}/{len(chunks_to_embed)} chunks")

            # Check for any missing embeddings and complete them
            if embedded_count < len(chunks_to_embed):
                missing_count = len(chunks_to_embed) - embedded_count
                print(f"[RLM] {missing_count} chunks still need embedding - scheduling completion...")
                asyncio.create_task(complete_embeddings_background(user_id))

            # ============================================================
            # STEP 5: Final status update (embeddings done)
            # ============================================================
            elapsed = time.time() - start
            await client.patch(
                f"{SUPABASE_URL}/rest/v1/user_profiles",
                params={"user_id": f"eq.{user_id}"},
                headers=headers,
                json={
                    "embedding_status": "complete",
                    "embedding_progress": 100,
                    "memory_status": "ready",  # Full memory now available
                    "processed_chunks": embedded_count,
                },
            )

            print(f"[RLM] Full processing complete in {elapsed:.1f}s (user chatting since {soulprint_time:.1f}s)")

            # Mark job complete
            if job_id:
                await complete_job(job_id, success=True)

            await alert_drew(
                f"✅ Full Processing Complete!\n\n"
                f"User: {user_id}\n"
                f"Chunks: {tier_counts}\n"
                f"Embedded: {embedded_count}\n"
                f"Time: {elapsed:.1f}s"
            )

    except Exception as e:
        error_msg = str(e)
        print(f"[RLM] Full processing failed: {error_msg}")

        # Mark job failed
        if job_id:
            await complete_job(job_id, success=False, error_message=error_msg[:500])

        try:
            async with httpx.AsyncClient() as client:
                await client.patch(
                    f"{SUPABASE_URL}/rest/v1/user_profiles",
                    params={"user_id": f"eq.{user_id}"},
                    headers={
                        "apikey": SUPABASE_SERVICE_KEY,
                        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={"import_status": "failed", "import_error": error_msg[:500]},
                )
        except:
            pass

        await alert_drew(f"❌ Full Processing Failed\n\nUser: {user_id}\nError: {error_msg[:500]}")


async def embed_chunks_background(user_id: str):
    """
    Background job to embed existing chunks for a user.
    Use process_full_background for full pipeline.

    Flow:
    1. Fetch all chunks without embeddings
    2. Embed with Bedrock Titan v2
    3. Update chunks in Supabase
    4. Generate SoulPrint
    5. Update user_profiles status
    """
    start = time.time()
    print(f"[RLM] Starting chunk embedding for user {user_id}")

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            headers = {
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type": "application/json",
            }

            # Update status to processing
            await client.patch(
                f"{SUPABASE_URL}/rest/v1/user_profiles",
                params={"user_id": f"eq.{user_id}"},
                headers=headers,
                json={"embedding_status": "processing"},
            )

            # Fetch all chunks without embeddings
            response = await client.get(
                f"{SUPABASE_URL}/rest/v1/conversation_chunks",
                params={
                    "user_id": f"eq.{user_id}",
                    "embedding": "is.null",
                    "select": "id,content,chunk_tier",
                    "limit": "5000",
                },
                headers=headers,
            )

            if response.status_code != 200:
                raise Exception(f"Failed to fetch chunks: {response.text}")

            chunks = response.json()
            total_chunks = len(chunks)
            print(f"[RLM] Found {total_chunks} chunks to embed")

            if total_chunks == 0:
                print(f"[RLM] No chunks to embed, generating soulprint...")
                await generate_soulprint_from_chunks(user_id, client, headers)
                return

            # Embed in batches
            BATCH_SIZE = 20
            embedded_count = 0
            failed_count = 0
            bedrock_client = get_bedrock_client()

            for i in range(0, total_chunks, BATCH_SIZE):
                batch = chunks[i:i+BATCH_SIZE]
                print(f"[RLM] Embedding batch {i//BATCH_SIZE + 1}/{(total_chunks + BATCH_SIZE - 1)//BATCH_SIZE}")

                for chunk in batch:
                    try:
                        embedding = await embed_text_bedrock(chunk["content"], bedrock_client)

                        if embedding:
                            # Update chunk with embedding
                            update_resp = await client.patch(
                                f"{SUPABASE_URL}/rest/v1/conversation_chunks",
                                params={"id": f"eq.{chunk['id']}"},
                                headers=headers,
                                json={"embedding": embedding},
                            )

                            if update_resp.status_code in [200, 204]:
                                embedded_count += 1
                            else:
                                failed_count += 1
                                print(f"[RLM] Failed to update chunk {chunk['id']}: {update_resp.text}")
                        else:
                            failed_count += 1

                        await asyncio.sleep(0.05)  # Rate limit

                    except Exception as e:
                        failed_count += 1
                        print(f"[RLM] Chunk embedding error: {e}")

                # Update progress
                progress = int((i + len(batch)) / total_chunks * 100)
                await client.patch(
                    f"{SUPABASE_URL}/rest/v1/user_profiles",
                    params={"user_id": f"eq.{user_id}"},
                    headers=headers,
                    json={
                        "embedding_progress": progress,
                        "processed_chunks": embedded_count,
                    },
                )

                await asyncio.sleep(0.2)  # Batch delay

            elapsed = time.time() - start
            print(f"[RLM] Embedded {embedded_count}/{total_chunks} chunks in {elapsed:.1f}s")

            # Generate SoulPrint now that embeddings are done
            await generate_soulprint_from_chunks(user_id, client, headers)

            # Update final status
            await client.patch(
                f"{SUPABASE_URL}/rest/v1/user_profiles",
                params={"user_id": f"eq.{user_id}"},
                headers=headers,
                json={
                    "embedding_status": "complete",
                    "embedding_progress": 100,
                    "import_status": "complete",
                    "total_chunks": total_chunks,
                    "processed_chunks": embedded_count,
                },
            )

            await alert_drew(
                f"✅ Embeddings Complete!\n\n"
                f"User: {user_id}\n"
                f"Chunks: {embedded_count}/{total_chunks}\n"
                f"Time: {elapsed:.1f}s"
            )

    except Exception as e:
        error_msg = str(e)
        print(f"[RLM] Embedding failed: {error_msg}")

        try:
            async with httpx.AsyncClient() as client:
                await client.patch(
                    f"{SUPABASE_URL}/rest/v1/user_profiles",
                    params={"user_id": f"eq.{user_id}"},
                    headers={
                        "apikey": SUPABASE_SERVICE_KEY,
                        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "embedding_status": "failed",
                        "import_error": error_msg[:500],
                    },
                )
        except:
            pass

        await alert_drew(f"❌ Embedding Failed\n\nUser: {user_id}\nError: {error_msg[:500]}")


async def generate_soulprint_from_chunks(user_id: str, client: httpx.AsyncClient, headers: dict):
    """Generate SoulPrint from existing chunks"""
    print(f"[RLM] Generating SoulPrint from chunks for user {user_id}")

    # Fetch chunks for soulprint generation (prefer macro for themes)
    response = await client.get(
        f"{SUPABASE_URL}/rest/v1/conversation_chunks",
        params={
            "user_id": f"eq.{user_id}",
            "select": "content,title,chunk_tier,created_at",
            "order": "created_at.desc",
            "limit": "500",
        },
        headers=headers,
    )

    if response.status_code != 200:
        print(f"[RLM] Failed to fetch chunks for soulprint: {response.text}")
        return

    chunks = response.json()
    if not chunks:
        print(f"[RLM] No chunks found for soulprint generation")
        return

    # Convert chunks to message format for synthesis
    messages = []
    for chunk in chunks:
        messages.append({
            "content": chunk.get("content", ""),
            "conversation_title": chunk.get("title", ""),
            "original_timestamp": chunk.get("created_at"),
        })

    print(f"[RLM] Running synthesis on {len(messages)} chunks...")
    profile = await recursive_synthesize(messages, user_id)
    archetype = profile.get("archetype", "Unique Individual")
    core_essence = profile.get("core_essence", archetype)

    # Log what we got from synthesis
    print(f"[RLM] Synthesis result - archetype: {archetype}, has_error: {'error' in profile}")
    if "error" in profile:
        print(f"[RLM] Synthesis error: {profile.get('error')}")

    print("[RLM] Generating SoulPrint files...")
    soul_files = await generate_soulprint_files(profile, messages, user_id)

    print("[RLM] Generating memory log...")
    memory_log = await generate_memory_log(messages, profile, user_id)

    # Save to user_profiles with fresh client to avoid connection issues
    async with httpx.AsyncClient(timeout=60.0) as save_client:
        save_headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }
        save_data = {
            "soulprint": profile,
            "soulprint_text": core_essence,
            "archetype": archetype,
            "soul_md": soul_files.get("soul_md"),
            "identity_md": soul_files.get("identity_md"),
            "agents_md": soul_files.get("agents_md"),
            "user_md": soul_files.get("user_md"),
            "memory_log": memory_log,
            "soulprint_generated_at": datetime.utcnow().isoformat(),
        }

        print(f"[RLM] Saving SoulPrint - archetype: {archetype}, core_essence length: {len(core_essence)}")

        save_resp = await save_client.patch(
            f"{SUPABASE_URL}/rest/v1/user_profiles",
            params={"user_id": f"eq.{user_id}"},
            headers=save_headers,
            json=save_data,
        )

        if save_resp.status_code not in (200, 204):
            print(f"[RLM] SoulPrint save FAILED: {save_resp.status_code} - {save_resp.text[:500]}")
        else:
            result = save_resp.json() if save_resp.text else []
            if result:
                saved_archetype = result[0].get("archetype", "UNKNOWN")
                print(f"[RLM] SoulPrint save VERIFIED - archetype in DB: {saved_archetype}")
            else:
                print(f"[RLM] SoulPrint save returned empty - may not have updated")

    print(f"[RLM] SoulPrint generated and saved for user {user_id}")


@app.get("/generate-soulprint/{user_id}")
async def generate_soulprint_endpoint(user_id: str):
    """
    Generate SoulPrint from existing chunks (use when embeddings are done but soulprint failed).
    """
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            headers = {
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type": "application/json",
            }

            # Check if chunks exist
            check_resp = await client.get(
                f"{SUPABASE_URL}/rest/v1/conversation_chunks",
                params={"user_id": f"eq.{user_id}", "select": "id", "limit": "1"},
                headers=headers,
            )
            chunks = check_resp.json() if check_resp.status_code == 200 else []
            if not chunks:
                return {"success": False, "error": "No chunks found for this user"}

            # Generate soulprint
            await generate_soulprint_from_chunks(user_id, client, headers)

            # Update import status
            await client.patch(
                f"{SUPABASE_URL}/rest/v1/user_profiles",
                params={"user_id": f"eq.{user_id}"},
                headers=headers,
                json={
                    "import_status": "complete",
                    "embedding_status": "complete",
                },
            )

            # Trigger completion callback
            vercel_url = VERCEL_API_URL
            try:
                callback_resp = await client.post(
                    f"{vercel_url}/api/import/complete",
                    json={"user_id": user_id},
                    timeout=30.0,
                )
                print(f"[RLM] Completion callback: {callback_resp.status_code}")
            except Exception as e:
                print(f"[RLM] Callback failed: {e}")

            return {"success": True, "message": "SoulPrint generated and saved"}
    except Exception as e:
        print(f"[RLM] generate-soulprint error: {e}")
        return {"success": False, "error": str(e)}


@app.get("/status")
async def status():
    """Detailed status for monitoring"""
    return {
        "service": "soulprint-rlm",
        "rlm_available": RLM_AVAILABLE,
        "rlm_error": RLM_IMPORT_ERROR if not RLM_AVAILABLE else None,
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_SERVICE_KEY),
        "anthropic_configured": bool(ANTHROPIC_API_KEY),
        "bedrock_configured": BEDROCK_AVAILABLE and bool(AWS_ACCESS_KEY_ID),
        "cohere_configured": bool(COHERE_API_KEY),
        "vercel_callback_configured": bool(VERCEL_API_URL),
        "timestamp": datetime.utcnow().isoformat(),
    }


# ============================================================================
# AUTOMATIC EMBEDDING COMPLETION
# ============================================================================

@app.get("/embedding-status/{user_id}")
async def embedding_status(user_id: str):
    """
    Check embedding status for a user.
    Returns counts by tier and overall completion percentage.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        }

        # Get counts by tier - with embeddings
        embedded_resp = await client.get(
            f"{SUPABASE_URL}/rest/v1/conversation_chunks",
            params={
                "user_id": f"eq.{user_id}",
                "embedding": "not.is.null",
                "select": "chunk_tier",
            },
            headers=headers,
        )
        embedded = embedded_resp.json() if embedded_resp.status_code == 200 else []

        # Get counts by tier - without embeddings
        missing_resp = await client.get(
            f"{SUPABASE_URL}/rest/v1/conversation_chunks",
            params={
                "user_id": f"eq.{user_id}",
                "embedding": "is.null",
                "select": "chunk_tier",
            },
            headers=headers,
        )
        missing = missing_resp.json() if missing_resp.status_code == 200 else []

        # Count by tier
        embedded_by_tier = {"macro": 0, "medium": 0, "micro": 0}
        missing_by_tier = {"macro": 0, "medium": 0, "micro": 0}

        for chunk in embedded:
            tier = chunk.get("chunk_tier", "unknown")
            if tier in embedded_by_tier:
                embedded_by_tier[tier] += 1

        for chunk in missing:
            tier = chunk.get("chunk_tier", "unknown")
            if tier in missing_by_tier:
                missing_by_tier[tier] += 1

        total_embedded = sum(embedded_by_tier.values())
        total_missing = sum(missing_by_tier.values())
        total = total_embedded + total_missing
        completion_pct = int((total_embedded / total * 100)) if total > 0 else 0

        return {
            "user_id": user_id,
            "total_chunks": total,
            "total_embedded": total_embedded,
            "total_missing": total_missing,
            "completion_percentage": completion_pct,
            "embedded_by_tier": embedded_by_tier,
            "missing_by_tier": missing_by_tier,
            "status": "complete" if total_missing == 0 else "incomplete",
        }


@app.post("/complete-embeddings/{user_id}")
async def complete_embeddings(user_id: str, background_tasks: BackgroundTasks):
    """
    Complete missing embeddings for a user.
    This endpoint is called automatically after import but can also be triggered manually.
    Runs in background and returns immediately.

    Use /embedding-status/{user_id} to check progress.
    """
    # Check current status first
    status = await embedding_status(user_id)

    if status["total_missing"] == 0:
        return {
            "success": True,
            "message": "All embeddings already complete",
            "status": status,
        }

    # Start background job
    background_tasks.add_task(complete_embeddings_background, user_id)

    return {
        "success": True,
        "message": f"Started embedding {status['total_missing']} missing chunks in background",
        "status": status,
        "note": "Use GET /embedding-status/{user_id} to check progress",
    }


async def complete_embeddings_background(user_id: str):
    """
    Background job to complete missing embeddings.
    Embeds all chunks with NULL embeddings until complete.
    """
    print(f"[RLM] Starting embedding completion for user {user_id}")
    start = time.time()
    total_embedded = 0
    batch_number = 0
    MAX_BATCHES = 100  # Safety limit

    bedrock_client = get_bedrock_client()
    if not bedrock_client:
        print(f"[RLM] ERROR: Bedrock client not available")
        return

    while batch_number < MAX_BATCHES:
        batch_number += 1

        async with httpx.AsyncClient(timeout=120.0) as client:
            headers = {
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type": "application/json",
            }

            # Fetch batch of chunks without embeddings
            resp = await client.get(
                f"{SUPABASE_URL}/rest/v1/conversation_chunks",
                params={
                    "user_id": f"eq.{user_id}",
                    "embedding": "is.null",
                    "select": "id,content,chunk_tier",
                    "limit": "50",  # Process 50 at a time
                },
                headers=headers,
            )

            if resp.status_code != 200:
                print(f"[RLM] Failed to fetch chunks: {resp.text[:200]}")
                break

            chunks = resp.json()
            if not chunks:
                print(f"[RLM] No more chunks to embed - all complete!")
                break

            print(f"[RLM] Embedding batch {batch_number}: {len(chunks)} chunks")

            # Embed each chunk
            batch_embedded = 0
            for chunk in chunks:
                try:
                    embedding = await embed_text_bedrock(chunk["content"], bedrock_client)

                    if embedding:
                        # Update chunk with embedding
                        patch_resp = await client.patch(
                            f"{SUPABASE_URL}/rest/v1/conversation_chunks",
                            params={"id": f"eq.{chunk['id']}"},
                            headers={**headers, "Prefer": "return=representation"},
                            json={"embedding": embedding},
                        )

                        if patch_resp.status_code in (200, 204):
                            batch_embedded += 1
                            total_embedded += 1
                        else:
                            print(f"[RLM] PATCH failed for {chunk['id']}: {patch_resp.status_code}")

                    await asyncio.sleep(0.05)  # Rate limit

                except Exception as e:
                    print(f"[RLM] Embed error for {chunk['id']}: {e}")

            print(f"[RLM] Batch {batch_number} complete: {batch_embedded}/{len(chunks)} embedded")

            # Update progress in user_profiles
            await client.patch(
                f"{SUPABASE_URL}/rest/v1/user_profiles",
                params={"user_id": f"eq.{user_id}"},
                headers=headers,
                json={"processed_chunks": total_embedded},
            )

            # Brief pause between batches
            await asyncio.sleep(0.5)

    elapsed = time.time() - start
    print(f"[RLM] Embedding completion finished: {total_embedded} chunks in {elapsed:.1f}s")

    # Update final status
    async with httpx.AsyncClient(timeout=30.0) as client:
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "Content-Type": "application/json",
        }
        await client.patch(
            f"{SUPABASE_URL}/rest/v1/user_profiles",
            params={"user_id": f"eq.{user_id}"},
            headers=headers,
            json={
                "embedding_status": "complete",
                "embedding_progress": 100,
            },
        )

    # Alert on completion
    await alert_drew(f"✅ Embeddings Complete\n\nUser: {user_id}\nChunks: {total_embedded}\nTime: {elapsed:.1f}s")
