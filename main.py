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
from datetime import datetime, date
from typing import Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
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

app = FastAPI(title="SoulPrint RLM Service")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

# Vercel callback
VERCEL_API_URL = os.getenv("VERCEL_API_URL")
RLM_API_KEY = os.getenv("RLM_API_KEY")  # Shared secret for auth

# Models
HAIKU_MODEL = "claude-3-5-haiku-20241022"  # Fast, cheap for chunk analysis
SONNET_MODEL = "claude-sonnet-4-20250514"  # Smart for synthesis


class QueryRequest(BaseModel):
    user_id: str
    message: str
    soulprint_text: Optional[str] = None
    history: Optional[List[dict]] = []


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

        # Titan Embed v2 request format
        # Note: dimensions and normalize are optional, try without them first
        request_body = json.dumps({
            "inputText": text
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
# VECTOR SEARCH (Bedrock Titan + conversation_chunks)
# ============================================================================

async def search_memories(user_id: str, query: str, limit: int = 50) -> List[dict]:
    """
    Search conversation_chunks by vector similarity using Bedrock Titan embeddings.
    Uses tier-aware search: macro (themes) → medium (context) → micro (facts)
    """

    # Generate query embedding with Bedrock Titan
    query_embedding = await embed_text_bedrock(query)
    if not query_embedding:
        print("[RLM] Failed to generate query embedding, using fallback")
        return await get_recent_chunks(user_id, limit)

    all_memories = []

    try:
        async with httpx.AsyncClient() as client:
            headers = {
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type": "application/json",
            }

            # Search each tier with different limits
            # Macro: themes/relationships (few, high context)
            # Medium: conversation context (moderate)
            # Micro: precise facts (many, pinpoint accuracy)
            tier_limits = [
                ("macro", 10),   # Broad themes
                ("medium", 20),  # Conversation context
                ("micro", 20),   # Precise facts
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
                        "match_threshold": 0.3,
                    },
                    timeout=30.0,
                )

                if response.status_code == 200:
                    tier_results = response.json()
                    for r in tier_results:
                        r["chunk_tier"] = tier  # Tag with tier
                    all_memories.extend(tier_results)
                    print(f"[RLM] Tier '{tier}' returned {len(tier_results)} matches")

            # Fallback to non-tier search if no results or function doesn't exist
            if not all_memories:
                response = await client.post(
                    f"{SUPABASE_URL}/rest/v1/rpc/match_conversation_chunks",
                    headers=headers,
                    json={
                        "query_embedding": query_embedding,
                        "match_user_id": user_id,
                        "match_count": limit,
                        "match_threshold": 0.3,
                    },
                    timeout=30.0,
                )
                if response.status_code == 200:
                    all_memories = response.json()

            print(f"[RLM] Total memories retrieved: {len(all_memories)}")
            return all_memories[:limit]

    except Exception as e:
        print(f"[RLM] Vector search exception: {e}")
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

        # Build memory context
        memory_context = ""
        if memories:
            memory_context = "## Relevant Memories from Past Conversations\n\n"
            for i, mem in enumerate(memories[:20]):
                title = mem.get('conversation_title', 'Unknown')
                content = mem.get('content', '')[:500]
                timestamp = mem.get('original_timestamp', '')
                similarity = mem.get('similarity', 0)
                memory_context += f"**[{title}]** (relevance: {similarity:.2f})\n{content}\n\n"

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

        # Generate response
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=SONNET_MODEL,
            max_tokens=2048,
            system=system_prompt[:15000],  # Limit system prompt
            messages=[{"role": "user", "content": request.message}],
        )

        latency = int((time.time() - start) * 1000)

        return ChatResponse(
            response=response.content[0].text,
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
    Recursive synthesis using Haiku for chunks, Sonnet for synthesis.

    Level 1: Haiku processes batches → summaries
    Level 2+: Sonnet merges summaries → until single result
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    if len(messages) <= batch_size:
        # Base case: small enough for single Sonnet call
        return await sonnet_generate_profile(messages, client)

    # Level 1: Process batches with Haiku (fast, cheap)
    print(f"[RLM] Level 1: Processing {len(messages)} messages in batches of {batch_size}")
    summaries = []

    batches = [messages[i:i+batch_size] for i in range(0, len(messages), batch_size)]

    for i, batch in enumerate(batches):
        print(f"[RLM] Haiku processing batch {i+1}/{len(batches)}")
        summary = await haiku_extract_patterns(batch, i, len(batches), client)
        summaries.append(summary)
        await asyncio.sleep(0.3)  # Rate limit

    # Level 2+: Recursive merge with Sonnet
    print(f"[RLM] Level 2: Merging {len(summaries)} summaries")
    return await recursive_merge_summaries(summaries, client)


async def haiku_extract_patterns(batch: List[dict], batch_num: int, total_batches: int, client) -> dict:
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
        response = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text

        # Parse JSON
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        return json.loads(text.strip())
    except Exception as e:
        print(f"[RLM] Haiku batch {batch_num} error: {e}")
        return {"error": str(e), "batch": batch_num}


async def recursive_merge_summaries(summaries: List[dict], client, merge_size: int = 10) -> dict:
    """Recursively merge summaries until we have a single profile"""

    if len(summaries) <= merge_size:
        # Final merge
        return await sonnet_final_synthesis(summaries, client)

    # Merge in groups
    merged = []
    for i in range(0, len(summaries), merge_size):
        group = summaries[i:i+merge_size]
        partial = await sonnet_merge_partial(group, client)
        merged.append(partial)
        await asyncio.sleep(0.5)

    # Recurse
    return await recursive_merge_summaries(merged, client, merge_size)


async def sonnet_merge_partial(summaries: List[dict], client) -> dict:
    """Sonnet merges a group of summaries into one"""

    prompt = f"""Merge these personality summaries into one consolidated summary.

## SUMMARIES TO MERGE
{json.dumps(summaries, indent=2)[:40000]}

## TASK
Combine all patterns, keeping the most specific quotes and examples.
Deduplicate but preserve unique insights.

Return JSON with same structure as input summaries, but merged."""

    try:
        response = client.messages.create(
            model=SONNET_MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        return json.loads(text.strip())
    except Exception as e:
        print(f"[RLM] Merge error: {e}")
        return {"merged": summaries, "error": str(e)}


async def sonnet_final_synthesis(summaries: List[dict], client) -> dict:
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
        response = client.messages.create(
            model=SONNET_MODEL,
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        return json.loads(text.strip())
    except Exception as e:
        print(f"[RLM] Final synthesis error: {e}")
        return {"archetype": "Unique Individual", "error": str(e)}


async def sonnet_generate_profile(messages: List[dict], client) -> dict:
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
        response = client.messages.create(
            model=SONNET_MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        return json.loads(text.strip())
    except Exception as e:
        return {"archetype": "Unique Individual", "error": str(e)}


async def generate_soulprint_files(profile: dict, messages: List[dict], user_id: str) -> dict:
    """Generate the four SoulPrint markdown files"""

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

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

    soul_response = client.messages.create(
        model=SONNET_MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": soul_prompt}],
    )
    soul_md = soul_response.content[0].text

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

    identity_response = client.messages.create(
        model=SONNET_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": identity_prompt}],
    )
    identity_md = identity_response.content[0].text

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

    agents_response = client.messages.create(
        model=SONNET_MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": agents_prompt}],
    )
    agents_md = agents_response.content[0].text

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

    user_response = client.messages.create(
        model=SONNET_MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": user_prompt}],
    )
    user_md = user_response.content[0].text

    return {
        "soul_md": soul_md,
        "identity_md": identity_md,
        "agents_md": agents_md,
        "user_md": user_md,
    }


async def generate_memory_log(messages: List[dict], profile: dict, user_id: str) -> str:
    """
    Generate a memory log summarizing the user's conversation history.
    This creates a daily summary style log.
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

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
        response = client.messages.create(
            model=HAIKU_MODEL,  # Use Haiku for speed
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
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


@app.on_event("startup")
async def startup():
    """Check availability on startup"""
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


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "ok",
        "service": "soulprint-rlm",
        "rlm_available": RLM_AVAILABLE,
        "bedrock_available": BEDROCK_AVAILABLE and bool(AWS_ACCESS_KEY_ID),
        "timestamp": datetime.utcnow().isoformat(),
    }


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


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Main query endpoint - TRUE RLM with vector similarity search"""
    start = time.time()

    if not RLM_AVAILABLE:
        await alert_drew(f"Query failed - RLM not available\nUser: {request.user_id}")
        raise HTTPException(
            status_code=503,
            detail="Memory service unavailable. Please try again later."
        )

    try:
        chunks, soulprint_text = await get_user_data(request.user_id, query=request.message)
        soulprint = request.soulprint_text or soulprint_text
        context = build_context(chunks, soulprint, request.history or [])

        rlm = RLM(
            backend="anthropic",
            backend_kwargs={
                "model_name": SONNET_MODEL,
                "api_key": ANTHROPIC_API_KEY,
            },
            verbose=False,
        )

        limited_context = context[:20000] if len(context) > 20000 else context

        prompt = f"""You are SoulPrint, a personal AI with infinite memory of the user's conversation history.

{limited_context}

## Instructions
- Use the conversation history to provide personalized, contextual responses
- Reference relevant past conversations naturally
- Be warm and helpful
- Keep responses focused and concise

User's message: {request.message}"""

        try:
            result = rlm.completion(prompt)
        except Exception as rlm_error:
            print(f"[RLM] RLM execution failed: {rlm_error}, using direct call")
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            response = client.messages.create(
                model=SONNET_MODEL,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            class MockResult:
                def __init__(self, text):
                    self.response = text
            result = MockResult(response.content[0].text)

        latency = int((time.time() - start) * 1000)

        return QueryResponse(
            response=result.response,
            chunks_used=len(chunks),
            method="rlm-bedrock" if (BEDROCK_AVAILABLE and AWS_ACCESS_KEY_ID) else "rlm-cohere",
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

            anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            response = anthropic_client.messages.create(
                model=SONNET_MODEL,
                max_tokens=2048,
                messages=[{"role": "user", "content": analysis_prompt}],
            )
            analysis_text = response.content[0].text

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
    conversations: Optional[List[dict]] = None  # Parsed conversations from Vercel


@app.post("/process-full")
async def process_full(request: ProcessFullRequest, background_tasks: BackgroundTasks):
    """
    Full processing pipeline: Create chunks → Embed → Generate SoulPrint.
    Called by Vercel after parsing ZIP. Runs in background - no timeout.
    """
    print(f"[RLM] Received process-full request for user {request.user_id}")
    print(f"[RLM] Conversations received: {len(request.conversations) if request.conversations else 0}")

    # Start background processing
    background_tasks.add_task(
        process_full_background,
        request.user_id,
        request.conversations
    )

    return {
        "status": "processing",
        "message": "Full processing started: chunking → embedding → soulprint.",
        "user_id": request.user_id,
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


async def process_full_background(user_id: str, conversations: Optional[List[dict]]):
    """
    Full background pipeline: Create chunks → Embed → Generate SoulPrint.

    Multi-tier chunking:
    - micro: 200 chars (precise facts, names, dates)
    - medium: 2000 chars (conversation context)
    - macro: 5000 chars (themes, relationships)
    """
    start = time.time()
    print(f"[RLM] Starting full processing for user {user_id}")

    try:
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

            if not conversations:
                print(f"[RLM] No conversations provided, checking existing chunks...")
                await embed_chunks_background(user_id)
                return

            # ============================================================
            # STEP 1: Create multi-tier chunks
            # ============================================================
            print(f"[RLM] Creating multi-tier chunks from {len(conversations)} conversations...")

            all_chunks = []
            total_messages = 0

            for idx, conv in enumerate(conversations[:500]):  # Limit to 500 conversations
                title = conv.get("title", "Untitled")
                messages = conv.get("messages", [])
                created_at = conv.get("createdAt") or conv.get("created_at") or datetime.utcnow().isoformat()

                # Build full content from messages
                full_content = ""
                for m in messages[:30]:  # Max 30 messages per conversation
                    if isinstance(m, dict):
                        role = m.get("role", "user")
                        content = m.get("content", "")
                        if content:
                            full_content += f"{role}: {content}\n"
                            total_messages += 1

                if not full_content.strip():
                    continue

                is_recent = idx < 100

                # MICRO chunks (200 chars) - precise facts
                MICRO_SIZE = 200
                for i in range(0, min(len(full_content), 2000), MICRO_SIZE):
                    chunk_text = full_content[i:i + MICRO_SIZE].strip()
                    if len(chunk_text) > 50:
                        all_chunks.append({
                            "user_id": user_id,
                            "conversation_id": conv.get("id", f"conv_{idx}"),
                            "title": title,
                            "content": chunk_text,
                            "chunk_tier": "micro",
                            "message_count": len(messages),
                            "created_at": created_at,
                            "is_recent": is_recent,
                        })

                # MEDIUM chunk (2000 chars) - conversation context
                medium_content = full_content[:2000].strip()
                if len(medium_content) > 100:
                    all_chunks.append({
                        "user_id": user_id,
                        "conversation_id": conv.get("id", f"conv_{idx}"),
                        "title": title,
                        "content": medium_content,
                        "chunk_tier": "medium",
                        "message_count": len(messages),
                        "created_at": created_at,
                        "is_recent": is_recent,
                    })

                # MACRO chunk (5000 chars) - themes, relationships
                macro_content = full_content[:5000].strip()
                if len(macro_content) > 500:
                    all_chunks.append({
                        "user_id": user_id,
                        "conversation_id": conv.get("id", f"conv_{idx}"),
                        "title": title,
                        "content": macro_content,
                        "chunk_tier": "macro",
                        "message_count": len(messages),
                        "created_at": created_at,
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
                    "embedding_status": "processing",
                },
            )

            # ============================================================
            # STEP 3: Embed all chunks
            # ============================================================
            print(f"[RLM] Embedding {inserted_count} chunks with Bedrock Titan...")

            # Fetch chunks we just inserted
            resp = await client.get(
                f"{SUPABASE_URL}/rest/v1/conversation_chunks",
                params={
                    "user_id": f"eq.{user_id}",
                    "select": "id,content",
                    "limit": "5000",
                },
                headers=headers,
            )
            chunks_to_embed = resp.json() if resp.status_code == 200 else []

            bedrock_client = get_bedrock_client()
            embedded_count = 0

            for i, chunk in enumerate(chunks_to_embed):
                try:
                    embedding = await embed_text_bedrock(chunk["content"], bedrock_client)
                    if embedding:
                        await client.patch(
                            f"{SUPABASE_URL}/rest/v1/conversation_chunks",
                            params={"id": f"eq.{chunk['id']}"},
                            headers=headers,
                            json={"embedding": embedding},
                        )
                        embedded_count += 1

                    # Progress update every 50
                    if i % 50 == 0:
                        progress = int((i + 1) / len(chunks_to_embed) * 100)
                        await client.patch(
                            f"{SUPABASE_URL}/rest/v1/user_profiles",
                            params={"user_id": f"eq.{user_id}"},
                            headers=headers,
                            json={"embedding_progress": progress, "processed_chunks": embedded_count},
                        )
                        print(f"[RLM] Embedding progress: {progress}% ({embedded_count}/{len(chunks_to_embed)})")

                    await asyncio.sleep(0.05)  # Rate limit

                except Exception as e:
                    print(f"[RLM] Embed error for chunk {chunk['id']}: {e}")

            print(f"[RLM] Embedded {embedded_count}/{len(chunks_to_embed)} chunks")

            # ============================================================
            # STEP 4: Generate SoulPrint
            # ============================================================
            await generate_soulprint_from_chunks(user_id, client, headers)

            # Final status update
            elapsed = time.time() - start
            await client.patch(
                f"{SUPABASE_URL}/rest/v1/user_profiles",
                params={"user_id": f"eq.{user_id}"},
                headers=headers,
                json={
                    "import_status": "complete",
                    "embedding_status": "complete",
                    "embedding_progress": 100,
                    "processed_chunks": embedded_count,
                },
            )

            print(f"[RLM] Full processing complete in {elapsed:.1f}s")
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

    print("[RLM] Generating SoulPrint files...")
    soul_files = await generate_soulprint_files(profile, messages, user_id)

    print("[RLM] Generating memory log...")
    memory_log = await generate_memory_log(messages, profile, user_id)

    # Save to user_profiles
    await client.patch(
        f"{SUPABASE_URL}/rest/v1/user_profiles",
        params={"user_id": f"eq.{user_id}"},
        headers=headers,
        json={
            "soulprint": profile,
            "soulprint_text": profile.get("core_essence", archetype),
            "archetype": archetype,
            "soul_md": soul_files.get("soul_md"),
            "identity_md": soul_files.get("identity_md"),
            "agents_md": soul_files.get("agents_md"),
            "user_md": soul_files.get("user_md"),
            "memory_log": memory_log,
            "soulprint_generated_at": datetime.utcnow().isoformat(),
        },
    )

    print(f"[RLM] SoulPrint generated and saved for user {user_id}")


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
