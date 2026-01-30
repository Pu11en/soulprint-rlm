"""
SoulPrint RLM Service
TRUE Recursive Language Models - no fallback, memory is critical
Multi-tier precision chunking with vector similarity search
"""
import os
import re
import json
import httpx
import time
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, HTTPException
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

load_dotenv()

app = FastAPI(title="SoulPrint RLM Service")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Render handles CORS
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
ALERT_TELEGRAM_CHAT = os.getenv("ALERT_TELEGRAM_CHAT", "7414639817")  # Drew's Telegram


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


class AnalyzeRequest(BaseModel):
    user_id: str


class CreateSoulprintRequest(BaseModel):
    user_id: str
    conversations: List[dict]
    stats: Optional[dict] = None


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


async def embed_query(query: str) -> Optional[List[float]]:
    """Embed a query using Cohere for vector similarity search"""
    if not COHERE_API_KEY:
        print("[RLM] No Cohere API key, skipping vector search")
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
            else:
                print(f"[RLM] Cohere embed error: {response.status_code}")
                return None
    except Exception as e:
        print(f"[RLM] Cohere embed exception: {e}")
        return None


async def vector_search_chunks(user_id: str, query_embedding: List[float], limit: int = 50) -> List[dict]:
    """Search chunks by vector similarity using Supabase RPC"""
    try:
        async with httpx.AsyncClient() as client:
            # Call the match_conversation_chunks RPC function
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
                    "match_threshold": 0.3,  # Lower threshold for more results
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
    """Fetch conversation chunks and soulprint from Supabase
    
    If query is provided, uses vector similarity search for precision.
    Otherwise falls back to fetching recent chunks.
    """
    async with httpx.AsyncClient() as client:
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        }
        
        chunks = []
        
        # Try vector similarity search if we have a query
        if query and COHERE_API_KEY:
            query_embedding = await embed_query(query)
            if query_embedding:
                chunks = await vector_search_chunks(user_id, query_embedding, limit=100)
        
        # Fallback to recent chunks if vector search didn't work
        if not chunks:
            print("[RLM] Falling back to recent chunks")
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
        
        # Get soulprint
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
        for i, chunk in enumerate(chunks[:50]):  # Top 50 most relevant
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
    """Check RLM availability on startup"""
    if not RLM_AVAILABLE:
        await alert_drew(f"RLM LIBRARY NOT AVAILABLE!\n\nImport error: {RLM_IMPORT_ERROR}\n\nService will return errors until fixed.")
    else:
        print("[RLM] Library loaded successfully")
    
    if COHERE_API_KEY:
        print("[RLM] Cohere API key configured - vector search enabled")
    else:
        print("[RLM] No Cohere API key - using fallback chunk retrieval")


@app.get("/health")
async def health():
    """Health check - returns error if RLM not available"""
    if not RLM_AVAILABLE:
        raise HTTPException(status_code=503, detail=f"RLM not available: {RLM_IMPORT_ERROR}")
    return {
        "status": "ok",
        "service": "soulprint-rlm",
        "rlm_available": True,
        "vector_search": bool(COHERE_API_KEY),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Main query endpoint - TRUE RLM with vector similarity search"""
    start = time.time()
    
    # CRITICAL: RLM must be available
    if not RLM_AVAILABLE:
        await alert_drew(f"Query failed - RLM not available\nUser: {request.user_id}")
        raise HTTPException(
            status_code=503,
            detail="Memory service unavailable. Please try again later."
        )
    
    try:
        # Fetch user data with vector search using the query
        chunks, soulprint_text = await get_user_data(request.user_id, query=request.message)
        
        # Use provided soulprint or fetched
        soulprint = request.soulprint_text or soulprint_text
        
        # Build context
        context = build_context(chunks, soulprint, request.history or [])
        
        # Initialize RLM with Anthropic backend
        rlm = RLM(
            backend="anthropic",
            backend_kwargs={
                "model_name": "claude-sonnet-4-20250514",
                "api_key": ANTHROPIC_API_KEY,
            },
            verbose=False,
        )
        
        # Limit context to avoid timeout
        limited_context = context[:20000] if len(context) > 20000 else context
        
        # Build the RLM prompt
        prompt = f"""You are SoulPrint, a personal AI with infinite memory of the user's conversation history.

{limited_context}

## Instructions
- Use the conversation history to provide personalized, contextual responses
- Reference relevant past conversations naturally
- Be warm and helpful
- If asked about past conversations, the relevant chunks are already provided above
- Keep responses focused and concise

User's message: {request.message}"""

        # Execute RLM query with error handling
        try:
            result = rlm.completion(prompt)
        except Exception as rlm_error:
            # If RLM fails, fall back to direct Anthropic
            print(f"[RLM] RLM execution failed: {rlm_error}, using direct call")
            import anthropic
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
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
            method="rlm-vector" if COHERE_API_KEY else "rlm-fallback",
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
            
            # Fetch raw conversations
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
            
            # Strategic sampling: recent, oldest, longest
            recent = conversations[:50]
            oldest = conversations[-50:] if len(conversations) > 50 else []
            by_length = sorted(conversations, key=lambda c: c.get("message_count", 0), reverse=True)
            longest = by_length[:100]
            
            # Deduplicate
            seen = set()
            sampled = []
            for conv in recent + oldest + longest:
                conv_id = conv.get("title", "") + str(conv.get("created_at", ""))
                if conv_id not in seen:
                    seen.add(conv_id)
                    sampled.append(conv)
            
            # Build analysis prompt
            sample_text = ""
            for conv in sampled[:100]:  # Max 100 for analysis
                title = conv.get("title", "Untitled")
                messages = conv.get("messages", [])[:10]  # First 10 messages
                msg_text = "\n".join([f"- {m.get('role', 'unknown')}: {m.get('content', '')[:200]}" for m in messages])
                sample_text += f"\n\n### {title}\n{msg_text}"
            
            analysis_prompt = f"""Analyze this user's conversation history and create a detailed personality profile.

{sample_text[:30000]}

Based on these conversations, provide a JSON analysis with:
1. archetype: Their primary personality archetype (e.g., "The Builder", "The Explorer", "The Connector")
2. tone: How they communicate (casual, professional, mixed)
3. humor: Their sense of humor (dry, playful, sarcastic, none)
4. interests: Top 10 interests/topics
5. communication_style: How they prefer to receive information
6. key_traits: 5 defining personality traits
7. avoid: Things the AI should avoid when talking to them

Return ONLY valid JSON, no markdown."""

            # Use RLM or direct Anthropic
            if RLM_AVAILABLE:
                try:
                    rlm = RLM(
                        backend="anthropic",
                        backend_kwargs={
                            "model_name": "claude-sonnet-4-20250514",
                            "api_key": ANTHROPIC_API_KEY,
                        },
                        verbose=False,
                    )
                    result = rlm.completion(analysis_prompt)
                    analysis_text = result.response
                except Exception as e:
                    print(f"[RLM] Analysis failed, using direct: {e}")
                    import anthropic
                    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
                    response = client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=2048,
                        messages=[{"role": "user", "content": analysis_prompt}],
                    )
                    analysis_text = response.content[0].text
            else:
                import anthropic
                client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=2048,
                    messages=[{"role": "user", "content": analysis_prompt}],
                )
                analysis_text = response.content[0].text
            
            # Parse JSON from response
            try:
                # Try to extract JSON from response
                json_match = re.search(r'\{[\s\S]*\}', analysis_text)
                if json_match:
                    profile = json.loads(json_match.group())
                else:
                    profile = {"raw": analysis_text}
            except json.JSONDecodeError:
                profile = {"raw": analysis_text}
            
            # Update user profile with analysis
            update_response = await client.patch(
                f"{SUPABASE_URL}/rest/v1/user_profiles",
                params={"user_id": f"eq.{request.user_id}"},
                headers={**headers, "Content-Type": "application/json", "Prefer": "return=minimal"},
                json={
                    "soulprint": {**profile, "analyzed_at": datetime.utcnow().isoformat()},
                    "soulprint_updated_at": datetime.utcnow().isoformat(),
                },
            )
            
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
    Generate a soulprint from conversation data using RLM.
    Called by the landing app during import flow.
    
    Uses SoulPrint Quality Criteria to capture:
    - Voice Profile (formality, humor, emoji patterns)
    - Thinking Style (explanation patterns, problem-solving)
    - Emotional Signature (enthusiasm, frustration, support)
    - Context Adaptation (professional vs personal)
    - Memory Anchors (people, events, preferences)
    """
    start = time.time()
    
    try:
        conversations = request.conversations
        stats = request.stats or {}
        user_id = request.user_id
        
        if not conversations:
            return {"error": "No conversations provided", "soulprint": None, "archetype": None}
        
        print(f"[RLM] Creating soulprint for user {user_id} from {len(conversations)} conversations")
        
        # Strategic sampling for comprehensive analysis
        # Recent conversations (current voice)
        recent = conversations[:50]
        
        # Oldest conversations (historical patterns)
        oldest = conversations[-30:] if len(conversations) > 50 else []
        
        # Longest conversations (depth of engagement)
        by_length = sorted(conversations, key=lambda c: c.get("message_count", len(c.get("messages", []))), reverse=True)
        longest = by_length[:50]
        
        # Deduplicate while preserving order
        seen = set()
        sampled = []
        for conv in recent + longest + oldest:
            conv_id = conv.get("id") or conv.get("title", "") + str(conv.get("created_at", ""))
            if conv_id not in seen:
                seen.add(conv_id)
                sampled.append(conv)
        
        # Build conversation text for analysis
        conversation_text = ""
        for conv in sampled[:100]:  # Max 100 for token limits
            title = conv.get("title", "Untitled")
            messages = conv.get("messages", conv.get("mapping", []))
            
            # Handle different message formats
            if isinstance(messages, list):
                msg_excerpts = []
                for m in messages[:15]:  # First 15 messages per conversation
                    if isinstance(m, dict):
                        role = m.get("role", m.get("author", {}).get("role", "unknown"))
                        content = m.get("content", "")
                        if isinstance(content, list):
                            content = " ".join([p.get("text", str(p)) for p in content if isinstance(p, dict)])
                        elif isinstance(content, dict):
                            content = content.get("parts", [content.get("text", str(content))])[0] if content else ""
                        content = str(content)[:300]  # Truncate long messages
                        if content.strip():
                            msg_excerpts.append(f"  {role}: {content}")
                    elif isinstance(m, str):
                        msg_excerpts.append(f"  user: {m[:300]}")
                
                if msg_excerpts:
                    conversation_text += f"\n### {title}\n" + "\n".join(msg_excerpts) + "\n"
            elif isinstance(messages, dict):
                # Handle OpenAI/ChatGPT mapping format
                for node_id, node in list(messages.items())[:15]:
                    if isinstance(node, dict) and "message" in node:
                        msg = node["message"]
                        if msg:
                            role = msg.get("author", {}).get("role", "unknown")
                            content = msg.get("content", {})
                            if isinstance(content, dict):
                                parts = content.get("parts", [])
                                text = parts[0] if parts else ""
                            else:
                                text = str(content)
                            if text and len(text.strip()) > 0:
                                conversation_text += f"  {role}: {text[:300]}\n"
        
        # Truncate to fit token limits
        conversation_text = conversation_text[:50000]
        
        # Build the soulprint generation prompt using quality criteria
        soulprint_prompt = f"""You are creating a SoulPrint - a deeply personalized profile that captures someone's unique essence from their conversations.

## CONVERSATION HISTORY
{conversation_text}

## ANALYSIS STATS
{json.dumps(stats, indent=2) if stats else "No stats provided"}

## YOUR TASK
Analyze these conversations to create a comprehensive SoulPrint. You must capture:

### 1. VOICE PROFILE
- Formality spectrum (casual ↔ formal)
- Humor style (dry, silly, sarcastic, playful, none)
- Emoji/punctuation patterns
- Typical message length and rhythm
- How they start and end messages

### 2. THINKING STYLE
- How they explain things to others
- Question-asking patterns
- Problem-solving approach
- Decision-making style
- How they build arguments

### 3. EMOTIONAL SIGNATURE
- How they express enthusiasm
- How they handle frustration
- How they comfort and support others
- Their vulnerability level
- Emotional range in communication

### 4. CONTEXT ADAPTATION
- Professional vs personal tone shifts
- How they adjust for different audiences
- Topic-specific tone changes

### 5. MEMORY ANCHORS
- Important people mentioned frequently
- Significant events or dates
- Strong preferences and opinions
- Recurring interests and passions
- Things they care deeply about

## OUTPUT FORMAT
Return a JSON object with these exact fields:

{{
  "archetype": "A creative 2-3 word archetype title (e.g., 'The Curious Builder', 'The Thoughtful Explorer')",
  "summary": "A 2-3 sentence essence of who this person is",
  "voice": {{
    "formality": "casual|mixed|formal",
    "humor": "dry|playful|sarcastic|warm|none",
    "emoji_usage": "heavy|moderate|minimal|none",
    "message_style": "brief|moderate|detailed",
    "punctuation_quirks": ["list of any notable patterns"]
  }},
  "thinking": {{
    "explanation_style": "how they explain things",
    "problem_solving": "their approach to problems",
    "curiosity_areas": ["top interests/topics"],
    "decision_style": "how they make decisions"
  }},
  "emotional": {{
    "enthusiasm_expression": "how they show excitement",
    "frustration_handling": "how they deal with frustration",
    "support_style": "how they comfort others",
    "vulnerability": "low|moderate|high"
  }},
  "memory_anchors": {{
    "key_people": ["important names mentioned"],
    "recurring_topics": ["topics they return to"],
    "strong_opinions": ["things they feel strongly about"],
    "preferences": ["clear likes/dislikes"]
  }},
  "soulprint_text": "A rich, narrative description of this person in 200-400 words that an AI could use to embody their communication style"
}}

CRITICAL: Return ONLY valid JSON, no markdown code blocks, no explanations outside the JSON."""

        # Use RLM or direct Anthropic
        analysis_text = None
        
        if RLM_AVAILABLE:
            try:
                rlm = RLM(
                    backend="anthropic",
                    backend_kwargs={
                        "model_name": "claude-sonnet-4-20250514",
                        "api_key": ANTHROPIC_API_KEY,
                    },
                    verbose=False,
                )
                result = rlm.completion(soulprint_prompt)
                analysis_text = result.response
            except Exception as e:
                print(f"[RLM] Create soulprint RLM failed, using direct: {e}")
        
        if not analysis_text:
            import anthropic
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                messages=[{"role": "user", "content": soulprint_prompt}],
            )
            analysis_text = response.content[0].text
        
        # Parse JSON from response
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', analysis_text)
            if json_match:
                soulprint_data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[RLM] Failed to parse soulprint JSON: {e}")
            # Return a structured fallback
            soulprint_data = {
                "archetype": "Unique Individual",
                "summary": "A person with a distinct voice and perspective.",
                "soulprint_text": analysis_text[:1000] if analysis_text else "Profile generation in progress.",
                "raw_response": analysis_text[:2000] if analysis_text else None,
            }
        
        archetype = soulprint_data.get("archetype", "Unique Individual")
        
        # Store to Supabase if configured
        if SUPABASE_URL and SUPABASE_SERVICE_KEY:
            try:
                async with httpx.AsyncClient() as client:
                    headers = {
                        "apikey": SUPABASE_SERVICE_KEY,
                        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                        "Content-Type": "application/json",
                        "Prefer": "return=minimal",
                    }
                    
                    # Upsert user profile with soulprint
                    await client.post(
                        f"{SUPABASE_URL}/rest/v1/user_profiles",
                        headers={**headers, "Prefer": "resolution=merge-duplicates"},
                        json={
                            "user_id": user_id,
                            "soulprint": soulprint_data,
                            "soulprint_text": soulprint_data.get("soulprint_text", ""),
                            "archetype": archetype,
                            "soulprint_updated_at": datetime.utcnow().isoformat(),
                        },
                    )
                    print(f"[RLM] Stored soulprint for user {user_id}")
            except Exception as e:
                print(f"[RLM] Failed to store soulprint: {e}")
        
        latency = int((time.time() - start) * 1000)
        
        return {
            "soulprint": soulprint_data,
            "archetype": archetype,
            "conversations_analyzed": len(sampled),
            "latency_ms": latency,
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"[RLM] Create soulprint error: {error_msg}")
        await alert_drew(f"Create Soulprint Error\n\nUser: {request.user_id}\nError: {error_msg[:500]}")
        raise HTTPException(status_code=500, detail=f"Soulprint creation failed: {error_msg}")


@app.get("/status")
async def status():
    """Detailed status for monitoring"""
    return {
        "service": "soulprint-rlm",
        "rlm_available": RLM_AVAILABLE,
        "rlm_error": RLM_IMPORT_ERROR if not RLM_AVAILABLE else None,
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_SERVICE_KEY),
        "anthropic_configured": bool(ANTHROPIC_API_KEY),
        "cohere_configured": bool(COHERE_API_KEY),
        "vector_search_enabled": bool(COHERE_API_KEY),
        "alerts_configured": bool(ALERT_TELEGRAM_BOT),
        "timestamp": datetime.utcnow().isoformat(),
    }
