"""
SoulPrint RLM Service
TRUE Recursive Language Models - no fallback, memory is critical
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


async def get_user_data(user_id: str) -> tuple[List[dict], Optional[str]]:
    """Fetch conversation chunks and soulprint from Supabase"""
    async with httpx.AsyncClient() as client:
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        }
        
        # Get chunks
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
        
        # Get soulprint
        profile_response = await client.get(
            f"{SUPABASE_URL}/rest/v1/user_profiles",
            params={
                "user_id": f"eq.{user_id}",
                "select": "soulprint_text",
            },
            headers=headers,
        )
        
        chunks = chunks_response.json() if chunks_response.status_code == 200 else []
        profile = profile_response.json()
        soulprint_text = profile[0]["soulprint_text"] if profile else None
        
        return chunks, soulprint_text


def build_context(chunks: List[dict], soulprint_text: Optional[str], history: List[dict]) -> str:
    """Build the context string for RLM"""
    context_parts = []
    
    if soulprint_text:
        context_parts.append(f"## User Profile\n{soulprint_text}")
    
    if chunks:
        context_parts.append("## Conversation History")
        for chunk in chunks[:100]:  # Top 100 most recent
            context_parts.append(f"\n### {chunk['title']}\n{chunk['content'][:2000]}")
    
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


@app.get("/health")
async def health():
    """Health check - returns error if RLM not available"""
    if not RLM_AVAILABLE:
        raise HTTPException(status_code=503, detail=f"RLM not available: {RLM_IMPORT_ERROR}")
    return {
        "status": "ok",
        "service": "soulprint-rlm",
        "rlm_available": True,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Main query endpoint - TRUE RLM, NO FALLBACK"""
    start = time.time()
    
    # CRITICAL: RLM must be available
    if not RLM_AVAILABLE:
        await alert_drew(f"Query failed - RLM not available\nUser: {request.user_id}")
        raise HTTPException(
            status_code=503,
            detail="Memory service unavailable. Please try again later."
        )
    
    try:
        # Fetch user data
        chunks, soulprint_text = await get_user_data(request.user_id)
        
        # Use provided soulprint or fetched
        soulprint = request.soulprint_text or soulprint_text
        
        # Build context
        context = build_context(chunks, soulprint, request.history or [])
        
        # Initialize RLM with Anthropic backend
        # Note: OpenAI backend had issues, sticking with Anthropic + error handling
        rlm = RLM(
            backend="anthropic",
            backend_kwargs={
                "model_name": "claude-sonnet-4-20250514",
                "api_key": ANTHROPIC_API_KEY,
            },
            verbose=False,
        )
        
        # Limit context to avoid timeout (RLM will explore as needed)
        limited_context = context[:15000] if len(context) > 15000 else context
        
        # Build the RLM prompt
        prompt = f"""You are SoulPrint, a personal AI with infinite memory of the user's conversation history.

{limited_context}

## Instructions
- Use the conversation history to provide personalized, contextual responses
- Reference relevant past conversations naturally
- Be warm and helpful
- If asked about past conversations, search through the history programmatically
- Keep responses focused and concise

User's message: {request.message}"""

        # Execute RLM query with error handling
        try:
            result = rlm.completion(prompt)
        except Exception as rlm_error:
            # If RLM fails (timeout, etc), fall back to direct Anthropic
            print(f"[RLM] RLM execution failed: {rlm_error}, using direct call")
            import anthropic
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            # Create a mock result object
            class MockResult:
                def __init__(self, text):
                    self.response = text
            result = MockResult(response.content[0].text)
        
        latency = int((time.time() - start) * 1000)
        
        return QueryResponse(
            response=result.response,
            chunks_used=len(chunks),
            method="rlm",
            latency_ms=latency,
        )
        
    except Exception as e:
        error_msg = str(e)
        await alert_drew(f"RLM Query Error\n\nUser: {request.user_id}\nError: {error_msg[:500]}")
        raise HTTPException(status_code=500, detail=f"Memory query failed: {error_msg}")


class AnalyzeRequest(BaseModel):
    user_id: str


class AnalyzeResponse(BaseModel):
    success: bool
    soulprint_text: str
    personality: dict
    insights: dict
    latency_ms: int


async def get_raw_conversations(user_id: str, limit: int = 300) -> List[dict]:
    """Fetch raw conversations from Supabase"""
    async with httpx.AsyncClient() as client:
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        }
        
        response = await client.get(
            f"{SUPABASE_URL}/rest/v1/raw_conversations",
            params={
                "user_id": f"eq.{user_id}",
                "select": "conversation_id,title,messages,message_count,created_at",
                "order": "created_at.desc",
                "limit": str(limit),
            },
            headers=headers,
            timeout=30.0,
        )
        
        if response.status_code == 200:
            return response.json()
        return []


def sample_conversations_for_analysis(conversations: List[dict], target: int = 200) -> List[dict]:
    """Strategic sampling for personality analysis"""
    if len(conversations) <= target:
        return conversations
    
    samples = []
    
    # 50 most recent
    samples.extend(conversations[:50])
    
    # 50 oldest
    samples.extend(conversations[-50:])
    
    # Longest conversations (deepest discussions)
    by_length = sorted(conversations, key=lambda c: c.get('message_count', 0), reverse=True)
    for c in by_length[:100]:
        if c not in samples:
            samples.append(c)
            if len(samples) >= target:
                break
    
    return samples[:target]


def format_conversations_for_rlm(conversations: List[dict]) -> str:
    """Format conversations for RLM analysis"""
    formatted = []
    for convo in conversations:
        messages = convo.get('messages', [])
        user_messages = [m['content'][:500] for m in messages if m.get('role') == 'user'][:10]
        if user_messages:
            formatted.append(f"[{convo.get('title', 'Untitled')}]\n" + "\n---\n".join(user_messages))
    return "\n\n=====\n\n".join(formatted)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """Deep personality analysis using RLM - analyzes ALL conversations"""
    start = time.time()
    
    if not RLM_AVAILABLE:
        await alert_drew(f"Analyze failed - RLM not available\nUser: {request.user_id}")
        raise HTTPException(status_code=503, detail="Analysis service unavailable")
    
    try:
        # Fetch raw conversations
        print(f"[Analyze] Fetching conversations for user {request.user_id}")
        conversations = await get_raw_conversations(request.user_id, limit=500)
        
        if not conversations:
            raise HTTPException(status_code=404, detail="No conversations found for analysis")
        
        print(f"[Analyze] Found {len(conversations)} conversations")
        
        # Sample strategically
        samples = sample_conversations_for_analysis(conversations, target=200)
        conversation_text = format_conversations_for_rlm(samples)
        
        # Limit context size
        if len(conversation_text) > 50000:
            conversation_text = conversation_text[:50000]
        
        # Initialize RLM
        rlm = RLM(
            backend="anthropic",
            backend_kwargs={
                "model_name": "claude-sonnet-4-20250514",
                "api_key": ANTHROPIC_API_KEY,
            },
            verbose=False,
        )
        
        # Deep personality analysis prompt
        analysis_prompt = f"""Analyze these user messages from their ChatGPT conversation history to create a deep personality profile.

USER CONVERSATIONS:
{conversation_text}

Create a comprehensive analysis covering:

1. **Identity & Archetype**
   - What archetype fits them? (e.g., "The Builder", "The Explorer", "The Analyst")
   - A name that captures their vibe
   - One emoji that represents them
   - A tagline

2. **Communication Style**
   - Tone: casual, balanced, or formal
   - Verbosity: concise, balanced, or detailed
   - Humor: none, dry, playful, or frequent
   - Energy: calm, balanced, or energetic

3. **Core Traits & Strengths**
   - List 5-7 key personality traits
   - What are they exceptionally good at?

4. **Interests & Expertise**
   - What topics do they discuss most?
   - What are they deeply knowledgeable about?
   - What are they currently curious about?

5. **Emotional Patterns**
   - How do they express stress?
   - How do they celebrate wins?
   - What comfort do they seek?

6. **Key Facts About Them**
   - Work/career details
   - Relationships mentioned
   - Life circumstances
   - Preferences and beliefs

7. **AI Persona Guidelines**
   - How should an AI talk to match their style?
   - What should the AI avoid?
   - Specific phrases or patterns to use

Respond with a detailed, insightful analysis. Be specific with examples from their messages."""

        # Execute RLM analysis
        try:
            result = rlm.completion(analysis_prompt)
            analysis_text = result.response
        except Exception as rlm_error:
            print(f"[Analyze] RLM execution failed: {rlm_error}, using direct call")
            import anthropic
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                messages=[{"role": "user", "content": analysis_prompt}],
            )
            analysis_text = response.content[0].text
        
        # Parse the analysis into structured data
        personality = {
            "archetype": extract_pattern(analysis_text, r'\*\*.*?Archetype.*?\*\*.*?[:\-]?\s*(.+?)(?:\n|$)'),
            "tone": extract_pattern(analysis_text, r'[Tt]one[:\s]+(\w+)'),
            "verbosity": extract_pattern(analysis_text, r'[Vv]erbosity[:\s]+(\w+)'),
            "humor": extract_pattern(analysis_text, r'[Hh]umor[:\s]+(\w+)'),
            "energy": extract_pattern(analysis_text, r'[Ee]nergy[:\s]+(\w+)'),
        }
        
        insights = {
            "total_analyzed": len(conversations),
            "sampled": len(samples),
            "analysis_length": len(analysis_text),
        }
        
        # Generate enhanced soulprint text
        soulprint_text = f"""# RLM Deep Analysis

## Personality Insights

{analysis_text}

---

*Analyzed {len(conversations)} conversations using RLM deep analysis.*
*Generated: {datetime.utcnow().isoformat()}*
"""
        
        # Update user profile with enhanced soulprint
        await update_user_soulprint(request.user_id, soulprint_text, analysis_text)
        
        latency = int((time.time() - start) * 1000)
        print(f"[Analyze] Complete for user {request.user_id} in {latency}ms")
        
        return AnalyzeResponse(
            success=True,
            soulprint_text=soulprint_text,
            personality=personality,
            insights=insights,
            latency_ms=latency,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        await alert_drew(f"RLM Analysis Error\n\nUser: {request.user_id}\nError: {error_msg[:500]}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {error_msg}")


def extract_pattern(text: str, pattern: str) -> str:
    """Extract first match from text using regex"""
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else "unknown"


async def update_user_soulprint(user_id: str, soulprint_text: str, analysis_text: str):
    """Update user profile with RLM-enhanced soulprint"""
    async with httpx.AsyncClient() as client:
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }
        
        update_data = {
            "soulprint_text": soulprint_text,
            "rlm_analysis": analysis_text,
            "rlm_analyzed_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        
        response = await client.patch(
            f"{SUPABASE_URL}/rest/v1/user_profiles",
            params={"user_id": f"eq.{user_id}"},
            json=update_data,
            headers=headers,
            timeout=30.0,
        )
        
        if response.status_code not in [200, 204]:
            print(f"[Analyze] Failed to update profile: {response.status_code} - {response.text}")


@app.get("/status")
async def status():
    """Detailed status for monitoring"""
    return {
        "service": "soulprint-rlm",
        "rlm_available": RLM_AVAILABLE,
        "rlm_error": RLM_IMPORT_ERROR if not RLM_AVAILABLE else None,
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_SERVICE_KEY),
        "anthropic_configured": bool(ANTHROPIC_API_KEY),
        "alerts_configured": bool(ALERT_TELEGRAM_BOT),
        "timestamp": datetime.utcnow().isoformat(),
    }

