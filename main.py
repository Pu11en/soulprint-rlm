"""
SoulPrint RLM Service
Provides memory-enhanced chat using Recursive Language Models
"""
import os
import json
import httpx
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="SoulPrint RLM Service")

# CORS for Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://*.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ALERT_WEBHOOK = os.getenv("ALERT_WEBHOOK")  # Optional: for failure alerts


class QueryRequest(BaseModel):
    user_id: str
    message: str
    soulprint_text: Optional[str] = None
    history: Optional[List[dict]] = []


class QueryResponse(BaseModel):
    response: str
    chunks_used: int
    method: str  # "rlm" or "fallback"
    latency_ms: int


async def get_conversation_chunks(user_id: str, recent_only: bool = True) -> List[dict]:
    """Fetch conversation chunks from Supabase"""
    async with httpx.AsyncClient() as client:
        query = f"{SUPABASE_URL}/rest/v1/conversation_chunks"
        params = {
            "user_id": f"eq.{user_id}",
            "select": "conversation_id,title,content,message_count,created_at",
            "order": "created_at.desc",
            "limit": "100",
        }
        if recent_only:
            params["is_recent"] = "eq.true"
        
        response = await client.get(
            query,
            params=params,
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            },
        )
        
        if response.status_code != 200:
            raise Exception(f"Supabase error: {response.text}")
        
        return response.json()


async def alert_failure(error: str, user_id: str, message: str):
    """Alert Drew about failures"""
    if not ALERT_WEBHOOK:
        print(f"[ALERT] RLM failure for user {user_id}: {error}")
        return
    
    try:
        async with httpx.AsyncClient() as client:
            await client.post(ALERT_WEBHOOK, json={
                "text": f"🚨 SoulPrint RLM Failure\nUser: {user_id}\nError: {error}\nMessage: {message[:100]}",
            })
    except Exception as e:
        print(f"Failed to send alert: {e}")


async def query_with_rlm(
    message: str,
    conversation_context: str,
    soulprint_text: str,
    history: List[dict],
) -> str:
    """Query using RLM for recursive memory exploration"""
    try:
        from rlm import RLM
        
        rlm = RLM(
            backend="anthropic",
            backend_kwargs={
                "model_name": "claude-sonnet-4-20250514",
                "api_key": ANTHROPIC_API_KEY,
            },
            verbose=False,
        )
        
        # Build the context for RLM
        context = f"""You are SoulPrint, a personal AI assistant with access to the user's conversation history.

## User Profile (SoulPrint)
{soulprint_text or "No profile available yet."}

## Conversation History
The following is the user's conversation history that you can explore and reference:

{conversation_context}

## Current Conversation
{json.dumps(history[-5:] if history else [], indent=2)}

## Task
Respond to the user's message naturally, using relevant context from their history when appropriate.
Don't explicitly mention "according to your history" unless it's natural.
Be helpful, personalized, and conversational.

User message: {message}"""

        result = rlm.completion(context)
        return result.response
        
    except ImportError:
        # RLM not installed, use direct Anthropic
        raise Exception("RLM library not available")


async def query_fallback(
    message: str,
    conversation_context: str,
    soulprint_text: str,
    history: List[dict],
) -> str:
    """Fallback to direct Anthropic API if RLM fails"""
    import anthropic
    
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    system_prompt = f"""You are SoulPrint, a personal AI assistant with memory of the user's past conversations.

## User Profile
{soulprint_text or "No profile available yet."}

## Recent Conversation History (for context)
{conversation_context[:8000] if conversation_context else "No history available yet."}

Guidelines:
- Be warm, personalized, and helpful
- Reference relevant memories naturally when appropriate
- Don't overwhelm with information
- If you don't have relevant context, just be helpful in the moment"""

    messages = []
    for h in (history or [])[-10:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": message})

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=system_prompt,
        messages=messages,
    )
    
    return response.content[0].text


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "service": "soulprint-rlm"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Main query endpoint - uses RLM with fallback"""
    import time
    start = time.time()
    
    try:
        # Fetch conversation chunks
        chunks = await get_conversation_chunks(request.user_id, recent_only=True)
        
        # Build context from chunks
        conversation_context = ""
        for chunk in chunks[:50]:  # Limit to 50 most recent
            conversation_context += f"\n---\n**{chunk.get('title', 'Untitled')}** ({chunk.get('created_at', 'unknown date')})\n"
            conversation_context += chunk.get('content', '')[:2000]  # Truncate long convos
        
        # Try RLM first
        try:
            response = await query_with_rlm(
                message=request.message,
                conversation_context=conversation_context,
                soulprint_text=request.soulprint_text or "",
                history=request.history or [],
            )
            method = "rlm"
        except Exception as rlm_error:
            # Log and alert on RLM failure
            print(f"[RLM] Falling back due to: {rlm_error}")
            await alert_failure(str(rlm_error), request.user_id, request.message)
            
            # Fallback to direct API
            response = await query_fallback(
                message=request.message,
                conversation_context=conversation_context,
                soulprint_text=request.soulprint_text or "",
                history=request.history or [],
            )
            method = "fallback"
        
        latency_ms = int((time.time() - start) * 1000)
        
        return QueryResponse(
            response=response,
            chunks_used=len(chunks),
            method=method,
            latency_ms=latency_ms,
        )
        
    except Exception as e:
        await alert_failure(str(e), request.user_id, request.message)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
