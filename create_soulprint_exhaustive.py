"""
Exhaustive Soulprint Generation
Processes ALL conversations in batches, then synthesizes
"""
import json
import time
import asyncio
from typing import List, Optional

# This will be integrated into main.py

async def extract_patterns_from_batch(batch: List[dict], batch_num: int, total_batches: int, anthropic_client) -> dict:
    """Extract patterns from a batch of conversations"""
    
    # Build conversation text for this batch
    conversation_text = ""
    for conv in batch:
        title = conv.get("title", "Untitled")
        messages = conv.get("messages", conv.get("mapping", []))
        
        if isinstance(messages, list):
            msg_excerpts = []
            for m in messages[:20]:  # More messages per convo
                if isinstance(m, dict):
                    role = m.get("role", m.get("author", {}).get("role", "unknown"))
                    content = m.get("content", "")
                    if isinstance(content, list):
                        content = " ".join([p.get("text", str(p)) for p in content if isinstance(p, dict)])
                    elif isinstance(content, dict):
                        content = content.get("parts", [content.get("text", str(content))])[0] if content else ""
                    content = str(content)[:500]  # More content per message
                    if content.strip():
                        msg_excerpts.append(f"{role}: {content}")
                elif isinstance(m, str):
                    msg_excerpts.append(f"user: {m[:500]}")
            
            if msg_excerpts:
                conversation_text += f"\n### {title}\n" + "\n".join(msg_excerpts) + "\n"
        elif isinstance(messages, dict):
            # Handle ChatGPT mapping format
            for node_id, node in list(messages.items())[:20]:
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
                            conversation_text += f"{role}: {text[:500]}\n"
    
    # Truncate batch text
    conversation_text = conversation_text[:40000]
    
    extraction_prompt = f"""You are analyzing conversations to extract personality patterns. This is batch {batch_num + 1} of {total_batches}.

## CONVERSATIONS
{conversation_text}

## TASK
Extract SPECIFIC patterns you observe. Be detailed and cite examples.

Return JSON:
{{
  "voice_patterns": {{
    "formality_examples": ["quote examples of formal/casual language"],
    "humor_examples": ["any jokes, sarcasm, playfulness"],
    "emoji_patterns": ["emojis used and when"],
    "punctuation_quirks": ["!!!", "...", etc],
    "greeting_styles": ["how they start messages"],
    "sign_off_styles": ["how they end messages"]
  }},
  "thinking_patterns": {{
    "explanation_style": "how they explain things (with examples)",
    "question_types": ["types of questions they ask"],
    "problem_approach": "how they tackle problems",
    "decision_language": ["phrases used when deciding"]
  }},
  "emotional_patterns": {{
    "enthusiasm_markers": ["words/phrases when excited"],
    "frustration_markers": ["words/phrases when frustrated"],
    "support_language": ["how they comfort others"],
    "vulnerability_examples": ["times they opened up"]
  }},
  "memory_anchors": {{
    "people_mentioned": ["names and relationships"],
    "places_mentioned": ["locations important to them"],
    "recurring_topics": ["subjects they return to"],
    "strong_opinions": ["things they feel strongly about"],
    "interests": ["hobbies, passions, curiosities"]
  }},
  "unique_markers": {{
    "catchphrases": ["repeated phrases"],
    "word_choices": ["distinctive vocabulary"],
    "sentence_structures": ["how they build sentences"]
  }}
}}

Be SPECIFIC. Quote actual text when possible. If a pattern isn't present in this batch, use empty arrays."""

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": extraction_prompt}],
    )
    
    # Parse JSON from response
    response_text = response.content[0].text
    try:
        # Try to extract JSON
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        return json.loads(response_text.strip())
    except:
        return {"raw_analysis": response_text}


async def synthesize_soulprint(all_patterns: List[dict], stats: dict, anthropic_client) -> dict:
    """Synthesize all extracted patterns into final soulprint"""
    
    # Merge all patterns
    merged = {
        "voice_patterns": [],
        "thinking_patterns": [],
        "emotional_patterns": [],
        "memory_anchors": [],
        "unique_markers": []
    }
    
    for patterns in all_patterns:
        for key in merged:
            if key in patterns:
                if isinstance(patterns[key], dict):
                    merged[key].append(patterns[key])
                elif isinstance(patterns[key], list):
                    merged[key].extend(patterns[key])
    
    synthesis_prompt = f"""You are creating the FINAL SoulPrint by synthesizing patterns extracted from ALL of someone's conversations.

## EXTRACTED PATTERNS FROM ALL CONVERSATIONS
{json.dumps(merged, indent=2)[:50000]}

## STATISTICS
{json.dumps(stats, indent=2) if stats else "No stats"}

## TASK
Create a comprehensive, deeply personalized SoulPrint. This should feel like you KNOW this person intimately.

Return JSON:
{{
  "archetype": "A creative 2-4 word title that captures their essence (e.g., 'The Relentless Builder', 'The Curious Connector')",
  
  "core_essence": "2-3 sentences capturing who they fundamentally are",
  
  "voice": {{
    "formality": "casual|mixed|formal",
    "humor": "dry|playful|sarcastic|warm|witty|none",
    "emoji_style": "heavy|moderate|minimal|strategic|none",
    "message_rhythm": "rapid-fire|thoughtful|varies-by-context",
    "signature_phrases": ["their catchphrases and verbal tics"],
    "punctuation_style": "description of their punctuation habits"
  }},
  
  "mind": {{
    "thinking_style": "how they process and explain ideas",
    "curiosity_drivers": ["what makes them want to learn more"],
    "problem_solving": "their approach to challenges",
    "decision_making": "how they make choices"
  }},
  
  "heart": {{
    "emotional_range": "how they express feelings",
    "enthusiasm_triggers": ["what excites them"],
    "frustration_patterns": "how they handle setbacks",
    "connection_style": "how they relate to others"
  }},
  
  "world": {{
    "key_relationships": ["important people in their life"],
    "core_interests": ["passions and hobbies"],
    "strong_beliefs": ["values and opinions they hold dear"],
    "recurring_themes": ["topics they return to"]
  }},
  
  "soulprint_text": "A rich 300-500 word narrative description that an AI could use to truly embody this person's communication style. Include specific examples and patterns. This should read like a deep character study.",
  
  "communication_guide": {{
    "do": ["specific things to do when writing as them"],
    "avoid": ["things that would feel inauthentic"],
    "context_shifts": ["how their style changes in different contexts"]
  }}
}}

Make this SPECIFIC and PERSONAL. Use the extracted examples. This soulprint should make anyone who reads it feel like they understand this person deeply."""

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
        messages=[{"role": "user", "content": synthesis_prompt}],
    )
    
    response_text = response.content[0].text
    try:
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        return json.loads(response_text.strip())
    except:
        return {
            "archetype": "The Unique Individual",
            "soulprint_text": response_text,
            "raw": True
        }


async def create_exhaustive_soulprint(conversations: List[dict], stats: dict, user_id: str, anthropic_client) -> dict:
    """
    Main function: Process ALL conversations exhaustively
    
    Flow:
    1. Split into batches of 50 conversations
    2. Extract patterns from each batch (can be parallelized)
    3. Synthesize all patterns into final soulprint
    """
    start = time.time()
    
    total = len(conversations)
    batch_size = 50
    batches = [conversations[i:i+batch_size] for i in range(0, total, batch_size)]
    total_batches = len(batches)
    
    print(f"[RLM] Exhaustive analysis: {total} conversations in {total_batches} batches")
    
    # Phase 1: Extract patterns from all batches
    all_patterns = []
    for i, batch in enumerate(batches):
        print(f"[RLM] Processing batch {i+1}/{total_batches} ({len(batch)} conversations)")
        patterns = await extract_patterns_from_batch(batch, i, total_batches, anthropic_client)
        all_patterns.append(patterns)
        # Small delay between batches to avoid rate limits
        if i < total_batches - 1:
            await asyncio.sleep(0.5)
    
    print(f"[RLM] Pattern extraction complete. Synthesizing final soulprint...")
    
    # Phase 2: Synthesize all patterns
    soulprint = await synthesize_soulprint(all_patterns, stats, anthropic_client)
    
    elapsed = time.time() - start
    print(f"[RLM] Exhaustive soulprint complete in {elapsed:.1f}s")
    
    return {
        "soulprint": soulprint,
        "archetype": soulprint.get("archetype", "Unknown"),
        "batches_processed": total_batches,
        "conversations_analyzed": total,
        "elapsed_seconds": round(elapsed, 1)
    }
