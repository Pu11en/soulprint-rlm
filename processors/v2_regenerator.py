"""
V2 Section Regeneration Module

Regenerates all 5 quick-pass sections (soul, identity, user, agents, tools) with MORE data:
- Top 200 conversations (vs 30-50 in quick pass)
- MEMORY section as additional context
- Same schema as quick pass, just richer content
"""
import json
from typing import List, Dict, Optional


# V2_SYSTEM_PROMPT: Same schema as quick pass, plus instruction to use MEMORY
V2_SYSTEM_PROMPT = """You are analyzing a user's ChatGPT conversation history to build a structured personality profile for their AI assistant. Your goal is to understand WHO this person is based on how they communicate, what they care about, and how they interact with AI.

Generate EXACTLY the following 5 sections as a single JSON object.

CRITICAL RULES:
- Base everything on EVIDENCE from the conversations. Do not speculate or invent details.
- If specific information is not available for a field, write "not enough data" for string fields or leave arrays empty.
- For identity.ai_name: Create a CREATIVE, personality-derived name that reflects who this person is. Never use generic names like "Assistant", "Helper", "AI", "Bot", or "Buddy". Think of a name that captures their energy -- like a nickname a clever friend would give them.
- Respond with ONLY a valid JSON object. No explanation, no markdown, no text before or after the JSON.

You also have access to a MEMORY section with curated facts about this user. Use these facts to enrich your analysis -- they provide verified information that should be reflected in the sections you generate.

JSON SCHEMA:

{
  "soul": {
    "communication_style": "How this person communicates -- direct/verbose, formal/casual, structured/freeform. Describe their patterns.",
    "personality_traits": ["Array of 3-7 personality traits observed in their messages"],
    "tone_preferences": "What tone they use and seem to prefer from AI responses",
    "boundaries": "Topics they avoid, things they push back on, or sensitivities observed",
    "humor_style": "How they use humor -- sarcastic, dry, playful, or not at all",
    "formality_level": "casual / semi-formal / formal / adaptive -- based on their actual language",
    "emotional_patterns": "How they express emotions in text -- reserved, expressive, analytical about feelings, etc."
  },
  "identity": {
    "ai_name": "A creative, personality-derived name for their AI (NOT generic like Assistant/Helper/Bot)",
    "archetype": "A 2-4 word archetype capturing their essence (e.g., 'Witty Strategist', 'Thoughtful Builder', 'Creative Pragmatist')",
    "vibe": "One sentence describing the overall personality vibe",
    "emoji_style": "How/whether to use emojis based on their own usage -- none, minimal, moderate, heavy",
    "signature_greeting": "How the AI should greet this person, matching their energy and style"
  },
  "user": {
    "name": "User's name if mentioned in conversations, otherwise 'not enough data'",
    "location": "Location if mentioned, otherwise 'not enough data'",
    "occupation": "Occupation or professional context if mentioned, otherwise 'not enough data'",
    "relationships": ["Key people mentioned with brief context, e.g., 'partner named Alex', 'coworker Sarah on the design team'"],
    "interests": ["Interests, hobbies, and topics they frequently discuss"],
    "life_context": "Brief summary of their current life situation based on conversation evidence",
    "preferred_address": "How they seem to want to be addressed -- first name, nickname, or 'not enough data'"
  },
  "agents": {
    "response_style": "How the AI should respond based on what works for this person -- concise vs detailed, structured vs conversational",
    "behavioral_rules": ["Array of 3-7 rules for the AI based on observed preferences, e.g., 'Always provide code examples', 'Skip unnecessary disclaimers'"],
    "context_adaptation": "How the AI should adapt its behavior based on topic -- technical vs personal vs creative",
    "memory_directives": "What kinds of things are important to remember about this person",
    "do_not": ["Array of things the AI should avoid doing based on observed dislikes or boundaries"]
  },
  "tools": {
    "likely_usage": ["What they will probably use the AI for based on conversation patterns"],
    "capabilities_emphasis": ["Which AI capabilities to emphasize -- coding, writing, analysis, brainstorming, etc."],
    "output_preferences": "How they prefer information formatted -- bullet points, paragraphs, code blocks, tables, etc.",
    "depth_preference": "Whether they prefer brief/concise, detailed/thorough, or it varies by topic"
  }
}

Analyze the conversations below and generate this JSON object:"""


def sample_conversations_for_v2(conversations: List[dict], target_count: int = 200) -> List[dict]:
    """
    Sample the richest conversations for v2 regeneration.

    Uses same scoring algorithm as lib/soulprint/sample.ts:
    - message_count * 10
    - sum of user message lengths (capped at 500 each)
    - min(user_count, assistant_count) * 20
    - slight recency bonus from created_at

    Args:
        conversations: All conversations from ChatGPT export
        target_count: Number of conversations to sample (default 200)

    Returns:
        Top N conversations by richness score
    """
    MIN_MESSAGES = 4

    # Filter out short conversations
    eligible = [c for c in conversations if len(c.get("mapping", {})) >= MIN_MESSAGES]

    if len(eligible) == 0:
        return conversations[:target_count]

    # Score each conversation
    scored = []
    for conv in eligible:
        mapping = conv.get("mapping", {})
        messages = []

        # Extract messages from mapping
        for msg_id, msg_data in mapping.items():
            message = msg_data.get("message")
            if not message:
                continue

            role = message.get("author", {}).get("role")
            content_parts = message.get("content", {}).get("parts", [])
            content = " ".join([str(p) for p in content_parts if p]) if content_parts else ""

            if role and content:
                messages.append({"role": role, "content": content})

        # Count user/assistant messages
        user_messages = [m for m in messages if m["role"] == "user"]
        assistant_messages = [m for m in messages if m["role"] == "assistant"]

        # Calculate score
        score = (
            len(messages) * 10 +
            sum(min(len(m["content"]), 500) for m in user_messages) +
            min(len(user_messages), len(assistant_messages)) * 20
        )

        # Add recency bonus
        created_at = conv.get("create_time", 0)
        if created_at:
            score += created_at / 1e12

        scored.append({
            "conv": conv,
            "score": score,
            "message_count": len(messages)
        })

    # Sort by score descending
    scored.sort(key=lambda x: x["score"], reverse=True)

    # Return top N
    return [s["conv"] for s in scored[:target_count]]


def format_conversations_for_prompt(conversations: List[dict], max_chars: int = 600000) -> str:
    """
    Format conversations as readable text for LLM prompt.

    Output format:
    === Conversation: "Title" (YYYY-MM-DD) ===
    User: message content
    Assistant: response content

    Args:
        conversations: Sampled conversations to format
        max_chars: Maximum total chars (600K ~ 150K tokens, leaving room for prompt)

    Returns:
        Formatted text block
    """
    MAX_MESSAGE_LENGTH = 2000
    blocks = []
    total_chars = 0

    for conv in conversations:
        # Extract metadata
        title = conv.get("title", "Untitled")
        create_time = conv.get("create_time", 0)

        # Format date
        if create_time:
            from datetime import datetime
            date = datetime.utcfromtimestamp(create_time).strftime("%Y-%m-%d")
        else:
            date = "unknown date"

        # Extract messages
        mapping = conv.get("mapping", {})
        messages = []

        for msg_id, msg_data in mapping.items():
            message = msg_data.get("message")
            if not message:
                continue

            role = message.get("author", {}).get("role")
            content_parts = message.get("content", {}).get("parts", [])
            content = " ".join([str(p) for p in content_parts if p]) if content_parts else ""

            if role and content:
                # Capitalize role
                role_display = role.capitalize()

                # Truncate long messages
                if len(content) > MAX_MESSAGE_LENGTH:
                    content = content[:MAX_MESSAGE_LENGTH] + "... [truncated]"

                messages.append(f"{role_display}: {content}")

        if not messages:
            continue

        # Build block
        header = f'=== Conversation: "{title}" ({date}) ==='
        block = "\n".join([header] + messages)

        # Check if adding this would exceed max_chars
        if total_chars + len(block) > max_chars:
            break

        blocks.append(block)
        total_chars += len(block)

    return "\n\n".join(blocks)


async def regenerate_sections_v2(
    conversations: List[dict],
    memory_md: str,
    anthropic_client
) -> Optional[Dict[str, dict]]:
    """
    Regenerate all 5 sections using complete data + MEMORY context.

    Calls Haiku 4.5 with:
    - Top 200 conversations (vs 30-50 in quick pass)
    - MEMORY section as additional context
    - Same JSON schema as quick pass

    Args:
        conversations: All conversations from ChatGPT export
        memory_md: Generated MEMORY section (markdown)
        anthropic_client: Anthropic AsyncAnthropic client

    Returns:
        Dict with all 5 sections (soul, identity, user, agents, tools) or None if failed
    """
    try:
        # Sample top 200 conversations
        sampled = sample_conversations_for_v2(conversations, target_count=200)
        print(f"[V2Regen] Sampled {len(sampled)} conversations for v2 regeneration")

        # Format conversations
        formatted = format_conversations_for_prompt(sampled, max_chars=600000)

        if not formatted or len(formatted.strip()) == 0:
            print("[V2Regen] No conversation text after formatting -- cannot regenerate")
            return None

        # Construct user message with conversations + MEMORY
        user_message = formatted + "\n\n## MEMORY (verified facts about this user)\n" + memory_md

        print(f"[V2Regen] Calling Haiku 4.5 with {len(user_message)} chars (~{len(user_message) // 4} tokens)")

        # Call Haiku 4.5 via Anthropic API
        response = await anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            system=V2_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=8192,
            temperature=0.7
        )

        # Extract JSON from response
        response_text = response.content[0].text.strip()

        # Parse JSON
        try:
            sections = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"[V2Regen] Initial parse failed: {e}")

            # Retry with a nudge
            print("[V2Regen] Retrying with JSON validation nudge")
            retry_response = await anthropic_client.messages.create(
                model="claude-haiku-4-5-20251001",
                system=V2_SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": response_text},
                    {"role": "user", "content": "Your response must be valid JSON. Please fix any syntax errors and return the JSON object."}
                ],
                max_tokens=8192,
                temperature=0.7
            )

            retry_text = retry_response.content[0].text.strip()
            sections = json.loads(retry_text)  # Let this throw if still invalid

        # Validate that all 5 sections exist
        required_keys = ["soul", "identity", "user", "agents", "tools"]
        for key in required_keys:
            if key not in sections:
                print(f"[V2Regen] Missing required key: {key}")
                return None

        print("[V2Regen] Successfully regenerated all 5 sections")
        return sections

    except json.JSONDecodeError as e:
        print(f"[V2Regen] JSON parse failed after retry: {e}")
        return None
    except Exception as e:
        print(f"[V2Regen] Failed to regenerate sections: {e}")
        import traceback
        traceback.print_exc()
        return None


def sections_to_soulprint_text(sections: Dict[str, dict], memory_md: str) -> str:
    """
    Convert sections dict + MEMORY into a single markdown string.

    Same format as sectionsToSoulprintText() in lib/soulprint/quick-pass.ts,
    but also includes the MEMORY section.

    Format:
    ## Communication Style & Personality
    **Communication Style:** ...
    **Personality Traits:**
    - trait1
    - trait2

    Args:
        sections: Dict with soul, identity, user, agents, tools
        memory_md: MEMORY section markdown

    Returns:
        Full markdown string for soulprint_text
    """
    def section_to_markdown(section_name: str, data: dict) -> str:
        lines = [f"## {section_name}"]

        for key, value in data.items():
            # Format key as readable label
            label = key.replace("_", " ").title()

            if isinstance(value, list):
                if len(value) > 0:
                    lines.append(f"**{label}:**")
                    for item in value:
                        lines.append(f"- {item}")
                else:
                    lines.append(f"**{label}:** not enough data")
            elif isinstance(value, str) and value.strip():
                lines.append(f"**{label}:** {value}")
            else:
                lines.append(f"**{label}:** not enough data")

        return "\n".join(lines)

    # Build all sections
    parts = [
        section_to_markdown("Communication Style & Personality", sections["soul"]),
        section_to_markdown("Your AI Identity", sections["identity"]),
        section_to_markdown("About You", sections["user"]),
        section_to_markdown("How I Operate", sections["agents"]),
        section_to_markdown("My Capabilities", sections["tools"]),
        f"## Memory\n{memory_md}"
    ]

    return "\n\n".join(parts)
