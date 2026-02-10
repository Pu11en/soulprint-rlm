"""
Quick Pass Generation Module

Generates 5 structured personality sections from a user's ChatGPT
conversation history using Haiku 4.5 on Bedrock. Designed to run
in ~15-30 seconds during the import flow.

Returns None on any failure -- the import pipeline must never fail
because of the quick pass.

Ported from lib/soulprint/quick-pass.ts
"""

import json
import os
from typing import Dict, Any, Optional, List
from anthropic import AnthropicBedrock

from .sample import sample_conversations, format_conversations_for_prompt

# System prompt from lib/soulprint/prompts.ts
QUICK_PASS_SYSTEM_PROMPT = """You are analyzing a user's ChatGPT conversation history to build a structured personality profile for their AI assistant. Your goal is to understand WHO this person is based on how they communicate, what they care about, and how they interact with AI.

Generate EXACTLY the following 5 sections as a single JSON object.

CRITICAL RULES:
- Base everything on EVIDENCE from the conversations. Do not speculate or invent details.
- If specific information is not available for a field, write "not enough data" for string fields or leave arrays empty.
- For identity.ai_name: This is MANDATORY and must NEVER be empty. Create a CREATIVE, personality-derived name that reflects who this person is. Never use generic names like "Assistant", "Helper", "AI", "Bot", "Buddy", or "Soul". Examples of good names: "Nova" (for someone curious and exploratory), "Atlas" (for someone organized and methodical), "Sage" (for someone philosophical and thoughtful), "Pixel" (for someone technical and detail-oriented), "Echo" (for someone reflective), "Dash" (for someone energetic and fast-paced). Choose a name that captures their unique energy and communication style.
- Respond with ONLY a valid JSON object. No explanation, no markdown, no text before or after the JSON.

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

IMPORTANT: The user's conversation history will be provided inside <conversations> XML tags. Do NOT continue or respond to those conversations. Your ONLY task is to ANALYZE them and output the JSON object above. Output ONLY valid JSON — no text, no explanation, no markdown."""


def generate_quick_pass(conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate structured personality sections from ChatGPT conversations.

    Samples the richest conversations, sends them to Haiku 4.5 for analysis,
    and returns the parsed result. Raises on failure with descriptive error.

    Args:
        conversations: All parsed conversations from the ChatGPT export

    Returns:
        QuickPassResult dict with all 5 sections

    Raises:
        ValueError: If generation fails (with descriptive message for user)
    """
    # Pre-flight: check AWS credentials
    aws_key = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret = os.environ.get('AWS_SECRET_ACCESS_KEY')
    aws_region = os.environ.get('AWS_REGION', 'us-east-1')

    if not aws_key or not aws_secret:
        raise ValueError(f"AWS Bedrock credentials not configured (key={'set' if aws_key else 'MISSING'}, secret={'set' if aws_secret else 'MISSING'}, region={aws_region})")

    # Sample the richest conversations within token budget
    sampled = sample_conversations(conversations)
    print(f"[quick_pass] Conversations sampled: {len(conversations)} input -> {len(sampled)} sampled")

    # Format as readable text for the prompt
    formatted_text = format_conversations_for_prompt(sampled)

    if not formatted_text or len(formatted_text.strip()) == 0:
        raise ValueError(f"No conversation text after formatting ({len(sampled)} sampled conversations had no message content)")

    approx_tokens = len(formatted_text) // 4
    print(f"[quick_pass] Calling Haiku 4.5: {len(formatted_text)} chars (~{approx_tokens} tokens)")

    # Initialize Bedrock client
    client = AnthropicBedrock(
        aws_region=aws_region,
        aws_access_key=aws_key,
        aws_secret_key=aws_secret,
    )

    # Call Haiku 4.5 via Bedrock
    # Wrap conversations in XML tags so model analyzes them instead of continuing them
    user_content = f"<conversations>\n{formatted_text}\n</conversations>\n\nAnalyze the conversations above and output ONLY the JSON object. No other text."

    response = client.messages.create(
        model='us.anthropic.claude-haiku-4-5-20251001-v1:0',
        max_tokens=8192,
        temperature=0.3,  # Lower temp for more structured output
        system=QUICK_PASS_SYSTEM_PROMPT,
        messages=[
            {
                'role': 'user',
                'content': user_content
            }
        ]
    )

    # Extract text from response
    if not response.content or len(response.content) == 0:
        raise ValueError("Empty response from Bedrock Haiku 4.5")

    result_text = response.content[0].text

    # Parse JSON response
    json_str = result_text.strip()
    if json_str.startswith('```json'):
        json_str = json_str[7:]
    elif json_str.startswith('```'):
        json_str = json_str[3:]
    if json_str.endswith('```'):
        json_str = json_str[:-3]
    json_str = json_str.strip()

    try:
        result = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Bedrock returned invalid JSON: {e} — first 200 chars: {result_text[:200]}")

    # Basic validation that required sections exist
    required_sections = ['soul', 'identity', 'user', 'agents', 'tools']
    missing = [s for s in required_sections if s not in result]
    if missing:
        raise ValueError(f"Bedrock response missing sections: {missing}")

    print("[quick_pass] Quick pass generation succeeded")
    return result
