"""
MEMORY Generator
Creates structured markdown MEMORY section from consolidated facts using Claude Haiku 4.5
"""
import json


MEMORY_GENERATION_PROMPT = """You are creating a MEMORY section for a personal AI assistant. This section captures durable facts about the user -- things that should be remembered long-term to provide personalized, context-aware responses.

From the extracted facts below, create a structured markdown document with these sections:

## Preferences
- Communication preferences, tool preferences, workflow preferences, aesthetic preferences
- How they like to work, learn, and receive information

## Projects
- Active and past projects with names, descriptions, and key details
- Include tech stacks, timelines, and status when available

## Important Dates
- Birthdays, anniversaries, milestones, deadlines
- Career events, project launches, personal milestones

## Beliefs & Values
- Core principles that guide their decisions
- What they care about deeply (privacy, quality, speed, etc.)
- Philosophical or professional stances

## Decisions & Context
- Key decisions they've made with context for why
- Tradeoffs they've considered
- Evolution of thinking (if they changed their mind on something, note both the old and new position)

RULES:
- Only include facts with actual substance -- skip vague or generic items
- Prefer specific over general ("prefers Tailwind CSS" over "likes CSS frameworks")
- Include dates/timeframes when available
- If a category has no substantive facts, include the heading with "No data yet."
- Keep each bullet point to one clear, concise sentence
- Total output should be 1000-5000 tokens (roughly 4-20KB of markdown)

Extracted facts:
"""


async def generate_memory_section(consolidated_facts: dict, anthropic_client) -> str:
    """
    Generate a structured MEMORY section from consolidated facts.

    Args:
        consolidated_facts: Dict with preferences, projects, dates, beliefs, decisions
        anthropic_client: AsyncAnthropic client instance

    Returns:
        Markdown string with MEMORY section (human-readable, NOT JSON)
        Falls back to minimal MEMORY section on errors
    """
    try:
        # Format facts as readable JSON
        facts_json = json.dumps(consolidated_facts, indent=2)

        print(f"[MemoryGenerator] Generating MEMORY section from {consolidated_facts.get('total_count', 0)} facts")

        # Call Haiku 4.5 to generate MEMORY section
        response = await anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            temperature=0.5,  # Moderate temperature for natural writing
            messages=[{
                "role": "user",
                "content": MEMORY_GENERATION_PROMPT + "\n" + facts_json
            }]
        )

        # Extract markdown from response
        if not response.content or len(response.content) == 0:
            print("[MemoryGenerator] Empty response from Haiku")
            return _fallback_memory(consolidated_facts)

        memory_md = response.content[0].text

        print(f"[MemoryGenerator] Generated MEMORY section ({len(memory_md)} chars)")

        return memory_md

    except Exception as e:
        print(f"[MemoryGenerator] Error generating MEMORY section: {e}")
        return _fallback_memory(consolidated_facts)


def _fallback_memory(consolidated_facts: dict) -> str:
    """
    Create a minimal fallback MEMORY section when generation fails.

    Args:
        consolidated_facts: Dict with fact counts

    Returns:
        Minimal markdown MEMORY section
    """
    total_facts = consolidated_facts.get("total_count", 0)

    return f"""# MEMORY

Memory generation failed. Facts extracted but not yet organized.

Raw fact count: {total_facts}

## Preferences
No data yet.

## Projects
No data yet.

## Important Dates
No data yet.

## Beliefs & Values
No data yet.

## Decisions & Context
No data yet.
"""
