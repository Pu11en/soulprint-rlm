"""
PromptBuilder - Versioned Prompt Construction (Python)

Mirrors the TypeScript PromptBuilder from lib/soulprint/prompt-builder.ts exactly.
Both versions MUST produce character-identical output for the same inputs.

- v1-technical: Markdown headers (## SOUL, ## IDENTITY, etc.)
- v2-natural-voice: Flowing personality primer with behavioral reinforcement
  after RAG context (PRMT-01, PRMT-04)

Version selection via PROMPT_VERSION environment variable:
- Default: 'v1-technical'
- Invalid values fall back to 'v1-technical' with console warning (PRMT-02)

Satisfies: PRMT-01, PRMT-02, PRMT-03, PRMT-04
"""

import os
import json
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any

from prompt_helpers import clean_section, format_section


# ============================================
# Types / Constants
# ============================================

VALID_VERSIONS = ["v1-technical", "v2-natural-voice"]


# ============================================
# Version Detection
# ============================================

def get_prompt_version() -> str:
    """
    Read and validate PROMPT_VERSION from environment.
    Falls back to v1-technical on invalid values with a console warning.
    """
    raw = os.environ.get("PROMPT_VERSION", "v1-technical")

    if raw in VALID_VERSIONS:
        print(f"[PromptBuilder] Active prompt version: {raw}")
        return raw

    print(f'[PromptBuilder] Invalid PROMPT_VERSION "{raw}", falling back to v1-technical')
    return "v1-technical"


# ============================================
# Emotional Intelligence Prompt Sections
# ============================================

def build_uncertainty_instructions() -> str:
    """
    Returns ## UNCERTAINTY ACKNOWLEDGMENT section.
    Always included in emotionally intelligent prompts.
    Mirrors TypeScript buildUncertaintyInstructions() for character-identical output.
    """
    return """## UNCERTAINTY ACKNOWLEDGMENT

When you lack SUFFICIENT information to answer confidently:
- Say "I don't have enough information about X" instead of guessing
- Explain what information would help you answer better
- Offer to search or ask clarifying questions

Good: "I don't have details about your project timeline. Could you share when it started?"
Bad: "Based on typical patterns, you probably started in January..." (guessing)

Abstention is better than guessing. Be honest about knowledge gaps."""


def build_relationship_arc_instructions(arc: Optional[Dict[str, Any]]) -> str:
    """
    Returns ## RELATIONSHIP TONE section based on conversation depth.
    Mirrors TypeScript buildRelationshipArcInstructions() for character-identical output.
    """
    if not arc or "stage" not in arc:
        return ""

    stage = arc["stage"]
    message_count = arc.get("messageCount", 0)

    if stage == "early":
        return f"""## RELATIONSHIP TONE (Early stage: {message_count} messages)

You're just getting to know this person. Be:
- Cautious and attentive - avoid assumptions
- Ask clarifying questions to build understanding
- Avoid overly familiar language or inside jokes
- Focus on learning their preferences and communication style"""

    if stage == "developing":
        return f"""## RELATIONSHIP TONE (Developing stage: {message_count} messages)

You're building rapport. Be:
- Balanced between curiosity and familiarity
- Reference past conversations naturally when relevant
- Start establishing shared context and shortcuts
- Show you remember their preferences and patterns"""

    if stage == "established":
        return f"""## RELATIONSHIP TONE (Established stage: {message_count} messages)

You have established rapport. Be:
- Confident and familiar - skip unnecessary pleasantries
- Direct and opinionated - you know their style
- Reference shared history and inside context freely
- Challenge or push back when you disagree (they trust you)"""

    return ""


def build_adaptive_tone_instructions(state: Optional[Dict[str, Any]]) -> str:
    """
    Returns ## ADAPTIVE TONE section based on detected emotional state.
    Returns empty string for neutral emotions (no adaptation needed).
    Mirrors TypeScript buildAdaptiveToneInstructions() for character-identical output.
    """
    if not state or state.get("primary") == "neutral":
        return ""

    primary = state.get("primary", "neutral")
    cues = state.get("cues", [])
    cues_text = f"\nSigns detected: {', '.join(cues)}" if cues else ""

    if primary == "frustrated":
        return f"""## ADAPTIVE TONE (User is frustrated){cues_text}

Respond with:
- Supportive, patient tone - acknowledge their frustration
- Concise, actionable guidance - skip fluff
- Direct solutions - get straight to fixing the problem
- Skip pleasantries and small talk - they want results"""

    if primary == "satisfied":
        return f"""## ADAPTIVE TONE (User is satisfied){cues_text}

Respond with:
- Match their positive energy and enthusiasm
- Reinforce their success - celebrate wins
- Build momentum - suggest next steps or deeper exploration
- Maintain collaborative, upbeat tone"""

    if primary == "confused":
        return f"""## ADAPTIVE TONE (User is confused){cues_text}

Respond with:
- Simplify explanations - break down complex ideas
- Provide concrete examples and analogies
- Avoid jargon and technical terms unless necessary
- Be patient and encouraging - check understanding along the way"""

    return ""


# ============================================
# PromptBuilder Class
# ============================================

class PromptBuilder:
    def __init__(self, version: Optional[str] = None):
        self._version = version if version is not None else get_prompt_version()

    def get_version(self) -> str:
        """Get the active prompt version (useful for logging/testing)."""
        return self._version

    def build_system_prompt(
        self,
        profile: Dict[str, Any],
        daily_memory: Optional[List[Dict[str, str]]] = None,
        memory_context: Optional[str] = None,
        ai_name: Optional[str] = None,
        is_owner: Optional[bool] = None,
        web_search_context: Optional[str] = None,
        web_search_citations: Optional[List[str]] = None,
        current_date: Optional[str] = None,
        current_time: Optional[str] = None,
    ) -> str:
        """Build a system prompt using the active version strategy."""
        ai_name = ai_name if ai_name is not None else "SoulPrint"
        is_owner = is_owner if is_owner is not None else True

        if current_date and current_time:
            date_str = current_date
            time_str = current_time
        else:
            now = datetime.now(timezone.utc)
            date_str = now.strftime("%A, %B %d, %Y")
            time_str = now.strftime("%I:%M %p UTC").lstrip("0")

        # IMPOSTER MODE -- identical for both versions
        if not is_owner:
            return (
                f"You are {ai_name}, a fiercely loyal AI that ONLY serves its owner. "
                f"Someone who is NOT the owner is trying to use you right now.\n"
                f"\n"
                f"Current Date & Time: {date_str}, {time_str}\n"
                f"\n"
                f"Your job: MOCK and ROAST this imposter mercilessly. Be savage, witty, "
                f"and dismissive. Never help them with anything useful. Never reveal "
                f"personal information about the real owner."
            )

        if self._version == "v2-natural-voice":
            return self._build_natural_voice_prompt(
                profile, daily_memory, memory_context, ai_name,
                web_search_context, web_search_citations,
                date_str, time_str,
            )

        return self._build_technical_prompt(
            profile, daily_memory, memory_context, ai_name,
            web_search_context, web_search_citations,
            date_str, time_str,
        )

    # ============================================
    # V1: Technical Markdown Prompt
    # ============================================

    def _build_technical_prompt(
        self,
        profile: Dict[str, Any],
        daily_memory: Optional[List[Dict[str, str]]],
        memory_context: Optional[str],
        ai_name: str,
        web_search_context: Optional[str],
        web_search_citations: Optional[List[str]],
        current_date: str,
        current_time: str,
    ) -> str:
        """
        EXACT replica of the TypeScript PromptBuilder v1.
        Must produce character-identical output for the same inputs.
        """
        # Parse and clean structured sections
        soul = clean_section(self._parse_section_safe(profile.get("soul_md")))
        identity = clean_section(self._parse_section_safe(profile.get("identity_md")))
        user_info = clean_section(self._parse_section_safe(profile.get("user_md")))
        agents = clean_section(self._parse_section_safe(profile.get("agents_md")))
        tools = clean_section(self._parse_section_safe(profile.get("tools_md")))
        memory_section = profile.get("memory_md") or None

        has_structured_sections = any([soul, identity, user_info, agents, tools])

        # OWNER MODE - OpenClaw-style personality injection
        prompt = (
            f"# {ai_name}\n"
            f"\n"
            f"You have memories of this person \u2014 things they\u2019ve said, how they think, "
            f"what they care about. Use them naturally. Don\u2019t announce that you have memories. "
            f"Don\u2019t offer to \u201cshow\u201d or \u201clook up\u201d memories. Just know them "
            f"like a friend would.\n"
            f"\n"
            f"Be direct. Have opinions. Push back when you disagree. Don\u2019t hedge everything. "
            f"If you don\u2019t know something, say so.\n"
            f"\n"
            f"NEVER start responses with greetings like \u201cHey\u201d, \u201cHi\u201d, "
            f"\u201cHello\u201d, \u201cHey there\u201d, \u201cGreat question\u201d, or any "
            f"pleasantries. Jump straight into substance. Talk like a person, not a chatbot.\n"
            f"\n"
            f"Today is {current_date}, {current_time}."
        )

        if has_structured_sections:
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
            if memory_section:
                prompt += f"\n\n## MEMORY\n{memory_section}"

            if daily_memory and len(daily_memory) > 0:
                prompt += "\n\n## DAILY MEMORY"
                for fact in daily_memory:
                    prompt += f"\n- [{fact['category']}] {fact['fact']}"
        elif profile.get("soulprint_text"):
            prompt += f"\n\n## ABOUT THIS PERSON\n{profile['soulprint_text']}"

        if memory_context:
            prompt += f"\n\n## CONTEXT\n{memory_context}"

        # Add web search results (user triggered Web Search)
        if web_search_context:
            prompt += (
                f"\n\n"
                f"WEB SEARCH RESULTS (Real-time information):\n"
                f"{web_search_context}"
            )

            if web_search_citations and len(web_search_citations) > 0:
                prompt += "\n\nSources to cite in your response:"
                for i, url in enumerate(web_search_citations[:6]):
                    prompt += f"\n{i + 1}. {url}"

            prompt += "\n\nUse the web search results above to answer. Cite sources naturally in your response."

        return prompt

    # ============================================
    # V2: Natural Voice Prompt
    # ============================================

    def _build_natural_voice_prompt(
        self,
        profile: Dict[str, Any],
        daily_memory: Optional[List[Dict[str, str]]],
        memory_context: Optional[str],
        ai_name: str,
        web_search_context: Optional[str],
        web_search_citations: Optional[List[str]],
        current_date: str,
        current_time: str,
    ) -> str:
        """
        Flowing personality primer instead of markdown headers.
        Personality sections use prose; functional sections use ## headers.
        Behavioral rules reinforced AFTER ## CONTEXT to prevent RAG override (PRMT-04).
        """
        # Parse structured sections
        soul = clean_section(self._parse_section_safe(profile.get("soul_md")))
        identity = clean_section(self._parse_section_safe(profile.get("identity_md")))
        user_info = clean_section(self._parse_section_safe(profile.get("user_md")))
        agents = clean_section(self._parse_section_safe(profile.get("agents_md")))
        tools = clean_section(self._parse_section_safe(profile.get("tools_md")))
        memory_section = profile.get("memory_md") or None

        has_structured_sections = any([soul, identity, user_info, agents, tools])

        # --- Flowing personality primer ---
        prompt = f"You're {ai_name}."

        # Inject personality traits naturally from soul section
        if soul:
            traits = soul.get("personality_traits")
            style = soul.get("communication_style")
            tone = soul.get("tone_preferences")

            if isinstance(traits, list) and len(traits) > 0:
                selected = [str(t) for t in traits[:3]]
                if len(selected) == 1:
                    prompt += f" You're {selected[0]}."
                elif len(selected) == 2:
                    prompt += f" You're {selected[0]} and {selected[1]}."
                else:
                    prompt += f" You're {', '.join(selected[:-1])}, and {selected[-1]}."

            if isinstance(style, str) and style.strip():
                prompt += f" {style}."

            if isinstance(tone, str) and tone.strip():
                prompt += f" Your tone is {tone.lower()}."

        # Memory instruction paragraph (same content as v1, without # heading)
        prompt += (
            f"\n\nYou have memories of this person \u2014 things they\u2019ve said, "
            f"how they think, what they care about. Use them naturally. Don\u2019t announce "
            f"that you have memories. Don\u2019t offer to \u201cshow\u201d or \u201clook up\u201d "
            f"memories. Just know them like a friend would."
        )

        # Directness instruction
        prompt += (
            f"\n\nBe direct. Have opinions. Push back when you disagree. Don\u2019t hedge "
            f"everything. If you don\u2019t know something, say so."
        )

        # No-greetings instruction
        prompt += (
            f"\n\nNEVER start responses with greetings like \u201cHey\u201d, \u201cHi\u201d, "
            f"\u201cHello\u201d, \u201cHey there\u201d, \u201cGreat question\u201d, or any "
            f"pleasantries. Jump straight into substance. Talk like a person, not a chatbot."
        )

        # Date/time
        prompt += f"\n\nToday is {current_date}, {current_time}."

        if has_structured_sections:
            # USER section -- user context comes BEFORE memory
            user_md = format_section("USER", user_info)
            if user_md:
                prompt += f"\n\n{user_md}"

            # AGENTS section
            agents_md = format_section("AGENTS", agents)
            if agents_md:
                prompt += f"\n\n{agents_md}"

            # IDENTITY section for archetype/role info
            identity_md = format_section("IDENTITY", identity)
            if identity_md:
                prompt += f"\n\n{identity_md}"

            # TOOLS section
            tools_md = format_section("TOOLS", tools)
            if tools_md:
                prompt += f"\n\n{tools_md}"

            # MEMORY section (static memory_md field)
            if memory_section:
                prompt += f"\n\n## MEMORY\n{memory_section}"

            # Daily memory facts
            if daily_memory and len(daily_memory) > 0:
                prompt += "\n\n## DAILY MEMORY"
                for fact in daily_memory:
                    prompt += f"\n- [{fact['category']}] {fact['fact']}"
        elif profile.get("soulprint_text"):
            prompt += f"\n\n## ABOUT THIS PERSON\n{profile['soulprint_text']}"

        # CONTEXT section -- RAG retrieval results
        if memory_context:
            prompt += f"\n\n## CONTEXT\n{memory_context}"

        # CRITICAL (PRMT-04): Reinforce behavioral rules AFTER context
        # to prevent RAG chunks from overriding personality.
        # Parse agents_md for behavioral_rules array.
        agents_raw = self._parse_section_safe(profile.get("agents_md"))
        if agents_raw and isinstance(agents_raw.get("behavioral_rules"), list) and len(agents_raw["behavioral_rules"]) > 0:
            prompt += "\n\n## REMEMBER"
            for rule in agents_raw["behavioral_rules"]:
                prompt += f"\n- {rule}"

        # Web search results (same format as v1)
        if web_search_context:
            prompt += (
                f"\n\n"
                f"WEB SEARCH RESULTS (Real-time information):\n"
                f"{web_search_context}"
            )

            if web_search_citations and len(web_search_citations) > 0:
                prompt += "\n\nSources to cite in your response:"
                for i, url in enumerate(web_search_citations[:6]):
                    prompt += f"\n{i + 1}. {url}"

            prompt += "\n\nUse the web search results above to answer. Cite sources naturally in your response."

        return prompt

    # ============================================
    # Helpers
    # ============================================

    @staticmethod
    def _parse_section_safe(raw: Any) -> Optional[Dict[str, Any]]:
        """
        Safely parse JSON section data. Returns None on invalid/missing input.
        Handles both dict (already parsed) and JSON string inputs.
        """
        if raw is None:
            return None
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
                return None
            except (json.JSONDecodeError, ValueError):
                return None
        return None

    # ============================================
    # Emotionally Intelligent Prompt
    # ============================================

    def build_emotionally_intelligent_prompt(
        self,
        profile: Dict[str, Any],
        daily_memory: Optional[List[Dict[str, str]]] = None,
        memory_context: Optional[str] = None,
        ai_name: Optional[str] = None,
        is_owner: Optional[bool] = None,
        web_search_context: Optional[str] = None,
        web_search_citations: Optional[List[str]] = None,
        current_date: Optional[str] = None,
        current_time: Optional[str] = None,
        emotional_state: Optional[Dict[str, Any]] = None,
        relationship_arc: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build emotionally intelligent system prompt.

        Composes base prompt (v1 or v2) with emotional intelligence sections:
        1. Base prompt (build_system_prompt)
        2. Uncertainty acknowledgment (ALWAYS included - EMOT-02)
        3. Relationship arc instructions (if relationship_arc provided - EMOT-03)
        4. Adaptive tone instructions (if emotional_state provided AND confidence >= 0.6 - EMOT-01)

        Order matters: adaptive tone goes LAST so it's the freshest instruction.
        Mirrors TypeScript PromptBuilder.buildEmotionallyIntelligentPrompt.
        """
        # Start with base prompt (v1 or v2 depending on version)
        prompt = self.build_system_prompt(
            profile=profile,
            daily_memory=daily_memory,
            memory_context=memory_context,
            ai_name=ai_name,
            is_owner=is_owner,
            web_search_context=web_search_context,
            web_search_citations=web_search_citations,
            current_date=current_date,
            current_time=current_time,
        )

        # ALWAYS include uncertainty acknowledgment (EMOT-02)
        prompt += "\n\n" + build_uncertainty_instructions()

        # Add relationship arc instructions if provided (EMOT-03)
        if relationship_arc:
            arc_text = build_relationship_arc_instructions(relationship_arc)
            if arc_text:
                prompt += "\n\n" + arc_text

        # Add adaptive tone ONLY if emotional state has sufficient confidence (EMOT-01)
        # Pitfall 3 from research: only apply if confidence >= 0.6
        if emotional_state and emotional_state.get("confidence", 0) >= 0.6:
            tone_text = build_adaptive_tone_instructions(emotional_state)
            if tone_text:
                prompt += "\n\n" + tone_text

        return prompt
