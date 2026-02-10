"""
Conversation Sampling for Quick Pass Pipeline

Selects the richest conversations from a user's ChatGPT export
and formats them as readable text for Haiku 4.5 analysis.

Ported from lib/soulprint/sample.ts
"""

from typing import List, Dict, Any
from datetime import datetime

# Constants matching TypeScript exactly
MIN_MESSAGES = 4
DEFAULT_TARGET_TOKENS = 50000
HARD_CAP = 50
MIN_SELECTED = 5
CHARS_PER_TOKEN = 4
MAX_MESSAGE_LENGTH = 2000


def sample_conversations(
    conversations: List[Dict[str, Any]],
    target_tokens: int = DEFAULT_TARGET_TOKENS
) -> List[Dict[str, Any]]:
    """
    Sample the richest conversations from a parsed ChatGPT export.

    Scoring algorithm:
    - Message count * 10 (prefer multi-turn conversations)
    - Sum of user message lengths capped at 500 chars each (prefer substantive messages)
    - Min(user, assistant) count * 20 (prefer balanced conversations)
    - Slight recency bonus from createdAt timestamp

    Filters out conversations with fewer than 4 messages.
    Selects conversations until the token budget is met.
    Forces at least 5 conversations if available (even if over budget).
    Hard-capped at 50 conversations.

    Args:
        conversations: Parsed conversation array from ChatGPT export
        target_tokens: Approximate token budget (default 50,000)

    Returns:
        Subset of conversations ranked by richness within token budget
    """
    # Filter out short conversations
    eligible = [c for c in conversations if len(c.get('messages', [])) >= MIN_MESSAGES]

    print(f"[sample_conversations] Filtering: {len(conversations)} total, {len(eligible)} eligible (min {MIN_MESSAGES} messages)")

    if len(eligible) == 0:
        print("[sample_conversations] WARNING: No conversations with enough messages, returning all conversations")
        return conversations[:HARD_CAP]

    # Score each conversation
    scored = []
    for conv in eligible:
        messages = conv.get('messages', [])
        total_chars = sum(len(m.get('content', '')) for m in messages)

        user_messages = [m for m in messages if m.get('role') == 'user']
        assistant_messages = [m for m in messages if m.get('role') == 'assistant']

        # Calculate score
        score = (
            # Prefer conversations with many messages (back-and-forth)
            len(messages) * 10 +
            # Prefer conversations with substantial user messages (capped at 500 chars each)
            sum(min(len(m.get('content', '')), 500) for m in user_messages) +
            # Prefer balanced conversations (both user and assistant)
            min(len(user_messages), len(assistant_messages)) * 20
        )

        # Add slight recency bonus
        try:
            created_at = conv.get('createdAt', '')
            if created_at:
                timestamp = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                score += timestamp.timestamp() / 1e12
        except (ValueError, AttributeError):
            pass  # Skip recency bonus if date parsing fails

        scored.append({
            'conv': conv,
            'score': score,
            'chars': total_chars
        })

    # Sort by score descending
    scored.sort(key=lambda x: x['score'], reverse=True)

    # Select conversations within token budget
    target_chars = target_tokens * CHARS_PER_TOKEN
    # Hard limit: Haiku 4.5 has 200K token context, system prompt ~2K tokens
    # At 4 chars/token, that's ~792K chars max. Use 600K to be safe.
    ABSOLUTE_CHAR_LIMIT = 600_000
    selected = []
    total_chars = 0

    for item in scored:
        conv = item['conv']
        chars = item['chars']

        # Never exceed absolute limit (prevents blowing past model context)
        if total_chars + chars > ABSOLUTE_CHAR_LIMIT and len(selected) > 0:
            break

        if total_chars + chars > target_chars:
            # Force-include up to MIN_SELECTED, but only if this single
            # conversation won't blow past the absolute limit on its own
            if len(selected) < MIN_SELECTED and total_chars + chars <= ABSOLUTE_CHAR_LIMIT:
                selected.append(conv)
                total_chars += chars
                continue
            # Over budget and have enough -- skip remaining
            continue

        selected.append(conv)
        total_chars += chars

        if len(selected) >= HARD_CAP:
            break

    top_score = scored[0]['score'] if scored else 0
    print(f"[sample_conversations] Selected: {len(selected)} conversations, {total_chars} chars (target: {target_chars}), top score: {top_score:.2f}")

    return selected


def format_conversations_for_prompt(conversations: List[Dict[str, Any]]) -> str:
    """
    Format sampled conversations as human-readable text for the LLM prompt.

    Output format:
    ```
    === Conversation: "Title Here" (YYYY-MM-DD) ===
    User: message content
    Assistant: response content
    ```

    Messages longer than 2000 characters are truncated with "... [truncated]".

    Args:
        conversations: Sampled conversations to format

    Returns:
        Formatted text block suitable for inclusion in a prompt
    """
    blocks = []

    for conv in conversations:
        # Extract date (YYYY-MM-DD)
        created_at = conv.get('createdAt', '')
        date = created_at[:10] if created_at else 'unknown'

        title = conv.get('title', 'Untitled')
        header = f'=== Conversation: "{title}" ({date}) ==='

        message_lines = []
        messages = conv.get('messages', [])

        for msg in messages:
            role = msg.get('role', 'unknown')
            # Capitalize role
            if role == 'user':
                role_label = 'User'
            elif role == 'assistant':
                role_label = 'Assistant'
            else:
                role_label = role

            content = msg.get('content', '')

            # Truncate long messages
            if len(content) > MAX_MESSAGE_LENGTH:
                content = content[:MAX_MESSAGE_LENGTH] + '... [truncated]'

            message_lines.append(f'{role_label}: {content}')

        blocks.append('\n'.join([header] + message_lines))

    result = '\n\n'.join(blocks)

    # Safety truncation: Haiku 4.5 has 200K token context (~800K chars).
    # System prompt uses ~2K tokens. Cap user content at 180K tokens = 720K chars.
    MAX_PROMPT_CHARS = 720_000
    if len(result) > MAX_PROMPT_CHARS:
        print(f"[format_conversations_for_prompt] Truncating from {len(result)} to {MAX_PROMPT_CHARS} chars (model context limit)")
        result = result[:MAX_PROMPT_CHARS] + "\n\n[... remaining conversations truncated to fit model context ...]"

    print(f"[format_conversations_for_prompt] Formatted {len(conversations)} conversations, {len(result)} chars")

    return result
