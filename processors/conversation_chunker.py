"""
Conversation Chunker
Splits conversations into ~2000 token segments with overlap for fact extraction and RAG
"""
from datetime import datetime
from typing import List, Dict

from .dag_parser import extract_active_path


def estimate_tokens(text: str) -> int:
    """
    Simple token estimator for chunking purposes.
    Approximates tokens as len(text) // 4 (chars per token).

    Args:
        text: Input text string

    Returns:
        Estimated token count
    """
    return len(text) // 4


def format_conversation(conversation: dict) -> str:
    """
    Format a conversation dict into a readable text string.
    Handles both ChatGPT export format (with mapping) and simplified format (with messages).

    Args:
        conversation: Dict with either {"title": str, "mapping": dict} or {"title": str, "messages": list}

    Returns:
        Formatted conversation string with User/Assistant exchanges
    """
    title = conversation.get("title", "Untitled")
    formatted_lines = [f"# {title}\n"]

    # Check if this is the simplified format (already parsed)
    if "messages" in conversation:
        # Simplified format: {"title": str, "messages": [{"role": str, "content": str}]}
        messages = conversation["messages"]
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            # Skip system messages
            if role == "system":
                continue

            # Truncate very long messages
            if len(content) > 5000:
                content = content[:5000] + "... [truncated]"

            # Format based on role
            if role == "user":
                formatted_lines.append(f"User: {content}")
            elif role == "assistant":
                formatted_lines.append(f"Assistant: {content}")

        return "\n".join(formatted_lines)

    # Otherwise, handle ChatGPT export format with mapping (or no messages)
    # Use DAG traversal to extract only active conversation path
    parsed_messages = extract_active_path(conversation)

    if not parsed_messages:
        return f"# {title}\n[No messages found]"

    for msg in parsed_messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        # Truncate very long messages
        if len(content) > 5000:
            content = content[:5000] + "... [truncated]"

        if role == "user":
            formatted_lines.append(f"User: {content}")
        elif role == "assistant":
            formatted_lines.append(f"Assistant: {content}")

    return "\n".join(formatted_lines)


def chunk_conversations(
    conversations: list,
    target_tokens: int = 2000,
    overlap_tokens: int = 200
) -> List[Dict]:
    """
    Chunk conversations into segments for fact extraction and storage.

    Each conversation is formatted and split at sentence boundaries with overlap.
    Small conversations (under target_tokens) remain as single chunks.

    Args:
        conversations: List of conversation dicts
        target_tokens: Target size for each chunk (default 2000)
        overlap_tokens: Token overlap between chunks for context continuity (default 200)

    Returns:
        List of chunk dicts with conversation_id, title, content, token_count, chunk_index, etc.
    """
    all_chunks = []

    for conv_idx, conversation in enumerate(conversations):
        # Get conversation metadata
        title = conversation.get("title", "Untitled")
        conversation_id = conversation.get("id", f"conv_{conv_idx}")

        # Get created_at timestamp (try different fields)
        created_at = conversation.get("created_at") or conversation.get("create_time")
        if created_at and isinstance(created_at, (int, float)):
            # Convert Unix timestamp to ISO format
            created_at = datetime.fromtimestamp(created_at).isoformat()
        elif not created_at:
            created_at = datetime.utcnow().isoformat()

        # Format conversation as text
        formatted_text = format_conversation(conversation)
        total_tokens = estimate_tokens(formatted_text)

        # If conversation fits in single chunk, don't split
        if total_tokens <= target_tokens:
            all_chunks.append({
                "conversation_id": str(conversation_id),
                "title": title,
                "content": formatted_text,
                "token_count": total_tokens,
                "chunk_index": 0,
                "total_chunks": 1,
                "chunk_tier": "medium",
                "created_at": created_at,
            })
            continue

        # Split at sentence boundaries with overlap
        # Split on common sentence endings: `. `, `? `, `! `, `\n`
        sentences = []
        current_pos = 0
        text = formatted_text

        while current_pos < len(text):
            # Find next sentence boundary
            next_period = text.find(". ", current_pos)
            next_question = text.find("? ", current_pos)
            next_exclaim = text.find("! ", current_pos)
            next_newline = text.find("\n", current_pos)

            # Get the earliest boundary
            boundaries = [b for b in [next_period, next_question, next_exclaim, next_newline] if b != -1]
            if not boundaries:
                # No more boundaries, take rest of text
                sentences.append(text[current_pos:])
                break

            boundary = min(boundaries)
            sentences.append(text[current_pos:boundary + 1])
            current_pos = boundary + 1

        # Group sentences into chunks
        chunks_for_conv = []
        current_chunk = ""
        current_chunk_tokens = 0

        for sentence in sentences:
            sentence_tokens = estimate_tokens(sentence)

            # If adding this sentence exceeds target, finalize current chunk
            if current_chunk_tokens + sentence_tokens > target_tokens and current_chunk:
                chunks_for_conv.append(current_chunk)

                # Start new chunk with overlap from end of previous chunk
                overlap_text = current_chunk[-overlap_tokens * 4:] if len(current_chunk) > overlap_tokens * 4 else current_chunk
                current_chunk = overlap_text + sentence
                current_chunk_tokens = estimate_tokens(current_chunk)
            else:
                current_chunk += sentence
                current_chunk_tokens += sentence_tokens

        # Add final chunk
        if current_chunk:
            chunks_for_conv.append(current_chunk)

        # Convert to chunk dicts
        total_chunks = len(chunks_for_conv)
        for chunk_idx, chunk_content in enumerate(chunks_for_conv):
            all_chunks.append({
                "conversation_id": str(conversation_id),
                "title": title,
                "content": chunk_content,
                "token_count": estimate_tokens(chunk_content),
                "chunk_index": chunk_idx,
                "total_chunks": total_chunks,
                "chunk_tier": "medium",
                "created_at": created_at,
            })

    return all_chunks
