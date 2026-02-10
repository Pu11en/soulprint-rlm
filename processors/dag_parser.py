"""
DAG Parser for ChatGPT Export Format

Extracts the active conversation path from ChatGPT's DAG-based
conversation structure. ChatGPT conversations use a directed acyclic
graph where edits and regenerations create branches. The `current_node`
field marks the tip of the active path.

Three helpers:
  - extract_active_path: Backward traversal from current_node through parent chain
  - is_visible_message: Filter hidden messages (tool outputs, system, browsing)
  - extract_content: Extract all text from polymorphic content.parts
"""

from typing import List, Dict, Optional


def extract_active_path(conversation: dict) -> List[Dict]:
    """Extract only the active conversation path via backward DAG traversal.

    Starts at conversation["current_node"], follows the parent chain
    backward to root, then reverses to chronological order. Filters
    hidden messages and extracts full content from each visible message.

    Fallback: If current_node is missing or not in mapping, falls back
    to forward root traversal (find node with no parent, follow first
    child at each level). If mapping is missing entirely, checks for
    a pre-parsed "messages" key and returns those directly.

    Args:
        conversation: A single conversation dict from ChatGPT export

    Returns:
        List of {"role": str, "content": str, "create_time": float} dicts
        with only visible, non-empty messages in chronological order
    """
    # Handle pre-parsed format (already has messages key, no mapping)
    if "messages" in conversation and "mapping" not in conversation:
        return conversation["messages"]

    mapping = conversation.get("mapping", {})
    if not mapping:
        return []

    current_node = conversation.get("current_node")

    # Use backward traversal if current_node is available and valid
    if current_node and current_node in mapping:
        raw_messages = _backward_traversal(mapping, current_node)
    else:
        # Fallback: forward root traversal
        conv_id = conversation.get("id", "unknown")
        print(f"[dag_parser] WARNING: No current_node in conversation {conv_id}, using fallback root traversal")
        raw_messages = _forward_root_traversal(mapping)

    # Filter visible messages and extract content
    parsed_messages = []
    for msg in raw_messages:
        if not is_visible_message(msg):
            continue

        author = msg.get("author", {})
        role = author.get("role", "unknown")
        content = extract_content(msg.get("content", {}))

        if content.strip():
            parsed_messages.append({
                "role": role,
                "content": content,
                "create_time": msg.get("create_time", 0),
            })

    return parsed_messages


def _backward_traversal(mapping: dict, current_node: str) -> list:
    """Traverse from current_node backward through parent chain to root.

    Returns raw message objects in chronological order (reversed from
    collection order).
    """
    raw_messages = []
    node_id = current_node

    while node_id and node_id in mapping:
        node = mapping[node_id]
        message = node.get("message")
        if message:
            raw_messages.append(message)
        node_id = node.get("parent")

    # Reverse to chronological order (root -> current)
    raw_messages.reverse()
    return raw_messages


def _forward_root_traversal(mapping: dict) -> list:
    """Fallback: forward traversal from root, following first child at each level.

    Used when current_node is missing. Less accurate than backward
    traversal (may pick wrong branch at edit points) but still
    produces a valid conversation path.
    """
    # Find root node (no parent or parent not in mapping)
    root_id = None
    for node_id, node in mapping.items():
        parent = node.get("parent")
        if not parent or parent not in mapping:
            root_id = node_id
            break

    if not root_id:
        return []

    raw_messages = []
    node_id = root_id

    while node_id and node_id in mapping:
        node = mapping[node_id]
        message = node.get("message")
        if message:
            raw_messages.append(message)

        children = node.get("children", [])
        if children:
            # Follow the LAST child (most recent edit/response)
            node_id = children[-1]
        else:
            break

    return raw_messages


def is_visible_message(message: dict) -> bool:
    """Determine if a message should be included in parsed output.

    Filters out tool outputs (DALL-E, browsing, code interpreter),
    system messages (unless explicitly user-created), and unknown roles.

    Args:
        message: A raw message object from the conversation mapping

    Returns:
        True if the message should be visible to the user/soulprint
    """
    author = message.get("author", {})
    role = author.get("role")
    metadata = message.get("metadata", {})

    # Filter out tool outputs (DALL-E, browsing, code interpreter)
    if role == "tool":
        return False

    # Filter out system messages unless explicitly user-created
    if role == "system":
        return metadata.get("is_user_system_message", False) is True

    # Keep user and assistant messages
    if role in ("user", "assistant"):
        return True

    # Defensive default: exclude unknown roles
    return False


def extract_content(content_data) -> str:
    """Extract all text from a message's content field.

    Handles the polymorphic content.parts structure:
    - String parts: used directly
    - Dict parts with "text" key: text extracted
    - Dict parts with "asset_pointer": skipped (images)
    - Direct "text" field on content_data: used as-is
    - None/empty: returns empty string

    Args:
        content_data: The content field from a message (dict, str, or None)

    Returns:
        Concatenated text content, stripped of whitespace
    """
    if content_data is None:
        return ""

    if not content_data:
        return ""

    # Handle case where content_data is a plain string
    if isinstance(content_data, str):
        return content_data.strip()

    # Handle dict with direct "text" key (some format variants)
    if isinstance(content_data, dict) and "text" in content_data:
        return str(content_data["text"]).strip()

    # Handle dict with "parts" array (most common format)
    if isinstance(content_data, dict):
        parts = content_data.get("parts", [])
        text_parts = []

        for part in parts:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            # Skip other dict parts (images with asset_pointer, etc.)

        return "\n".join(text_parts).strip()

    return ""
