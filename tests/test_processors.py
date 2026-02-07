"""
Unit tests for processor-specific logic.

Tests pure functions in conversation_chunker, fact_extractor, and memory_generator.
External API calls (Anthropic) are NOT tested here - those require integration tests.
"""
import pytest
from datetime import datetime

from processors.conversation_chunker import (
    estimate_tokens,
    chunk_conversations,
    format_conversation
)
from processors.fact_extractor import consolidate_facts
from processors.memory_generator import _fallback_memory


# ============================================================================
# conversation_chunker tests
# ============================================================================

def test_estimate_tokens_empty_string():
    """estimate_tokens returns 0 for empty string."""
    assert estimate_tokens("") == 0


def test_estimate_tokens_known_length():
    """estimate_tokens calculates correctly for known lengths."""
    # 40 characters / 4 = 10 tokens
    assert estimate_tokens("a" * 40) == 10


def test_estimate_tokens_short_string():
    """estimate_tokens handles short strings (< 4 chars)."""
    # 3 chars / 4 = 0 tokens (integer division)
    assert estimate_tokens("abc") == 0
    # 4 chars / 4 = 1 token
    assert estimate_tokens("test") == 1


def test_chunk_conversations_single_small_conversation():
    """Single small conversation creates exactly 1 chunk."""
    conversations = [
        {
            "id": "conv-123",
            "title": "Small Chat",
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"}
            ],
            "created_at": 1704067200  # Unix timestamp
        }
    ]

    chunks = chunk_conversations(conversations, target_tokens=2000)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["conversation_id"] == "conv-123"
    assert chunk["title"] == "Small Chat"
    assert chunk["chunk_index"] == 0
    assert chunk["total_chunks"] == 1
    assert chunk["chunk_tier"] == "medium"
    assert "User: Hi" in chunk["content"]
    assert "Assistant: Hello" in chunk["content"]


def test_chunk_conversations_empty_list():
    """Empty conversation list returns empty chunk list."""
    chunks = chunk_conversations([])
    assert chunks == []


def test_chunk_conversations_defaults_created_at():
    """chunk_conversations defaults created_at if not provided."""
    conversations = [
        {
            "id": "conv-456",
            "title": "No Date Chat",
            "messages": [
                {"role": "user", "content": "Test"}
            ]
        }
    ]

    chunks = chunk_conversations(conversations)

    assert len(chunks) == 1
    assert "created_at" in chunks[0]
    # Should be ISO format datetime
    assert isinstance(chunks[0]["created_at"], str)
    assert "T" in chunks[0]["created_at"]


def test_format_conversation_with_messages():
    """format_conversation handles simplified messages format."""
    conversation = {
        "title": "Test Chat",
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4."},
            {"role": "system", "content": "System message"},  # Should be skipped
            {"role": "user", "content": "Thanks!"}
        ]
    }

    formatted = format_conversation(conversation)

    assert "# Test Chat" in formatted
    assert "User: What is 2+2?" in formatted
    assert "Assistant: The answer is 4." in formatted
    assert "User: Thanks!" in formatted
    # System messages should be skipped
    assert "System message" not in formatted


def test_format_conversation_with_mapping():
    """format_conversation handles ChatGPT export mapping format."""
    conversation = {
        "id": "conv-789",
        "title": "Mapping Test",
        "mapping": {
            "node-1": {
                "parent": None,  # Root node
                "message": {
                    "author": {"role": "user"},
                    "content": {"parts": ["Hello there"]},
                    "create_time": 1704067200
                },
                "children": ["node-2"]
            },
            "node-2": {
                "parent": "node-1",
                "message": {
                    "author": {"role": "assistant"},
                    "content": {"parts": ["Hi! How can I help?"]},
                    "create_time": 1704067210
                },
                "children": []
            }
        }
    }

    formatted = format_conversation(conversation)

    assert "# Mapping Test" in formatted
    assert "User: Hello there" in formatted
    assert "Assistant: Hi! How can I help?" in formatted


def test_format_conversation_truncates_long_messages():
    """format_conversation truncates messages over 5000 chars."""
    long_content = "x" * 6000
    conversation = {
        "title": "Long Message",
        "messages": [
            {"role": "user", "content": long_content}
        ]
    }

    formatted = format_conversation(conversation)

    assert "[truncated]" in formatted
    assert len(formatted) < 6000  # Should be truncated


# ============================================================================
# fact_extractor tests
# ============================================================================

def test_consolidate_facts_deduplication():
    """consolidate_facts deduplicates facts across categories."""
    all_facts = [
        {
            "preferences": ["coffee", "tea", "coffee"],  # Duplicate "coffee"
            "projects": [
                {"name": "Project A", "description": "First"},
                {"name": "Project B", "description": "Second"}
            ],
            "dates": [],
            "beliefs": ["privacy matters"],
            "decisions": []
        },
        {
            "preferences": ["tea", "music"],  # Duplicate "tea"
            "projects": [
                {"name": "Project A", "description": "Duplicate"}  # Duplicate project name
            ],
            "dates": [{"event": "Launch", "date": "2024-01-01"}],
            "beliefs": ["privacy matters", "quality over speed"],  # Duplicate belief
            "decisions": [{"decision": "Use Python", "context": "Fast development"}]
        }
    ]

    consolidated = consolidate_facts(all_facts)

    # Check deduplication worked
    assert len(consolidated["preferences"]) == 3  # coffee, tea, music (deduplicated)
    assert "coffee" in consolidated["preferences"]
    assert "tea" in consolidated["preferences"]
    assert "music" in consolidated["preferences"]

    # Projects deduplicated by name (case-insensitive)
    assert len(consolidated["projects"]) == 2  # Project A once, Project B once

    # Dates should have 1 entry
    assert len(consolidated["dates"]) == 1
    assert consolidated["dates"][0]["event"] == "Launch"

    # Beliefs deduplicated
    assert len(consolidated["beliefs"]) == 2  # privacy matters, quality over speed

    # Decisions should have 1 entry
    assert len(consolidated["decisions"]) == 1

    # Total count should match
    expected_total = 3 + 2 + 1 + 2 + 1  # 9 unique facts
    assert consolidated["total_count"] == expected_total


def test_consolidate_facts_empty_input():
    """consolidate_facts handles empty input gracefully."""
    consolidated = consolidate_facts([])

    assert consolidated["preferences"] == []
    assert consolidated["projects"] == []
    assert consolidated["dates"] == []
    assert consolidated["beliefs"] == []
    assert consolidated["decisions"] == []
    assert consolidated["total_count"] == 0


def test_consolidate_facts_empty_categories():
    """consolidate_facts handles facts with empty categories."""
    all_facts = [
        {
            "preferences": ["coffee"],
            "projects": [],
            "dates": [],
            "beliefs": [],
            "decisions": []
        }
    ]

    consolidated = consolidate_facts(all_facts)

    assert len(consolidated["preferences"]) == 1
    assert consolidated["total_count"] == 1


# ============================================================================
# memory_generator tests
# ============================================================================

def test_fallback_memory_with_facts():
    """_fallback_memory creates valid markdown with fact count."""
    facts = {
        "preferences": ["coffee", "vim"],
        "projects": [{"name": "Project X"}],
        "dates": [],
        "beliefs": ["privacy matters"],
        "decisions": [],
        "total_count": 4
    }

    memory = _fallback_memory(facts)

    assert isinstance(memory, str)
    assert "# MEMORY" in memory
    assert "4" in memory  # Total count should appear
    assert "## Preferences" in memory
    assert "## Projects" in memory
    assert "## Important Dates" in memory
    assert "## Beliefs & Values" in memory
    assert "## Decisions & Context" in memory


def test_fallback_memory_empty_facts():
    """_fallback_memory handles empty facts dict."""
    facts = {
        "preferences": [],
        "projects": [],
        "dates": [],
        "beliefs": [],
        "decisions": [],
        "total_count": 0
    }

    memory = _fallback_memory(facts)

    assert isinstance(memory, str)
    assert "# MEMORY" in memory
    assert "0" in memory  # Zero facts
    assert "No data yet." in memory


def test_fallback_memory_missing_total_count():
    """_fallback_memory handles missing total_count field."""
    facts = {
        "preferences": [],
        "projects": [],
        "dates": [],
        "beliefs": [],
        "decisions": []
        # total_count missing
    }

    memory = _fallback_memory(facts)

    assert isinstance(memory, str)
    assert "# MEMORY" in memory
    # Should still work, showing 0 as default
