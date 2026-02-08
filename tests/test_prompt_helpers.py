"""
Unit tests for prompt_helpers module.

Tests clean_section and format_section functions to ensure they:
1. Remove "not enough data" placeholders
2. Generate consistent markdown formatting
3. Match TypeScript implementation behavior
"""
import pytest
from prompt_helpers import clean_section, format_section


def test_clean_section_removes_not_enough_data_strings():
    """clean_section should remove 'not enough data' string values."""
    data = {
        "name": "Drew",
        "placeholder": "not enough data",
        "bio": "Software engineer",
    }
    result = clean_section(data)
    assert result == {"name": "Drew", "bio": "Software engineer"}
    assert "placeholder" not in result


def test_clean_section_removes_not_enough_data_from_lists():
    """clean_section should filter out 'not enough data' from list values."""
    data = {
        "interests": ["coding", "not enough data", "music"],
        "skills": ["Python", "JavaScript"],
    }
    result = clean_section(data)
    assert result == {
        "interests": ["coding", "music"],
        "skills": ["Python", "JavaScript"],
    }


def test_clean_section_case_insensitive():
    """clean_section should match 'not enough data' case-insensitively."""
    data = {
        "field1": "Not Enough Data",
        "field2": "NOT ENOUGH DATA",
        "field3": "  not enough data  ",  # with whitespace
        "field4": "Valid data",
    }
    result = clean_section(data)
    assert result == {"field4": "Valid data"}


def test_clean_section_removes_empty_lists():
    """clean_section should remove lists that become empty after filtering."""
    data = {
        "valid": ["item1", "item2"],
        "all_placeholders": ["not enough data", "Not Enough Data"],
        "empty": [],
    }
    result = clean_section(data)
    assert result == {"valid": ["item1", "item2"]}
    assert "all_placeholders" not in result
    assert "empty" not in result


def test_clean_section_preserves_other_types():
    """clean_section should preserve numbers, booleans, and objects."""
    data = {
        "count": 42,
        "active": True,
        "ratio": 3.14,
        "metadata": {"key": "value"},
    }
    result = clean_section(data)
    assert result == data


def test_clean_section_returns_none_for_invalid_input():
    """clean_section should return None for None, non-dict, or empty dict."""
    assert clean_section(None) is None
    assert clean_section("string") is None
    assert clean_section(123) is None
    assert clean_section({}) is None


def test_format_section_basic():
    """format_section should generate markdown with heading and labeled fields."""
    data = {"name": "Drew", "role": "Developer"}
    result = format_section("Test Section", data)
    expected = "## Test Section\n**Name:** Drew\n**Role:** Developer"
    assert result == expected


def test_format_section_snake_case_to_title_case():
    """format_section should convert snake_case keys to Title Case."""
    data = {"user_name": "Drew", "favorite_color": "blue"}
    result = format_section("Test", data)
    assert "**User Name:** Drew" in result
    assert "**Favorite Color:** blue" in result


def test_format_section_handles_lists():
    """format_section should format list values as bulleted items."""
    data = {"interests": ["coding", "music", "hiking"]}
    result = format_section("Interests", data)
    expected = "## Interests\n**Interests:**\n- coding\n- music\n- hiking"
    assert result == expected


def test_format_section_defensive_filtering():
    """format_section should filter placeholders even if clean_section wasn't called."""
    data = {
        "name": "Drew",
        "placeholder": "not enough data",
        "items": ["valid", "Not Enough Data", "another"],
    }
    result = format_section("Test", data)
    assert "placeholder" not in result  # String placeholder filtered
    assert "Not Enough Data" not in result  # List item filtered
    assert "**Name:** Drew" in result
    assert "- valid" in result
    assert "- another" in result


def test_format_section_returns_empty_for_empty_data():
    """format_section should return empty string for None or empty dict."""
    assert format_section("Test", None) == ""
    assert format_section("Test", {}) == ""
    assert format_section("Test", []) == ""


def test_format_section_returns_empty_if_all_filtered():
    """format_section should return empty string if all values are filtered out."""
    data = {
        "field1": "not enough data",
        "field2": "Not Enough Data",
        "list": ["not enough data"],
    }
    result = format_section("Test", data)
    assert result == ""


def test_format_section_sorted_keys():
    """format_section should sort keys alphabetically for deterministic output."""
    data = {"zebra": "last", "apple": "first", "middle": "mid"}
    result = format_section("Test", data)
    lines = result.split("\n")
    # After heading, should be apple, middle, zebra
    assert lines[1].startswith("**Apple:")
    assert lines[2].startswith("**Middle:")
    assert lines[3].startswith("**Zebra:")


def test_integration_clean_then_format():
    """Integration test: clean_section then format_section should produce clean output."""
    raw_data = {
        "name": "Drew",
        "bio": "not enough data",
        "interests": ["coding", "Not Enough Data", "music"],
        "age": 30,
    }
    cleaned = clean_section(raw_data)
    formatted = format_section("Profile", cleaned)

    assert "not enough data" not in formatted.lower()
    assert "Not Enough Data" not in formatted
    assert "**Name:** Drew" in formatted
    assert "**Interests:**" in formatted
    assert "- coding" in formatted
    assert "- music" in formatted
    assert "**Age:** 30" in formatted
