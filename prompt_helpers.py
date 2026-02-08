"""
Prompt Helpers for SoulPrint Section Processing (Python)

Pure functions for validating and formatting section data for prompts.
These helpers ensure:
1. No "not enough data" placeholders in prompts (MEM-02)
2. Consistent markdown formatting (PROMPT-02)
3. Deterministic output (same input always produces same output)

Must produce identical output to TypeScript lib/soulprint/prompt-helpers.ts
"""

from typing import Optional


def clean_section(data: Optional[dict]) -> Optional[dict]:
    """Remove 'not enough data' placeholders and empty values from a section dict."""
    if not data or not isinstance(data, dict):
        return None

    cleaned = {}
    for key, val in data.items():
        if isinstance(val, list):
            # Filter out "not enough data" items
            filtered = [item for item in val if not (isinstance(item, str) and item.strip().lower() == "not enough data")]
            if filtered:
                cleaned[key] = filtered
        elif isinstance(val, str):
            stripped = val.strip()
            if stripped and stripped.lower() != "not enough data":
                cleaned[key] = val
        # Preserve other value types (numbers, booleans, objects)
        elif val is not None:
            cleaned[key] = val

    return cleaned if cleaned else None


def format_section(section_name: str, data: Optional[dict]) -> str:
    """Convert a section dict to markdown. Must produce identical output to TypeScript formatSection()."""
    if not data or not isinstance(data, dict) or len(data) == 0:
        return ""

    lines = [f"## {section_name}"]

    # Sort keys for deterministic output (matches TypeScript)
    for key in sorted(data.keys()):
        val = data[key]
        # Convert snake_case to Title Case
        label = " ".join(word.capitalize() for word in key.split("_"))

        # Handle string values
        if isinstance(val, str):
            stripped = val.strip()
            # Defensive: skip empty/placeholder even if cleanSection wasn't called
            if stripped and stripped.lower() != "not enough data":
                lines.append(f"**{label}:** {val}")

        # Handle array values
        elif isinstance(val, list):
            # Defensive: filter out "not enough data" items
            filtered = [item for item in val if not (isinstance(item, str) and item.strip().lower() == "not enough data")]
            if filtered:
                lines.append(f"**{label}:**")
                for item in filtered:
                    lines.append(f"- {item}")

        # Handle other types (numbers, booleans, etc.)
        elif val is not None:
            lines.append(f"**{label}:** {val}")

    # If only heading remains (all values filtered out), return empty string
    return "\n".join(lines) if len(lines) > 1 else ""
