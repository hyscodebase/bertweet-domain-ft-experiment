from __future__ import annotations


def normalize_text(text: str) -> str:
    """Keep a single preprocessing hook for future task-specific normalization."""

    return str(text).strip()
