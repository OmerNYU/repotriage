"""Deterministic feature-text construction for model-ready records."""

from __future__ import annotations

TEXT_REPRESENTATION_VERSION = "1"

_TITLE_MARKER = "[TITLE]"
_BODY_MARKER = "[BODY]"


def normalize_line_endings(text: str) -> str:
    """Normalize CRLF and bare CR to LF without other transformations."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def build_feature_text_v1(title: str, body: str) -> str:
    """Build the v1 classifier feature text from normalized title and body."""
    normalized_title = normalize_line_endings(title)
    normalized_body = normalize_line_endings(body)
    return f"{_TITLE_MARKER}\n{normalized_title}\n\n{_BODY_MARKER}\n{normalized_body}"
