"""Tests for model-ready feature text construction."""

from repotriage.model_dataset.text import build_feature_text_v1, normalize_line_endings


def test_exact_marker_format() -> None:
    text = build_feature_text_v1("Fix bug", "Details here")
    assert text == "[TITLE]\nFix bug\n\n[BODY]\nDetails here"


def test_crlf_normalization() -> None:
    text = build_feature_text_v1("Line\r\nbreak", "Body\r\nhere")
    assert text == "[TITLE]\nLine\nbreak\n\n[BODY]\nBody\nhere"


def test_bare_cr_normalization() -> None:
    assert normalize_line_endings("a\rb") == "a\nb"


def test_empty_body() -> None:
    text = build_feature_text_v1("Title only", "")
    assert text == "[TITLE]\nTitle only\n\n[BODY]\n"


def test_unicode_and_markdown_preserved() -> None:
    title = "日本語 #标题"
    body = "```python\nprint('hi')\n```\n\nSee https://example.com"
    text = build_feature_text_v1(title, body)
    assert title in text
    assert body in text


def test_deterministic_bytes() -> None:
    first = build_feature_text_v1("T", "B")
    second = build_feature_text_v1("T", "B")
    assert first == second
    assert first.encode("utf-8") == second.encode("utf-8")


def test_literal_markers_in_source_preserved() -> None:
    text = build_feature_text_v1("[TITLE]", "[BODY]")
    assert "[TITLE]\n[TITLE]\n\n[BODY]\n[BODY]" == text
