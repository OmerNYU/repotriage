"""Tests for inference feature-text construction."""

from repotriage.inference.text import (
    build_inference_feature_text,
    feature_text_sha256,
)
from repotriage.model_dataset.text import build_feature_text_v1


def test_inference_feature_text_matches_model_ready_contract() -> None:
    title = "Line\r\nbreak"
    body = "Body\r\nhere"
    expected = build_feature_text_v1(title, body)
    actual = build_inference_feature_text(title, body, text_representation_version="1")
    assert actual == expected


def test_feature_text_sha256_is_deterministic() -> None:
    text = build_inference_feature_text("T", "B", text_representation_version="1")
    assert feature_text_sha256(text) == feature_text_sha256(text)


def test_empty_body_default() -> None:
    text = build_inference_feature_text("Title only", "", text_representation_version="1")
    assert text == "[TITLE]\nTitle only\n\n[BODY]\n"
