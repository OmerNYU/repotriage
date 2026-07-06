"""Feature-text construction for inference using the model-ready contract."""

from __future__ import annotations

import hashlib

from repotriage.inference.models import InferenceBundleError
from repotriage.model_dataset.text import (
    TEXT_REPRESENTATION_VERSION,
    build_feature_text_v1,
)

BODY_PREVIEW_MAX_LENGTH = 200


def build_inference_feature_text(
    title: str,
    body: str,
    *,
    text_representation_version: str,
) -> str:
    """Build feature text for inference using the supported text-representation version."""
    if text_representation_version == TEXT_REPRESENTATION_VERSION:
        return build_feature_text_v1(title, body)
    raise InferenceBundleError(
        f"Unsupported text_representation_version {text_representation_version!r}."
    )


def feature_text_sha256(feature_text: str) -> str:
    """Return the SHA-256 hex digest of UTF-8 feature text bytes."""
    return hashlib.sha256(feature_text.encode("utf-8")).hexdigest()


def body_preview(body: str, *, max_length: int = BODY_PREVIEW_MAX_LENGTH) -> str:
    """Return a truncated body preview for response display."""
    if len(body) <= max_length:
        return body
    return body[:max_length]
