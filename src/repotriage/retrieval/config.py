"""Human-authored retrieval-baseline configuration: schema, loading, and canonical hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from repotriage.retrieval.models import (
    RETRIEVAL_BASELINE_VERSION,
    RetrievalConfigDocument,
    RetrievalConfigError,
)


def config_source_sha256(config_bytes: bytes) -> str:
    return hashlib.sha256(config_bytes).hexdigest()


def config_semantic_sha256(config: RetrievalConfigDocument) -> str:
    """Hash validated configuration semantics, not incidental file formatting."""
    payload = config.model_dump(mode="json")
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_retrieval_config(
    path: Path,
) -> tuple[RetrievalConfigDocument, bytes, str, str]:
    """Load and validate a retrieval-baseline configuration file."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RetrievalConfigError(f"Unable to read retrieval config at {path}: {exc}") from exc

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RetrievalConfigError(f"Invalid JSON in retrieval config at {path}: {exc}") from exc

    try:
        config = RetrievalConfigDocument.model_validate(payload)
    except ValidationError as exc:
        raise RetrievalConfigError(f"Invalid retrieval config at {path}: {exc}") from exc

    if config.retrieval_baseline_version != RETRIEVAL_BASELINE_VERSION:
        raise RetrievalConfigError(
            f"Unsupported retrieval_baseline_version {config.retrieval_baseline_version!r}; "
            f"expected {RETRIEVAL_BASELINE_VERSION!r}."
        )

    source_hash = config_source_sha256(raw)
    semantic_hash = config_semantic_sha256(config)
    return config, raw, source_hash, semantic_hash
