"""Human-authored baseline configuration: schema, loading, and canonical hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from repotriage.baseline.models import (
    BASELINE_VERSION,
    BaselineCandidateConfig,
    BaselineConfigDocument,
    BaselineConfigError,
)


def config_source_sha256(config_bytes: bytes) -> str:
    return hashlib.sha256(config_bytes).hexdigest()


def config_semantic_sha256(config: BaselineConfigDocument) -> str:
    """Hash validated configuration semantics, not incidental file formatting."""
    payload = config.model_dump(mode="json")
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_baseline_config(path: Path) -> tuple[BaselineConfigDocument, bytes, str, str]:
    """Load and validate a baseline configuration file."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise BaselineConfigError(f"Unable to read baseline config at {path}: {exc}") from exc

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BaselineConfigError(f"Invalid JSON in baseline config at {path}: {exc}") from exc

    try:
        config = BaselineConfigDocument.model_validate(payload)
    except ValidationError as exc:
        raise BaselineConfigError(f"Invalid baseline config at {path}: {exc}") from exc

    if config.baseline_version != BASELINE_VERSION:
        raise BaselineConfigError(
            f"Unsupported baseline_version {config.baseline_version!r}; "
            f"expected {BASELINE_VERSION!r}."
        )

    candidate_ids = [candidate.candidate_id for candidate in config.candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise BaselineConfigError("candidate_id values must be unique")

    source_hash = config_source_sha256(raw)
    semantic_hash = config_semantic_sha256(config)
    return config, raw, source_hash, semantic_hash


def get_candidate_by_id(
    config: BaselineConfigDocument, candidate_id: str
) -> BaselineCandidateConfig:
    for candidate in config.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    raise BaselineConfigError(f"Unknown candidate_id {candidate_id!r}")
