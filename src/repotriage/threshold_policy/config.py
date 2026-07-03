"""Human-authored threshold-policy configuration: schema, loading, and canonical hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from repotriage.threshold_policy.models import (
    THRESHOLD_POLICY_VERSION,
    ThresholdPolicyConfigDocument,
    ThresholdPolicyConfigError,
)


def config_source_sha256(config_bytes: bytes) -> str:
    return hashlib.sha256(config_bytes).hexdigest()


def config_semantic_sha256(config: ThresholdPolicyConfigDocument) -> str:
    """Hash validated configuration semantics, not incidental file formatting."""
    payload = config.model_dump(mode="json")
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_threshold_policy_config(
    path: Path,
) -> tuple[ThresholdPolicyConfigDocument, bytes, str, str]:
    """Load and validate a threshold-policy configuration file."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ThresholdPolicyConfigError(
            f"Unable to read threshold-policy config at {path}: {exc}"
        ) from exc

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ThresholdPolicyConfigError(
            f"Invalid JSON in threshold-policy config at {path}: {exc}"
        ) from exc

    try:
        config = ThresholdPolicyConfigDocument.model_validate(payload)
    except ValidationError as exc:
        raise ThresholdPolicyConfigError(
            f"Invalid threshold-policy config at {path}: {exc}"
        ) from exc

    if config.threshold_policy_version != THRESHOLD_POLICY_VERSION:
        raise ThresholdPolicyConfigError(
            f"Unsupported threshold_policy_version {config.threshold_policy_version!r}; "
            f"expected {THRESHOLD_POLICY_VERSION!r}."
        )

    source_hash = config_source_sha256(raw)
    semantic_hash = config_semantic_sha256(config)
    return config, raw, source_hash, semantic_hash
