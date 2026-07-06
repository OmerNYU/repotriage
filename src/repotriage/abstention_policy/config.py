"""Human-authored abstention-policy configuration: schema, loading, and canonical hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from repotriage.abstention_policy.models import (
    ABSTENTION_POLICY_VERSION,
    AbstentionPolicyConfigDocument,
    AbstentionPolicyConfigError,
)


def config_source_sha256(config_bytes: bytes) -> str:
    return hashlib.sha256(config_bytes).hexdigest()


def config_semantic_sha256(config: AbstentionPolicyConfigDocument) -> str:
    """Hash validated configuration semantics, not incidental file formatting."""
    payload = config.model_dump(mode="json")
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_abstention_policy_config(
    path: Path,
) -> tuple[AbstentionPolicyConfigDocument, bytes, str, str]:
    """Load and validate an abstention-policy configuration file."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise AbstentionPolicyConfigError(
            f"Unable to read abstention-policy config at {path}: {exc}"
        ) from exc

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AbstentionPolicyConfigError(
            f"Invalid JSON in abstention-policy config at {path}: {exc}"
        ) from exc

    try:
        config = AbstentionPolicyConfigDocument.model_validate(payload)
    except ValidationError as exc:
        raise AbstentionPolicyConfigError(
            f"Invalid abstention-policy config at {path}: {exc}"
        ) from exc

    if config.abstention_policy_version != ABSTENTION_POLICY_VERSION:
        raise AbstentionPolicyConfigError(
            f"Unsupported abstention_policy_version {config.abstention_policy_version!r}; "
            f"expected {ABSTENTION_POLICY_VERSION!r}."
        )

    source_hash = config_source_sha256(raw)
    semantic_hash = config_semantic_sha256(config)
    return config, raw, source_hash, semantic_hash
