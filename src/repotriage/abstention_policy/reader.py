"""Load upstream threshold-policy and baseline score bundles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from repotriage.abstention_policy.models import AbstentionPolicyInputError
from repotriage.github.models import RepositoryRef
from repotriage.paths import resolve_within_directory
from repotriage.threshold_policy.models import (
    MANIFEST_JSON_FILE,
    POLICY_JSON_FILE,
    PolicyDocument,
    ThresholdPolicyCorruptionError,
    ThresholdPolicyManifest,
)
from repotriage.threshold_policy.reader import (
    load_test_scores,
    load_validation_scores,
)

__all__ = [
    "ThresholdPolicyInputs",
    "load_test_scores",
    "load_threshold_policy_inputs",
    "load_validation_scores",
]


@dataclass(frozen=True)
class ThresholdPolicyInputs:
    """Frozen threshold-policy manifest and selection document."""

    policy_dir: Path
    manifest: ThresholdPolicyManifest
    policy_document: PolicyDocument
    policy_bytes: bytes
    policy_sha256: str


def _load_threshold_policy_manifest(policy_dir: Path) -> ThresholdPolicyManifest:
    manifest_path = resolve_within_directory(policy_dir, MANIFEST_JSON_FILE)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ThresholdPolicyCorruptionError(
            f"Unable to read threshold-policy manifest at {manifest_path}: {exc}"
        ) from exc
    try:
        return ThresholdPolicyManifest.model_validate(payload)
    except ValidationError as exc:
        raise ThresholdPolicyCorruptionError(
            f"Invalid threshold-policy manifest at {manifest_path}: {exc}"
        ) from exc


def load_threshold_policy_inputs(
    policy_dir: Path,
    *,
    expected_policy_id: str | None = None,
    expected_repository: RepositoryRef | None = None,
) -> ThresholdPolicyInputs:
    """Load threshold-policy manifest and policy document from an on-disk artifact."""
    if not policy_dir.is_dir():
        raise AbstentionPolicyInputError(f"Threshold-policy directory does not exist: {policy_dir}")

    manifest = _load_threshold_policy_manifest(policy_dir)
    if expected_policy_id is not None and manifest.policy_id != expected_policy_id:
        raise AbstentionPolicyInputError(
            f"Threshold-policy id {manifest.policy_id!r} does not match expected "
            f"{expected_policy_id!r}."
        )
    if expected_repository is not None and manifest.repository != expected_repository.full_name:
        raise AbstentionPolicyInputError(
            f"Threshold-policy repository {manifest.repository!r} does not match expected "
            f"{expected_repository.full_name!r}."
        )

    policy_path = resolve_within_directory(policy_dir, POLICY_JSON_FILE)
    policy_bytes = policy_path.read_bytes()
    try:
        policy_document = PolicyDocument.model_validate_json(policy_bytes)
    except ValidationError as exc:
        raise ThresholdPolicyCorruptionError(
            f"Invalid threshold-policy document at {policy_path}: {exc}"
        ) from exc

    selected_bp = policy_document.selection.selected_threshold_basis_points
    if selected_bp != manifest.selected_threshold_basis_points:
        raise ThresholdPolicyCorruptionError(
            "threshold-policy policy.json selected threshold does not match manifest"
        )

    import hashlib

    policy_sha256 = hashlib.sha256(policy_bytes).hexdigest()
    if policy_sha256 != manifest.policy_sha256:
        raise ThresholdPolicyCorruptionError(
            "threshold-policy policy.json hash does not match manifest"
        )

    return ThresholdPolicyInputs(
        policy_dir=policy_dir,
        manifest=manifest,
        policy_document=policy_document,
        policy_bytes=policy_bytes,
        policy_sha256=policy_sha256,
    )
