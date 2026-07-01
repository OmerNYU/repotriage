"""Tests for validate_label_policy_artifact_integrity downstream validator."""

from __future__ import annotations

from pathlib import Path

import pytest

from repotriage.label_policy.builder import (
    validate_label_policy_artifact,
    validate_label_policy_artifact_integrity,
)
from repotriage.label_policy.models import LabelPolicyCorruptionError
from tests.test_label_policy_builder import _build
from tests.test_label_policy_builder import _setup as _policy_setup


def test_integrity_validator_without_audit_config_expectations(tmp_path: Path) -> None:
    fixture = _policy_setup(tmp_path)
    _build(fixture)
    policy_dir = fixture.policies_root / fixture.repository.slug / fixture.policy_id
    manifest, document = validate_label_policy_artifact_integrity(
        policy_dir,
        expected_repository=fixture.repository,
        expected_dataset_id=fixture.dataset_id,
        expected_dataset_output_sha256=fixture.processed_manifest.output_sha256,
        expected_policy_id=fixture.policy_id,
    )
    assert manifest.policy_id == fixture.policy_id
    assert document.coverage.included_label_count >= 1


def test_builder_validator_still_checks_audit_fields(tmp_path: Path) -> None:
    fixture = _policy_setup(tmp_path)
    _build(fixture)
    policy_dir = fixture.policies_root / fixture.repository.slug / fixture.policy_id
    with pytest.raises(LabelPolicyCorruptionError, match="audit_id"):
        validate_label_policy_artifact(
            policy_dir,
            expected_repository=fixture.repository,
            expected_dataset_id=fixture.dataset_id,
            expected_dataset_output_sha256=fixture.processed_manifest.output_sha256,
            expected_audit_id="wrong-audit-id",
            expected_audit_json_sha256=fixture.audit_manifest.audit_json_sha256,
            expected_audit_version=fixture.audit_manifest.audit_version,
            expected_config_schema_version="2",
            expected_config_sha256=fixture.config_sha256,
            expected_issue_schema_version=fixture.processed_manifest.issue_schema_version,
            expected_normalizer_version=fixture.processed_manifest.normalizer_version,
            expected_policy_id=fixture.policy_id,
            expected_policy_input_sha256=fixture.policy_input_sha256,
        )
