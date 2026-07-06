"""Tests for abstention-policy builder orchestration."""

from __future__ import annotations

import hashlib
import json
import shutil

import pytest

from repotriage.abstention_policy.builder import (
    build_abstention_policy,
    publish_abstention_policy,
    validate_abstention_policy_against_inputs,
    validate_abstention_policy_artifact_integrity,
)
from repotriage.abstention_policy.config import (
    config_semantic_sha256,
    load_abstention_policy_config,
)
from repotriage.abstention_policy.models import (
    AbstentionPolicyBuildError,
    AbstentionPolicyCorruptionError,
    compute_policy_input_sha256,
)
from repotriage.threshold_policy.builder import build_threshold_policy
from repotriage.threshold_policy.models import ThresholdGridConfig
from tests.helpers import write_abstention_policy_config, write_threshold_policy_config
from tests.test_threshold_policy_builder import _train_baseline


def _build_threshold_policy_artifact(tmp_path):
    fixture, baseline_result, baselines_root = _train_baseline(tmp_path)
    policy_config = write_threshold_policy_config(
        tmp_path / "threshold-policy.json",
        baseline_run_id=baseline_result.manifest.baseline_run_id,
    )
    policies_root = tmp_path / "threshold_policies"
    result = build_threshold_policy(
        fixture.repository,
        policy_config,
        baselines_root=baselines_root,
        threshold_policies_root=policies_root,
    )
    return fixture, baseline_result, baselines_root, result


def _build_valid_abstention_artifact(tmp_path):
    fixture, baseline_result, baselines_root, threshold_result = _build_threshold_policy_artifact(
        tmp_path
    )
    abstention_config = write_abstention_policy_config(
        tmp_path / "abstention.json",
        threshold_policy_id=threshold_result.manifest.policy_id,
    )
    abstention_root = tmp_path / "abstention_policies"
    result = build_abstention_policy(
        fixture.repository,
        abstention_config,
        threshold_policy_id=threshold_result.manifest.policy_id,
        baselines_root=baselines_root,
        threshold_policies_root=tmp_path / "threshold_policies",
        abstention_policies_root=abstention_root,
    )
    return fixture, baseline_result, baselines_root, threshold_result, result


def _copy_policy_dir(policy_dir, tmp_path):
    dest = tmp_path / policy_dir.name
    shutil.copytree(policy_dir, dest)
    return dest


def test_synthetic_end_to_end_build(tmp_path) -> None:
    fixture, baseline_result, baselines_root, threshold_result, result = (
        _build_valid_abstention_artifact(tmp_path)
    )
    assert result.cache_hit is False
    assert "-ap1-" in result.manifest.policy_id
    validate_abstention_policy_artifact_integrity(
        result.policy_dir,
        expected_repository=fixture.repository,
    )
    validate_abstention_policy_against_inputs(
        result.policy_dir,
        baseline_result.baseline_dir,
        threshold_result.policy_dir,
        expected_repository=fixture.repository,
    )


def test_cache_hit(tmp_path) -> None:
    fixture, baseline_result, baselines_root, threshold_result = _build_threshold_policy_artifact(
        tmp_path
    )
    abstention_config = write_abstention_policy_config(
        tmp_path / "abstention.json",
        threshold_policy_id=threshold_result.manifest.policy_id,
    )
    abstention_root = tmp_path / "abstention_policies"
    kwargs = dict(
        baselines_root=baselines_root,
        threshold_policies_root=tmp_path / "threshold_policies",
        abstention_policies_root=abstention_root,
    )
    first = build_abstention_policy(
        fixture.repository,
        abstention_config,
        threshold_policy_id=threshold_result.manifest.policy_id,
        **kwargs,
    )
    second = build_abstention_policy(
        fixture.repository,
        abstention_config,
        threshold_policy_id=threshold_result.manifest.policy_id,
        **kwargs,
    )
    assert second.cache_hit is True
    assert first.manifest.policy_id == second.manifest.policy_id


def test_stable_semantic_config_hash(tmp_path) -> None:
    _, _, _, threshold_result = _build_threshold_policy_artifact(tmp_path)
    config_a = write_abstention_policy_config(
        tmp_path / "a.json",
        threshold_policy_id=threshold_result.manifest.policy_id,
    )
    config_b = tmp_path / "b.json"
    payload = json.loads(config_a.read_text(encoding="utf-8"))
    config_b.write_text(
        json.dumps(payload, indent=4, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    _, _, _, hash_a = load_abstention_policy_config(config_a)
    config_doc_b, _, _, hash_b = load_abstention_policy_config(config_b)
    assert hash_a != hashlib.sha256(config_b.read_bytes()).hexdigest()
    assert hash_a == config_semantic_sha256(config_doc_b)


def test_identity_changes_when_minimum_coverage_changes() -> None:
    grid = ThresholdGridConfig(
        start_basis_points=39,
        stop_basis_points=95,
        step_basis_points=1,
        denominator=100,
    )
    common = {
        "abstention_policy_version": "1",
        "baseline_run_id": "run",
        "baseline_experiment_sha256": "a" * 64,
        "model_semantic_sha256": "b" * 64,
        "predictions_validation_sha256": "c" * 64,
        "predictions_test_sha256": "d" * 64,
        "threshold_policy_id": "tp",
        "threshold_policy_sha256": "e" * 64,
        "classification_threshold_basis_points": 39,
        "confidence_definition": "max_predicted_label_score",
        "abstention_grid": grid,
        "selection_rule_version": "1",
        "metric_contract_version": "2",
    }
    hash_a = compute_policy_input_sha256(minimum_coverage=0.25, **common)
    hash_b = compute_policy_input_sha256(minimum_coverage=0.30, **common)
    assert hash_a != hash_b


def test_publish_refuses_overwrite(tmp_path) -> None:
    staging = tmp_path / "staging"
    final = tmp_path / "final"
    staging.mkdir()
    (staging / "manifest.json").write_text("{}", encoding="utf-8")
    publish_abstention_policy(staging, final)
    staging.mkdir()
    with pytest.raises(AbstentionPolicyBuildError, match="overwrite"):
        publish_abstention_policy(staging, final)


def test_corruption_rejected(tmp_path) -> None:
    fixture, baseline_result, _, threshold_result, result = _build_valid_abstention_artifact(
        tmp_path
    )
    corrupted = _copy_policy_dir(result.policy_dir, tmp_path)
    manifest_path = corrupted / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["policy_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(AbstentionPolicyCorruptionError):
        validate_abstention_policy_artifact_integrity(
            corrupted,
            expected_repository=fixture.repository,
        )

    validate_abstention_policy_against_inputs(
        result.policy_dir,
        baseline_result.baseline_dir,
        threshold_result.policy_dir,
        expected_repository=fixture.repository,
    )


def test_repeated_build_byte_equality(tmp_path) -> None:
    fixture, _, baselines_root, threshold_result = _build_threshold_policy_artifact(tmp_path)
    abstention_config = write_abstention_policy_config(
        tmp_path / "abstention.json",
        threshold_policy_id=threshold_result.manifest.policy_id,
    )
    root_a = tmp_path / "policies_a"
    root_b = tmp_path / "policies_b"
    kwargs = dict(
        baselines_root=baselines_root,
        threshold_policies_root=tmp_path / "threshold_policies",
    )
    first = build_abstention_policy(
        fixture.repository,
        abstention_config,
        threshold_policy_id=threshold_result.manifest.policy_id,
        abstention_policies_root=root_a,
        **kwargs,
    )
    second = build_abstention_policy(
        fixture.repository,
        abstention_config,
        threshold_policy_id=threshold_result.manifest.policy_id,
        abstention_policies_root=root_b,
        **kwargs,
    )
    assert first.manifest.policy_id == second.manifest.policy_id
    for filename in (
        "config.json",
        "policy.json",
        "sweep_validation.json",
        "metrics_validation.json",
        "metrics_test.json",
        "comparison.json",
    ):
        assert (first.policy_dir / filename).read_bytes() == (
            second.policy_dir / filename
        ).read_bytes()
