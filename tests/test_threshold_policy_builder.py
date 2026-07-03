"""Tests for threshold-policy builder orchestration."""

from __future__ import annotations

import hashlib
import json
import shutil

import pytest

from repotriage.baseline.builder import train_baseline
from repotriage.baseline.models import BaselineCorruptionError
from repotriage.threshold_policy.builder import (
    build_threshold_policy,
    publish_threshold_policy,
    validate_threshold_policy_against_baseline,
    validate_threshold_policy_artifact_integrity,
)
from repotriage.threshold_policy.config import config_semantic_sha256, load_threshold_policy_config
from repotriage.threshold_policy.models import (
    ThresholdGridConfig,
    ThresholdPolicyBuildError,
    ThresholdPolicyCorruptionError,
    compute_policy_input_sha256,
)
from tests.helpers import write_baseline_config, write_threshold_policy_config
from tests.test_baseline_builder import _build_model_ready


def _train_baseline(tmp_path):
    fixture, model_result = _build_model_ready(tmp_path)
    config_path = write_baseline_config(tmp_path / "baseline.json")
    baselines_root = tmp_path / "baselines"
    result = train_baseline(
        fixture.repository,
        model_result.manifest.model_dataset_id,
        config_path,
        model_ready_root=fixture.model_ready_root,
        baselines_root=baselines_root,
    )
    return fixture, result, baselines_root


def _build_valid_policy_artifact(tmp_path):
    fixture, baseline_result, baselines_root = _train_baseline(tmp_path)
    policy_config = write_threshold_policy_config(
        tmp_path / "policy.json",
        baseline_run_id=baseline_result.manifest.baseline_run_id,
    )
    policies_root = tmp_path / "policies"
    result = build_threshold_policy(
        fixture.repository,
        policy_config,
        baselines_root=baselines_root,
        threshold_policies_root=policies_root,
    )
    return fixture, baseline_result, baselines_root, result


def _copy_policy_dir(policy_dir, tmp_path):
    dest = tmp_path / policy_dir.name
    shutil.copytree(policy_dir, dest)
    return dest


def test_synthetic_end_to_end_build(tmp_path) -> None:
    fixture, baseline_result, baselines_root = _train_baseline(tmp_path)
    policy_config = write_threshold_policy_config(
        tmp_path / "policy.json",
        baseline_run_id=baseline_result.manifest.baseline_run_id,
    )
    policies_root = tmp_path / "policies"
    result = build_threshold_policy(
        fixture.repository,
        policy_config,
        baselines_root=baselines_root,
        threshold_policies_root=policies_root,
    )
    assert result.cache_hit is False
    assert "-tp1-" in result.manifest.policy_id
    assert result.selected_threshold_basis_points in range(5, 96)
    validate_threshold_policy_artifact_integrity(
        result.policy_dir,
        expected_repository=fixture.repository,
    )
    validate_threshold_policy_against_baseline(
        result.policy_dir,
        baseline_result.baseline_dir,
        expected_repository=fixture.repository,
    )


def test_cache_hit(tmp_path) -> None:
    fixture, baseline_result, baselines_root = _train_baseline(tmp_path)
    policy_config = write_threshold_policy_config(
        tmp_path / "policy.json",
        baseline_run_id=baseline_result.manifest.baseline_run_id,
    )
    policies_root = tmp_path / "policies"
    first = build_threshold_policy(
        fixture.repository,
        policy_config,
        baselines_root=baselines_root,
        threshold_policies_root=policies_root,
    )
    second = build_threshold_policy(
        fixture.repository,
        policy_config,
        baselines_root=baselines_root,
        threshold_policies_root=policies_root,
    )
    assert second.cache_hit is True
    assert first.manifest.policy_id == second.manifest.policy_id


def test_stable_semantic_config_hash(tmp_path) -> None:
    fixture, baseline_result, _ = _train_baseline(tmp_path)
    config_a = write_threshold_policy_config(
        tmp_path / "a.json",
        baseline_run_id=baseline_result.manifest.baseline_run_id,
    )
    config_b = tmp_path / "b.json"
    payload = json.loads(config_a.read_text(encoding="utf-8"))
    config_b.write_text(
        json.dumps(payload, indent=4, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    _, _, _, hash_a = load_threshold_policy_config(config_a)
    config_doc_b, _, _, hash_b = load_threshold_policy_config(config_b)
    assert hash_a != hashlib.sha256(config_b.read_bytes()).hexdigest()
    assert hash_a == config_semantic_sha256(config_doc_b)


def test_identity_changes_when_baseline_run_changes(tmp_path) -> None:
    grid = ThresholdGridConfig(
        start_basis_points=5,
        stop_basis_points=95,
        step_basis_points=1,
        denominator=100,
    )
    base_kwargs = {
        "threshold_policy_version": "1",
        "baseline_experiment_sha256": "a" * 64,
        "model_semantic_sha256": "b" * 64,
        "predictions_validation_sha256": "c" * 64,
        "predictions_test_sha256": "d" * 64,
        "selected_candidate_id": "c3_bigram_balanced",
        "threshold_grid": grid,
        "selection_rule_version": "1",
        "metric_contract_version": "2",
    }
    hash_a = compute_policy_input_sha256(
        baseline_run_id="run-a",
        **base_kwargs,
    )
    hash_b = compute_policy_input_sha256(
        baseline_run_id="run-b",
        **base_kwargs,
    )
    assert hash_a != hash_b


def test_identity_changes_when_validation_prediction_hash_changes(tmp_path) -> None:
    grid = ThresholdGridConfig(
        start_basis_points=5,
        stop_basis_points=95,
        step_basis_points=1,
        denominator=100,
    )
    base_kwargs = {
        "threshold_policy_version": "1",
        "baseline_run_id": "run",
        "baseline_experiment_sha256": "a" * 64,
        "model_semantic_sha256": "b" * 64,
        "predictions_test_sha256": "d" * 64,
        "selected_candidate_id": "c3_bigram_balanced",
        "threshold_grid": grid,
        "selection_rule_version": "1",
        "metric_contract_version": "2",
    }
    hash_a = compute_policy_input_sha256(
        predictions_validation_sha256="c" * 64,
        **base_kwargs,
    )
    hash_b = compute_policy_input_sha256(
        predictions_validation_sha256="e" * 64,
        **base_kwargs,
    )
    assert hash_a != hash_b


def test_corruption_rejected(tmp_path) -> None:
    fixture, _, _, result = _build_valid_policy_artifact(tmp_path)
    manifest_path = result.policy_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["policy_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ThresholdPolicyCorruptionError):
        validate_threshold_policy_artifact_integrity(
            result.policy_dir,
            expected_repository=fixture.repository,
        )


def test_corruption_rejected_modified_sweep_metric(tmp_path) -> None:
    fixture, _, _, result = _build_valid_policy_artifact(tmp_path)
    policy_dir = _copy_policy_dir(result.policy_dir, tmp_path)
    sweep_path = policy_dir / "sweep_validation.json"
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    sweep["rows"][10]["metrics"]["aggregate"]["macro_f1"] = 0.999
    sweep_path.write_text(json.dumps(sweep), encoding="utf-8")
    with pytest.raises(ThresholdPolicyCorruptionError):
        validate_threshold_policy_artifact_integrity(
            policy_dir,
            expected_repository=fixture.repository,
        )


def test_corruption_rejected_modified_selected_threshold(tmp_path) -> None:
    fixture, _, _, result = _build_valid_policy_artifact(tmp_path)
    policy_dir = _copy_policy_dir(result.policy_dir, tmp_path)
    policy_path = policy_dir / "policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["selection"]["selected_threshold_basis_points"] = 42
    policy_path.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ThresholdPolicyCorruptionError):
        validate_threshold_policy_artifact_integrity(
            policy_dir,
            expected_repository=fixture.repository,
        )


def test_corruption_rejected_modified_baseline_lineage_hash(tmp_path) -> None:
    fixture, _, _, result = _build_valid_policy_artifact(tmp_path)
    policy_dir = _copy_policy_dir(result.policy_dir, tmp_path)
    manifest_path = policy_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["baseline_experiment_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ThresholdPolicyCorruptionError):
        validate_threshold_policy_artifact_integrity(
            policy_dir,
            expected_repository=fixture.repository,
        )


def test_corruption_rejected_missing_threshold_row(tmp_path) -> None:
    fixture, _, _, result = _build_valid_policy_artifact(tmp_path)
    policy_dir = _copy_policy_dir(result.policy_dir, tmp_path)
    sweep_path = policy_dir / "sweep_validation.json"
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    sweep["rows"] = sweep["rows"][:-1]
    sweep_path.write_text(json.dumps(sweep), encoding="utf-8")
    with pytest.raises(ThresholdPolicyCorruptionError):
        validate_threshold_policy_artifact_integrity(
            policy_dir,
            expected_repository=fixture.repository,
        )


def test_corruption_rejected_duplicate_threshold_row(tmp_path) -> None:
    fixture, _, _, result = _build_valid_policy_artifact(tmp_path)
    policy_dir = _copy_policy_dir(result.policy_dir, tmp_path)
    sweep_path = policy_dir / "sweep_validation.json"
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    sweep["rows"].append(sweep["rows"][0])
    sweep_path.write_text(json.dumps(sweep), encoding="utf-8")
    with pytest.raises(ThresholdPolicyCorruptionError):
        validate_threshold_policy_artifact_integrity(
            policy_dir,
            expected_repository=fixture.repository,
        )


def test_corruption_rejected_mismatched_comparison_metric(tmp_path) -> None:
    fixture, _, _, result = _build_valid_policy_artifact(tmp_path)
    policy_dir = _copy_policy_dir(result.policy_dir, tmp_path)
    comparison_path = policy_dir / "comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    comparison["test"]["selected"]["macro_f1"] = 0.9
    comparison_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ThresholdPolicyCorruptionError):
        validate_threshold_policy_artifact_integrity(
            policy_dir,
            expected_repository=fixture.repository,
        )


def test_corruption_rejected_changed_baseline_prediction(tmp_path) -> None:
    fixture, baseline_result, _, result = _build_valid_policy_artifact(tmp_path)
    baseline_copy = tmp_path / "baseline_copy"
    shutil.copytree(baseline_result.baseline_dir, baseline_copy)
    val_path = baseline_copy / "predictions_validation.jsonl"
    val_path.write_text(
        val_path.read_text(encoding="utf-8").replace(
            '"score_vector":[0.',
            '"score_vector":[0.9',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(BaselineCorruptionError):
        validate_threshold_policy_against_baseline(
            result.policy_dir,
            baseline_copy,
            expected_repository=fixture.repository,
        )


def test_publish_does_not_overwrite(tmp_path) -> None:
    staging = tmp_path / "staging"
    final = tmp_path / "final"
    staging.mkdir()
    (staging / "manifest.json").write_text("{}", encoding="utf-8")
    final.mkdir()
    with pytest.raises(ThresholdPolicyBuildError):
        publish_threshold_policy(staging, final)


def test_repeated_build_byte_equality(tmp_path) -> None:
    fixture, baseline_result, baselines_root = _train_baseline(tmp_path)
    policy_config = write_threshold_policy_config(
        tmp_path / "policy.json",
        baseline_run_id=baseline_result.manifest.baseline_run_id,
    )
    policies_root_a = tmp_path / "policies_a"
    policies_root_b = tmp_path / "policies_b"
    result_a = build_threshold_policy(
        fixture.repository,
        policy_config,
        baselines_root=baselines_root,
        threshold_policies_root=policies_root_a,
    )
    result_b = build_threshold_policy(
        fixture.repository,
        policy_config,
        baselines_root=baselines_root,
        threshold_policies_root=policies_root_b,
    )
    assert result_a.manifest.policy_id == result_b.manifest.policy_id
    for filename in (
        "config.json",
        "policy.json",
        "sweep_validation.json",
        "metrics_validation.json",
        "metrics_test.json",
        "comparison.json",
    ):
        assert (
            (result_a.policy_dir / filename).read_bytes()
            == (result_b.policy_dir / filename).read_bytes()
        )
