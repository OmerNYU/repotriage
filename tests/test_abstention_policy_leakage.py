"""Leakage tests for abstention-policy selection."""

from __future__ import annotations

import inspect
from unittest.mock import patch

import numpy as np

from repotriage.abstention_policy.builder import build_abstention_policy
from repotriage.abstention_policy.models import compute_policy_input_sha256
from repotriage.abstention_policy.selector import select_abstention_threshold
from repotriage.abstention_policy.sweep import build_abstention_sweep
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


def test_load_test_scores_not_called_before_selection(tmp_path) -> None:
    fixture, baseline_result, baselines_root, threshold_result = _build_threshold_policy_artifact(
        tmp_path
    )
    abstention_config = write_abstention_policy_config(
        tmp_path / "abstention.json",
        threshold_policy_id=threshold_result.manifest.policy_id,
    )
    abstention_root = tmp_path / "abstention_policies"
    selection_completed = {"value": False}

    original_pipeline = __import__(
        "repotriage.abstention_policy.builder", fromlist=["_run_validation_only_pipeline"]
    )._run_validation_only_pipeline

    def wrapped_pipeline(*args, **kwargs):
        result = original_pipeline(*args, **kwargs)
        selection_completed["value"] = True
        return result

    def guarded_load(*_args, **_kwargs):
        if not selection_completed["value"]:
            raise AssertionError("load_test_scores must not be called before selection")
        raise RuntimeError("stop-after-guard-check")

    with patch(
        "repotriage.abstention_policy.builder._run_validation_only_pipeline",
        side_effect=wrapped_pipeline,
    ):
        with patch(
            "repotriage.abstention_policy.builder.load_test_scores",
            side_effect=guarded_load,
        ):
            try:
                build_abstention_policy(
                    fixture.repository,
                    abstention_config,
                    threshold_policy_id=threshold_result.manifest.policy_id,
                    baselines_root=baselines_root,
                    threshold_policies_root=tmp_path / "threshold_policies",
                    abstention_policies_root=abstention_root,
                )
            except RuntimeError as exc:
                assert str(exc) == "stop-after-guard-check"
            else:
                raise AssertionError("expected guarded load_test_scores to stop build")


def test_mutating_test_scores_does_not_change_selected_threshold() -> None:
    labels = ["Bug", "Docs"]
    y_true = np.array([[1, 0], [0, 1]], dtype=np.int8)
    y_score = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=np.float64)
    grid = ThresholdGridConfig(
        start_basis_points=39,
        stop_basis_points=69,
        step_basis_points=10,
        denominator=100,
    )
    _, sweep = build_abstention_sweep(
        labels=labels,
        y_true=y_true,
        y_score=y_score,
        issue_ids=[1, 2],
        classification_threshold=0.39,
        grid=grid,
    )
    first = select_abstention_threshold(
        sweep=sweep,
        minimum_coverage=0.25,
        classification_threshold_basis_points=39,
    )
    second = select_abstention_threshold(
        sweep=sweep,
        minimum_coverage=0.25,
        classification_threshold_basis_points=39,
    )
    assert first.selected_abstention_basis_points == second.selected_abstention_basis_points


def test_mutating_validation_scores_can_change_selected_threshold() -> None:
    labels = ["Bug"]
    y_true = np.array([[1], [0]], dtype=np.int8)
    y_score = np.array([[0.96], [0.45]], dtype=np.float64)
    grid = ThresholdGridConfig(
        start_basis_points=39,
        stop_basis_points=95,
        step_basis_points=56,
        denominator=100,
    )
    _, sweep = build_abstention_sweep(
        labels=labels,
        y_true=y_true,
        y_score=y_score,
        issue_ids=[1, 2],
        classification_threshold=0.39,
        grid=grid,
    )
    first = select_abstention_threshold(
        sweep=sweep,
        minimum_coverage=0.0,
        classification_threshold_basis_points=39,
    )
    y_score_mutated = y_score.copy()
    y_score_mutated[1, 0] = 0.96
    _, sweep_mutated = build_abstention_sweep(
        labels=labels,
        y_true=y_true,
        y_score=y_score_mutated,
        issue_ids=[1, 2],
        classification_threshold=0.39,
        grid=grid,
    )
    second = select_abstention_threshold(
        sweep=sweep_mutated,
        minimum_coverage=0.0,
        classification_threshold_basis_points=39,
    )
    assert first.selected_abstention_basis_points == 95
    assert second.selected_abstention_basis_points == 39


def test_policy_input_hash_has_no_test_metric_fields() -> None:
    grid = ThresholdGridConfig(
        start_basis_points=39,
        stop_basis_points=95,
        step_basis_points=1,
        denominator=100,
    )
    payload_text = compute_policy_input_sha256(
        abstention_policy_version="1",
        baseline_run_id="run",
        baseline_experiment_sha256="a" * 64,
        model_semantic_sha256="b" * 64,
        predictions_validation_sha256="c" * 64,
        predictions_test_sha256="d" * 64,
        threshold_policy_id="tp",
        threshold_policy_sha256="e" * 64,
        classification_threshold_basis_points=39,
        confidence_definition="max_predicted_label_score",
        abstention_grid=grid,
        minimum_coverage=0.25,
        selection_rule_version="1",
        metric_contract_version="2",
    )
    assert "test_metric" not in payload_text
    assert "selected_abstention" not in payload_text


def test_test_prediction_hash_changes_identity_not_selection() -> None:
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
        "threshold_policy_id": "tp",
        "threshold_policy_sha256": "e" * 64,
        "classification_threshold_basis_points": 39,
        "confidence_definition": "max_predicted_label_score",
        "abstention_grid": grid,
        "minimum_coverage": 0.25,
        "selection_rule_version": "1",
        "metric_contract_version": "2",
    }
    hash_a = compute_policy_input_sha256(
        predictions_test_sha256="d" * 64,
        **common,
    )
    hash_b = compute_policy_input_sha256(
        predictions_test_sha256="f" * 64,
        **common,
    )
    assert hash_a != hash_b

    labels = ["Bug"]
    y_true = np.array([[1], [1]], dtype=np.int8)
    y_score = np.array([[0.9], [0.8]], dtype=np.float64)
    _, sweep = build_abstention_sweep(
        labels=labels,
        y_true=y_true,
        y_score=y_score,
        issue_ids=[1, 2],
        classification_threshold=0.39,
        grid=grid,
    )
    first = select_abstention_threshold(
        sweep=sweep,
        minimum_coverage=0.25,
        classification_threshold_basis_points=39,
    )
    second = select_abstention_threshold(
        sweep=sweep,
        minimum_coverage=0.25,
        classification_threshold_basis_points=39,
    )
    assert first.selected_abstention_basis_points == second.selected_abstention_basis_points


def test_build_abstention_sweep_has_no_test_parameters() -> None:
    signature = inspect.signature(build_abstention_sweep)
    for name in signature.parameters:
        assert "test" not in name.lower()
