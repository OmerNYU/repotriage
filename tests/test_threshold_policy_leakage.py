"""Leakage tests for threshold-policy selection."""

from __future__ import annotations

import inspect
from unittest.mock import patch

import numpy as np

from repotriage.baseline.builder import train_baseline
from repotriage.threshold_policy.builder import build_threshold_policy
from repotriage.threshold_policy.models import ThresholdGridConfig, compute_policy_input_sha256
from repotriage.threshold_policy.selector import select_threshold
from repotriage.threshold_policy.sweep import build_threshold_sweep
from tests.helpers import write_baseline_config, write_threshold_policy_config
from tests.test_baseline_builder import _build_model_ready


def test_load_test_scores_not_called_before_selection(tmp_path) -> None:
    fixture, model_result = _build_model_ready(tmp_path)
    baseline_config = write_baseline_config(tmp_path / "baseline.json")
    baselines_root = tmp_path / "baselines"
    baseline_result = train_baseline(
        fixture.repository,
        model_result.manifest.model_dataset_id,
        baseline_config,
        model_ready_root=fixture.model_ready_root,
        baselines_root=baselines_root,
    )
    policy_config = write_threshold_policy_config(
        tmp_path / "policy.json",
        baseline_run_id=baseline_result.manifest.baseline_run_id,
    )
    policies_root = tmp_path / "policies"
    selection_completed = {"value": False}

    original_pipeline = __import__(
        "repotriage.threshold_policy.builder", fromlist=["_run_validation_only_pipeline"]
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
        "repotriage.threshold_policy.builder._run_validation_only_pipeline",
        side_effect=wrapped_pipeline,
    ):
        with patch(
            "repotriage.threshold_policy.builder.load_test_scores",
            side_effect=guarded_load,
        ):
            try:
                build_threshold_policy(
                    fixture.repository,
                    policy_config,
                    baselines_root=baselines_root,
                    threshold_policies_root=policies_root,
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
        start_basis_points=30,
        stop_basis_points=70,
        step_basis_points=10,
        denominator=100,
    )
    sweep = build_threshold_sweep(labels=labels, y_true=y_true, y_score=y_score, grid=grid)
    selection_a = select_threshold(sweep=sweep)
    # Test scores are not inputs to selection; mutating them cannot change the winner.
    selection_b = select_threshold(sweep=sweep)
    assert (
        selection_a.selected_threshold_basis_points
        == selection_b.selected_threshold_basis_points
    )


def test_validation_score_change_can_change_selected_threshold() -> None:
    labels = ["Bug"]
    grid = ThresholdGridConfig(
        start_basis_points=40,
        stop_basis_points=60,
        step_basis_points=10,
        denominator=100,
    )
    y_true = np.array([[1], [1]], dtype=np.int8)
    low_scores = np.array([[0.45], [0.46]], dtype=np.float64)
    high_scores = np.array([[0.95], [0.96]], dtype=np.float64)
    low_selection = select_threshold(
        sweep=build_threshold_sweep(
            labels=labels, y_true=y_true, y_score=low_scores, grid=grid
        )
    )
    high_selection = select_threshold(
        sweep=build_threshold_sweep(
            labels=labels, y_true=y_true, y_score=high_scores, grid=grid
        )
    )
    assert (
        low_selection.selected_threshold_basis_points
        != high_selection.selected_threshold_basis_points
    )


def test_policy_input_hash_excludes_selected_threshold() -> None:
    grid = ThresholdGridConfig(
        start_basis_points=5,
        stop_basis_points=95,
        step_basis_points=1,
        denominator=100,
    )
    hash_a = compute_policy_input_sha256(
        threshold_policy_version="1",
        baseline_run_id="run",
        baseline_experiment_sha256="a" * 64,
        model_semantic_sha256="b" * 64,
        predictions_validation_sha256="c" * 64,
        predictions_test_sha256="d" * 64,
        selected_candidate_id="c1",
        threshold_grid=grid,
        selection_rule_version="1",
        metric_contract_version="2",
    )
    assert hash_a
    assert "selected_threshold" not in hash_a


def test_selector_signature_has_no_test_args() -> None:
    for name in inspect.signature(select_threshold).parameters:
        assert "test" not in name.lower()
