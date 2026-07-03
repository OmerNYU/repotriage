"""Tests for threshold sweep and grid construction."""

from __future__ import annotations

import numpy as np
import pytest

from repotriage.baseline.models import PredictionRecord
from repotriage.threshold_policy.models import ThresholdGridConfig, ThresholdPolicyInputError
from repotriage.threshold_policy.sweep import (
    build_threshold_grid,
    build_threshold_sweep,
    predictions_from_scores,
)


def test_basis_point_grid_produces_91_values_without_float_drift() -> None:
    grid = ThresholdGridConfig(
        start_basis_points=5,
        stop_basis_points=95,
        step_basis_points=1,
        denominator=100,
    )
    basis_points = build_threshold_grid(grid=grid)
    assert len(basis_points) == 91
    assert basis_points[0] == 5
    assert basis_points[-1] == 95
    thresholds = [bp / 100 for bp in basis_points]
    assert thresholds[45] == 0.50
    assert thresholds[34] == 0.39


def test_score_equal_to_threshold_predicts_positive() -> None:
    scores = np.array([[0.5, 0.49]], dtype=np.float64)
    predictions = predictions_from_scores(scores, 0.5)
    assert predictions.tolist() == [[1, 0]]


def test_all_zero_predictions_at_high_threshold() -> None:
    scores = np.array([[0.1, 0.2, 0.3]], dtype=np.float64)
    predictions = predictions_from_scores(scores, 0.95)
    assert predictions.sum() == 0


def test_all_positive_predictions_at_low_threshold() -> None:
    scores = np.array([[0.1, 0.2, 0.3]], dtype=np.float64)
    predictions = predictions_from_scores(scores, 0.05)
    assert predictions.sum() == 3


def test_nan_scores_rejected_by_validate_score_matrix() -> None:
    from repotriage.baseline.scores import validate_score_matrix

    scores = np.array([[0.1, np.nan]], dtype=np.float64)
    with pytest.raises(ValueError, match="finite"):
        validate_score_matrix(scores, target_count=2)


def test_vector_label_dimension_mismatch_rejected() -> None:
    labels = ["a", "b"]
    y_true = np.array([[1, 0]], dtype=np.int8)
    y_score = np.array([[0.6, 0.4, 0.1]], dtype=np.float64)
    grid = ThresholdGridConfig(
        start_basis_points=50,
        stop_basis_points=50,
        step_basis_points=1,
        denominator=100,
    )
    with pytest.raises(ValueError):
        build_threshold_sweep(labels=labels, y_true=y_true, y_score=y_score, grid=grid)


def test_duplicate_validation_issue_rejected(tmp_path) -> None:
    from repotriage.baseline.models import BaselineManifest
    from repotriage.threshold_policy.reader import load_validation_scores

    record_kwargs = {
        "repository": "pandas-dev/pandas",
        "model_dataset_id": "md1",
        "baseline_run_id": "run1",
        "issue_number": 1,
        "split": "validation",
        "true_labels": ["Bug"],
        "true_vector": [1, 0],
        "predicted_labels": ["Bug"],
        "predicted_vector": [1, 0],
        "score_type": "probability_estimates",
        "threshold": 0.5,
        "score_vector": [0.8, 0.2],
        "candidate_id": "c1",
    }
    lines = [
        PredictionRecord(issue_id=1, **record_kwargs).model_dump_json(),
        PredictionRecord(issue_id=1, **record_kwargs).model_dump_json(),
    ]
    path = tmp_path / "predictions_validation.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = BaselineManifest.model_construct(
        baseline_run_id="run1",
        baseline_experiment_sha256="a" * 64,
        numerical_environment_sha256="b" * 64,
        baseline_run_sha256="c" * 64,
        repository="pandas-dev/pandas",
        model_dataset_id="md1",
        records_sha256="d" * 64,
        label_map_sha256="e" * 64,
        config_semantic_sha256="f" * 64,
        config_source_sha256="0" * 64,
        random_state=42,
        threshold=0.5,
        selected_candidate_id="c1",
        built_at="2026-01-01T00:00:00Z",
        validation_record_count=2,
        validation_prediction_count=2,
        test_record_count=0,
        target_count=2,
        config_sha256="1" * 64,
        candidate_results_sha256="2" * 64,
        metrics_test_sha256="3" * 64,
        metrics_markdown_sha256="4" * 64,
        predictions_validation_sha256="5" * 64,
        predictions_test_sha256="6" * 64,
        feature_summary_sha256="7" * 64,
        model_sha256="8" * 64,
        model_semantic_sha256="9" * 64,
        environment={
            "python_implementation": "CPython",
            "python_version": "3.13",
            "os_system": "Linux",
            "platform": "Linux",
            "machine_architecture": "x86_64",
            "numerical_thread_limit": 1,
            "reproducibility_note": "note",
            "serialization_security_warning": "warn",
        },
    )
    with pytest.raises(ThresholdPolicyInputError, match="Duplicate validation issue_ids"):
        load_validation_scores(tmp_path, baseline_manifest=manifest, candidate_id="c1")
