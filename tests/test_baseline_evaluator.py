"""Tests for baseline evaluator metrics."""

from __future__ import annotations

import numpy as np
import pytest

from repotriage.baseline.evaluator import compute_split_metrics
from repotriage.baseline.models_ml import AllZeroPredictor
from repotriage.baseline.scores import validate_score_matrix


def test_all_zero_predictor_metrics() -> None:
    labels = ["Bug", "Docs"]
    y_true = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.int8)
    predictor = AllZeroPredictor(labels=labels)
    y_pred = predictor.predict_matrix(len(y_true))
    metrics = compute_split_metrics(
        split="validation",
        labels=labels,
        y_true=y_true,
        y_pred=y_pred,
        y_score=None,
        threshold=0.5,
        score_type="none",
    )
    assert metrics.aggregate.subset_accuracy == 1 / 3
    assert metrics.aggregate.micro_recall == 0.0
    assert metrics.aggregate.micro_f1 is None
    assert metrics.aggregate.macro_precision == 0.0
    assert metrics.aggregate.macro_label_count == 2
    assert metrics.per_label[0].precision is None
    assert metrics.per_label[0].precision_undefined_reason == "no_positive_predictions"


def test_hand_computed_micro_f1() -> None:
    labels = ["Bug"]
    y_true = np.array([[1], [0], [1]], dtype=np.int8)
    y_pred = np.array([[1], [1], [0]], dtype=np.int8)
    y_score = np.array([[0.9], [0.6], [0.1]], dtype=np.float64)
    metrics = compute_split_metrics(
        split="test",
        labels=labels,
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        threshold=0.5,
        score_type="probability_estimates",
    )
    assert metrics.aggregate.micro_precision == 0.5
    assert metrics.aggregate.micro_recall == 0.5
    assert metrics.aggregate.micro_f1 == 0.5


def test_f1_zero_when_precision_and_recall_zero() -> None:
    labels = ["Regression"]
    y_true = np.array([[1], [0]], dtype=np.int8)
    y_pred = np.array([[0], [1]], dtype=np.int8)
    metrics = compute_split_metrics(
        split="test",
        labels=labels,
        y_true=y_true,
        y_pred=y_pred,
        y_score=np.array([[0.1], [0.9]], dtype=np.float64),
        threshold=0.5,
        score_type="probability_estimates",
    )
    assert metrics.per_label[0].precision == 0.0
    assert metrics.per_label[0].recall == 0.0
    assert metrics.per_label[0].f1 == 0.0
    assert metrics.aggregate.macro_f1 == 0.0


def test_samples_f1_empty_empty_policy() -> None:
    labels = ["Bug", "Docs"]
    y_true = np.array([[0, 0]], dtype=np.int8)
    y_pred = np.array([[0, 0]], dtype=np.int8)
    metrics = compute_split_metrics(
        split="validation",
        labels=labels,
        y_true=y_true,
        y_pred=y_pred,
        y_score=None,
        threshold=0.5,
        score_type="none",
    )
    assert metrics.aggregate.samples_f1 == 1.0
    assert metrics.aggregate.samples_f1_empty_empty_policy == "one"


def test_macro_zero_filled_includes_undefined_labels() -> None:
    labels = ["A", "B"]
    y_true = np.array([[1, 0], [0, 0]], dtype=np.int8)
    y_pred = np.array([[1, 0], [0, 0]], dtype=np.int8)
    metrics = compute_split_metrics(
        split="validation",
        labels=labels,
        y_true=y_true,
        y_pred=y_pred,
        y_score=None,
        threshold=0.5,
        score_type="none",
    )
    assert metrics.aggregate.macro_precision == 0.5
    assert metrics.aggregate.macro_precision_defined_only == 1.0
    assert metrics.aggregate.undefined_precision_label_count == 1


def test_validate_score_matrix_rejects_out_of_bounds() -> None:
    with pytest.raises(ValueError, match="within"):
        validate_score_matrix(np.array([[1.5]]), target_count=1)
