"""Tests for explicit per-label logistic regression models."""

from __future__ import annotations

import numpy as np

from repotriage.baseline.config import load_baseline_config
from repotriage.baseline.models_ml import AllZeroPredictor, train_model
from tests.helpers import write_baseline_config


def test_train_model_score_dimensions(tmp_path) -> None:
    config, _, _, _ = load_baseline_config(write_baseline_config(tmp_path / "baseline.json"))
    candidate = config.candidates[0]
    labels = ["Bug", "Docs"]
    train_texts = [
        "bug crash",
        "documentation fix",
        "another bug",
        "more docs",
    ]
    train_targets = np.array(
        [
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
        ],
        dtype=np.int8,
    )
    model, report = train_model(
        candidate=candidate,
        labels=labels,
        train_texts=train_texts,
        train_targets=train_targets,
        random_state=42,
        threshold=0.5,
    )
    scores = model.predict_proba_matrix(train_texts)
    assert scores.shape == (4, 2)
    assert len(model.estimators) == 2
    assert report.vocabulary_size > 0


def test_all_zero_predictor() -> None:
    predictor = AllZeroPredictor(labels=["Bug", "Docs"])
    preds = predictor.predict_matrix(3)
    assert preds.shape == (3, 2)
    assert preds.sum() == 0
