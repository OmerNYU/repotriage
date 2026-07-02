"""Tests for baseline feature extraction."""

from __future__ import annotations

from repotriage.baseline.config import load_baseline_config
from repotriage.baseline.features import candidate_vectorizer, fit_vectorizer, transform_texts
from tests.helpers import write_baseline_config


def test_vectorizer_fit_train_only_vocab(tmp_path) -> None:
    config, _, _, _ = load_baseline_config(write_baseline_config(tmp_path / "baseline.json"))
    vectorizer = candidate_vectorizer(config.candidates[0])
    train_texts = ["bug in read_csv", "docs update"]
    val_texts = ["unique_validation_token_xyz"]
    fit_vectorizer(vectorizer, train_texts)
    assert "unique_validation_token_xyz" not in vectorizer.vocabulary_
    matrix = transform_texts(vectorizer, val_texts)
    assert matrix.shape == (1, len(vectorizer.vocabulary_))
