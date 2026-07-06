"""Tests for train-only TF-IDF feature fitting in retrieval."""

from __future__ import annotations

from repotriage.baseline.reader import load_training_splits
from repotriage.model_dataset.builder import build_model_dataset
from repotriage.retrieval.config import load_retrieval_config
from repotriage.retrieval.index import build_retrieval_index
from tests.helpers import write_retrieval_baseline_config
from tests.test_model_dataset_builder import _setup


def test_vectorizer_fit_train_only(tmp_path) -> None:
    fixture = _setup(tmp_path)
    model_result = build_model_dataset(
        fixture.repository,
        fixture.dataset_id,
        fixture.policy_id,
        fixture.split_config_path,
        processed_root=fixture.processed_root,
        policies_root=fixture.policies_root,
        model_ready_root=fixture.model_ready_root,
    )
    config_path = write_retrieval_baseline_config(tmp_path / "retrieval.json", min_df=1)
    config, _, _, _ = load_retrieval_config(config_path)
    splits = load_training_splits(
        model_result.model_dataset_dir,
        expected_repository=fixture.repository,
        expected_model_dataset_id=model_result.manifest.model_dataset_id,
    )
    index = build_retrieval_index(splits.train, config.tfidf)
    val_token = "zz_unique_val_token_for_retrieval_leakage"
    assert val_token not in index.vectorizer.vocabulary_
    splits.validation.texts.append(val_token)
    matrix = index.vectorizer.transform(splits.validation.texts)
    assert matrix[-1, :].nnz == 0


def test_vectorizer_excludes_test_only_token(tmp_path) -> None:
    fixture = _setup(tmp_path)
    model_result = build_model_dataset(
        fixture.repository,
        fixture.dataset_id,
        fixture.policy_id,
        fixture.split_config_path,
        processed_root=fixture.processed_root,
        policies_root=fixture.policies_root,
        model_ready_root=fixture.model_ready_root,
    )
    config_path = write_retrieval_baseline_config(tmp_path / "retrieval.json", min_df=1)
    config, _, _, _ = load_retrieval_config(config_path)
    splits = load_training_splits(
        model_result.model_dataset_dir,
        expected_repository=fixture.repository,
        expected_model_dataset_id=model_result.manifest.model_dataset_id,
    )
    index = build_retrieval_index(splits.train, config.tfidf)
    test_only_token = "zz_unique_test_token_for_retrieval_leakage"
    assert test_only_token not in index.vectorizer.vocabulary_
    matrix = index.vectorizer.transform([test_only_token])
    assert matrix[0, :].nnz == 0
