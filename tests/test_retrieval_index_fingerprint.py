"""Tests for retrieval index semantic fingerprinting."""

from __future__ import annotations

import io

import joblib

from repotriage.baseline.reader import load_training_splits
from repotriage.model_dataset.builder import build_model_dataset
from repotriage.retrieval.config import load_retrieval_config
from repotriage.retrieval.index import (
    build_retrieval_index,
    compute_index_semantic_sha256,
    load_corpus_matrix,
    save_corpus_matrix,
)
from tests.helpers import write_retrieval_baseline_config
from tests.test_model_dataset_builder import _setup


def test_semantic_fingerprint_stable_across_joblib_roundtrip(tmp_path) -> None:
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
    env_hash = "a" * 64
    direct = compute_index_semantic_sha256(
        index,
        retrieval_baseline_version="1",
        model_dataset_id=model_result.manifest.model_dataset_id,
        repository=fixture.repository.full_name,
        label_order=splits.label_map.labels,
        top_k=config.top_k,
        similarity_metric=config.similarity_metric,
        metric_contract_version=config.metric_contract_version,
        numerical_environment_sha256=env_hash,
    )

    vectorizer_buffer = io.BytesIO()
    joblib.dump(index.vectorizer, vectorizer_buffer)
    matrix_buffer = io.BytesIO()
    save_corpus_matrix(matrix_buffer, index.corpus_matrix)
    matrix_path = tmp_path / "matrix.npz"
    matrix_path.write_bytes(matrix_buffer.getvalue())
    from repotriage.retrieval.index import RetrievalIndex

    reloaded = RetrievalIndex(
        vectorizer=joblib.load(io.BytesIO(vectorizer_buffer.getvalue())),
        corpus_matrix=load_corpus_matrix(str(matrix_path)),
        corpus_records=index.corpus_records,
        train_issue_ids=index.train_issue_ids,
    )
    roundtrip = compute_index_semantic_sha256(
        reloaded,
        retrieval_baseline_version="1",
        model_dataset_id=model_result.manifest.model_dataset_id,
        repository=fixture.repository.full_name,
        label_order=splits.label_map.labels,
        top_k=config.top_k,
        similarity_metric=config.similarity_metric,
        metric_contract_version=config.metric_contract_version,
        numerical_environment_sha256=env_hash,
    )
    assert direct == roundtrip
