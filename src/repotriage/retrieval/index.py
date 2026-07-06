"""Train-only TF-IDF index construction and semantic fingerprinting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from repotriage.baseline.models import TfidfParams
from repotriage.baseline.reader import SplitBundle
from repotriage.baseline.runtime import numerical_thread_limits
from repotriage.model_dataset.models import ModelReadyRecord
from repotriage.retrieval.features import build_vectorizer, fit_vectorizer, transform_texts
from repotriage.retrieval.models import (
    CORPUS_MATRIX_NPZ_FILE,
    INDEX_SEMANTIC_CONTRACT_VERSION,
    CorpusRecord,
    SimilarityMetric,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=str)}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _array_fingerprint(array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


def _sparse_matrix_fingerprint(matrix: sparse.csr_matrix) -> dict[str, Any]:
    return {
        "data": _array_fingerprint(matrix.data),
        "indices": _array_fingerprint(matrix.indices),
        "indptr": _array_fingerprint(matrix.indptr),
        "shape": list(matrix.shape),
    }


def build_corpus_records(records: list[ModelReadyRecord]) -> list[CorpusRecord]:
    corpus_records: list[CorpusRecord] = []
    for index, record in enumerate(records):
        corpus_records.append(
            CorpusRecord(
                issue_id=record.issue_id,
                issue_number=record.issue_number,
                created_at=record.model_dump(mode="json")["created_at"],
                selected_labels=list(record.selected_labels),
                corpus_index=index,
            )
        )
    return corpus_records


@dataclass(frozen=True)
class RetrievalIndex:
    """Fitted train-only TF-IDF corpus index."""

    vectorizer: TfidfVectorizer
    corpus_matrix: sparse.csr_matrix
    corpus_records: list[CorpusRecord]
    train_issue_ids: list[int]

    @property
    def corpus_size(self) -> int:
        return len(self.corpus_records)

    @property
    def vocabulary_size(self) -> int:
        return len(self.vectorizer.vocabulary_)


def build_retrieval_index(train: SplitBundle, tfidf: TfidfParams) -> RetrievalIndex:
    """Fit a vectorizer on train texts only and transform the train corpus."""
    if not train.records:
        raise ValueError("train split must not be empty")
    vectorizer = build_vectorizer(tfidf)
    with numerical_thread_limits():
        fit_vectorizer(vectorizer, train.texts)
        corpus_matrix = transform_texts(vectorizer, train.texts)
    if not sparse.isspmatrix_csr(corpus_matrix):
        corpus_matrix = corpus_matrix.tocsr()
    corpus_records = build_corpus_records(train.records)
    train_issue_ids = [record.issue_id for record in corpus_records]
    return RetrievalIndex(
        vectorizer=vectorizer,
        corpus_matrix=corpus_matrix,
        corpus_records=corpus_records,
        train_issue_ids=train_issue_ids,
    )


def compute_index_semantic_sha256(
    index: RetrievalIndex,
    *,
    retrieval_baseline_version: str,
    model_dataset_id: str,
    repository: str,
    label_order: list[str],
    top_k: int,
    similarity_metric: SimilarityMetric,
    metric_contract_version: str,
    numerical_environment_sha256: str,
) -> str:
    """Canonical fingerprint over inference-relevant fitted index state."""
    vectorizer = index.vectorizer
    vocabulary = sorted(vectorizer.vocabulary_.items(), key=lambda item: item[0])
    payload = {
        "corpus_matrix": _sparse_matrix_fingerprint(index.corpus_matrix),
        "corpus_issue_ids": list(index.train_issue_ids),
        "index_semantic_contract_version": INDEX_SEMANTIC_CONTRACT_VERSION,
        "label_order": list(label_order),
        "metric_contract_version": metric_contract_version,
        "model_dataset_id": model_dataset_id,
        "numerical_environment_sha256": numerical_environment_sha256,
        "repository": repository,
        "retrieval_baseline_version": retrieval_baseline_version,
        "similarity_metric": similarity_metric,
        "top_k": top_k,
        "vectorizer": {
            "class_name": type(vectorizer).__name__,
            "idf_": _array_fingerprint(vectorizer.idf_),
            "params": _json_safe(vectorizer.get_params()),
            "vocabulary": [[term, int(idx)] for term, idx in vocabulary],
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def save_corpus_matrix(target, matrix: sparse.csr_matrix) -> None:
    sparse.save_npz(target, matrix)


def load_corpus_matrix(path: str) -> sparse.csr_matrix:
    loaded = sparse.load_npz(path)
    if not sparse.isspmatrix_csr(loaded):
        loaded = loaded.tocsr()
    return loaded


CORPUS_MATRIX_NPZ = CORPUS_MATRIX_NPZ_FILE
