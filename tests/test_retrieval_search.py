"""Tests for cosine-similarity search and deterministic ranking."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from repotriage.baseline.reader import SplitBundle
from repotriage.model_dataset.models import ModelReadyRecord
from repotriage.retrieval.index import CorpusRecord, RetrievalIndex
from repotriage.retrieval.search import search_split


def _record(issue_id: int, text: str, labels: list[str]) -> ModelReadyRecord:
    return ModelReadyRecord(
        repository="pandas-dev/pandas",
        issue_id=issue_id,
        issue_number=issue_id,
        created_at="2026-01-01T00:00:00Z",
        title=text,
        body="",
        feature_text=text,
        selected_labels=labels,
        target_vector=[1 if label == "Bug" else 0 for label in ["Bug"]],
        split="train",
    )


def _build_index(texts: list[str], issue_ids: list[int]) -> RetrievalIndex:
    vectorizer = TfidfVectorizer(norm="l2")
    vectorizer.fit(texts)
    matrix = vectorizer.transform(texts)
    records = [
        CorpusRecord(
            issue_id=issue_id,
            issue_number=issue_id,
            created_at="2026-01-01T00:00:00Z",
            selected_labels=["Bug"],
            corpus_index=index,
        )
        for index, issue_id in enumerate(issue_ids)
    ]
    return RetrievalIndex(
        vectorizer=vectorizer,
        corpus_matrix=matrix if sparse.isspmatrix_csr(matrix) else matrix.tocsr(),
        corpus_records=records,
        train_issue_ids=issue_ids,
    )


def test_cosine_ordering_and_issue_id_tiebreak() -> None:
    corpus_texts = ["alpha beta", "alpha beta", "alpha beta"]
    issue_ids = [30, 10, 20]
    index = _build_index(corpus_texts, issue_ids)
    query = SplitBundle(
        records=[_record(99, "alpha beta", ["Bug"])],
        texts=["alpha beta"],
        targets=np.zeros((1, 1), dtype=np.int8),
    )
    results = search_split(index, query, split_name="validation", top_k=3, label_order=["Bug"])
    neighbors = results[0].neighbors
    assert [neighbor.neighbor_issue_id for neighbor in neighbors] == [10, 20, 30]
    assert neighbors[0].similarity == pytest.approx(1.0, abs=1e-9)


def test_top_k_limits_neighbors() -> None:
    corpus_texts = ["one", "two", "three"]
    issue_ids = [1, 2, 3]
    index = _build_index(corpus_texts, issue_ids)
    query = SplitBundle(
        records=[_record(99, "one", ["Bug"])],
        texts=["one"],
        targets=np.zeros((1, 1), dtype=np.int8),
    )
    results = search_split(index, query, split_name="test", top_k=2, label_order=["Bug"])
    assert len(results[0].neighbors) == 2
