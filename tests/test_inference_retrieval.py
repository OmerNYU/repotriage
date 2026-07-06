"""Tests for query-time retrieval and predicted-label overlap."""

from __future__ import annotations

from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from repotriage.inference.retrieval import predicted_label_overlap, search_query_text
from repotriage.retrieval.index import CorpusRecord, RetrievalIndex


def _build_index(texts: list[str], issue_ids: list[int]) -> RetrievalIndex:
    vectorizer = TfidfVectorizer(norm="l2")
    vectorizer.fit(texts)
    matrix = vectorizer.transform(texts)
    records = [
        CorpusRecord(
            issue_id=issue_id,
            issue_number=issue_id,
            created_at="2026-01-01T00:00:00Z",
            selected_labels=["Bug", "Indexing"],
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


def test_predicted_label_overlap_preserves_predicted_order() -> None:
    overlap = predicted_label_overlap(["Indexing", "Bug"], ["Bug", "Docs"])
    assert overlap == ["Bug"]


def test_search_query_text_returns_neighbors() -> None:
    index = _build_index(["alpha beta", "gamma delta"], [10, 20])
    results = search_query_text(
        index,
        "alpha beta",
        top_k=2,
        predicted_labels=["Bug"],
    )
    assert len(results) == 2
    assert results[0].rank == 1
    assert results[0].predicted_label_overlap == ["Bug"]
    assert results[0].similarity >= results[1].similarity
