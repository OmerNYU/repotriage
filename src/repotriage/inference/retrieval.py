"""Query-time retrieval over a train-only corpus index."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from repotriage.inference.models import SimilarIssueResult
from repotriage.retrieval.index import RetrievalIndex
from repotriage.retrieval.models import RetrievalCorruptionError


def predicted_label_overlap(
    predicted_labels: list[str],
    neighbor_labels: list[str],
) -> list[str]:
    """Return overlap between predicted labels and neighbor labels in predicted-label order."""
    neighbor_set = set(neighbor_labels)
    return [label for label in predicted_labels if label in neighbor_set]


def _validate_similarity(value: float) -> float:
    if not math.isfinite(value):
        raise RetrievalCorruptionError(f"Non-finite similarity value: {value!r}")
    return float(value)


def search_query_text(
    index: RetrievalIndex,
    query_feature_text: str,
    *,
    top_k: int,
    predicted_labels: list[str],
) -> list[SimilarIssueResult]:
    """Retrieve top-k train neighbors for one query feature text."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    effective_k = min(top_k, index.corpus_size)
    if index.corpus_size == 0:
        return []

    query_matrix = index.vectorizer.transform([query_feature_text])
    similarities = cosine_similarity(query_matrix, index.corpus_matrix)[0]
    corpus_issue_ids = np.asarray(index.train_issue_ids, dtype=np.int64)
    order = np.lexsort((corpus_issue_ids, -similarities))
    top_indices = order[:effective_k]

    results: list[SimilarIssueResult] = []
    for rank, corpus_index in enumerate(top_indices, start=1):
        corpus_record = index.corpus_records[int(corpus_index)]
        similarity = _validate_similarity(float(similarities[int(corpus_index)]))
        similarity = max(-1.0, min(1.0, similarity))
        neighbor_labels = list(corpus_record.selected_labels)
        results.append(
            SimilarIssueResult(
                rank=rank,
                issue_id=corpus_record.issue_id,
                issue_number=corpus_record.issue_number,
                similarity=similarity,
                neighbor_selected_labels=neighbor_labels,
                predicted_label_overlap=predicted_label_overlap(
                    predicted_labels,
                    neighbor_labels,
                ),
            )
        )
    return results
