"""Cosine-similarity top-k retrieval over a train-only corpus."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from repotriage.baseline.reader import SplitBundle
from repotriage.model_dataset.models import SplitName
from repotriage.retrieval.evaluator import annotate_neighbor
from repotriage.retrieval.index import RetrievalIndex
from repotriage.retrieval.models import (
    NeighborRecord,
    QueryNeighborRecord,
    RetrievalCorruptionError,
)


def _validate_similarity(value: float) -> float:
    if not math.isfinite(value):
        raise RetrievalCorruptionError(f"Non-finite similarity value: {value!r}")
    return float(value)


def search_split(
    index: RetrievalIndex,
    query_split: SplitBundle,
    *,
    split_name: SplitName,
    top_k: int,
    label_order: list[str],
) -> list[QueryNeighborRecord]:
    """Retrieve top-k train neighbors for each query in a split."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    effective_k = min(top_k, index.corpus_size)
    if not query_split.records:
        return []

    query_matrix = index.vectorizer.transform(query_split.texts)
    similarities = cosine_similarity(query_matrix, index.corpus_matrix)
    corpus_issue_ids = np.asarray(index.train_issue_ids, dtype=np.int64)

    results: list[QueryNeighborRecord] = []
    for row_index, record in enumerate(query_split.records):
        row = similarities[row_index]
        # Deterministic ranking: higher similarity first, then lower issue_id.
        order = np.lexsort((corpus_issue_ids, -row))
        top_indices = order[:effective_k]

        neighbors: list[NeighborRecord] = []
        for rank, corpus_index in enumerate(top_indices, start=1):
            corpus_record = index.corpus_records[int(corpus_index)]
            similarity = _validate_similarity(float(row[int(corpus_index)]))
            similarity = max(-1.0, min(1.0, similarity))
            neighbor = annotate_neighbor(
                rank=rank,
                query_labels=record.selected_labels,
                neighbor_issue_id=corpus_record.issue_id,
                neighbor_issue_number=corpus_record.issue_number,
                neighbor_labels=corpus_record.selected_labels,
                similarity=similarity,
                label_order=label_order,
            )
            neighbors.append(neighbor)

        results.append(
            QueryNeighborRecord(
                query_issue_id=record.issue_id,
                query_issue_number=record.issue_number,
                query_split=split_name,
                query_selected_labels=list(record.selected_labels),
                query_has_positive_labels=bool(record.selected_labels),
                neighbors=neighbors,
            )
        )
    return results


def assert_neighbors_descending(neighbors: list[NeighborRecord]) -> None:
    """Validate rank ordering and similarity monotonicity."""
    ranks = [neighbor.rank for neighbor in neighbors]
    if ranks != list(range(1, len(neighbors) + 1)):
        raise RetrievalCorruptionError(f"Neighbor ranks are not contiguous from 1: {ranks}")
    seen_pairs: set[tuple[int, int]] = set()
    previous_similarity = float("inf")
    for neighbor in neighbors:
        pair = (neighbor.rank, neighbor.neighbor_issue_id)
        if pair in seen_pairs:
            raise RetrievalCorruptionError(f"Duplicate rank/neighbor pair: {pair}")
        seen_pairs.add(pair)
        if neighbor.similarity > previous_similarity + 1e-12:
            raise RetrievalCorruptionError("Neighbor similarities are not in descending order")
        previous_similarity = neighbor.similarity
