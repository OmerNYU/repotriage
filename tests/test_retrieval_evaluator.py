"""Tests for label-overlap retrieval metrics."""

from __future__ import annotations

from repotriage.retrieval.evaluator import (
    compute_retrieval_metrics,
    is_relevant,
    label_jaccard,
    shared_labels,
)
from repotriage.retrieval.models import NeighborRecord, QueryNeighborRecord


def _neighbor(
    rank: int,
    labels: list[str],
    *,
    query_labels: list[str],
    similarity: float = 0.5,
) -> NeighborRecord:
    overlap = shared_labels(query_labels, labels)
    return NeighborRecord(
        rank=rank,
        neighbor_issue_id=rank,
        neighbor_issue_number=rank,
        similarity=similarity,
        neighbor_selected_labels=labels,
        shared_labels=overlap,
        label_jaccard=label_jaccard(query_labels, labels),
        is_relevant=is_relevant(query_labels, labels),
    )


def test_label_overlap_math() -> None:
    assert shared_labels(["Bug", "Docs"], ["Docs", "IO CSV"]) == ["Docs"]
    assert label_jaccard(["Bug", "Docs"], ["Docs"]) == 1 / 2
    assert label_jaccard([], []) == 0.0
    assert is_relevant(["Bug"], ["Docs"]) is False
    assert is_relevant(["Bug"], ["Bug", "Docs"]) is True


def test_metrics_exclude_zero_label_queries() -> None:
    scored = QueryNeighborRecord(
        query_issue_id=1,
        query_issue_number=1,
        query_split="validation",
        query_selected_labels=["Bug"],
        query_has_positive_labels=True,
        neighbors=[
            _neighbor(1, ["Bug"], query_labels=["Bug"]),
            _neighbor(2, ["Docs"], query_labels=["Bug"]),
        ],
    )
    zero = QueryNeighborRecord(
        query_issue_id=2,
        query_issue_number=2,
        query_split="validation",
        query_selected_labels=[],
        query_has_positive_labels=False,
        neighbors=[
            _neighbor(1, ["Bug"], query_labels=[]),
            _neighbor(2, ["Docs"], query_labels=[]),
        ],
    )
    metrics = compute_retrieval_metrics([scored, zero], split="validation")
    assert metrics.total_query_count == 2
    assert metrics.scored_query_count == 1
    assert metrics.all_zero_label_query_count == 1
    assert metrics.recall_at_5 == 1.0
    assert metrics.precision_at_5 == 0.2
    assert metrics.mrr_at_10 == 1.0
    assert metrics.mean_best_label_jaccard_at_10 == 1.0
    assert metrics.mean_best_shared_label_count_at_10 == 1.0


def test_recall_zero_when_no_relevant_neighbor() -> None:
    query = QueryNeighborRecord(
        query_issue_id=1,
        query_issue_number=1,
        query_split="test",
        query_selected_labels=["Bug"],
        query_has_positive_labels=True,
        neighbors=[
            _neighbor(1, ["Docs"], query_labels=["Bug"]),
            _neighbor(2, ["Enhancement"], query_labels=["Bug"]),
        ],
    )
    metrics = compute_retrieval_metrics([query], split="test")
    assert metrics.recall_at_10 == 0.0
    assert metrics.mrr_at_10 == 0.0
