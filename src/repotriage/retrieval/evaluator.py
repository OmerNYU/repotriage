"""Label-overlap retrieval relevance and aggregate metrics."""

from __future__ import annotations

from repotriage.model_dataset.models import SplitName
from repotriage.retrieval.models import (
    RETRIEVAL_METRIC_CONTRACT_VERSION,
    NeighborRecord,
    QueryNeighborRecord,
    RetrievalMetricsDocument,
)


def shared_labels(query_labels: list[str], neighbor_labels: list[str]) -> list[str]:
    """Return shared labels preserving canonical query-label order."""
    neighbor_set = set(neighbor_labels)
    return [label for label in query_labels if label in neighbor_set]


def label_jaccard(query_labels: list[str], neighbor_labels: list[str]) -> float:
    """Compute |Q ∩ N| / |Q ∪ N|; returns 0.0 when both label sets are empty."""
    query_set = set(query_labels)
    neighbor_set = set(neighbor_labels)
    union = query_set | neighbor_set
    if not union:
        return 0.0
    return len(query_set & neighbor_set) / len(union)


def is_relevant(query_labels: list[str], neighbor_labels: list[str]) -> bool:
    """A neighbor is relevant when it shares at least one selected target label."""
    return bool(set(query_labels) & set(neighbor_labels))


def annotate_neighbor(
    *,
    rank: int,
    query_labels: list[str],
    neighbor_issue_id: int,
    neighbor_issue_number: int,
    neighbor_labels: list[str],
    similarity: float,
    label_order: list[str],
) -> NeighborRecord:
    """Build one neighbor record with overlap diagnostics."""
    del label_order  # canonical order already reflected in selected_labels lists
    overlap = shared_labels(query_labels, neighbor_labels)
    return NeighborRecord(
        rank=rank,
        neighbor_issue_id=neighbor_issue_id,
        neighbor_issue_number=neighbor_issue_number,
        similarity=similarity,
        neighbor_selected_labels=list(neighbor_labels),
        shared_labels=overlap,
        label_jaccard=label_jaccard(query_labels, neighbor_labels),
        is_relevant=is_relevant(query_labels, neighbor_labels),
    )


def _metric_at_k(
    queries: list[QueryNeighborRecord],
    *,
    k: int,
    metric: str,
) -> float:
    scored = [query for query in queries if query.query_has_positive_labels]
    if not scored:
        return 0.0

    if metric == "recall":
        hits = 0
        for query in scored:
            top_neighbors = query.neighbors[:k]
            if any(neighbor.is_relevant for neighbor in top_neighbors):
                hits += 1
        return hits / len(scored)

    if metric == "precision":
        total = 0.0
        for query in scored:
            top_neighbors = query.neighbors[:k]
            relevant_count = sum(1 for neighbor in top_neighbors if neighbor.is_relevant)
            total += relevant_count / k
        return total / len(scored)

    raise ValueError(f"Unsupported metric: {metric!r}")


def compute_mrr_at_10(queries: list[QueryNeighborRecord]) -> float:
    scored = [query for query in queries if query.query_has_positive_labels]
    if not scored:
        return 0.0
    total = 0.0
    for query in scored:
        reciprocal = 0.0
        for neighbor in query.neighbors[:10]:
            if neighbor.is_relevant:
                reciprocal = 1.0 / neighbor.rank
                break
        total += reciprocal
    return total / len(scored)


def compute_mean_best_label_jaccard_at_10(queries: list[QueryNeighborRecord]) -> float:
    scored = [query for query in queries if query.query_has_positive_labels]
    if not scored:
        return 0.0
    total = 0.0
    for query in scored:
        best = 0.0
        for neighbor in query.neighbors[:10]:
            best = max(best, neighbor.label_jaccard)
        total += best
    return total / len(scored)


def compute_mean_best_shared_label_count_at_10(queries: list[QueryNeighborRecord]) -> float:
    scored = [query for query in queries if query.query_has_positive_labels]
    if not scored:
        return 0.0
    total = 0.0
    for query in scored:
        best = 0
        for neighbor in query.neighbors[:10]:
            best = max(best, len(neighbor.shared_labels))
        total += best
    return total / len(scored)


def compute_retrieval_metrics(
    queries: list[QueryNeighborRecord],
    *,
    split: SplitName,
) -> RetrievalMetricsDocument:
    """Aggregate label-overlap retrieval metrics for one query split."""
    total = len(queries)
    all_zero = sum(1 for query in queries if not query.query_has_positive_labels)
    scored_count = total - all_zero
    return RetrievalMetricsDocument(
        split=split,
        total_query_count=total,
        scored_query_count=scored_count,
        all_zero_label_query_count=all_zero,
        recall_at_5=_metric_at_k(queries, k=5, metric="recall"),
        recall_at_10=_metric_at_k(queries, k=10, metric="recall"),
        precision_at_5=_metric_at_k(queries, k=5, metric="precision"),
        precision_at_10=_metric_at_k(queries, k=10, metric="precision"),
        mrr_at_10=compute_mrr_at_10(queries),
        mean_best_label_jaccard_at_10=compute_mean_best_label_jaccard_at_10(queries),
        mean_best_shared_label_count_at_10=compute_mean_best_shared_label_count_at_10(queries),
        metric_contract_version=RETRIEVAL_METRIC_CONTRACT_VERSION,
    )
