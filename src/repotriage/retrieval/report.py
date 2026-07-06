"""Deterministic serialization of retrieval-baseline artifacts."""

from __future__ import annotations

import hashlib
import json

from repotriage.retrieval.models import (
    CorpusRecord,
    FrozenRetrievalConfig,
    IndexMetadataDocument,
    QueryNeighborRecord,
    RetrievalManifest,
    RetrievalMetricsDocument,
)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _serialize_json(payload: object, *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    else:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    return (text + "\n").encode("utf-8")


def serialize_config_json(config: FrozenRetrievalConfig) -> bytes:
    return _serialize_json(config.model_dump(mode="json"), pretty=True)


def serialize_index_metadata_json(document: IndexMetadataDocument) -> bytes:
    return _serialize_json(document.model_dump(mode="json"), pretty=True)


def serialize_metrics_json(metrics: RetrievalMetricsDocument) -> bytes:
    return _serialize_json(metrics.model_dump(mode="json"), pretty=True)


def serialize_manifest_json(manifest: RetrievalManifest) -> bytes:
    return _serialize_json(manifest.model_dump(mode="json"), pretty=True)


def serialize_corpus_records_jsonl(records: list[CorpusRecord]) -> bytes:
    lines = [record.model_dump_json() for record in records]
    if not lines:
        return b""
    return ("\n".join(lines) + "\n").encode("utf-8")


def serialize_query_neighbors_jsonl(records: list[QueryNeighborRecord]) -> bytes:
    lines = [record.model_dump_json() for record in records]
    if not lines:
        return b""
    return ("\n".join(lines) + "\n").encode("utf-8")


def serialize_report_markdown(
    *,
    repository: str,
    model_dataset_id: str,
    retrieval_run_id: str,
    corpus_size: int,
    validation_query_count: int,
    test_query_count: int,
    top_k: int,
    validation_metrics: RetrievalMetricsDocument,
    test_metrics: RetrievalMetricsDocument,
    index_semantic_sha256: str,
) -> bytes:
    lines: list[str] = []
    lines.append("# Similar-issue retrieval baseline report")
    lines.append("")
    lines.append("## Context")
    lines.append("")
    lines.append(f"- Repository: `{repository}`")
    lines.append(f"- Model dataset id: `{model_dataset_id}`")
    lines.append(f"- Retrieval run id: `{retrieval_run_id}`")
    lines.append(f"- Corpus size (train only): {corpus_size}")
    lines.append(f"- Top-k: {top_k}")
    lines.append("")
    lines.append("## Retrieval method")
    lines.append("")
    lines.append(
        "This artifact indexes train-split issues with a word TF-IDF representation "
        "(unigrams and bigrams) and retrieves nearest neighbors by cosine similarity. "
        "Results are **similar historical issues** under this text representation — "
        "not guaranteed duplicates and not evidence of semantic understanding."
    )
    lines.append("")
    lines.append("## Train-only corpus protocol")
    lines.append("")
    lines.append(
        "The vectorizer is fit on train records only. Validation and test records are "
        "queries only and never appear as neighbors. Test metrics are informational and "
        "did not influence configuration or artifact identity."
    )
    lines.append("")
    lines.append("## Label-overlap relevance")
    lines.append("")
    lines.append(
        "A retrieved train issue is relevant when it shares at least one selected target "
        "label with the query. Queries with no selected labels are stored but excluded "
        "from metric denominators."
    )
    lines.append("")
    lines.append("## Validation metrics")
    lines.append("")
    lines.append(f"- Scored queries: {validation_metrics.scored_query_count}")
    lines.append(
        f"- All-zero-label queries (excluded): "
        f"{validation_metrics.all_zero_label_query_count}"
    )
    lines.append(f"- Recall@5: {validation_metrics.recall_at_5:.6f}")
    lines.append(f"- Recall@10: {validation_metrics.recall_at_10:.6f}")
    lines.append(f"- Precision@5: {validation_metrics.precision_at_5:.6f}")
    lines.append(f"- Precision@10: {validation_metrics.precision_at_10:.6f}")
    lines.append(f"- MRR@10: {validation_metrics.mrr_at_10:.6f}")
    lines.append(
        f"- Mean best label Jaccard@10: "
        f"{validation_metrics.mean_best_label_jaccard_at_10:.6f}"
    )
    lines.append(
        f"- Mean best shared-label count@10: "
        f"{validation_metrics.mean_best_shared_label_count_at_10:.6f}"
    )
    lines.append("")
    lines.append("## Frozen test metrics (informational)")
    lines.append("")
    lines.append(f"- Scored queries: {test_metrics.scored_query_count}")
    lines.append(
        f"- All-zero-label queries (excluded): {test_metrics.all_zero_label_query_count}"
    )
    lines.append(f"- Recall@5: {test_metrics.recall_at_5:.6f}")
    lines.append(f"- Recall@10: {test_metrics.recall_at_10:.6f}")
    lines.append(f"- Precision@5: {test_metrics.precision_at_5:.6f}")
    lines.append(f"- Precision@10: {test_metrics.precision_at_10:.6f}")
    lines.append(f"- MRR@10: {test_metrics.mrr_at_10:.6f}")
    lines.append("")
    lines.append("## Serialized index files")
    lines.append("")
    lines.append(
        "`vectorizer.joblib` and `corpus_matrix.npz` are stored for serving convenience. "
        "Joblib bytes are not guaranteed deterministic across environments; "
        f"`index_semantic_sha256` (`{index_semantic_sha256}`) is the authoritative "
        "fitted-state identity."
    )
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "This baseline does not use issue resolution or closure metadata. Neighbors are "
        "ranked by lexical similarity only and should not be interpreted as proven fixes."
    )
    lines.append("")
    return ("\n".join(lines) + "\n").encode("utf-8")
