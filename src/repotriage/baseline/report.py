"""Deterministic serialization of baseline artifacts."""

from __future__ import annotations

import hashlib
import json

from repotriage.baseline.models import (
    BaselineManifest,
    CandidateResultsDocument,
    FeatureSummary,
    FrozenConfigDocument,
    PredictionRecord,
    SplitMetrics,
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


def serialize_predictions_jsonl(records: list[PredictionRecord]) -> bytes:
    lines: list[str] = []
    for record in records:
        payload = record.model_dump(mode="json")
        line = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        lines.append(line + "\n")
    return "".join(lines).encode("utf-8")


def serialize_metrics_json(metrics: SplitMetrics) -> bytes:
    return _serialize_json(metrics.model_dump(mode="json"), pretty=True)


def serialize_candidate_results_json(document: CandidateResultsDocument) -> bytes:
    return _serialize_json(document.model_dump(mode="json"), pretty=True)


def serialize_feature_summary_json(summary: FeatureSummary) -> bytes:
    return _serialize_json(summary.model_dump(mode="json"), pretty=True)


def serialize_frozen_config_json(document: FrozenConfigDocument) -> bytes:
    return _serialize_json(document.model_dump(mode="json"), pretty=True)


def serialize_manifest_json(manifest: BaselineManifest) -> bytes:
    return _serialize_json(manifest.model_dump(mode="json"), pretty=True)


def serialize_metrics_markdown(
    *,
    candidate_results: CandidateResultsDocument,
    test_metrics: SplitMetrics,
    selected_candidate_id: str,
) -> bytes:
    lines: list[str] = []
    lines.append("# Baseline evaluation report")
    lines.append("")
    lines.append("## Selected candidate")
    lines.append("")
    lines.append(f"- Candidate id: {selected_candidate_id}")
    lines.append(
        f"- Winner selection rule version: "
        f"{candidate_results.selection.selection_rule_version}"
    )
    lines.append(
        f"- Ranked candidates: {', '.join(candidate_results.selection.ranked_candidate_ids)}"
    )
    lines.append("")

    lines.append("## Validation candidate comparison")
    lines.append("")
    for candidate in candidate_results.candidates:
        aggregate = candidate.metrics.aggregate
        lines.append(f"### {candidate.candidate_id}")
        lines.append("")
        lines.append(f"- Macro average precision: {aggregate.macro_average_precision}")
        lines.append(f"- Macro F1 (zero-filled): {aggregate.macro_f1}")
        lines.append(f"- Macro F1 (defined-only): {aggregate.macro_f1_defined_only}")
        lines.append(f"- Micro F1: {aggregate.micro_f1}")
        lines.append(f"- Subset accuracy: {aggregate.subset_accuracy:.4f}")
        lines.append(f"- Hamming loss: {aggregate.hamming_loss:.4f}")
        lines.append("")

    if candidate_results.dummy_baseline is not None:
        dummy = candidate_results.dummy_baseline.metrics.aggregate
        lines.append("## Dummy all-zero baseline (validation)")
        lines.append("")
        lines.append(f"- Subset accuracy: {dummy.subset_accuracy:.4f}")
        lines.append(f"- Hamming loss: {dummy.hamming_loss:.4f}")
        lines.append(f"- Micro F1: {dummy.micro_f1}")
        lines.append(f"- Macro F1: {dummy.macro_f1}")
        lines.append("")

    test_aggregate = test_metrics.aggregate
    lines.append("## Frozen test metrics")
    lines.append("")
    lines.append(f"- Macro average precision: {test_aggregate.macro_average_precision}")
    lines.append(f"- Macro F1 (zero-filled): {test_aggregate.macro_f1}")
    lines.append(f"- Macro F1 (defined-only): {test_aggregate.macro_f1_defined_only}")
    lines.append(f"- Micro F1: {test_aggregate.micro_f1}")
    lines.append(f"- Subset accuracy: {test_aggregate.subset_accuracy:.4f}")
    lines.append(f"- Hamming loss: {test_aggregate.hamming_loss:.4f}")
    lines.append(f"- Mean true label cardinality: {test_aggregate.mean_true_label_cardinality:.4f}")
    lines.append(
        f"- Mean predicted label cardinality: {test_aggregate.mean_predicted_label_cardinality:.4f}"
    )
    lines.append(
        f"- Fraction with no predicted label: {test_aggregate.fraction_no_prediction:.4f}"
    )
    lines.append("")

    lines.append("## Metric interpretation notes")
    lines.append("")
    lines.append("- Subset accuracy is harsh for multilabel problems.")
    lines.append("- Hamming-based accuracy can look strong when labels are sparse.")
    lines.append(
        "- Micro metrics favor frequent labels; macro metrics expose rare-label performance."
    )
    lines.append(
        "- Macro precision/recall/F1 use zero-filled averaging over all labels in metric v2."
    )
    lines.append("- Scores are probability estimates, not calibrated confidence.")
    lines.append("")

    text = "\n".join(lines)
    if not text.endswith("\n"):
        text += "\n"
    return text.encode("utf-8")
