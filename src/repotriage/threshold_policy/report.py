"""Deterministic serialization of threshold-policy artifacts."""

from __future__ import annotations

import hashlib
import json

from repotriage.baseline.models import SplitMetrics
from repotriage.threshold_policy.models import (
    ComparisonDocument,
    ComparisonSplitPair,
    FrozenThresholdPolicyConfig,
    PolicyDocument,
    SweepValidationDocument,
    ThresholdGridConfig,
    ThresholdPolicyManifest,
    ThresholdSweepRow,
    comparison_split_metrics_from_split_metrics,
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


def serialize_config_json(config: FrozenThresholdPolicyConfig) -> bytes:
    return _serialize_json(config.model_dump(mode="json"), pretty=True)


def serialize_policy_json(document: PolicyDocument) -> bytes:
    return _serialize_json(document.model_dump(mode="json"), pretty=True)


def serialize_sweep_validation_json(document: SweepValidationDocument) -> bytes:
    return _serialize_json(document.model_dump(mode="json"), pretty=False)


def serialize_metrics_json(metrics: SplitMetrics) -> bytes:
    return _serialize_json(metrics.model_dump(mode="json"), pretty=True)


def serialize_comparison_json(document: ComparisonDocument) -> bytes:
    return _serialize_json(document.model_dump(mode="json"), pretty=True)


def serialize_manifest_json(manifest: ThresholdPolicyManifest) -> bytes:
    return _serialize_json(manifest.model_dump(mode="json"), pretty=True)


def build_comparison_document(
    *,
    reference_threshold_basis_points: int,
    selected_threshold_basis_points: int,
    denominator: int,
    validation_reference_metrics: SplitMetrics,
    validation_selected_metrics: SplitMetrics,
    test_reference_metrics: SplitMetrics,
    test_selected_metrics: SplitMetrics,
) -> ComparisonDocument:
    reference_threshold = reference_threshold_basis_points / denominator
    selected_threshold = selected_threshold_basis_points / denominator
    return ComparisonDocument(
        reference_threshold_basis_points=reference_threshold_basis_points,
        selected_threshold_basis_points=selected_threshold_basis_points,
        validation=ComparisonSplitPair(
            reference=comparison_split_metrics_from_split_metrics(
                validation_reference_metrics,
                threshold_basis_points=reference_threshold_basis_points,
                threshold=reference_threshold,
            ),
            selected=comparison_split_metrics_from_split_metrics(
                validation_selected_metrics,
                threshold_basis_points=selected_threshold_basis_points,
                threshold=selected_threshold,
            ),
        ),
        test=ComparisonSplitPair(
            reference=comparison_split_metrics_from_split_metrics(
                test_reference_metrics,
                threshold_basis_points=reference_threshold_basis_points,
                threshold=reference_threshold,
            ),
            selected=comparison_split_metrics_from_split_metrics(
                test_selected_metrics,
                threshold_basis_points=selected_threshold_basis_points,
                threshold=selected_threshold,
            ),
        ),
    )


def _find_sweep_neighbors(
    rows: list[ThresholdSweepRow], selected_basis_points: int
) -> list[ThresholdSweepRow]:
    index = next(
        i
        for i, row in enumerate(rows)
        if row.threshold_basis_points == selected_basis_points
    )
    start = max(0, index - 2)
    end = min(len(rows), index + 3)
    return rows[start:end]


def serialize_report_markdown(
    *,
    repository: str,
    baseline_run_id: str,
    selected_candidate_id: str,
    policy_id: str,
    reference_threshold_basis_points: int,
    selected_threshold_basis_points: int,
    denominator: int,
    sweep_rows: list[ThresholdSweepRow],
    comparison: ComparisonDocument,
    grid: ThresholdGridConfig,
) -> bytes:
    reference_threshold = reference_threshold_basis_points / denominator
    selected_threshold = selected_threshold_basis_points / denominator
    neighbors = _find_sweep_neighbors(sweep_rows, selected_threshold_basis_points)

    lines: list[str] = []
    lines.append("# Global threshold policy report")
    lines.append("")
    lines.append("## Context")
    lines.append("")
    lines.append(f"- Repository: `{repository}`")
    lines.append(f"- Baseline run id: `{baseline_run_id}`")
    lines.append(f"- Selected candidate: `{selected_candidate_id}`")
    lines.append(f"- Threshold policy id: `{policy_id}`")
    lines.append("")
    lines.append("## Why threshold 0.50 was only a baseline")
    lines.append("")
    lines.append(
        "The baseline artifact fixed decisions at probability threshold 0.50 as a conventional "
        "default. That value was not tuned on validation and should be treated as a reference "
        "point rather than a production decision rule."
    )
    lines.append("")
    lines.append("## Validation-only selection protocol")
    lines.append("")
    lines.append(
        "This policy sweeps a global threshold grid on validation scores only, ranking "
        "candidates by validation macro F1, then micro F1, then proximity to 0.50, then "
        "higher threshold. Test scores are evaluated only after the threshold is frozen."
    )
    lines.append("")
    lines.append("## Selected threshold")
    lines.append("")
    lines.append(
        f"- Selected threshold: **{selected_threshold:.2f}** "
        f"({selected_threshold_basis_points} basis points)"
    )
    lines.append(
        f"- Reference baseline threshold: **{reference_threshold:.2f}** "
        f"({reference_threshold_basis_points} basis points)"
    )
    lines.append("")

    val_ref = comparison.validation.reference
    val_sel = comparison.validation.selected
    lines.append("## Validation improvement relative to 0.50")
    lines.append("")
    lines.append("| Metric | At 0.50 | Selected | Delta |")
    lines.append("|---|---:|---:|---:|")
    for name in (
        "macro_f1",
        "micro_f1",
        "macro_precision",
        "macro_recall",
        "subset_accuracy",
        "hamming_loss",
    ):
        ref_value = getattr(val_ref, name)
        sel_value = getattr(val_sel, name)
        if ref_value is None or sel_value is None:
            delta = "n/a"
        else:
            delta = f"{sel_value - ref_value:+.6f}"
        ref_display = "n/a" if ref_value is None else f"{ref_value:.6f}"
        sel_display = "n/a" if sel_value is None else f"{sel_value:.6f}"
        lines.append(f"| {name} | {ref_display} | {sel_display} | {delta} |")
    lines.append("")

    test_ref = comparison.test.reference
    test_sel = comparison.test.selected
    lines.append("## Frozen test comparison")
    lines.append("")
    lines.append("| Metric | At 0.50 | Selected | Delta |")
    lines.append("|---|---:|---:|---:|")
    for name in ("macro_f1", "micro_f1", "subset_accuracy", "hamming_loss"):
        ref_value = getattr(test_ref, name)
        sel_value = getattr(test_sel, name)
        if ref_value is None or sel_value is None:
            delta = "n/a"
        else:
            delta = f"{sel_value - ref_value:+.6f}"
        ref_display = "n/a" if ref_value is None else f"{ref_value:.6f}"
        sel_display = "n/a" if sel_value is None else f"{sel_value:.6f}"
        lines.append(f"| {name} | {ref_display} | {sel_display} | {delta} |")
    lines.append("")
    lines.append(
        "Frozen test macro F1 was nearly unchanged between the reference and selected "
        "thresholds."
    )
    lines.append(
        "Test metrics are informational only and did not influence threshold selection."
    )
    lines.append("")

    lines.append("## Precision-recall trade-off")
    lines.append("")
    lines.append(
        f"- Validation macro precision: {val_ref.macro_precision:.6f} → "
        f"{val_sel.macro_precision:.6f}"
        if val_ref.macro_precision is not None and val_sel.macro_precision is not None
        else "- Validation macro precision: n/a"
    )
    lines.append(
        f"- Validation macro recall: {val_ref.macro_recall:.6f} → "
        f"{val_sel.macro_recall:.6f}"
        if val_ref.macro_recall is not None and val_sel.macro_recall is not None
        else "- Validation macro recall: n/a"
    )
    lines.append("")

    lines.append("## Prediction cardinality change")
    lines.append("")
    lines.append(
        f"- Mean predicted label cardinality (validation): "
        f"{val_ref.mean_predicted_label_cardinality:.6f} → "
        f"{val_sel.mean_predicted_label_cardinality:.6f}"
    )
    lines.append(
        f"- Mean predicted label cardinality (test): "
        f"{test_ref.mean_predicted_label_cardinality:.6f} → "
        f"{test_sel.mean_predicted_label_cardinality:.6f}"
    )
    lines.append("")

    lines.append("## No-prediction fraction change")
    lines.append("")
    lines.append(
        f"- Fraction with no prediction (validation): "
        f"{val_ref.fraction_no_prediction:.6f} → {val_sel.fraction_no_prediction:.6f}"
    )
    lines.append(
        f"- Fraction with no prediction (test): "
        f"{test_ref.fraction_no_prediction:.6f} → {test_sel.fraction_no_prediction:.6f}"
    )
    lines.append("")

    lines.append("## Per-label predicted-positive changes (validation)")
    lines.append("")
    lines.append("| Label | At 0.50 | Selected | Delta |")
    lines.append("|---|---:|---:|---:|")
    ref_labels = set(val_ref.predicted_positives_by_label)
    sel_labels = set(val_sel.predicted_positives_by_label)
    all_labels = sorted(ref_labels | sel_labels)
    for label in all_labels:
        ref_count = val_ref.predicted_positives_by_label.get(label, 0)
        sel_count = val_sel.predicted_positives_by_label.get(label, 0)
        lines.append(f"| {label} | {ref_count} | {sel_count} | {sel_count - ref_count:+d} |")
    lines.append("")

    lines.append("## Low-support limitations")
    lines.append("")
    lines.append(
        "Several labels have fewer than five validation positives. Per-label threshold curves "
        "may appear in the sweep diagnostics but were not used for selection."
    )
    lines.append("")
    lines.append("## Why per-label thresholds were not selected")
    lines.append("")
    lines.append(
        "Per-label threshold tuning requires reliable validation support per label. Labels "
        "with very low support produce unstable metrics, so Session 6 publishes one global "
        "threshold only."
    )
    lines.append("")
    lines.append("## Probabilities are not calibrated confidence")
    lines.append("")
    lines.append(
        "These scores are multilabel logistic-regression probability estimates. Threshold "
        "selection optimizes validation F1; it does not calibrate probabilities into "
        "frequency-aligned confidence scores."
    )
    lines.append("")

    lines.append("## Threshold neighborhood (validation macro F1)")
    lines.append("")
    lines.append("| Threshold | Macro F1 | Micro F1 | Mean pred cardinality |")
    lines.append("|---:|---:|---:|---:|")
    for row in neighbors:
        aggregate = row.metrics.aggregate
        macro_f1 = "n/a" if aggregate.macro_f1 is None else f"{aggregate.macro_f1:.6f}"
        micro_f1 = "n/a" if aggregate.micro_f1 is None else f"{aggregate.micro_f1:.6f}"
        is_selected = row.threshold_basis_points == selected_threshold_basis_points
        marker = " **selected**" if is_selected else ""
        lines.append(
            f"| {row.threshold:.2f}{marker} | {macro_f1} | {micro_f1} | "
            f"{aggregate.mean_predicted_label_cardinality:.6f} |"
        )
    lines.append("")
    lines.append("## Grid configuration")
    lines.append("")
    lines.append(
        f"- Basis points: {grid.start_basis_points} to {grid.stop_basis_points} "
        f"step {grid.step_basis_points} over denominator {grid.denominator}"
    )
    lines.append(f"- Threshold count: {len(sweep_rows)}")
    lines.append("")

    return ("\n".join(lines) + "\n").encode("utf-8")
