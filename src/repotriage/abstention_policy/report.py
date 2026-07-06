"""Deterministic serialization of abstention-policy artifacts."""

from __future__ import annotations

import hashlib
import json

from repotriage.abstention_policy.models import (
    AbstentionPolicyManifest,
    AbstentionSplitMetrics,
    AbstentionSweepRow,
    ComparisonDocument,
    ComparisonSplitPair,
    ConfidenceBinsDocument,
    FrozenAbstentionPolicyConfig,
    HandledMetrics,
    PolicyDocument,
    SweepValidationDocument,
    comparison_split_metrics_from_handled,
)
from repotriage.baseline.models import SplitMetrics
from repotriage.threshold_policy.models import ThresholdGridConfig


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


def serialize_config_json(config: FrozenAbstentionPolicyConfig) -> bytes:
    return _serialize_json(config.model_dump(mode="json"), pretty=True)


def serialize_policy_json(document: PolicyDocument) -> bytes:
    return _serialize_json(document.model_dump(mode="json"), pretty=True)


def serialize_sweep_validation_json(document: SweepValidationDocument) -> bytes:
    return _serialize_json(document.model_dump(mode="json"), pretty=False)


def serialize_split_metrics_json(metrics: AbstentionSplitMetrics) -> bytes:
    return _serialize_json(metrics.model_dump(mode="json"), pretty=True)


def serialize_confidence_bins_json(document: ConfidenceBinsDocument) -> bytes:
    return _serialize_json(document.model_dump(mode="json"), pretty=True)


def serialize_comparison_json(document: ComparisonDocument) -> bytes:
    return _serialize_json(document.model_dump(mode="json"), pretty=True)


def serialize_manifest_json(manifest: AbstentionPolicyManifest) -> bytes:
    return _serialize_json(manifest.model_dump(mode="json"), pretty=True)


def full_set_reference_metrics(split_metrics: SplitMetrics) -> HandledMetrics:
    from repotriage.abstention_policy.models import handled_metrics_from_split_metrics

    return handled_metrics_from_split_metrics(split_metrics)


def build_comparison_document(
    *,
    classification_threshold_basis_points: int,
    selected_abstention_basis_points: int,
    denominator: int,
    validation_full_set_metrics: HandledMetrics,
    validation_selected_metrics: HandledMetrics,
    validation_full_set_coverage: float,
    validation_full_set_handled_count: int,
    validation_selected_coverage: float,
    validation_selected_handled_count: int,
    test_full_set_metrics: HandledMetrics,
    test_selected_metrics: HandledMetrics,
    test_full_set_coverage: float,
    test_full_set_handled_count: int,
    test_selected_coverage: float,
    test_selected_handled_count: int,
) -> ComparisonDocument:
    classification_threshold = classification_threshold_basis_points / denominator
    selected_abstention_threshold = selected_abstention_basis_points / denominator
    return ComparisonDocument(
        classification_threshold_basis_points=classification_threshold_basis_points,
        selected_abstention_basis_points=selected_abstention_basis_points,
        validation=ComparisonSplitPair(
            classification_threshold_full_set=comparison_split_metrics_from_handled(
                handled_metrics=validation_full_set_metrics,
                coverage=validation_full_set_coverage,
                handled_count=validation_full_set_handled_count,
                abstention_threshold=classification_threshold,
            ),
            selected_abstention_handled=comparison_split_metrics_from_handled(
                handled_metrics=validation_selected_metrics,
                coverage=validation_selected_coverage,
                handled_count=validation_selected_handled_count,
                abstention_basis_points=selected_abstention_basis_points,
                abstention_threshold=selected_abstention_threshold,
            ),
        ),
        test=ComparisonSplitPair(
            classification_threshold_full_set=comparison_split_metrics_from_handled(
                handled_metrics=test_full_set_metrics,
                coverage=test_full_set_coverage,
                handled_count=test_full_set_handled_count,
                abstention_threshold=classification_threshold,
            ),
            selected_abstention_handled=comparison_split_metrics_from_handled(
                handled_metrics=test_selected_metrics,
                coverage=test_selected_coverage,
                handled_count=test_selected_handled_count,
                abstention_basis_points=selected_abstention_basis_points,
                abstention_threshold=selected_abstention_threshold,
            ),
        ),
    )


def _find_sweep_neighbors(
    rows: list[AbstentionSweepRow], selected_basis_points: int
) -> list[AbstentionSweepRow]:
    index = next(
        i for i, row in enumerate(rows) if row.abstention_basis_points == selected_basis_points
    )
    start = max(0, index - 2)
    end = min(len(rows), index + 3)
    return rows[start:end]


def serialize_report_markdown(
    *,
    repository: str,
    baseline_run_id: str,
    threshold_policy_id: str,
    policy_id: str,
    classification_threshold_basis_points: int,
    selected_abstention_basis_points: int,
    minimum_coverage: float,
    denominator: int,
    confidence_definition: str,
    sweep_rows: list[AbstentionSweepRow],
    comparison: ComparisonDocument,
    grid: ThresholdGridConfig,
    validation_bins: ConfidenceBinsDocument,
    test_bins: ConfidenceBinsDocument,
) -> bytes:
    classification_threshold = classification_threshold_basis_points / denominator
    selected_abstention_threshold = selected_abstention_basis_points / denominator
    neighbors = _find_sweep_neighbors(sweep_rows, selected_abstention_basis_points)
    val_ref = comparison.validation.classification_threshold_full_set
    val_sel = comparison.validation.selected_abstention_handled
    test_sel = comparison.test.selected_abstention_handled

    lines: list[str] = []
    lines.append("# Abstention policy report")
    lines.append("")
    lines.append("## What abstention means")
    lines.append("")
    lines.append(
        "Abstention decides which issues receive automated label suggestions and which are "
        "routed to maintainer review. The policy unit is the issue, not the individual label."
    )
    lines.append("")
    lines.append("## Forced abstention when no labels are predicted")
    lines.append("")
    lines.append(
        "If the Session 6 classification threshold produces zero predicted labels for an "
        "issue, that issue must abstain regardless of the abstention threshold."
    )
    lines.append("")
    lines.append("## Confidence definition")
    lines.append("")
    lines.append(f"- Confidence definition: `{confidence_definition}`")
    lines.append(
        "- Issue confidence is the maximum score among predicted labels at the classification "
        "threshold."
    )
    lines.append("")
    lines.append("## Classification threshold source")
    lines.append("")
    lines.append(f"- Threshold policy id: `{threshold_policy_id}`")
    lines.append(
        f"- Classification threshold: **{classification_threshold:.2f}** "
        f"({classification_threshold_basis_points} basis points)"
    )
    lines.append("")
    lines.append("## Validation-only abstention selection")
    lines.append("")
    lines.append(
        "Abstention threshold selection uses validation issue confidences only. Eligible rows "
        f"must have validation coverage >= {minimum_coverage:.2f}. Among eligible rows the "
        "selector ranks by handled subset accuracy, then handled samples F1, then coverage, "
        "then lower abstention threshold."
    )
    lines.append("")
    lines.append("## Selected abstention threshold")
    lines.append("")
    lines.append(
        f"- Selected abstention threshold: **{selected_abstention_threshold:.2f}** "
        f"({selected_abstention_basis_points} basis points)"
    )
    lines.append(
        f"- Validation coverage: {val_sel.coverage:.6f} ({val_sel.handled_count} handled issues)"
    )
    lines.append(
        f"- Validation handled subset accuracy: {val_sel.subset_accuracy:.6f}"
        if val_sel.subset_accuracy is not None
        else "- Validation handled subset accuracy: n/a"
    )
    lines.append("")
    lines.append("## Frozen test coverage and handled quality")
    lines.append("")
    lines.append(
        f"- Test coverage: {test_sel.coverage:.6f} ({test_sel.handled_count} handled issues)"
    )
    lines.append(
        f"- Test handled subset accuracy: {test_sel.subset_accuracy:.6f}"
        if test_sel.subset_accuracy is not None
        else "- Test handled subset accuracy: n/a"
    )
    lines.append("")
    lines.append(
        "Test metrics are informational only and did not influence abstention threshold selection."
    )
    lines.append("")
    lines.append("## Confidence-bin diagnostics")
    lines.append("")
    lines.append("These bins are diagnostics only; they are not calibration evidence.")
    lines.append("")
    lines.append("### Validation bins")
    lines.append("")
    lines.append("| Bin | Issues | Fraction | Subset accuracy | Samples F1 |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in validation_bins.bins:
        subset = "n/a" if row.subset_accuracy is None else f"{row.subset_accuracy:.6f}"
        samples = "n/a" if row.samples_f1 is None else f"{row.samples_f1:.6f}"
        lines.append(
            f"| {row.bin_label} | {row.issue_count} | {row.fraction_of_all_issues:.6f} | "
            f"{subset} | {samples} |"
        )
    lines.append(
        f"| no_prediction | {validation_bins.no_prediction_bucket.issue_count} | "
        f"{validation_bins.no_prediction_bucket.fraction_of_all_issues:.6f} | n/a | n/a |"
    )
    lines.append("")
    lines.append("### Test bins")
    lines.append("")
    lines.append("| Bin | Issues | Fraction | Subset accuracy | Samples F1 |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in test_bins.bins:
        subset = "n/a" if row.subset_accuracy is None else f"{row.subset_accuracy:.6f}"
        samples = "n/a" if row.samples_f1 is None else f"{row.samples_f1:.6f}"
        lines.append(
            f"| {row.bin_label} | {row.issue_count} | {row.fraction_of_all_issues:.6f} | "
            f"{subset} | {samples} |"
        )
    lines.append(
        f"| no_prediction | {test_bins.no_prediction_bucket.issue_count} | "
        f"{test_bins.no_prediction_bucket.fraction_of_all_issues:.6f} | n/a | n/a |"
    )
    lines.append("")
    lines.append("## Precision-recall-coverage trade-off")
    lines.append("")
    lines.append(
        f"- Validation full-set handled subset accuracy at classification threshold: "
        f"{val_ref.subset_accuracy:.6f}"
        if val_ref.subset_accuracy is not None
        else "- Validation full-set handled subset accuracy at classification threshold: n/a"
    )
    lines.append(
        f"- Validation selected handled subset accuracy: {val_sel.subset_accuracy:.6f}"
        if val_sel.subset_accuracy is not None
        else "- Validation selected handled subset accuracy: n/a"
    )
    lines.append(f"- Validation coverage change: {val_ref.coverage:.6f} -> {val_sel.coverage:.6f}")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- Probabilities are not calibrated confidence scores.")
    lines.append("- This policy does not use per-label abstention thresholds.")
    lines.append("- Test data did not influence abstention threshold selection.")
    lines.append("- Abstention is not claimed production-safe from test evidence alone.")
    lines.append("")
    lines.append("## Context")
    lines.append("")
    lines.append(f"- Repository: `{repository}`")
    lines.append(f"- Baseline run id: `{baseline_run_id}`")
    lines.append(f"- Abstention policy id: `{policy_id}`")
    lines.append("")
    lines.append("## Abstention threshold neighborhood (validation handled subset accuracy)")
    lines.append("")
    lines.append("| Threshold | Coverage | Handled subset accuracy | Handled samples F1 |")
    lines.append("|---:|---:|---:|---:|")
    for row in neighbors:
        subset = (
            "n/a"
            if row.handled_metrics.subset_accuracy is None
            else f"{row.handled_metrics.subset_accuracy:.6f}"
        )
        samples = (
            "n/a"
            if row.handled_metrics.samples_f1 is None
            else f"{row.handled_metrics.samples_f1:.6f}"
        )
        marker = (
            " **selected**"
            if row.abstention_basis_points == selected_abstention_basis_points
            else ""
        )
        lines.append(
            f"| {row.abstention_threshold:.2f}{marker} | {row.coverage:.6f} | "
            f"{subset} | {samples} |"
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
