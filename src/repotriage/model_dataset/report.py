"""Deterministic serialization of model-ready artifacts into JSON, JSONL, and Markdown."""

from __future__ import annotations

import hashlib
import json

from repotriage.model_dataset.models import LabelMap, ModelReadyRecord, SplitReport


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def serialize_records_jsonl(records: list[ModelReadyRecord]) -> bytes:
    """Serialize model-ready records to deterministic UTF-8 JSON Lines bytes."""
    lines: list[str] = []
    for record in records:
        payload = record.model_dump(mode="json")
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        lines.append(line + "\n")
    return "".join(lines).encode("utf-8")


def serialize_label_map_json(label_map: LabelMap) -> bytes:
    payload = label_map.model_dump(mode="json")
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    return (text + "\n").encode("utf-8")


def serialize_split_report_json(report: SplitReport) -> bytes:
    payload = report.model_dump(mode="json")
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    return (text + "\n").encode("utf-8")


def _fraction(value: float) -> str:
    return f"{value:.4f}"


def serialize_split_report_markdown(report: SplitReport) -> bytes:
    """Serialize the split report to deterministic UTF-8 Markdown bytes."""
    lines: list[str] = []
    lines.append("# Model-ready dataset split report")
    lines.append("")
    lines.append("## Split configuration")
    lines.append("")
    lines.append(f"- Split strategy: {report.split_strategy}")
    lines.append(f"- Validation start: {report.model_dump(mode='json')['validation_start']}")
    lines.append(f"- Test start: {report.model_dump(mode='json')['test_start']}")
    lines.append("")
    lines.append("### Boundary semantics")
    lines.append("")
    for split_name in ("train", "validation", "test"):
        lines.append(f"- {split_name}: {report.boundary_semantics[split_name]}")
    lines.append("")

    gts = report.global_target_statistics
    lines.append("## Global target statistics")
    lines.append("")
    lines.append(f"- Total records: {gts.total_records}")
    lines.append(f"- Target count: {gts.target_count}")
    lines.append(f"- Issues with included target: {gts.issues_with_included_target}")
    lines.append(f"- Issues without included target: {gts.issues_without_included_target}")
    lines.append(f"- Target coverage fraction: {_fraction(gts.target_coverage_fraction)}")
    lines.append(f"- Positive assignments: {gts.positive_assignments}")
    lines.append(f"- All-zero target count: {gts.all_zero_target_count}")
    lines.append("")

    for split_name in ("train", "validation", "test"):
        stats = report.splits[split_name]
        lines.append(f"## Split: {split_name}")
        lines.append("")
        lines.append(f"- Issue count: {stats.issue_count}")
        lines.append(f"- Fraction: {_fraction(stats.fraction)}")
        if stats.earliest_created_at is not None and stats.latest_created_at is not None:
            earliest = stats.model_dump(mode="json")["earliest_created_at"]
            latest = stats.model_dump(mode="json")["latest_created_at"]
            lines.append(f"- Date range: {earliest} .. {latest}")
        lines.append(f"- All-zero target count: {stats.all_zero_target_count}")
        lines.append("")
        lines.append("### Positives per label")
        lines.append("")
        for label, count in sorted(stats.positives_per_label.items()):
            lines.append(f"- {label}: {count}")
        lines.append("")

    if report.warnings:
        lines.append("## Warnings")
        lines.append("")
        for warning in report.warnings:
            lines.append(
                f"- {warning.code}: {warning.label} in {warning.split} "
                f"has {warning.count} positive(s) (threshold {warning.threshold})"
            )
        lines.append("")

    text = "\n".join(lines)
    if not text.endswith("\n"):
        text += "\n"
    return text.encode("utf-8")
