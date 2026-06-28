"""Deterministic serialization of an audit into JSON and Markdown, plus hashing.

Determinism rules:

- ``audit.json`` is UTF-8, ``ensure_ascii=False``, ``sort_keys=True``, two-space
  indented, ``\\n`` newlines, with a single trailing newline. Numbers are stored at
  full precision. It contains no build timestamp.
- ``audit.md`` is UTF-8 with ``\\n`` newlines and a single trailing newline. Counts are
  integers; fractions/ratios are rendered to four decimals; character-length
  means/percentiles to one decimal; datetimes use the canonical UTC format. Long lists
  are truncated to a deterministic top subset (the full data lives in ``audit.json``).
- SHA-256 hashes are computed over the exact serialized bytes of each file.
"""

from __future__ import annotations

import hashlib
import json

from repotriage.audit.models import AuditDocument, TextFieldStats
from repotriage.dataset.models import format_utc_datetime

TOP_LABELS = 25
TOP_LABEL_PAIRS = 25


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def serialize_audit_json(document: AuditDocument) -> bytes:
    """Serialize the audit document to deterministic UTF-8 JSON bytes."""
    payload = document.model_dump(mode="json")
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    return (text + "\n").encode("utf-8")


def _fraction(value: float) -> str:
    return f"{value:.4f}"


def _stat(value: float | int | None, decimals: int = 1) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{decimals}f}"


def _stats_row(label: str, stats: TextFieldStats) -> str:
    return (
        f"| {label} | {_stat(stats.min)} | {_stat(stats.median)} | {_stat(stats.mean)} "
        f"| {_stat(stats.p90)} | {_stat(stats.p95)} | {_stat(stats.max)} |"
    )


def _datetime(value: object) -> str:
    if value is None:
        return "n/a"
    return format_utc_datetime(value)  # type: ignore[arg-type]


def serialize_audit_markdown(document: AuditDocument) -> bytes:
    """Serialize the audit document to deterministic UTF-8 Markdown bytes."""
    identity = document.identity
    summary = document.repository_summary
    text = document.text_metrics
    labels = document.label_metrics
    temporal = document.temporal_metrics

    lines: list[str] = []

    lines.append(f"# Dataset audit {identity.audit_id}")
    lines.append("")

    lines.append("## 1. Dataset identity")
    lines.append("")
    lines.append(f"- Audit ID: `{identity.audit_id}`")
    lines.append(f"- Audit version: {identity.audit_version}")
    lines.append(f"- Repository: {identity.repository}")
    lines.append(f"- Dataset ID: `{identity.dataset_id}`")
    lines.append(f"- Dataset output SHA-256: `{identity.dataset_output_sha256}`")
    lines.append(f"- Issue schema version: {identity.issue_schema_version}")
    lines.append(f"- Normalizer version: {identity.normalizer_version}")
    lines.append("")

    lines.append("## 2. Repository overview")
    lines.append("")
    lines.append(f"- Total issues: {summary.total_issues}")
    lines.append(
        f"- Labelled issues: {summary.labelled_issues} "
        f"(fraction {_fraction(summary.labelled_fraction)})"
    )
    lines.append(
        f"- Unlabelled issues: {summary.unlabelled_issues} "
        f"(fraction {_fraction(summary.unlabelled_fraction)})"
    )
    lines.append(f"- Open issues: {summary.open_issues}")
    lines.append(f"- Closed issues: {summary.closed_issues}")
    lines.append(f"- Null-author issues: {summary.null_author_issues}")
    lines.append(f"- Earliest created_at: {_datetime(summary.earliest_created_at)}")
    lines.append(f"- Latest created_at: {_datetime(summary.latest_created_at)}")
    lines.append(f"- Temporal span (days): {_stat(summary.temporal_span_days)}")
    lines.append(f"- Active months: {summary.active_month_count}")
    lines.append(f"- Calendar span (months): {summary.calendar_span_months}")
    lines.append("")

    lines.append("## 3. Text quality")
    lines.append("")
    lines.append("| Field | min | median | mean | p90 | p95 | max |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    lines.append(_stats_row("Title characters", text.title_chars))
    lines.append(_stats_row("Body characters", text.body_chars))
    lines.append(_stats_row("Total text characters", text.total_text_chars))
    lines.append("")
    structural = text.structural
    lines.append(
        f"- Empty bodies: {structural.empty_bodies.count} "
        f"(fraction {_fraction(structural.empty_bodies.fraction)})"
    )
    lines.append(
        f"- Bodies shorter than 100 chars: {structural.short_bodies_lt_100.count} "
        f"(fraction {_fraction(structural.short_bodies_lt_100.fraction)})"
    )
    lines.append(
        f"- Bodies longer than 10,000 chars: {structural.long_bodies_gt_10000.count} "
        f"(fraction {_fraction(structural.long_bodies_gt_10000.fraction)})"
    )
    lines.append(
        f"- Bodies with a fenced code block: {structural.with_code_fence.count} "
        f"(fraction {_fraction(structural.with_code_fence.fraction)})"
    )
    lines.append(
        f"- Bodies with a URL: {structural.with_url.count} "
        f"(fraction {_fraction(structural.with_url.fraction)})"
    )
    lines.append(
        f"- Bodies with a Markdown heading: {structural.with_heading.count} "
        f"(fraction {_fraction(structural.with_heading.fraction)})"
    )
    lines.append("")

    lines.append("## 4. Label distribution")
    lines.append("")
    lines.append(f"- Unique labels: {labels.unique_label_count}")
    lines.append(f"- Total label assignments: {labels.total_label_assignments}")
    lines.append(f"- Zero-label issues: {labels.zero_label_issue_count}")
    lines.append(f"- Label cardinality (assignments/issue): {_stat(labels.label_cardinality, 4)}")
    lines.append(f"- Label density (cardinality/unique label): {_stat(labels.label_density, 4)}")
    lpi = labels.labels_per_issue
    lines.append(
        f"- Labels per issue: min {lpi.min}, median {_stat(lpi.median)}, "
        f"mean {_stat(lpi.mean)}, max {lpi.max}"
    )
    lines.append("")

    lines.append("## 5. Rare-label summary")
    lines.append("")
    buckets = labels.rare_label_buckets
    lines.append(f"- Labels with support < 5: {buckets.lt_5}")
    lines.append(f"- Labels with support < 10: {buckets.lt_10}")
    lines.append(f"- Labels with support < 20: {buckets.lt_20}")
    lines.append(f"- Labels with support < 50: {buckets.lt_50}")
    lines.append(f"- Labels with support < 100: {buckets.lt_100}")
    lines.append("")

    lines.append(f"## 6. Top labels (top {TOP_LABELS})")
    lines.append("")
    if labels.labels:
        lines.append("| Label | count | fraction | first seen | last seen |")
        lines.append("| --- | --- | --- | --- | --- |")
        for label in labels.labels[:TOP_LABELS]:
            lines.append(
                f"| {label.name} | {label.count} | {_fraction(label.fraction)} "
                f"| {_datetime(label.first_created_at)} | {_datetime(label.last_created_at)} |"
            )
        remainder = len(labels.labels) - TOP_LABELS
        if remainder > 0:
            lines.append("")
            lines.append(
                f"_{remainder} more label(s) omitted; see `audit.json` for the full list._"
            )
    else:
        lines.append("_No labels present._")
    lines.append("")

    lines.append(f"## 7. Top co-occurring label pairs (top {TOP_LABEL_PAIRS})")
    lines.append("")
    if labels.label_pairs:
        lines.append("| Label A | Label B | co-occurrence |")
        lines.append("| --- | --- | --- |")
        for pair in labels.label_pairs[:TOP_LABEL_PAIRS]:
            lines.append(f"| {pair.label_a} | {pair.label_b} | {pair.count} |")
        remainder = len(labels.label_pairs) - TOP_LABEL_PAIRS
        if remainder > 0:
            lines.append("")
            lines.append(
                f"_{remainder} more pair(s) omitted; see `audit.json` for the full list._"
            )
    else:
        lines.append("_No label pairs with co-occurrence of at least 2._")
    lines.append("")

    lines.append("## 8. Temporal coverage")
    lines.append("")
    lines.append(f"- Earliest created_at: {_datetime(temporal.earliest_created_at)}")
    lines.append(f"- Latest created_at: {_datetime(temporal.latest_created_at)}")
    lines.append(f"- Active months: {temporal.active_month_count}")
    lines.append(f"- Calendar span (months): {temporal.calendar_span_months}")
    lines.append("- Per-month issue counts are available in `audit.json`.")
    lines.append("")

    lines.append("## 9. Suitability warnings")
    lines.append("")
    if document.warnings:
        lines.append("| Code | severity | value | threshold | explanation |")
        lines.append("| --- | --- | --- | --- | --- |")
        for warning in document.warnings:
            lines.append(
                f"| {warning.code} | {warning.severity} | {_stat(warning.value, 4)} "
                f"| {_stat(warning.threshold, 4)} | {warning.message} |"
            )
    else:
        lines.append("No suitability warnings were raised.")
    lines.append("")

    lines.append("## 10. Interpretation caveat")
    lines.append("")
    lines.append(
        "These metrics are objective descriptions of the normalized dataset. Suitability "
        "warnings are versioned heuristics, not universal scientific rules, and there is no "
        "single aggregate quality score. Label-role classification (workflow, type, or "
        "component labels) is still manual and is out of scope for this audit version."
    )
    lines.append("")
    lines.append(
        "- All conclusions apply only to the explicitly selected dataset identified above."
    )
    lines.append(
        "- This dataset may be a bounded, recent slice of the repository rather than its "
        "full issue history; metrics describe the snapshot, not the project's entire past."
    )
    lines.append(
        "- Per-label first/last dates refer only to occurrences inside this dataset, not to "
        "when a label was first or last used in the repository."
    )
    lines.append(
        "- A label or pattern being absent here does not prove it is absent from the "
        "repository's full history."
    )
    lines.append(
        "- Suitability conclusions are bound to this dataset id; a different snapshot may "
        "yield different warnings."
    )
    lines.append("")

    return ("\n".join(lines)).encode("utf-8")
