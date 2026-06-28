"""Deterministic serialization of a label policy into JSON and Markdown, plus hashing.

Determinism rules mirror the audit subsystem:

- ``label_policy.json`` is UTF-8, ``ensure_ascii=False``, ``sort_keys=True``, two-space
  indented, ``\\n`` newlines, with a single trailing newline, and no build timestamp.
- ``label_policy.md`` is UTF-8 with ``\\n`` newlines and a single trailing newline. The
  full 125-label inventory lives only in JSON; Markdown uses concise deterministic
  sections. Fractions are rendered to four decimals.
- SHA-256 hashes are computed over the exact serialized bytes of each file.
"""

from __future__ import annotations

import hashlib
import json

from repotriage.label_policy.models import LabelDecisionRecord, LabelPolicyDocument


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def serialize_policy_json(document: LabelPolicyDocument) -> bytes:
    """Serialize the policy document to deterministic UTF-8 JSON bytes."""
    payload = document.model_dump(mode="json")
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    return (text + "\n").encode("utf-8")


def _fraction(value: float) -> str:
    return f"{value:.4f}"


def _months(record: LabelDecisionRecord) -> str:
    if record.first_month is None or record.last_month is None:
        return "n/a"
    return f"{record.first_month} .. {record.last_month}"


def serialize_policy_markdown(document: LabelPolicyDocument) -> bytes:
    """Serialize the policy document to deterministic UTF-8 Markdown bytes."""
    identity = document.identity
    coverage = document.coverage
    decisions = document.decisions

    # Markdown groups are derived strictly from decision and decision_source, never from
    # explanation strings or roles.
    included = [r for r in decisions if r.decision == "include"]
    deferred = [r for r in decisions if r.decision == "defer"]
    reviewed_exclusions = [
        r for r in decisions if r.decision == "exclude" and r.decision_source == "explicit"
    ]
    default_excluded = [
        r for r in decisions if r.decision == "exclude" and r.decision_source == "default"
    ]

    lines: list[str] = []

    lines.append(f"# Target-label policy {identity.policy_id}")
    lines.append("")

    lines.append("## 1. Dataset and audit identity")
    lines.append("")
    lines.append(f"- Repository: {identity.repository}")
    lines.append(f"- Dataset ID: `{identity.dataset_id}`")
    lines.append(f"- Dataset output SHA-256: `{identity.dataset_output_sha256}`")
    lines.append(f"- Audit ID: `{identity.audit_id}`")
    lines.append(f"- Audit JSON SHA-256: `{identity.audit_json_sha256}`")
    lines.append(f"- Audit version: {identity.audit_version}")
    lines.append(f"- Issue schema version: {identity.issue_schema_version}")
    lines.append(f"- Normalizer version: {identity.normalizer_version}")
    lines.append("")

    lines.append("## 2. Policy identity")
    lines.append("")
    lines.append(f"- Policy ID: `{identity.policy_id}`")
    lines.append(f"- Policy version: {identity.policy_version}")
    lines.append(f"- Document schema version: {document.schema_version}")
    lines.append(f"- Policy-input SHA-256: `{identity.policy_input_sha256}`")
    lines.append(f"- Configuration schema version: {identity.config_schema_version}")
    lines.append(f"- Configuration SHA-256: `{identity.config_sha256}`")
    lines.append("")

    lines.append("## 3. Selection criteria")
    lines.append("")
    criteria = document.selection_criteria
    lines.append(f"- Minimum total support: {criteria.min_total_support}")
    lines.append(f"- Minimum active months: {criteria.min_active_months}")
    lines.append(f"- Minimum recent-window support: {criteria.min_recent_support}")
    lines.append(f"- Recent active-month window (months): {criteria.recent_window_months}")
    lines.append("")

    lines.append("## 4. Coverage summary")
    lines.append("")
    lines.append(f"- Total issues: {coverage.total_issues}")
    lines.append(f"- Total audited labels: {coverage.total_audited_labels}")
    lines.append(f"- Included labels: {coverage.included_label_count}")
    lines.append(f"- Deferred labels: {coverage.deferred_label_count}")
    lines.append(f"- Excluded labels: {coverage.excluded_label_count}")
    lines.append(f"- Explicitly reviewed labels: {coverage.explicit_label_count}")
    lines.append(f"- Default-applied labels: {coverage.default_label_count}")
    lines.append(f"- Dataset active months: {coverage.dataset_active_month_count}")
    lines.append(
        f"- Issues with at least one included target: {coverage.issues_with_included_target}"
    )
    lines.append(
        f"- Issues with no included target: {coverage.issues_without_included_target}"
    )
    lines.append(
        f"- Target coverage fraction: {_fraction(coverage.target_coverage_fraction)}"
    )
    lines.append(f"- Included-target assignments: {coverage.included_target_assignments}")
    lines.append(
        f"- Included-target cardinality: {_fraction(coverage.included_target_cardinality)}"
    )
    lpi = coverage.included_labels_per_issue
    lines.append(
        f"- Included labels per issue: min {lpi.min}, median {lpi.median:.4f}, "
        f"mean {lpi.mean:.4f}, max {lpi.max}"
    )
    lines.append("")

    lines.append("## 5. Included targets")
    lines.append("")
    if included:
        lines.append(
            "| Label | role | support | fraction | active months "
            "| recent-window support | leakage |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for record in included:
            lines.append(
                f"| {record.label} | {record.role} | {record.total_support} "
                f"| {_fraction(record.issue_fraction)} | {record.active_month_count} "
                f"| {record.recent_support} | {record.leakage_risk} |"
            )
    else:
        lines.append("No labels were included as targets.")
    lines.append("")

    lines.append("## 6. Deferred labels")
    lines.append("")
    if deferred:
        lines.append("| Label | role | support | recent-window support | months | reason |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for record in deferred:
            lines.append(
                f"| {record.label} | {record.role} | {record.total_support} "
                f"| {record.recent_support} | {_months(record)} | {record.reason_code} |"
            )
    else:
        lines.append("No labels were deferred.")
    lines.append("")

    lines.append("## 7. Explicit semantic exclusions")
    lines.append("")
    if reviewed_exclusions:
        lines.append("| Label | role | reason | explanation |")
        lines.append("| --- | --- | --- | --- |")
        for record in reviewed_exclusions:
            lines.append(
                f"| {record.label} | {record.role} | {record.reason_code} "
                f"| {record.explanation} |"
            )
    else:
        lines.append("No labels were explicitly excluded for semantic reasons.")
    lines.append("")

    lines.append("## 8. Safe-default exclusions")
    lines.append("")
    lines.append(
        f"- {len(default_excluded)} label(s) were excluded by the safe default "
        "(unreviewed_default) and were not individually reviewed."
    )
    lines.append(
        "- The complete per-label inventory, including every default-excluded label, is "
        "available in `label_policy.json`."
    )
    lines.append("")

    lines.append("## 9. Weak-supervision caveat")
    lines.append("")
    lines.append(
        "- Labels are applied by maintainers and are incomplete; an issue without a given "
        "label is treated as a negative example under a weak-supervision assumption."
    )
    lines.append(
        "- Absent labels therefore do not prove an issue is unrelated to a target; they "
        "only reflect what maintainers chose to apply."
    )
    lines.append(
        "- Coverage and support counts describe applied labels in this dataset, not ground "
        "truth about the issues."
    )
    lines.append("")

    lines.append("## 10. Interpretation caveat")
    lines.append("")
    lines.append(
        "- All conclusions apply only to the explicitly selected dataset and audit "
        "identified above."
    )
    lines.append(
        "- This dataset may be a bounded, recent slice of the repository rather than its "
        "full issue history; support and recent-window support counts describe the snapshot."
    )
    lines.append(
        "- The recent active-month window is the final months of this dataset's distinct "
        "active YYYY-MM keys; it is a policy-selection heuristic, not an ML data split."
    )
    lines.append(
        "- Per-label first/last months and recent-window support refer only to occurrences "
        "inside this dataset."
    )
    lines.append(
        "- Suitability conclusions are bound to this policy id; a different dataset, audit, "
        "or configuration may yield different decisions."
    )
    lines.append("")

    return ("\n".join(lines)).encode("utf-8")
