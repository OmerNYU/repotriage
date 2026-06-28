"""Tests for deterministic audit JSON/Markdown serialization and hashing."""

from __future__ import annotations

import hashlib

from repotriage.audit.analyzer import Analyzer
from repotriage.audit.models import AuditDocument, DatasetIdentity
from repotriage.audit.policy import build_warnings
from repotriage.audit.report import (
    TOP_LABELS,
    serialize_audit_json,
    serialize_audit_markdown,
    sha256_hex,
)
from tests.helpers import make_normalized_issue

_DATASET_ID = "20260101T000000000000Z-n1-aaaaaaaaaaaa"


def _document(issues: list) -> AuditDocument:
    analyzer = Analyzer()
    for issue in issues:
        analyzer.add(issue)
    analysis = analyzer.finalize()
    identity = DatasetIdentity(
        audit_version="2",
        audit_id=f"{_DATASET_ID}-a2",
        repository="pandas-dev/pandas",
        dataset_id=_DATASET_ID,
        dataset_output_sha256="a" * 64,
        issue_schema_version="1",
        normalizer_version="1",
    )
    return AuditDocument(
        identity=identity,
        repository_summary=analysis.repository_summary,
        text_metrics=analysis.text_metrics,
        label_metrics=analysis.label_metrics,
        temporal_metrics=analysis.temporal_metrics,
        warnings=build_warnings(analysis),
    )


def test_audit_json_is_deterministic_and_round_trips() -> None:
    document = _document([make_normalized_issue(1, labels=["Bug"]), make_normalized_issue(2)])
    first = serialize_audit_json(document)
    second = serialize_audit_json(document)
    assert first == second
    assert first.endswith(b"\n")
    assert b'"schema_version": "2"' in first
    reparsed = AuditDocument.model_validate_json(first.decode("utf-8"))
    assert serialize_audit_json(reparsed) == first


def test_audit_markdown_is_deterministic() -> None:
    document = _document([make_normalized_issue(1, labels=["Bug"]), make_normalized_issue(2)])
    first = serialize_audit_markdown(document)
    second = serialize_audit_markdown(document)
    assert first == second
    assert first.endswith(b"\n")
    assert b"# Dataset audit 20260101T000000000000Z-n1-aaaaaaaaaaaa-a2" in first
    text = first.decode("utf-8")
    assert "Total text characters" in text
    assert "Active months" in text
    assert "Calendar span (months)" in text
    assert "All conclusions apply only to the explicitly selected dataset" in text


def test_audit_hashes_match_recomputed() -> None:
    document = _document([make_normalized_issue(1, labels=["Bug"])])
    json_bytes = serialize_audit_json(document)
    md_bytes = serialize_audit_markdown(document)
    assert sha256_hex(json_bytes) == hashlib.sha256(json_bytes).hexdigest()
    assert sha256_hex(md_bytes) == hashlib.sha256(md_bytes).hexdigest()


def test_markdown_truncates_labels_to_top_subset() -> None:
    issues = [make_normalized_issue(i, labels=[f"label-{i:03d}"]) for i in range(1, 31)]
    document = _document(issues)
    markdown = serialize_audit_markdown(document).decode("utf-8")

    assert document.label_metrics.unique_label_count == 30
    label_rows = [line for line in markdown.splitlines() if line.startswith("| label-")]
    assert len(label_rows) == TOP_LABELS
    omitted = 30 - TOP_LABELS
    assert f"_{omitted} more label(s) omitted" in markdown
