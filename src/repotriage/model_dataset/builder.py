"""Build (or reuse) an immutable model-ready dataset artifact (md1 contract).

The model-ready dataset binds itself to the normalized dataset bytes, the label-policy
document bytes, the policy id, and the canonical temporal-split configuration via a single
``model_dataset_input_sha256``. It constructs feature text, target vectors, temporal split
assignments, and publishes ``records.jsonl``, ``label_map.json``, ``split_report.json``,
``split_report.md``, and ``manifest.json`` atomically from a hidden staging directory.
Immutable model-dataset ids are never overwritten.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from repotriage.dataset.builder import (
    DEFAULT_PROCESSED_ROOT,
    validate_processed_dataset_integrity,
)
from repotriage.dataset.models import ISSUE_SCHEMA_VERSION, ProcessedManifest
from repotriage.dataset.reader import read_dataset_issues
from repotriage.filesystem import atomic_write_bytes, best_effort_remove_tree
from repotriage.github.models import RepositoryRef
from repotriage.label_policy.builder import (
    DEFAULT_POLICIES_ROOT,
    validate_label_policy_artifact_integrity,
)
from repotriage.label_policy.models import (
    LABEL_POLICY_JSON_FILE,
    LabelPolicyCorruptionError,
    LabelPolicyDocument,
)
from repotriage.model_dataset.config import TemporalSplitConfig, load_split_config
from repotriage.model_dataset.models import (
    LABEL_MAP_JSON_FILE,
    MODEL_DATASET_MANIFEST_SCHEMA_VERSION,
    MODEL_DATASET_VERSION,
    MODEL_READY_RECORD_SCHEMA_VERSION,
    RECORDS_JSONL_FILE,
    SPLIT_REPORT_JSON_FILE,
    SPLIT_REPORT_MARKDOWN_FILE,
    TEMPORAL_SPLITTER_VERSION,
    TEXT_REPRESENTATION_VERSION,
    GlobalTargetStatistics,
    LabelMap,
    ModelDatasetBuildError,
    ModelDatasetConfigError,
    ModelDatasetCorruptionError,
    ModelDatasetInputError,
    ModelDatasetManifest,
    ModelDatasetTransformError,
    ModelReadyRecord,
    SplitReport,
    SplitStatistics,
    SupportValidationSummary,
    _floats_consistent,
    compute_model_dataset_id,
    compute_model_dataset_input_sha256,
)
from repotriage.model_dataset.report import (
    serialize_label_map_json,
    serialize_records_jsonl,
    serialize_split_report_json,
    serialize_split_report_markdown,
    sha256_hex,
)
from repotriage.model_dataset.splitter import (
    assign_split,
    canonicalize_warnings,
    raise_on_hard_support_errors,
    validate_split_support,
)
from repotriage.model_dataset.targets import (
    assert_canonical_order_matches_policy,
    build_target_labels,
)
from repotriage.model_dataset.text import build_feature_text_v1
from repotriage.paths import resolve_within_directory

logger = logging.getLogger(__name__)

DEFAULT_MODEL_READY_ROOT = Path("data/model_ready/github")


@dataclass(frozen=True)
class ModelDatasetBuildResult:
    """Summary of a model-dataset build or model-dataset-cache hit."""

    repository: RepositoryRef
    model_dataset_dir: Path
    manifest: ModelDatasetManifest
    label_map: LabelMap
    split_report: SplitReport
    cache_hit: bool


def _load_and_validate_dataset(
    repository: RepositoryRef, dataset_id: str, processed_root: Path
) -> tuple[Path, ProcessedManifest]:
    dataset_dir = processed_root / repository.slug / dataset_id
    if not dataset_dir.is_dir():
        raise ModelDatasetInputError(
            f"No normalized dataset found for {repository.full_name} with dataset id "
            f"{dataset_id!r} at {dataset_dir}."
        )
    manifest = validate_processed_dataset_integrity(
        dataset_dir,
        expected_repository=repository,
        expected_dataset_id=dataset_id,
        expected_issue_schema_version=ISSUE_SCHEMA_VERSION,
    )
    return dataset_dir, manifest


def _load_and_validate_policy(
    repository: RepositoryRef,
    processed_manifest: ProcessedManifest,
    policy_id: str,
    policies_root: Path,
) -> tuple[Path, LabelPolicyDocument, str]:
    policy_dir = policies_root / repository.slug / policy_id
    if not policy_dir.is_dir():
        raise ModelDatasetInputError(
            f"No label policy found for {repository.full_name} with policy id "
            f"{policy_id!r} at {policy_dir}."
        )

    policy_json_path = policy_dir / LABEL_POLICY_JSON_FILE
    if not policy_json_path.is_file():
        raise ModelDatasetInputError(f"Missing policy document at {policy_json_path}")
    policy_json_bytes = policy_json_path.read_bytes()
    policy_json_sha256 = hashlib.sha256(policy_json_bytes).hexdigest()

    try:
        _, document = validate_label_policy_artifact_integrity(
            policy_dir,
            expected_repository=repository,
            expected_dataset_id=processed_manifest.dataset_id,
            expected_dataset_output_sha256=processed_manifest.output_sha256,
            expected_policy_id=policy_id,
        )
    except LabelPolicyCorruptionError as exc:
        raise ModelDatasetInputError(str(exc)) from exc

    if document.identity.dataset_id != processed_manifest.dataset_id:
        raise ModelDatasetInputError(
            "Policy document dataset_id disagrees with the requested dataset."
        )
    return policy_dir, document, policy_json_sha256


def _verify_output_file(
    model_dataset_dir: Path,
    *,
    relative_path: str,
    expected_sha256: str,
    parse_json: bool = False,
) -> dict | None:
    try:
        resolved = resolve_within_directory(model_dataset_dir, relative_path)
    except ValueError as exc:
        raise ModelDatasetCorruptionError(
            f"Model-dataset manifest references an unsafe path: {relative_path!r}"
        ) from exc
    if not resolved.is_file():
        raise ModelDatasetCorruptionError(f"Missing model-dataset output file: {relative_path}")
    data = resolved.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise ModelDatasetCorruptionError(
            f"Model-dataset output hash mismatch for {relative_path}: "
            f"expected {expected_sha256}, found {actual}."
        )
    if not parse_json:
        return None
    try:
        return json.loads(data.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ModelDatasetCorruptionError(
            f"Invalid JSON in model-dataset file {relative_path}: {exc}"
        ) from exc


def _build_label_map(policy_id: str, canonical_labels: list[str]) -> LabelMap:
    return LabelMap(
        policy_id=policy_id,
        target_count=len(canonical_labels),
        labels=list(canonical_labels),
        label_to_index={label: index for index, label in enumerate(canonical_labels)},
    )


def _transform_issues(
    *,
    dataset_dir: Path,
    processed_manifest: ProcessedManifest,
    canonical_labels: list[str],
    config: TemporalSplitConfig,
) -> list[ModelReadyRecord]:
    records: list[ModelReadyRecord] = []
    for issue in read_dataset_issues(dataset_dir, processed_manifest):
        try:
            selected_labels, target_vector = build_target_labels(issue.labels, canonical_labels)
            if len(target_vector) != len(canonical_labels):
                raise ModelDatasetTransformError("target_vector length mismatch")
            split = assign_split(issue.created_at, config)
            feature_text = build_feature_text_v1(issue.title, issue.body)
            records.append(
                ModelReadyRecord(
                    repository=issue.repository,
                    issue_id=issue.issue_id,
                    issue_number=issue.issue_number,
                    created_at=issue.created_at,
                    title=issue.title,
                    body=issue.body,
                    feature_text=feature_text,
                    selected_labels=selected_labels,
                    target_vector=target_vector,
                    split=split,
                )
            )
        except ModelDatasetTransformError:
            raise
        except Exception as exc:
            raise ModelDatasetTransformError(
                f"Failed to transform issue {issue.issue_id}: {exc}"
            ) from exc
    records.sort(key=lambda record: (record.created_at, record.issue_id))
    return records


def _compute_split_statistics(
    records: list[ModelReadyRecord],
    canonical_labels: list[str],
    config: TemporalSplitConfig,
) -> tuple[SplitReport, dict[str, Counter[str]]]:
    total = len(records)
    positives_per_split: dict[str, Counter[str]] = {
        "train": Counter(),
        "validation": Counter(),
        "test": Counter(),
    }
    split_records: dict[str, list[ModelReadyRecord]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for record in records:
        split_records[record.split].append(record)
        for label, value in zip(canonical_labels, record.target_vector, strict=True):
            if value == 1:
                positives_per_split[record.split][label] += 1

    splits: dict[str, SplitStatistics] = {}
    for split_name in ("train", "validation", "test"):
        split_list = split_records[split_name]
        count = len(split_list)
        all_zero = sum(1 for record in split_list if sum(record.target_vector) == 0)
        cardinality_hist: Counter[int] = Counter()
        for record in split_list:
            cardinality_hist[sum(record.target_vector)] += 1

        earliest = min((r.created_at for r in split_list), default=None)
        latest = max((r.created_at for r in split_list), default=None)
        positives = {
            label: positives_per_split[split_name].get(label, 0) for label in canonical_labels
        }

        splits[split_name] = SplitStatistics(
            issue_count=count,
            fraction=(count / total) if total else 0.0,
            earliest_created_at=earliest,
            latest_created_at=latest,
            all_zero_target_count=all_zero,
            target_cardinality_histogram={
                str(key): value for key, value in sorted(cardinality_hist.items())
            },
            positives_per_label=positives,
        )

    issues_with_target = sum(1 for record in records if sum(record.target_vector) > 0)
    issues_without = total - issues_with_target
    positive_assignments = sum(sum(record.target_vector) for record in records)
    all_zero_count = sum(1 for record in records if sum(record.target_vector) == 0)

    hard_errors, warnings = validate_split_support(
        canonical_labels=canonical_labels,
        positives_per_split=positives_per_split,
        config=config,
    )
    warnings = canonicalize_warnings(warnings, canonical_labels=canonical_labels)

    global_stats = GlobalTargetStatistics(
        total_records=total,
        target_count=len(canonical_labels),
        issues_with_included_target=issues_with_target,
        issues_without_included_target=issues_without,
        target_coverage_fraction=(issues_with_target / total) if total else 0.0,
        positive_assignments=positive_assignments,
        all_zero_target_count=all_zero_count,
    )

    report = SplitReport(
        split_strategy=config.split_strategy,
        validation_start=config.validation_start,
        test_start=config.test_start,
        boundary_semantics=config.boundary_semantics.model_dump(),
        total_records=total,
        global_target_statistics=global_stats,
        splits=splits,
        warnings=warnings,
        support_validation=SupportValidationSummary(hard_errors=hard_errors, warnings=warnings),
    )
    return report, positives_per_split


def _iter_model_ready_records(records_path: Path):
    """Yield validated model-ready records from a JSONL file."""
    try:
        handle = records_path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ModelDatasetCorruptionError(
            f"Unable to read records file {records_path}: {exc}"
        ) from exc

    with handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            if line.strip() == "":
                raise ModelDatasetCorruptionError(
                    f"Blank line in records at line {line_number}"
                )
            try:
                payload = json.loads(line)
                yield ModelReadyRecord.model_validate(payload)
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ModelDatasetCorruptionError(
                    f"Invalid model-ready record at line {line_number}: {exc}"
                ) from exc


def _build_feature_text(text_representation_version: str, title: str, body: str) -> str:
    """Build feature text for the supported text-representation version."""
    if text_representation_version == TEXT_REPRESENTATION_VERSION:
        return build_feature_text_v1(title, body)
    raise ModelDatasetCorruptionError(
        f"Unsupported text_representation_version {text_representation_version!r}."
    )


def _expected_record_from_issue(
    issue,
    *,
    canonical_labels: list[str],
    config: TemporalSplitConfig,
    text_representation_version: str,
) -> ModelReadyRecord:
    """Transform one normalized issue into the expected model-ready record."""
    selected_labels, target_vector = build_target_labels(issue.labels, canonical_labels)
    return ModelReadyRecord(
        repository=issue.repository,
        issue_id=issue.issue_id,
        issue_number=issue.issue_number,
        created_at=issue.created_at,
        title=issue.title,
        body=issue.body,
        feature_text=_build_feature_text(text_representation_version, issue.title, issue.body),
        selected_labels=selected_labels,
        target_vector=target_vector,
        split=assign_split(issue.created_at, config),
    )


def _split_config_from_manifest(
    manifest: ModelDatasetManifest, split_report: SplitReport
) -> TemporalSplitConfig:
    """Build a minimal split config for boundary checks during validation."""
    from repotriage.model_dataset.config import BoundarySemantics, MinimumPositiveSupport

    threshold = (
        split_report.warnings[0].threshold if split_report.warnings else 5
    )
    return TemporalSplitConfig(
        repository=manifest.repository,
        validation_start=manifest.validation_start,
        test_start=manifest.test_start,
        boundary_semantics=BoundarySemantics(**split_report.boundary_semantics),
        minimum_positive_support=MinimumPositiveSupport(train=1, validation=1, test=1),
        low_support_warning_threshold=threshold,
    )


@dataclass
class _RecordScanStats:
    total: int
    issues_with_target: int
    positive_assignments: int
    all_zero_count: int
    split_counts: Counter[str]
    split_all_zero: dict[str, int]
    split_positives: dict[str, Counter[str]]
    split_cardinality: dict[str, Counter[int]]
    split_earliest: dict[str, datetime | None]
    split_latest: dict[str, datetime | None]


def _init_record_scan_stats() -> _RecordScanStats:
    return _RecordScanStats(
        total=0,
        issues_with_target=0,
        positive_assignments=0,
        all_zero_count=0,
        split_counts=Counter(),
        split_all_zero={"train": 0, "validation": 0, "test": 0},
        split_positives={
            "train": Counter(),
            "validation": Counter(),
            "test": Counter(),
        },
        split_cardinality={
            "train": Counter(),
            "validation": Counter(),
            "test": Counter(),
        },
        split_earliest={"train": None, "validation": None, "test": None},
        split_latest={"train": None, "validation": None, "test": None},
    )


def _update_record_scan_stats(
    stats: _RecordScanStats,
    record: ModelReadyRecord,
    label_map: LabelMap,
) -> None:
    stats.total += 1
    cardinality = sum(record.target_vector)
    if cardinality > 0:
        stats.issues_with_target += 1
    else:
        stats.all_zero_count += 1
    stats.positive_assignments += cardinality
    split_name = record.split
    stats.split_counts[split_name] += 1
    if cardinality == 0:
        stats.split_all_zero[split_name] += 1
    stats.split_cardinality[split_name][cardinality] += 1
    for label, value in zip(label_map.labels, record.target_vector, strict=True):
        if value == 1:
            stats.split_positives[split_name][label] += 1
    earliest = stats.split_earliest[split_name]
    latest = stats.split_latest[split_name]
    if earliest is None or record.created_at < earliest:
        stats.split_earliest[split_name] = record.created_at
    if latest is None or record.created_at > latest:
        stats.split_latest[split_name] = record.created_at


def _validate_scanned_records_against_report(
    stats: _RecordScanStats,
    *,
    manifest: ModelDatasetManifest,
    label_map: LabelMap,
    split_report: SplitReport,
    split_config: TemporalSplitConfig,
) -> None:
    """Compare record-derived statistics against split_report.json."""
    gts = split_report.global_target_statistics
    if stats.total != split_report.total_records:
        raise ModelDatasetCorruptionError(
            "split_report total_records disagrees with scanned records"
        )
    if stats.total != manifest.records_written:
        raise ModelDatasetCorruptionError(
            "manifest records_written disagrees with scanned records"
        )
    if gts.total_records != stats.total:
        raise ModelDatasetCorruptionError(
            "split_report global total_records disagrees with scanned records"
        )
    if gts.target_count != manifest.target_count:
        raise ModelDatasetCorruptionError(
            "split_report target_count disagrees with manifest"
        )
    if gts.target_count != len(label_map.labels):
        raise ModelDatasetCorruptionError("label_map length disagrees with target_count")
    if gts.issues_with_included_target != stats.issues_with_target:
        raise ModelDatasetCorruptionError(
            "split_report issues_with_included_target disagrees with records"
        )
    if gts.issues_without_included_target != stats.total - stats.issues_with_target:
        raise ModelDatasetCorruptionError(
            "split_report issues_without_included_target disagrees with records"
        )
    if gts.positive_assignments != stats.positive_assignments:
        raise ModelDatasetCorruptionError(
            "split_report positive_assignments disagrees with records"
        )
    if gts.all_zero_target_count != stats.all_zero_count:
        raise ModelDatasetCorruptionError(
            "split_report all_zero_target_count disagrees with records"
        )
    if stats.total > 0:
        expected_coverage = stats.issues_with_target / stats.total
        if not _floats_consistent(gts.target_coverage_fraction, expected_coverage):
            raise ModelDatasetCorruptionError(
                "split_report target_coverage_fraction disagrees with records"
            )

    for split_name in ("train", "validation", "test"):
        reported = split_report.splits[split_name]
        count = stats.split_counts[split_name]
        if reported.issue_count != count:
            raise ModelDatasetCorruptionError(
                f"split_report {split_name} issue_count disagrees with records"
            )
        if stats.total > 0:
            expected_fraction = count / stats.total
            if not _floats_consistent(reported.fraction, expected_fraction):
                raise ModelDatasetCorruptionError(
                    f"split_report {split_name} fraction disagrees with records"
                )
        if reported.all_zero_target_count != stats.split_all_zero[split_name]:
            raise ModelDatasetCorruptionError(
                f"split_report {split_name} all_zero_target_count disagrees with records"
            )
        if reported.earliest_created_at != stats.split_earliest[split_name]:
            raise ModelDatasetCorruptionError(
                f"split_report {split_name} earliest_created_at disagrees with records"
            )
        if reported.latest_created_at != stats.split_latest[split_name]:
            raise ModelDatasetCorruptionError(
                f"split_report {split_name} latest_created_at disagrees with records"
            )
        expected_histogram = {
            str(key): value
            for key, value in sorted(stats.split_cardinality[split_name].items())
        }
        if reported.target_cardinality_histogram != expected_histogram:
            raise ModelDatasetCorruptionError(
                f"split_report {split_name} target_cardinality_histogram disagrees "
                "with records"
            )
        for label in label_map.labels:
            if reported.positives_per_label.get(label, 0) != stats.split_positives[
                split_name
            ].get(label, 0):
                raise ModelDatasetCorruptionError(
                    f"split_report {split_name} positives for {label!r} disagree "
                    "with records"
                )

    _, expected_warnings = validate_split_support(
        canonical_labels=label_map.labels,
        positives_per_split=stats.split_positives,
        config=split_config,
    )
    expected_warnings = canonicalize_warnings(
        expected_warnings, canonical_labels=label_map.labels
    )
    if len(split_report.warnings) != len(
        {(w.split, w.label, w.code) for w in split_report.warnings}
    ):
        raise ModelDatasetCorruptionError("split_report warnings contain duplicates")
    if split_report.warnings != expected_warnings:
        raise ModelDatasetCorruptionError(
            "split_report warnings are not in canonical order or disagree with records"
        )


def _scan_and_validate_records(
    records_path: Path,
    *,
    manifest: ModelDatasetManifest,
    label_map: LabelMap,
    split_config: TemporalSplitConfig,
) -> _RecordScanStats:
    """Stream records, validating artifact invariants and accumulating statistics."""
    stats = _init_record_scan_stats()
    seen_issue_ids: set[int] = set()
    seen_issue_numbers: set[int] = set()
    prev_key: tuple[datetime, int] | None = None

    for record in _iter_model_ready_records(records_path):
        if record.schema_version != MODEL_READY_RECORD_SCHEMA_VERSION:
            raise ModelDatasetCorruptionError(
                f"Unsupported record schema_version {record.schema_version!r}"
            )
        if record.repository != manifest.repository:
            raise ModelDatasetCorruptionError(
                f"Record repository {record.repository!r} disagrees with manifest"
            )
        if record.issue_id in seen_issue_ids:
            raise ModelDatasetCorruptionError(f"Duplicate issue_id {record.issue_id}")
        if record.issue_number in seen_issue_numbers:
            raise ModelDatasetCorruptionError(
                f"Duplicate issue_number {record.issue_number}"
            )
        seen_issue_ids.add(record.issue_id)
        seen_issue_numbers.add(record.issue_number)

        order_key = (record.created_at, record.issue_id)
        if prev_key is not None and order_key < prev_key:
            raise ModelDatasetCorruptionError(
                "records.jsonl is not sorted by (created_at, issue_id)"
            )
        prev_key = order_key

        if len(record.target_vector) != manifest.target_count:
            raise ModelDatasetCorruptionError(
                f"target_vector length {len(record.target_vector)} != "
                f"target_count {manifest.target_count}"
            )
        if any(type(value) is not int or value not in (0, 1) for value in record.target_vector):
            raise ModelDatasetCorruptionError(
                "target_vector contains non-strict-binary JSON integer values"
            )

        expected_selected = [
            label for label, value in zip(label_map.labels, record.target_vector, strict=True)
            if value == 1
        ]
        if record.selected_labels != expected_selected:
            raise ModelDatasetCorruptionError("selected_labels disagrees with target_vector")

        expected_feature_text = _build_feature_text(
            manifest.text_representation_version, record.title, record.body
        )
        if record.feature_text != expected_feature_text:
            raise ModelDatasetCorruptionError(
                f"feature_text for issue_id {record.issue_id} disagrees with "
                "title/body and text-representation version"
            )

        expected_split = assign_split(record.created_at, split_config)
        if record.split != expected_split:
            raise ModelDatasetCorruptionError(
                f"Record split {record.split!r} disagrees with created_at boundary "
                f"(expected {expected_split!r})"
            )

        _update_record_scan_stats(stats, record, label_map)

    if stats.total != manifest.records_written:
        raise ModelDatasetCorruptionError(
            f"records.jsonl contains {stats.total} records but manifest declares "
            f"{manifest.records_written}"
        )
    return stats


def _load_model_dataset_manifest(model_dataset_dir: Path) -> ModelDatasetManifest:
    manifest_path = model_dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ModelDatasetCorruptionError(f"Missing model-dataset manifest at {manifest_path}")
    try:
        return ModelDatasetManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (ValidationError, json.JSONDecodeError) as exc:
        raise ModelDatasetCorruptionError(
            f"Invalid model-dataset manifest at {manifest_path}: {exc}"
        ) from exc
    except OSError as exc:
        raise ModelDatasetCorruptionError(
            f"Unable to read model-dataset manifest at {manifest_path}: {exc}"
        ) from exc


def validate_model_dataset_artifact_integrity(
    model_dataset_dir: Path,
    *,
    expected_repository: RepositoryRef,
    expected_model_dataset_id: str,
    check_dir_name: bool = True,
) -> tuple[ModelDatasetManifest, LabelMap, SplitReport]:
    """Validate a model-ready artifact using only on-disk bytes and minimal identity.

    Downstream consumers need only the artifact directory, repository, and model-dataset id.
    This does not require the normalized dataset, policy artifact, or split-config file.
    """
    if not model_dataset_dir.is_dir():
        raise ModelDatasetCorruptionError(
            f"Model-dataset directory does not exist: {model_dataset_dir}"
        )

    manifest = _load_model_dataset_manifest(model_dataset_dir)

    if check_dir_name and model_dataset_dir.name != manifest.model_dataset_id:
        raise ModelDatasetCorruptionError(
            f"Model-dataset directory {model_dataset_dir.name!r} does not match "
            f"manifest model_dataset_id {manifest.model_dataset_id!r}."
        )
    if manifest.model_dataset_id != expected_model_dataset_id:
        raise ModelDatasetCorruptionError(
            f"Model-dataset manifest model_dataset_id {manifest.model_dataset_id!r} "
            f"does not match expected {expected_model_dataset_id!r}."
        )
    if manifest.repository != expected_repository.full_name:
        raise ModelDatasetCorruptionError(
            f"Model-dataset manifest repository {manifest.repository!r} does not match "
            f"expected {expected_repository.full_name!r}."
        )
    if manifest.model_dataset_version != MODEL_DATASET_VERSION:
        raise ModelDatasetCorruptionError(
            f"Unsupported model_dataset_version {manifest.model_dataset_version!r}."
        )
    if manifest.schema_version != MODEL_DATASET_MANIFEST_SCHEMA_VERSION:
        raise ModelDatasetCorruptionError(
            f"Unsupported manifest schema_version {manifest.schema_version!r}."
        )
    if manifest.text_representation_version != TEXT_REPRESENTATION_VERSION:
        raise ModelDatasetCorruptionError(
            f"Unsupported text_representation_version "
            f"{manifest.text_representation_version!r}."
        )
    if manifest.temporal_splitter_version != TEMPORAL_SPLITTER_VERSION:
        raise ModelDatasetCorruptionError(
            f"Unsupported temporal_splitter_version "
            f"{manifest.temporal_splitter_version!r}."
        )

    label_map_payload = _verify_output_file(
        model_dataset_dir,
        relative_path=manifest.label_map_file,
        expected_sha256=manifest.label_map_sha256,
        parse_json=True,
    )
    split_report_payload = _verify_output_file(
        model_dataset_dir,
        relative_path=manifest.split_report_json_file,
        expected_sha256=manifest.split_report_json_sha256,
        parse_json=True,
    )
    _verify_output_file(
        model_dataset_dir,
        relative_path=manifest.split_report_markdown_file,
        expected_sha256=manifest.split_report_markdown_sha256,
    )
    _verify_output_file(
        model_dataset_dir,
        relative_path=manifest.records_file,
        expected_sha256=manifest.records_sha256,
    )

    assert label_map_payload is not None
    assert split_report_payload is not None
    try:
        label_map = LabelMap.model_validate(label_map_payload)
    except ValidationError as exc:
        raise ModelDatasetCorruptionError(f"Invalid label_map.json: {exc}") from exc
    try:
        split_report = SplitReport.model_validate(split_report_payload)
    except ValidationError as exc:
        raise ModelDatasetCorruptionError(f"Invalid split_report.json: {exc}") from exc

    if label_map.policy_id != manifest.policy_id:
        raise ModelDatasetCorruptionError("label_map policy_id disagrees with manifest")
    if label_map.target_count != manifest.target_count:
        raise ModelDatasetCorruptionError("label_map target_count disagrees with manifest")
    if split_report.validation_start != manifest.validation_start:
        raise ModelDatasetCorruptionError(
            "split_report validation_start disagrees with manifest"
        )
    if split_report.test_start != manifest.test_start:
        raise ModelDatasetCorruptionError("split_report test_start disagrees with manifest")

    records_path = resolve_within_directory(model_dataset_dir, manifest.records_file)
    split_config = _split_config_from_manifest(manifest, split_report)
    stats = _scan_and_validate_records(
        records_path,
        manifest=manifest,
        label_map=label_map,
        split_config=split_config,
    )
    _validate_scanned_records_against_report(
        stats,
        manifest=manifest,
        label_map=label_map,
        split_report=split_report,
        split_config=split_config,
    )
    return manifest, label_map, split_report


def _validate_policy_label_order(
    label_map: LabelMap, policy_document: LabelPolicyDocument
) -> None:
    """Require label_map.labels to match policy included-label order exactly."""
    policy_labels = list(policy_document.coverage.included_labels)
    include_ordered = [
        record.label for record in policy_document.decisions if record.decision == "include"
    ]
    try:
        assert_canonical_order_matches_policy(policy_labels, include_ordered)
    except ModelDatasetTransformError as exc:
        raise ModelDatasetCorruptionError(str(exc)) from exc
    if label_map.labels != policy_labels:
        raise ModelDatasetCorruptionError(
            "label_map.labels does not match policy coverage.included_labels order"
        )


def _validate_records_derive_from_source(
    *,
    dataset_dir: Path,
    processed_manifest: ProcessedManifest,
    records_path: Path,
    canonical_labels: list[str],
    config: TemporalSplitConfig,
    text_representation_version: str,
) -> None:
    """Verify every model-ready record derives exactly from the normalized source dataset."""
    source_by_id: dict[int, object] = {}
    for issue in read_dataset_issues(dataset_dir, processed_manifest):
        source_by_id[issue.issue_id] = issue

    model_records = list(_iter_model_ready_records(records_path))
    if len(model_records) != len(source_by_id):
        raise ModelDatasetCorruptionError(
            f"Model-ready record count {len(model_records)} disagrees with source "
            f"issue count {len(source_by_id)}"
        )

    for record in model_records:
        issue = source_by_id.get(record.issue_id)
        if issue is None:
            raise ModelDatasetCorruptionError(
                f"Model-ready issue_id {record.issue_id} has no source normalized issue"
            )
        expected = _expected_record_from_issue(
            issue,
            canonical_labels=canonical_labels,
            config=config,
            text_representation_version=text_representation_version,
        )
        if record != expected:
            raise ModelDatasetCorruptionError(
                f"Model-ready record for issue_id {record.issue_id} does not derive "
                "from the normalized source dataset"
            )


def validate_model_dataset_against_inputs(
    model_dataset_dir: Path,
    *,
    expected_repository: RepositoryRef,
    dataset_dir: Path,
    processed_manifest: ProcessedManifest,
    policy_document: LabelPolicyDocument,
    policy_id: str,
    policy_json_sha256: str,
    config: TemporalSplitConfig,
    config_hash: str,
    expected_model_dataset_id: str,
    expected_model_dataset_input_sha256: str,
    check_dir_name: bool = True,
) -> tuple[ModelDatasetManifest, LabelMap, SplitReport]:
    """Validate artifact integrity and prove it derives from the exact current inputs."""
    manifest, label_map, split_report = validate_model_dataset_artifact_integrity(
        model_dataset_dir,
        expected_repository=expected_repository,
        expected_model_dataset_id=expected_model_dataset_id,
        check_dir_name=check_dir_name,
    )

    lineage_expectations: list[tuple[str, object, object]] = [
        (
            "model_dataset_input_sha256",
            manifest.model_dataset_input_sha256,
            expected_model_dataset_input_sha256,
        ),
        ("dataset_id", manifest.dataset_id, processed_manifest.dataset_id),
        (
            "dataset_output_sha256",
            manifest.dataset_output_sha256,
            processed_manifest.output_sha256,
        ),
        ("policy_id", manifest.policy_id, policy_id),
        ("policy_json_sha256", manifest.policy_json_sha256, policy_json_sha256),
        ("split_config_sha256", manifest.split_config_sha256, config_hash),
        (
            "split_config_schema_version",
            manifest.split_config_schema_version,
            config.config_schema_version,
        ),
        ("validation_start", manifest.validation_start, config.validation_start),
        ("test_start", manifest.test_start, config.test_start),
    ]
    for field_name, actual, expected in lineage_expectations:
        if actual != expected:
            raise ModelDatasetCorruptionError(
                f"Model-dataset manifest {field_name} {actual!r} does not match expected "
                f"{expected!r}."
            )

    _validate_policy_label_order(label_map, policy_document)
    records_path = resolve_within_directory(model_dataset_dir, manifest.records_file)
    canonical_labels = list(policy_document.coverage.included_labels)
    _validate_records_derive_from_source(
        dataset_dir=dataset_dir,
        processed_manifest=processed_manifest,
        records_path=records_path,
        canonical_labels=canonical_labels,
        config=config,
        text_representation_version=manifest.text_representation_version,
    )
    return manifest, label_map, split_report


def validate_model_dataset_integrity(
    model_dataset_dir: Path,
    *,
    expected_repository: RepositoryRef,
    expected_dataset_id: str,
    expected_dataset_output_sha256: str,
    expected_policy_id: str,
    expected_policy_json_sha256: str,
    expected_split_config_sha256: str,
    expected_validation_start: datetime,
    expected_test_start: datetime,
    expected_model_dataset_id: str,
    expected_model_dataset_input_sha256: str,
    check_dir_name: bool = True,
    dataset_dir: Path | None = None,
    processed_manifest: ProcessedManifest | None = None,
    policy_document: LabelPolicyDocument | None = None,
    config: TemporalSplitConfig | None = None,
    config_hash: str | None = None,
) -> tuple[ModelDatasetManifest, LabelMap, SplitReport]:
    """Validate a model-ready artifact against explicit input lineage (builder cache path).

    When ``dataset_dir``, ``processed_manifest``, ``policy_document``, and ``config`` are
    supplied, performs full source-derivation validation. Otherwise validates manifest
    lineage fields only (legacy call sites).
    """
    if (
        dataset_dir is not None
        and processed_manifest is not None
        and policy_document is not None
        and config is not None
        and config_hash is not None
    ):
        return validate_model_dataset_against_inputs(
            model_dataset_dir,
            expected_repository=expected_repository,
            dataset_dir=dataset_dir,
            processed_manifest=processed_manifest,
            policy_document=policy_document,
            policy_id=expected_policy_id,
            policy_json_sha256=expected_policy_json_sha256,
            config=config,
            config_hash=config_hash,
            expected_model_dataset_id=expected_model_dataset_id,
            expected_model_dataset_input_sha256=expected_model_dataset_input_sha256,
            check_dir_name=check_dir_name,
        )

    manifest, label_map, split_report = validate_model_dataset_artifact_integrity(
        model_dataset_dir,
        expected_repository=expected_repository,
        expected_model_dataset_id=expected_model_dataset_id,
        check_dir_name=check_dir_name,
    )
    expectations: list[tuple[str, object, object]] = [
        (
            "model_dataset_input_sha256",
            manifest.model_dataset_input_sha256,
            expected_model_dataset_input_sha256,
        ),
        ("dataset_id", manifest.dataset_id, expected_dataset_id),
        ("dataset_output_sha256", manifest.dataset_output_sha256, expected_dataset_output_sha256),
        ("policy_id", manifest.policy_id, expected_policy_id),
        ("policy_json_sha256", manifest.policy_json_sha256, expected_policy_json_sha256),
        ("split_config_sha256", manifest.split_config_sha256, expected_split_config_sha256),
        ("validation_start", manifest.validation_start, expected_validation_start),
        ("test_start", manifest.test_start, expected_test_start),
    ]
    for field_name, actual, expected in expectations:
        if actual != expected:
            raise ModelDatasetCorruptionError(
                f"Model-dataset manifest {field_name} {actual!r} does not match expected "
                f"{expected!r}."
            )
    return manifest, label_map, split_report


def build_model_dataset(
    repository: RepositoryRef,
    dataset_id: str,
    policy_id: str,
    config_path: Path,
    *,
    processed_root: Path = DEFAULT_PROCESSED_ROOT,
    policies_root: Path = DEFAULT_POLICIES_ROOT,
    model_ready_root: Path = DEFAULT_MODEL_READY_ROOT,
) -> ModelDatasetBuildResult:
    """Build one model-ready dataset, publishing or reusing an immutable artifact."""
    dataset_dir, processed_manifest = _load_and_validate_dataset(
        repository, dataset_id, processed_root
    )
    _, policy_document, policy_json_sha256 = _load_and_validate_policy(
        repository, processed_manifest, policy_id, policies_root
    )

    config, config_hash = load_split_config(config_path)
    if config.repository != repository.full_name:
        raise ModelDatasetConfigError(
            f"Configuration repository {config.repository!r} does not match requested "
            f"repository {repository.full_name!r}."
        )

    canonical_labels = list(policy_document.coverage.included_labels)
    include_ordered = [
        record.label for record in policy_document.decisions if record.decision == "include"
    ]
    assert_canonical_order_matches_policy(canonical_labels, include_ordered)

    model_dataset_input_sha256 = compute_model_dataset_input_sha256(
        model_dataset_version=MODEL_DATASET_VERSION,
        dataset_id=processed_manifest.dataset_id,
        dataset_output_sha256=processed_manifest.output_sha256,
        policy_id=policy_id,
        policy_json_sha256=policy_json_sha256,
        text_representation_version=TEXT_REPRESENTATION_VERSION,
        temporal_splitter_version=TEMPORAL_SPLITTER_VERSION,
        split_config_schema_version=config.config_schema_version,
        split_config_sha256=config_hash,
    )
    model_dataset_id = compute_model_dataset_id(
        processed_manifest.dataset_id, model_dataset_input_sha256
    )
    final_dir = model_ready_root / repository.slug / model_dataset_id

    if final_dir.exists():
        manifest, label_map, split_report = validate_model_dataset_against_inputs(
            final_dir,
            expected_repository=repository,
            dataset_dir=dataset_dir,
            processed_manifest=processed_manifest,
            policy_document=policy_document,
            policy_id=policy_id,
            policy_json_sha256=policy_json_sha256,
            config=config,
            config_hash=config_hash,
            expected_model_dataset_id=model_dataset_id,
            expected_model_dataset_input_sha256=model_dataset_input_sha256,
        )
        logger.info("Model-dataset-cache hit for %s at %s", repository.full_name, final_dir)
        return ModelDatasetBuildResult(
            repository=repository,
            model_dataset_dir=final_dir,
            manifest=manifest,
            label_map=label_map,
            split_report=split_report,
            cache_hit=True,
        )

    records = _transform_issues(
        dataset_dir=dataset_dir,
        processed_manifest=processed_manifest,
        canonical_labels=canonical_labels,
        config=config,
    )
    split_report, _ = _compute_split_statistics(records, canonical_labels, config)
    raise_on_hard_support_errors(split_report.support_validation.hard_errors)

    label_map = _build_label_map(policy_id, canonical_labels)
    records_bytes = serialize_records_jsonl(records)
    label_map_bytes = serialize_label_map_json(label_map)
    split_report_json_bytes = serialize_split_report_json(split_report)
    split_report_md_bytes = serialize_split_report_markdown(split_report)

    records_sha256 = sha256_hex(records_bytes)
    label_map_sha256 = sha256_hex(label_map_bytes)
    split_report_json_sha256 = sha256_hex(split_report_json_bytes)
    split_report_md_sha256 = sha256_hex(split_report_md_bytes)

    slug_dir = model_ready_root / repository.slug
    slug_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{repository.slug}.{model_dataset_id}.staging-", dir=slug_dir
        )
    )

    logger.info(
        "Building model dataset %s for %s (%d records, %d targets)",
        model_dataset_id,
        repository.full_name,
        len(records),
        len(canonical_labels),
    )

    try:
        atomic_write_bytes(staging_dir / RECORDS_JSONL_FILE, records_bytes)
        atomic_write_bytes(staging_dir / LABEL_MAP_JSON_FILE, label_map_bytes)
        atomic_write_bytes(staging_dir / SPLIT_REPORT_JSON_FILE, split_report_json_bytes)
        atomic_write_bytes(staging_dir / SPLIT_REPORT_MARKDOWN_FILE, split_report_md_bytes)

        manifest = ModelDatasetManifest(
            model_dataset_version=MODEL_DATASET_VERSION,
            model_dataset_id=model_dataset_id,
            model_dataset_input_sha256=model_dataset_input_sha256,
            repository=processed_manifest.repository,
            dataset_id=processed_manifest.dataset_id,
            dataset_output_sha256=processed_manifest.output_sha256,
            policy_id=policy_id,
            policy_json_sha256=policy_json_sha256,
            text_representation_version=TEXT_REPRESENTATION_VERSION,
            temporal_splitter_version=TEMPORAL_SPLITTER_VERSION,
            split_config_schema_version=config.config_schema_version,
            split_config_sha256=config_hash,
            validation_start=config.validation_start,
            test_start=config.test_start,
            built_at=datetime.now(UTC),
            records_written=len(records),
            target_count=len(canonical_labels),
            records_sha256=records_sha256,
            label_map_sha256=label_map_sha256,
            split_report_json_sha256=split_report_json_sha256,
            split_report_markdown_sha256=split_report_md_sha256,
        )
        atomic_write_bytes(
            staging_dir / "manifest.json",
            (manifest.model_dump_json() + "\n").encode("utf-8"),
        )

        validate_model_dataset_against_inputs(
            staging_dir,
            expected_repository=repository,
            dataset_dir=dataset_dir,
            processed_manifest=processed_manifest,
            policy_document=policy_document,
            policy_id=policy_id,
            policy_json_sha256=policy_json_sha256,
            config=config,
            config_hash=config_hash,
            expected_model_dataset_id=model_dataset_id,
            expected_model_dataset_input_sha256=model_dataset_input_sha256,
            check_dir_name=False,
        )
        publish_model_dataset(staging_dir, final_dir)
    except BaseException:
        best_effort_remove_tree(staging_dir)
        raise

    logger.info(
        "Published model dataset %s for %s at %s",
        model_dataset_id,
        repository.full_name,
        final_dir,
    )
    return ModelDatasetBuildResult(
        repository=repository,
        model_dataset_dir=final_dir,
        manifest=manifest,
        label_map=label_map,
        split_report=split_report,
        cache_hit=False,
    )


def format_model_dataset_summary(result: ModelDatasetBuildResult) -> str:
    """Build the user-facing model-dataset summary."""
    manifest = result.manifest
    report = result.split_report
    gts = report.global_target_statistics
    lines = [
        f"Repository: {manifest.repository}",
        f"Dataset ID: {manifest.dataset_id}",
        f"Policy ID: {manifest.policy_id}",
        f"Model-dataset ID: {manifest.model_dataset_id}",
        f"Model-dataset-input SHA-256: {manifest.model_dataset_input_sha256}",
        f"Records written: {manifest.records_written}",
        f"Target count: {manifest.target_count}",
        f"Target coverage: {gts.target_coverage_fraction:.4f}",
        f"All-zero target records: {gts.all_zero_target_count}",
        f"Train records: {report.splits['train'].issue_count}",
        f"Validation records: {report.splits['validation'].issue_count}",
        f"Test records: {report.splits['test'].issue_count}",
        f"Split warnings: {len(report.warnings)}",
        f"Output directory: {result.model_dataset_dir}",
        f"Model-dataset-cache hit: {'yes' if result.cache_hit else 'no'}",
    ]
    return "\n".join(lines)


def publish_model_dataset(staging_dir: Path, final_dir: Path) -> None:
    """Atomically publish a completed staging directory to its immutable final path."""
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    if final_dir.exists():
        raise ModelDatasetBuildError(
            f"Refusing to overwrite existing model-dataset directory {final_dir}."
        )
    os.rename(staging_dir, final_dir)
