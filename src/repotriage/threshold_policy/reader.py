"""Load frozen validation and test score bundles from baseline prediction artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from repotriage.baseline.models import BaselineManifest, PredictionRecord
from repotriage.baseline.scores import validate_score_matrix
from repotriage.github.models import RepositoryRef
from repotriage.paths import resolve_within_directory
from repotriage.threshold_policy.models import (
    ThresholdPolicyCorruptionError,
    ThresholdPolicyInputError,
)


@dataclass(frozen=True)
class ValidationScoreBundle:
    labels: list[str]
    repository: str
    model_dataset_id: str
    baseline_run_id: str
    candidate_id: str
    issue_ids: list[int]
    y_true: np.ndarray
    y_score: np.ndarray
    record_count: int
    target_count: int


@dataclass(frozen=True)
class TestScoreBundle:
    labels: list[str]
    repository: str
    model_dataset_id: str
    baseline_run_id: str
    candidate_id: str
    issue_ids: list[int]
    y_true: np.ndarray
    y_score: np.ndarray
    record_count: int
    target_count: int


def _prediction_sort_key(record: PredictionRecord) -> tuple[str, int]:
    return (record.split, record.issue_id)


def _read_predictions_jsonl(path: Path) -> list[PredictionRecord]:
    records: list[PredictionRecord] = []
    for line_number, line in enumerate(path.open("r", encoding="utf-8"), start=1):
        stripped = line.strip()
        if not stripped:
            raise ThresholdPolicyCorruptionError(
                f"Blank line in prediction file {path} at line {line_number}"
            )
        try:
            record = PredictionRecord.model_validate_json(stripped)
        except Exception as exc:
            raise ThresholdPolicyCorruptionError(
                f"Invalid prediction record in {path} at line {line_number}: {exc}"
            ) from exc
        records.append(record)
    return records


def _label_names_for_record(record: PredictionRecord) -> dict[int, str]:
    positive_labels = iter(record.true_labels)
    mapping: dict[int, str] = {}
    for index, value in enumerate(record.true_vector):
        if value == 1:
            mapping[index] = next(positive_labels)
    return mapping


def _extract_canonical_labels(records: list[PredictionRecord], *, target_count: int) -> list[str]:
    labels: list[str | None] = [None] * target_count
    for record in records:
        mapping = _label_names_for_record(record)
        for index, label in mapping.items():
            if labels[index] is None:
                labels[index] = label
            elif labels[index] != label:
                raise ThresholdPolicyCorruptionError(
                    f"Label name mismatch at index {index}: {labels[index]!r} vs {label!r}"
                )
    unresolved = [index for index, label in enumerate(labels) if label is None]
    if unresolved:
        raise ThresholdPolicyCorruptionError(
            f"Could not resolve label names for indices {unresolved}"
        )
    return [label for label in labels if label is not None]


def _records_to_arrays(
    records: list[PredictionRecord],
    *,
    labels: list[str],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    target_count = len(labels)
    y_true = np.array([record.true_vector for record in records], dtype=np.int8)
    score_rows: list[list[float]] = []
    for record in records:
        if record.score_vector is None:
            raise ThresholdPolicyCorruptionError("prediction record missing score_vector")
        if len(record.score_vector) != target_count:
            raise ThresholdPolicyCorruptionError("score_vector length mismatch")
        score_rows.append([float(value) for value in record.score_vector])
    y_score = np.array(score_rows, dtype=np.float64)
    validate_score_matrix(y_score, target_count=target_count)
    issue_ids = [record.issue_id for record in records]
    return y_true, y_score, issue_ids


def load_validation_scores(
    baseline_dir: Path,
    *,
    baseline_manifest: BaselineManifest,
    candidate_id: str,
    expected_repository: RepositoryRef | None = None,
) -> ValidationScoreBundle:
    """Load validation probability scores for one candidate from a baseline artifact."""
    if expected_repository is not None:
        if baseline_manifest.repository != expected_repository.full_name:
            raise ThresholdPolicyInputError(
                f"Baseline repository {baseline_manifest.repository!r} does not match "
                f"expected {expected_repository.full_name!r}."
            )

    val_path = resolve_within_directory(
        baseline_dir, baseline_manifest.predictions_validation_file
    )
    all_records = _read_predictions_jsonl(val_path)
    filtered = [record for record in all_records if record.candidate_id == candidate_id]
    if len(filtered) != baseline_manifest.validation_record_count:
        raise ThresholdPolicyInputError(
            f"Expected {baseline_manifest.validation_record_count} validation records for "
            f"candidate {candidate_id!r}; found {len(filtered)}."
        )

    filtered.sort(key=_prediction_sort_key)
    issue_ids = [record.issue_id for record in filtered]
    if len(set(issue_ids)) != len(issue_ids):
        raise ThresholdPolicyInputError(
            f"Duplicate validation issue_ids for candidate {candidate_id!r}"
        )
    labels = _extract_canonical_labels(all_records, target_count=baseline_manifest.target_count)
    y_true, y_score, issue_ids = _records_to_arrays(filtered, labels=labels)
    first = filtered[0]
    return ValidationScoreBundle(
        labels=labels,
        repository=first.repository,
        model_dataset_id=first.model_dataset_id,
        baseline_run_id=first.baseline_run_id,
        candidate_id=candidate_id,
        issue_ids=issue_ids,
        y_true=y_true,
        y_score=y_score,
        record_count=len(filtered),
        target_count=len(labels),
    )


def load_test_scores(
    baseline_dir: Path,
    *,
    baseline_manifest: BaselineManifest,
    candidate_id: str,
    expected_repository: RepositoryRef | None = None,
) -> TestScoreBundle:
    """Load test probability scores for the frozen candidate from a baseline artifact."""
    if expected_repository is not None:
        if baseline_manifest.repository != expected_repository.full_name:
            raise ThresholdPolicyInputError(
                f"Baseline repository {baseline_manifest.repository!r} does not match "
                f"expected {expected_repository.full_name!r}."
            )

    test_path = resolve_within_directory(
        baseline_dir, baseline_manifest.predictions_test_file
    )
    records = _read_predictions_jsonl(test_path)
    if len(records) != baseline_manifest.test_record_count:
        raise ThresholdPolicyInputError(
            f"Expected {baseline_manifest.test_record_count} test records; found {len(records)}."
        )

    for record in records:
        if record.candidate_id != candidate_id:
            raise ThresholdPolicyInputError(
                f"Test prediction candidate_id {record.candidate_id!r} does not match "
                f"expected {candidate_id!r}."
            )

    records.sort(key=_prediction_sort_key)
    issue_ids = [record.issue_id for record in records]
    if len(set(issue_ids)) != len(issue_ids):
        raise ThresholdPolicyInputError("Duplicate test issue_ids")
    labels = _extract_canonical_labels(records, target_count=baseline_manifest.target_count)
    y_true, y_score, issue_ids = _records_to_arrays(records, labels=labels)
    first = records[0]
    return TestScoreBundle(
        labels=labels,
        repository=first.repository,
        model_dataset_id=first.model_dataset_id,
        baseline_run_id=first.baseline_run_id,
        candidate_id=candidate_id,
        issue_ids=issue_ids,
        y_true=y_true,
        y_score=y_score,
        record_count=len(records),
        target_count=len(labels),
    )
