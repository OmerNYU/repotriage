"""Load model-ready artifacts into train/validation/test splits for baseline training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from repotriage.baseline.models import BaselineInputError
from repotriage.github.models import RepositoryRef
from repotriage.model_dataset.builder import validate_model_dataset_artifact_integrity
from repotriage.model_dataset.models import (
    LabelMap,
    ModelDatasetManifest,
    ModelReadyRecord,
    SplitName,
)
from repotriage.model_dataset.reader import read_model_ready_records


@dataclass(frozen=True)
class SplitBundle:
    """Texts, target matrix, and metadata for one split."""

    records: list[ModelReadyRecord]
    texts: list[str]
    targets: np.ndarray


@dataclass(frozen=True)
class TrainingSplits:
    """Train and validation bundles from one model-ready artifact."""

    manifest: ModelDatasetManifest
    label_map: LabelMap
    train: SplitBundle
    validation: SplitBundle


@dataclass(frozen=True)
class ModelReadySplits:
    """Train, validation, and test bundles from one model-ready artifact."""

    manifest: ModelDatasetManifest
    label_map: LabelMap
    train: SplitBundle
    validation: SplitBundle
    test: SplitBundle


def _record_sort_key(record: ModelReadyRecord) -> tuple[str, int]:
    return (record.model_dump(mode="json")["created_at"], record.issue_id)


def _load_split(
    model_dataset_dir: Path,
    manifest: ModelDatasetManifest,
    label_map: LabelMap,
    split: SplitName,
) -> SplitBundle:
    records = sorted(
        read_model_ready_records(model_dataset_dir, manifest, split=split),
        key=_record_sort_key,
    )
    texts = [record.feature_text for record in records]
    target_count = label_map.target_count
    targets = np.zeros((len(records), target_count), dtype=np.int8)
    for row_index, record in enumerate(records):
        if len(record.target_vector) != target_count:
            raise BaselineInputError(
                f"Record issue_id {record.issue_id} has target_vector length "
                f"{len(record.target_vector)}; expected {target_count}."
            )
        targets[row_index, :] = record.target_vector
    return SplitBundle(records=records, texts=texts, targets=targets)


def load_training_splits(
    model_dataset_dir: Path,
    *,
    expected_repository: RepositoryRef,
    expected_model_dataset_id: str,
) -> TrainingSplits:
    """Validate and load train+validation splits without touching test."""
    manifest, label_map, _split_report = validate_model_dataset_artifact_integrity(
        model_dataset_dir,
        expected_repository=expected_repository,
        expected_model_dataset_id=expected_model_dataset_id,
    )
    if manifest.repository != expected_repository.full_name:
        raise BaselineInputError(
            f"Model-ready repository {manifest.repository!r} does not match "
            f"expected {expected_repository.full_name!r}."
        )

    train = _load_split(model_dataset_dir, manifest, label_map, "train")
    validation = _load_split(model_dataset_dir, manifest, label_map, "validation")

    train_ids = {record.issue_id for record in train.records}
    val_ids = {record.issue_id for record in validation.records}
    if train_ids & val_ids:
        raise BaselineInputError("Train and validation issue_id sets must be disjoint")

    return TrainingSplits(
        manifest=manifest,
        label_map=label_map,
        train=train,
        validation=validation,
    )


def load_test_split(
    model_dataset_dir: Path,
    *,
    manifest: ModelDatasetManifest,
    label_map: LabelMap,
    training_splits: TrainingSplits,
) -> SplitBundle:
    """Load the held-out test split after candidate selection is frozen."""
    test = _load_split(model_dataset_dir, manifest, label_map, "test")
    train_ids = {record.issue_id for record in training_splits.train.records}
    val_ids = {record.issue_id for record in training_splits.validation.records}
    test_ids = {record.issue_id for record in test.records}
    if train_ids & test_ids or val_ids & test_ids:
        raise BaselineInputError("Test issue_id set must be disjoint from train and validation")
    return test


def load_model_ready_splits(
    model_dataset_dir: Path,
    *,
    expected_repository: RepositoryRef,
    expected_model_dataset_id: str,
) -> ModelReadySplits:
    """Validate and load all splits (legacy convenience wrapper)."""
    training = load_training_splits(
        model_dataset_dir,
        expected_repository=expected_repository,
        expected_model_dataset_id=expected_model_dataset_id,
    )
    test = load_test_split(
        model_dataset_dir,
        manifest=training.manifest,
        label_map=training.label_map,
        training_splits=training,
    )
    return ModelReadySplits(
        manifest=training.manifest,
        label_map=training.label_map,
        train=training.train,
        validation=training.validation,
        test=test,
    )
