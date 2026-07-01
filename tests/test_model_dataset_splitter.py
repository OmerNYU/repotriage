"""Tests for temporal split assignment and support validation."""

from collections import Counter
from datetime import datetime

import pytest

from repotriage.model_dataset.config import TemporalSplitConfig
from repotriage.model_dataset.models import ModelDatasetSplitSupportError, SplitWarning
from repotriage.model_dataset.splitter import (
    assign_split,
    canonicalize_warnings,
    raise_on_hard_support_errors,
    validate_split_support,
)
from tests.helpers import write_temporal_split_config


def _config(
  validation_start: str = "2026-02-01T00:00:00Z",
  test_start: str = "2026-04-01T00:00:00Z",
) -> TemporalSplitConfig:
    import tempfile
    from pathlib import Path

    path = Path(tempfile.mkdtemp()) / "split.json"
    write_temporal_split_config(path, validation_start=validation_start, test_start=test_start)
    from repotriage.model_dataset.config import load_split_config

    config, _ = load_split_config(path)
    return config


def _at(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def test_exact_cutoff_boundary() -> None:
    config = _config()
    assert assign_split(_at("2026-01-31T23:59:59Z"), config) == "train"
    assert assign_split(_at("2026-02-01T00:00:00Z"), config) == "validation"
    assert assign_split(_at("2026-04-01T00:00:00Z"), config) == "test"


def test_timestamp_before_at_after_cutoff() -> None:
    config = _config()
    assert assign_split(_at("2026-03-31T23:59:59Z"), config) == "validation"
    assert assign_split(_at("2026-04-01T00:00:00Z"), config) == "test"


def test_identical_timestamps_same_split() -> None:
    config = _config()
    ts = _at("2026-02-15T10:00:00Z")
    assert assign_split(ts, config) == assign_split(ts, config) == "validation"


def test_zero_train_positives_rejected() -> None:
    from collections import Counter

    config = _config()
    positives = {
        "train": Counter({"Bug": 0}),
        "validation": Counter({"Bug": 2}),
        "test": Counter({"Bug": 2}),
    }
    hard_errors, _ = validate_split_support(
        canonical_labels=["Bug"], positives_per_split=positives, config=config
    )
    with pytest.raises(ModelDatasetSplitSupportError):
        raise_on_hard_support_errors(hard_errors)


def test_low_support_creates_warnings() -> None:
    config = _config()
    positives = {
        "train": Counter({"Bug": 10}),
        "validation": Counter({"Bug": 3}),
        "test": Counter({"Bug": 4}),
    }
    hard_errors, warnings = validate_split_support(
        canonical_labels=["Bug"], positives_per_split=positives, config=config
    )
    assert not hard_errors
    assert len(warnings) == 2
    assert warnings[0].code == "low_positive_support"


def test_warnings_follow_canonical_split_and_label_order() -> None:
    config = _config()
    canonical_labels = ["Zebra", "Alpha", "Mango"]
    positives = {
        "train": Counter(),
        "validation": Counter({"Zebra": 2, "Alpha": 3, "Mango": 1}),
        "test": Counter({"Alpha": 2, "Mango": 4}),
    }
    _, warnings = validate_split_support(
        canonical_labels=canonical_labels,
        positives_per_split=positives,
        config=config,
    )
    canonical = canonicalize_warnings(warnings, canonical_labels=canonical_labels)
    assert [warning.split for warning in canonical] == [
        "validation",
        "validation",
        "validation",
        "test",
        "test",
    ]
    assert [warning.label for warning in canonical] == [
        "Zebra",
        "Alpha",
        "Mango",
        "Alpha",
        "Mango",
    ]


def test_canonicalize_reorders_noncanonical_warning_serialization() -> None:
    warnings = [
        SplitWarning(label="Mango", split="test", count=1, threshold=5),
        SplitWarning(label="Alpha", split="validation", count=2, threshold=5),
    ]
    canonical_labels = ["Zebra", "Alpha", "Mango"]
    canonical = canonicalize_warnings(warnings, canonical_labels=canonical_labels)
    assert [(w.split, w.label) for w in canonical] == [
        ("validation", "Alpha"),
        ("test", "Mango"),
    ]


def test_no_warning_at_or_above_threshold() -> None:
    config = _config()
    positives = {
        "train": Counter({"Bug": 10}),
        "validation": Counter({"Bug": 5}),
        "test": Counter({"Bug": 5}),
    }
    hard_errors, warnings = validate_split_support(
        canonical_labels=["Bug"], positives_per_split=positives, config=config
    )
    assert not hard_errors
    assert warnings == []


def test_zero_positive_remains_hard_failure_not_warning() -> None:
    config = _config()
    positives = {
        "train": Counter({"Bug": 0}),
        "validation": Counter({"Bug": 2}),
        "test": Counter({"Bug": 2}),
    }
    hard_errors, warnings = validate_split_support(
        canonical_labels=["Bug"], positives_per_split=positives, config=config
    )
    assert any("train" in error for error in hard_errors)
    assert not any("validation" in error for error in hard_errors)
    assert not any("test" in error for error in hard_errors)
    assert len(warnings) == 2
    with pytest.raises(ModelDatasetSplitSupportError):
        raise_on_hard_support_errors(hard_errors)
