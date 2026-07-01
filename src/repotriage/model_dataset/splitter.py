"""Temporal split assignment and support validation."""

from __future__ import annotations

from collections import Counter
from datetime import datetime

from repotriage.dataset.models import _ensure_utc
from repotriage.model_dataset.config import TemporalSplitConfig
from repotriage.model_dataset.models import (
    ModelDatasetSplitSupportError,
    SplitName,
    SplitWarning,
)


def assign_split(created_at: datetime, config: TemporalSplitConfig) -> SplitName:
    """Assign a record to train, validation, or test by created_at and config cutoffs."""
    timestamp = _ensure_utc(created_at)
    validation_start = _ensure_utc(config.validation_start)
    test_start = _ensure_utc(config.test_start)
    if timestamp < validation_start:
        return "train"
    if timestamp < test_start:
        return "validation"
    return "test"


def validate_split_support(
    *,
    canonical_labels: list[str],
    positives_per_split: dict[str, Counter[str]],
    config: TemporalSplitConfig,
) -> tuple[list[str], list[SplitWarning]]:
    """Validate per-label positive support; return hard errors and low-support warnings."""
    hard_errors: list[str] = []
    warnings: list[SplitWarning] = []
    minimums = config.minimum_positive_support
    threshold = config.low_support_warning_threshold

    for split_name in ("train", "validation", "test"):
        minimum = getattr(minimums, split_name)
        counts = positives_per_split.get(split_name, Counter())
        for label in canonical_labels:
            count = counts.get(label, 0)
            if count < minimum:
                hard_errors.append(
                    f"Label {label!r} has {count} positive(s) in {split_name} "
                    f"(minimum {minimum})."
                )
            elif split_name in ("validation", "test") and 0 < count < threshold:
                warnings.append(
                    SplitWarning(
                        label=label,
                        split=split_name,  # type: ignore[arg-type]
                        count=count,
                        threshold=threshold,
                    )
                )

    return hard_errors, warnings


def raise_on_hard_support_errors(hard_errors: list[str]) -> None:
    if hard_errors:
        raise ModelDatasetSplitSupportError("; ".join(hard_errors))


_SPLIT_RANK = {"train": 0, "validation": 1, "test": 2}


def canonicalize_warnings(
    warnings: list[SplitWarning], *, canonical_labels: list[str]
) -> list[SplitWarning]:
    """Return warnings in canonical order: split rank, label_map index, then code."""
    label_index = {label: index for index, label in enumerate(canonical_labels)}
    return sorted(
        warnings,
        key=lambda warning: (
            _SPLIT_RANK[warning.split],
            label_index[warning.label],
            warning.code,
        ),
    )


def sort_warnings(
    warnings: list[SplitWarning], *, canonical_labels: list[str]
) -> list[SplitWarning]:
    """Alias for :func:`canonicalize_warnings`."""
    return canonicalize_warnings(warnings, canonical_labels=canonical_labels)
