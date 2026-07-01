"""Tests for target-label vector construction."""

import pytest

from repotriage.model_dataset.models import ModelDatasetTransformError
from repotriage.model_dataset.targets import (
    assert_canonical_order_matches_policy,
    build_target_labels,
)

CANONICAL = ["Bug", "Docs", "Arrow"]


def test_canonical_policy_order() -> None:
    selected, vector = build_target_labels(["Docs", "Bug"], CANONICAL)
    assert selected == ["Bug", "Docs"]
    assert vector == [1, 1, 0]


def test_multi_label_vector() -> None:
    selected, vector = build_target_labels(["Bug", "Arrow"], CANONICAL)
    assert selected == ["Bug", "Arrow"]
    assert vector == [1, 0, 1]


def test_all_zero_vector() -> None:
    selected, vector = build_target_labels(["Other"], CANONICAL)
    assert selected == []
    assert vector == [0, 0, 0]


def test_duplicate_source_labels() -> None:
    _, vector = build_target_labels(["Bug", "Bug", "Docs"], CANONICAL)
    assert vector == [1, 1, 0]


def test_selected_labels_vector_agreement() -> None:
    selected, vector = build_target_labels(["Docs"], CANONICAL)
    assert selected == [label for label, value in zip(CANONICAL, vector) if value == 1]


def test_empty_canonical_order_rejected() -> None:
    with pytest.raises(ModelDatasetTransformError, match="empty"):
        build_target_labels(["Bug"], [])


def test_duplicate_canonical_order_rejected() -> None:
    with pytest.raises(ModelDatasetTransformError, match="duplicate"):
        build_target_labels(["Bug"], ["Bug", "Bug"])


def test_assert_canonical_order_matches_policy() -> None:
    assert_canonical_order_matches_policy(["Bug", "Docs"], ["Bug", "Docs"])


def test_reordered_label_map_rejected() -> None:
    with pytest.raises(ModelDatasetTransformError, match="ordered include"):
        assert_canonical_order_matches_policy(["Docs", "Bug"], ["Bug", "Docs"])
