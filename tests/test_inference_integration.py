"""Integration tests for canonical pandas inference artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest

from repotriage.github.models import RepositoryRef
from repotriage.inference.artifact_loader import load_inference_bundle
from repotriage.inference.models import InferenceIssueInput
from repotriage.inference.pipeline import infer_issue

_CONFIG = Path("configs/inference/pandas-dev__pandas/local-v1.json")
_MODEL_DATASET_ID = "20260628T161306010651Z-n1-074402d21505-md1-14a9768bded7"
_BASELINE_RUN_ID = f"{_MODEL_DATASET_ID}-bl4-46227a0ec602"
_THRESHOLD_POLICY_ID = f"{_BASELINE_RUN_ID}-tp1-ccaab0996458"
_ABSTENTION_POLICY_ID = f"{_THRESHOLD_POLICY_ID}-ap1-9c3c140e7ccb"
_RETRIEVAL_RUN_ID = f"{_MODEL_DATASET_ID}-rb1-deb29b6da4eb"


def _artifacts_present() -> bool:
    slug = "pandas-dev__pandas"
    return (
        _CONFIG.is_file()
        and (Path("data/model_ready/github") / slug / _MODEL_DATASET_ID).is_dir()
        and (Path("data/baselines/github") / slug / _BASELINE_RUN_ID).is_dir()
        and (Path("data/threshold_policies/github") / slug / _THRESHOLD_POLICY_ID).is_dir()
        and (Path("data/abstention_policies/github") / slug / _ABSTENTION_POLICY_ID).is_dir()
        and (Path("data/retrieval_baselines/github") / slug / _RETRIEVAL_RUN_ID).is_dir()
    )


@pytest.mark.integration
@pytest.mark.skipif(not _artifacts_present(), reason="canonical local artifacts not present")
def test_load_inference_bundle_and_infer_issue() -> None:
    repository = RepositoryRef(owner="pandas-dev", name="pandas")
    bundle = load_inference_bundle(_CONFIG, repository=repository)

    assert bundle.classification_threshold == pytest.approx(0.39)
    assert bundle.abstention_threshold == pytest.approx(0.84)
    assert len(bundle.label_order) == 15
    assert bundle.label_order[0] == "Bug"

    response = infer_issue(
        bundle,
        InferenceIssueInput(
            title="BUG: loc indexing returns unexpected result",
            body="When using .loc with a list indexer, result dtype is wrong.",
            top_k=5,
        ),
    )

    assert response.repository == "pandas-dev/pandas"
    assert len(response.classification.scores) == 15
    assert response.classification.threshold == pytest.approx(0.39)
    assert response.abstention.threshold == pytest.approx(0.84)
    assert response.artifacts.baseline_run_id == _BASELINE_RUN_ID
    assert response.artifacts.retrieval_run_id == _RETRIEVAL_RUN_ID
    assert len(response.retrieval.similar_issues) <= 5
    if response.retrieval.similar_issues:
        assert "predicted_label_overlap" in response.retrieval.similar_issues[0].model_dump()
