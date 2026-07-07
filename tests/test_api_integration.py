"""Integration tests for the FastAPI inference API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from repotriage.api.app import create_app
from repotriage.api.settings import ApiSettings
from repotriage.github.models import RepositoryRef
from repotriage.inference.artifact_loader import load_inference_bundle
from repotriage.inference.models import InferenceIssueInput
from repotriage.inference.pipeline import infer_issue
from repotriage.inference.report import format_inference_response_json

_CONFIG = Path("configs/inference/pandas-dev__pandas/local-v1.json")
_MODEL_DATASET_ID = "20260628T161306010651Z-n1-074402d21505-md1-14a9768bded7"
_BASELINE_RUN_ID = f"{_MODEL_DATASET_ID}-bl4-46227a0ec602"
_THRESHOLD_POLICY_ID = f"{_BASELINE_RUN_ID}-tp1-ccaab0996458"
_ABSTENTION_POLICY_ID = f"{_THRESHOLD_POLICY_ID}-ap1-9c3c140e7ccb"
_RETRIEVAL_RUN_ID = f"{_MODEL_DATASET_ID}-rb1-deb29b6da4eb"

_ISSUE_INPUT = InferenceIssueInput(
    title="BUG: loc indexing returns unexpected result",
    body="When using .loc with a list indexer, result dtype is wrong.",
    top_k=5,
)


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
def test_api_infer_matches_direct_pipeline() -> None:
    settings = ApiSettings(inference_config_path=_CONFIG)
    repository = RepositoryRef(owner="pandas-dev", name="pandas")
    bundle = load_inference_bundle(_CONFIG, repository=repository)
    direct_response = infer_issue(bundle, _ISSUE_INPUT)

    app = create_app(settings=settings)
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["repository"] == "pandas-dev/pandas"

        api_response = client.post(
            "/api/v1/infer",
            json=_ISSUE_INPUT.model_dump(mode="json"),
        )

    assert api_response.status_code == 200
    api_payload = api_response.json()
    direct_payload = json.loads(format_inference_response_json(direct_response))

    assert api_payload["schema_version"] == direct_payload["schema_version"]
    assert api_payload["repository"] == direct_payload["repository"]
    assert api_payload["input"] == direct_payload["input"]
    assert api_payload["classification"] == direct_payload["classification"]
    assert api_payload["abstention"] == direct_payload["abstention"]
    assert api_payload["retrieval"] == direct_payload["retrieval"]
    assert api_payload["artifacts"] == direct_payload["artifacts"]
    assert api_payload["reproducibility"] == direct_payload["reproducibility"]
    assert api_payload["warnings"] == direct_payload["warnings"]
