"""Tests for the build-label-policy CLI command."""

from __future__ import annotations

from pathlib import Path

import pytest

from repotriage.audit.builder import audit_dataset
from repotriage.audit.models import compute_audit_id
from repotriage.cli import run_build_label_policy
from repotriage.github.models import RepositoryRef
from tests.helpers import (
    make_normalized_issue,
    write_label_policy_config,
    write_processed_dataset,
)


class Args:
    def __init__(
        self,
        repo: str,
        dataset_id: str,
        audit_id: str,
        config: Path,
        processed_root: Path,
        audits_root: Path,
        policies_root: Path,
    ) -> None:
        self.repo = repo
        self.dataset_id = dataset_id
        self.audit_id = audit_id
        self.config = config
        self.processed_root = processed_root
        self.audits_root = audits_root
        self.policies_root = policies_root


def _labels() -> list[dict[str, object]]:
    return [
        {
            "label": "Bug",
            "decision": "include",
            "role": "issue_type",
            "leakage_risk": "low",
            "reason_code": "selected_target",
            "explanation": "bug",
        }
    ]


def _setup(tmp_path: Path) -> tuple[Args, str]:
    repository = RepositoryRef(owner="pandas-dev", name="pandas")
    processed_root = tmp_path / "processed"
    audits_root = tmp_path / "audits"
    policies_root = tmp_path / "policies"
    _, dataset_id = write_processed_dataset(
        processed_root,
        repository,
        [make_normalized_issue(1, labels=["Bug"]), make_normalized_issue(2)],
    )
    audit_dataset(
        repository, dataset_id, processed_root=processed_root, audits_root=audits_root
    )
    config = write_label_policy_config(tmp_path / "policy.json", labels=_labels())
    audit_id = compute_audit_id(dataset_id, "2")
    args = Args(
        "pandas-dev/pandas",
        dataset_id,
        audit_id,
        config,
        processed_root,
        audits_root,
        policies_root,
    )
    return args, dataset_id


def test_cli_success_prints_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args, dataset_id = _setup(tmp_path)

    exit_code = run_build_label_policy(args)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Repository: pandas-dev/pandas" in captured.out
    assert f"Dataset ID: {dataset_id}" in captured.out
    assert f"Policy ID: {dataset_id}-lp2-" in captured.out
    assert "Policy-input SHA-256: " in captured.out
    assert "Included labels: 1" in captured.out
    assert "Explicitly reviewed labels: 1" in captured.out
    assert "Policy-cache hit: no" in captured.out


def test_cli_invalid_dataset_id_exits_2(tmp_path: Path) -> None:
    args, _ = _setup(tmp_path)
    args.dataset_id = "not-a-dataset-id"
    assert run_build_label_policy(args) == 2


def test_cli_invalid_audit_id_exits_2(tmp_path: Path) -> None:
    args, _ = _setup(tmp_path)
    args.audit_id = "not-an-audit-id"
    assert run_build_label_policy(args) == 2


def test_cli_invalid_repo_exits_2(tmp_path: Path) -> None:
    args, _ = _setup(tmp_path)
    args.repo = "not a repo"
    assert run_build_label_policy(args) == 2


def test_cli_missing_dataset_exits_1(tmp_path: Path) -> None:
    args, _ = _setup(tmp_path)
    args.dataset_id = "20990101T000000000000Z-n1-ffffffffffff"
    args.audit_id = "20990101T000000000000Z-n1-ffffffffffff-a2"
    assert run_build_label_policy(args) == 1
