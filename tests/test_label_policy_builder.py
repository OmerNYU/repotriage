"""Tests for label-policy orchestration, publication, determinism, and immutability."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from repotriage.audit.builder import audit_dataset
from repotriage.audit.models import AuditManifest
from repotriage.dataset.models import ProcessedManifest
from repotriage.github.models import RepositoryRef
from repotriage.label_policy.builder import (
    build_label_policy,
    publish_label_policy,
)
from repotriage.label_policy.config import load_config
from repotriage.label_policy.models import (
    LabelPolicyCorruptionError,
    LabelPolicyError,
    LabelPolicyInputError,
    compute_policy_id,
    compute_policy_input_sha256,
)
from tests.helpers import (
    make_normalized_issue,
    write_label_policy_config,
    write_processed_dataset,
)

_CRITERIA = {
    "min_total_support": 2,
    "min_active_months": 2,
    "min_recent_support": 1,
    "recent_window_months": 2,
}


def _at(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, 12, 0, 0, tzinfo=UTC)


def _issues() -> list:
    return [
        make_normalized_issue(1, labels=["Bug"], created_at=_at(2025, 1, 5)),
        make_normalized_issue(2, labels=["Bug", "Docs"], created_at=_at(2025, 5, 5)),
        make_normalized_issue(3, labels=["Docs"], created_at=_at(2025, 6, 5)),
        make_normalized_issue(4, labels=["Groupby"], created_at=_at(2025, 3, 5)),
        make_normalized_issue(5, labels=["Needs Triage"], created_at=_at(2025, 4, 5)),
        make_normalized_issue(6, labels=["Misc"], created_at=_at(2025, 2, 5)),
    ]


def _labels() -> list[dict[str, object]]:
    return [
        {
            "label": "Bug",
            "decision": "include",
            "role": "issue_type",
            "leakage_risk": "low",
            "reason_code": "selected_target",
            "explanation": "bug",
        },
        {
            "label": "Docs",
            "decision": "include",
            "role": "component",
            "leakage_risk": "low",
            "reason_code": "selected_target",
            "explanation": "docs",
        },
        {
            "label": "Groupby",
            "decision": "defer",
            "role": "component",
            "leakage_risk": "low",
            "reason_code": "insufficient_recent_support",
            "explanation": "groupby",
        },
        {
            "label": "Needs Triage",
            "decision": "exclude",
            "role": "workflow",
            "leakage_risk": "high",
            "reason_code": "workflow_label",
            "explanation": "workflow",
        },
    ]


@dataclass
class Fixture:
    repository: RepositoryRef
    dataset_id: str
    audit_id: str
    config_path: Path
    processed_root: Path
    audits_root: Path
    policies_root: Path
    processed_manifest: ProcessedManifest
    audit_manifest: AuditManifest
    config_sha256: str
    policy_input_sha256: str
    policy_id: str


def _setup(tmp_path: Path, *, labels: list[dict[str, object]] | None = None) -> Fixture:
    repository = RepositoryRef(owner="pandas-dev", name="pandas")
    processed_root = tmp_path / "processed"
    audits_root = tmp_path / "audits"
    policies_root = tmp_path / "policies"
    dataset_dir, dataset_id = write_processed_dataset(processed_root, repository, _issues())
    processed_manifest = ProcessedManifest.model_validate_json(
        (dataset_dir / "manifest.json").read_text(encoding="utf-8")
    )
    audit = audit_dataset(
        repository, dataset_id, processed_root=processed_root, audits_root=audits_root
    )
    config_path = write_label_policy_config(
        tmp_path / "policy.json",
        labels=labels if labels is not None else _labels(),
        selection_criteria=_CRITERIA,
    )
    _, config_hash = load_config(config_path)
    policy_input_sha256 = compute_policy_input_sha256(
        policy_version="2",
        dataset_id=dataset_id,
        dataset_output_sha256=processed_manifest.output_sha256,
        audit_id=audit.manifest.audit_id,
        audit_json_sha256=audit.manifest.audit_json_sha256,
        config_schema_version="2",
        config_sha256=config_hash,
    )
    return Fixture(
        repository=repository,
        dataset_id=dataset_id,
        audit_id=audit.manifest.audit_id,
        config_path=config_path,
        processed_root=processed_root,
        audits_root=audits_root,
        policies_root=policies_root,
        processed_manifest=processed_manifest,
        audit_manifest=audit.manifest,
        config_sha256=config_hash,
        policy_input_sha256=policy_input_sha256,
        policy_id=compute_policy_id(dataset_id, policy_input_sha256, "2"),
    )


def _build(fx: Fixture, *, policies_root: Path | None = None):
    return build_label_policy(
        fx.repository,
        fx.dataset_id,
        fx.audit_id,
        fx.config_path,
        processed_root=fx.processed_root,
        audits_root=fx.audits_root,
        policies_root=policies_root or fx.policies_root,
    )


def test_build_success(tmp_path: Path) -> None:
    fx = _setup(tmp_path)
    result = _build(fx)

    assert result.cache_hit is False
    assert "-lp2-" in result.manifest.policy_id
    assert result.policy_dir == fx.policies_root / fx.repository.slug / fx.policy_id
    assert (result.policy_dir / "label_policy.json").is_file()
    assert (result.policy_dir / "label_policy.md").is_file()
    assert (result.policy_dir / "manifest.json").is_file()
    assert result.manifest.included_label_count == 2
    assert result.manifest.deferred_label_count == 1
    assert result.manifest.excluded_label_count == 2


def test_manifest_lineage(tmp_path: Path) -> None:
    fx = _setup(tmp_path)
    manifest = _build(fx).manifest

    assert manifest.repository == fx.repository.full_name
    assert manifest.dataset_id == fx.dataset_id
    assert manifest.dataset_output_sha256 == fx.processed_manifest.output_sha256
    assert manifest.audit_id == fx.audit_id
    assert manifest.audit_json_sha256 == fx.audit_manifest.audit_json_sha256
    assert manifest.audit_version == "2"
    assert manifest.config_schema_version == "2"
    assert manifest.config_sha256 == fx.config_sha256
    assert manifest.policy_input_sha256 == fx.policy_input_sha256
    assert manifest.policy_id == fx.policy_id
    assert manifest.explicit_label_count == 4
    assert manifest.default_label_count == 1


def test_deterministic_reports(tmp_path: Path) -> None:
    fx = _setup(tmp_path)
    first = _build(fx, policies_root=tmp_path / "p1")
    second = _build(fx, policies_root=tmp_path / "p2")

    assert first.manifest.policy_id == second.manifest.policy_id
    assert first.manifest.label_policy_json_sha256 == second.manifest.label_policy_json_sha256
    assert (first.policy_dir / "label_policy.json").read_bytes() == (
        second.policy_dir / "label_policy.json"
    ).read_bytes()


def test_repeated_build_is_cache_hit(tmp_path: Path) -> None:
    fx = _setup(tmp_path)
    first = _build(fx)
    second = _build(fx)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.policy_dir == first.policy_dir
    assert second.manifest.built_at == first.manifest.built_at


def test_tampered_policy_json_detected(tmp_path: Path) -> None:
    fx = _setup(tmp_path)
    result = _build(fx)
    (result.policy_dir / "label_policy.json").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(LabelPolicyCorruptionError, match="hash mismatch"):
        _build(fx)


def _rehash_manifest(policy_dir: Path) -> None:
    import hashlib

    new_hash = hashlib.sha256((policy_dir / "label_policy.json").read_bytes()).hexdigest()
    manifest_path = policy_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["label_policy_json_sha256"] = new_hash
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")


def test_semantic_cross_check_detects_disagreement(tmp_path: Path) -> None:
    fx = _setup(tmp_path)
    result = _build(fx)

    json_path = result.policy_dir / "label_policy.json"
    document = json.loads(json_path.read_text(encoding="utf-8"))
    document["identity"]["repository"] = "other/repo"
    json_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _rehash_manifest(result.policy_dir)

    with pytest.raises(LabelPolicyCorruptionError):
        _build(fx)


def test_incompatible_artifact_not_overwritten(tmp_path: Path) -> None:
    fx = _setup(tmp_path)
    final_dir = fx.policies_root / fx.repository.slug / fx.policy_id
    final_dir.mkdir(parents=True)
    marker = '{"schema_version": "0"}\n'
    (final_dir / "manifest.json").write_text(marker, encoding="utf-8")

    with pytest.raises(LabelPolicyCorruptionError):
        _build(fx)

    assert (final_dir / "manifest.json").read_text(encoding="utf-8") == marker


def test_lp1_artifact_left_untouched(tmp_path: Path) -> None:
    fx = _setup(tmp_path)
    slug_dir = fx.policies_root / fx.repository.slug
    lp1_dir = slug_dir / f"{fx.dataset_id}-lp1-0c024bcdd789"
    lp1_dir.mkdir(parents=True)
    lp1_file = lp1_dir / "label_policy.json"
    lp1_bytes = b'{"frozen": true}\n'
    lp1_file.write_bytes(lp1_bytes)

    result = _build(fx)

    assert result.policy_dir != lp1_dir
    assert lp1_dir.is_dir()
    assert lp1_file.read_bytes() == lp1_bytes


def test_unsupported_audit_version_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fx = _setup(tmp_path)
    monkeypatch.setattr(
        "repotriage.label_policy.builder.SUPPORTED_AUDIT_VERSIONS", frozenset()
    )
    with pytest.raises(LabelPolicyInputError, match="Audit version"):
        _build(fx)


def test_failure_cleans_staging_and_leaves_no_final_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fx = _setup(tmp_path)

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("repotriage.label_policy.builder.atomic_write_bytes", boom)

    with pytest.raises(RuntimeError, match="boom"):
        _build(fx)

    final_dir = fx.policies_root / fx.repository.slug / fx.policy_id
    assert not final_dir.exists()
    slug_dir = fx.policies_root / fx.repository.slug
    if slug_dir.exists():
        assert list(slug_dir.glob(".*staging-*")) == []


def test_keyboard_interrupt_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fx = _setup(tmp_path)

    def interrupt(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("repotriage.label_policy.builder.atomic_write_bytes", interrupt)

    with pytest.raises(KeyboardInterrupt):
        _build(fx)

    slug_dir = fx.policies_root / fx.repository.slug
    if slug_dir.exists():
        assert list(slug_dir.glob(".*staging-*")) == []


def test_config_change_changes_policy_id(tmp_path: Path) -> None:
    fx = _setup(tmp_path)
    first = _build(fx)

    changed = _labels()
    changed[0]["leakage_risk"] = "medium"
    write_label_policy_config(
        fx.config_path, labels=changed, selection_criteria=_CRITERIA
    )
    second = _build(fx)

    assert second.policy_dir != first.policy_dir
    assert first.policy_dir.is_dir()
    assert second.policy_dir.is_dir()


def test_publish_refuses_to_overwrite(tmp_path: Path) -> None:
    final_dir = tmp_path / "policy"
    final_dir.mkdir()
    (final_dir / "marker").write_text("keep", encoding="utf-8")
    staging = tmp_path / "staging"
    staging.mkdir()

    with pytest.raises(LabelPolicyError, match="Refusing to overwrite"):
        publish_label_policy(staging, final_dir)
    assert (final_dir / "marker").read_text(encoding="utf-8") == "keep"
