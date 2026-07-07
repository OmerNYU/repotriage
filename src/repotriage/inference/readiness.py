"""Artifact readiness checks for local inference bundles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

from repotriage.abstention_policy.builder import (
    DEFAULT_ABSTENTION_POLICIES_ROOT,
    validate_abstention_policy_artifact_integrity,
)
from repotriage.abstention_policy.models import AbstentionPolicyCorruptionError
from repotriage.baseline.builder import DEFAULT_BASELINES_ROOT, validate_baseline_artifact_integrity
from repotriage.baseline.models import BaselineCorruptionError
from repotriage.github.models import RepositoryRef, parse_repository
from repotriage.inference.artifact_loader import (
    DEFAULT_RETRIEVAL_BASELINES_ROOT,
    load_inference_bundle,
)
from repotriage.inference.config import InferenceConfigDocument, load_inference_config
from repotriage.inference.models import InferenceBundleError, InferenceConfigError
from repotriage.model_dataset.builder import (
    DEFAULT_MODEL_READY_ROOT,
    validate_model_dataset_artifact_integrity,
)
from repotriage.model_dataset.models import ModelDatasetCorruptionError
from repotriage.retrieval.models import RetrievalCorruptionError
from repotriage.retrieval.validators import validate_retrieval_artifact_integrity
from repotriage.threshold_policy.builder import (
    DEFAULT_THRESHOLD_POLICIES_ROOT,
    validate_threshold_policy_artifact_integrity,
)
from repotriage.threshold_policy.models import ThresholdPolicyCorruptionError

MANIFEST_JSON_FILE = "manifest.json"

ArtifactStatus = Literal["ok", "missing", "invalid"]


class ReadinessMode(StrEnum):
    """Depth of artifact readiness verification."""

    PRESENCE = "presence"
    INTEGRITY = "integrity"
    STRICT = "strict"


@dataclass(frozen=True)
class ArtifactRoots:
    """Filesystem roots for inference-bound artifact families."""

    model_ready_root: Path = DEFAULT_MODEL_READY_ROOT
    baselines_root: Path = DEFAULT_BASELINES_ROOT
    threshold_policies_root: Path = DEFAULT_THRESHOLD_POLICIES_ROOT
    abstention_policies_root: Path = DEFAULT_ABSTENTION_POLICIES_ROOT
    retrieval_baselines_root: Path = DEFAULT_RETRIEVAL_BASELINES_ROOT


@dataclass(frozen=True)
class ArtifactSpec:
    """One inference-bound artifact family."""

    config_field: str
    display_name: str
    root_attr: str


ARTIFACT_SPECS: tuple[ArtifactSpec, ...] = (
    ArtifactSpec("model_dataset_id", "model-ready dataset", "model_ready_root"),
    ArtifactSpec("baseline_run_id", "baseline classifier", "baselines_root"),
    ArtifactSpec("threshold_policy_id", "threshold policy", "threshold_policies_root"),
    ArtifactSpec("abstention_policy_id", "abstention policy", "abstention_policies_root"),
    ArtifactSpec("retrieval_run_id", "retrieval baseline", "retrieval_baselines_root"),
)

PANDAS_BOOTSTRAP_HINTS: dict[str, str] = {
    "model_dataset_id": (
        "repotriage build-model-dataset \\\n"
        "  --repo {repository} \\\n"
        "  --dataset-id 20260628T161306010651Z-n1-074402d21505 \\\n"
        "  --policy-id 20260628T161306010651Z-n1-074402d21505-lp2-95899f0f5b37 \\\n"
        "  --config configs/model_datasets/pandas-dev__pandas/temporal-v1.json\n"
        "  (upstream prerequisites: see docs/demo.md#full-artifact-pipeline)"
    ),
    "baseline_run_id": (
        "repotriage train-baseline \\\n"
        "  --repo {repository} \\\n"
        "  --model-dataset-id {model_dataset_id} \\\n"
        "  --config configs/baselines/pandas-dev__pandas/tfidf-logreg-v1.json"
    ),
    "threshold_policy_id": (
        "repotriage build-threshold-policy \\\n"
        "  --repo {repository} \\\n"
        "  --baseline-run-id {baseline_run_id} \\\n"
        "  --config configs/threshold_policies/pandas-dev__pandas/global-v1.json"
    ),
    "abstention_policy_id": (
        "repotriage build-abstention-policy \\\n"
        "  --repo {repository} \\\n"
        "  --threshold-policy-id {threshold_policy_id} \\\n"
        "  --config configs/abstention_policies/pandas-dev__pandas/issue-confidence-v1.json"
    ),
    "retrieval_run_id": (
        "repotriage build-retrieval-baseline \\\n"
        "  --repo {repository} \\\n"
        "  --model-dataset-id {model_dataset_id} \\\n"
        "  --config configs/retrieval_baselines/pandas-dev__pandas/tfidf-cosine-v1.json"
    ),
}


@dataclass(frozen=True)
class ArtifactCheckResult:
    """Readiness result for one artifact family."""

    name: str
    status: ArtifactStatus
    path: Path
    detail: str | None = None
    next_command: str | None = None


@dataclass(frozen=True)
class ReadinessReport:
    """Aggregate artifact readiness report."""

    config_path: Path
    repository: str
    mode: ReadinessMode
    results: tuple[ArtifactCheckResult, ...]
    ready: bool
    strict_error: str | None = None


def resolve_artifact_paths(
    config: InferenceConfigDocument,
    repository: RepositoryRef,
    roots: ArtifactRoots,
) -> dict[str, Path]:
    """Resolve artifact directories from an inference config and roots."""
    resolved: dict[str, Path] = {}
    for spec in ARTIFACT_SPECS:
        root = getattr(roots, spec.root_attr)
        artifact_id = getattr(config, spec.config_field)
        resolved[spec.config_field] = root / repository.slug / artifact_id
    return resolved


def _format_next_command(config: InferenceConfigDocument, config_field: str) -> str | None:
    if config.repository != "pandas-dev/pandas":
        return None
    template = PANDAS_BOOTSTRAP_HINTS.get(config_field)
    if template is None:
        return None
    return template.format(
        repository=config.repository,
        model_dataset_id=config.model_dataset_id,
        baseline_run_id=config.baseline_run_id,
        threshold_policy_id=config.threshold_policy_id,
        abstention_policy_id=config.abstention_policy_id,
        retrieval_run_id=config.retrieval_run_id,
    )


def _check_artifact_presence(
    *,
    spec: ArtifactSpec,
    path: Path,
    config: InferenceConfigDocument,
) -> ArtifactCheckResult:
    next_command = _format_next_command(config, spec.config_field)
    if not path.is_dir():
        return ArtifactCheckResult(
            name=spec.display_name,
            status="missing",
            path=path,
            next_command=next_command,
        )
    manifest_path = path / MANIFEST_JSON_FILE
    if not manifest_path.is_file():
        return ArtifactCheckResult(
            name=spec.display_name,
            status="missing",
            path=path,
            detail=f"Directory exists but {MANIFEST_JSON_FILE} is missing.",
            next_command=next_command,
        )
    return ArtifactCheckResult(
        name=spec.display_name,
        status="ok",
        path=path,
    )


def _check_artifact_integrity(
    *,
    spec: ArtifactSpec,
    path: Path,
    config: InferenceConfigDocument,
    repository: RepositoryRef,
) -> ArtifactCheckResult:
    presence = _check_artifact_presence(spec=spec, path=path, config=config)
    if presence.status != "ok":
        return presence

    artifact_id = getattr(config, spec.config_field)
    try:
        if spec.config_field == "model_dataset_id":
            validate_model_dataset_artifact_integrity(
                path,
                expected_repository=repository,
                expected_model_dataset_id=artifact_id,
            )
        elif spec.config_field == "baseline_run_id":
            validate_baseline_artifact_integrity(
                path,
                expected_repository=repository,
                expected_baseline_run_id=artifact_id,
            )
        elif spec.config_field == "threshold_policy_id":
            validate_threshold_policy_artifact_integrity(
                path,
                expected_repository=repository,
                expected_policy_id=artifact_id,
            )
        elif spec.config_field == "abstention_policy_id":
            validate_abstention_policy_artifact_integrity(
                path,
                expected_repository=repository,
                expected_policy_id=artifact_id,
            )
        elif spec.config_field == "retrieval_run_id":
            validate_retrieval_artifact_integrity(
                path,
                expected_repository=repository,
                expected_retrieval_run_id=artifact_id,
            )
    except (
        ModelDatasetCorruptionError,
        BaselineCorruptionError,
        ThresholdPolicyCorruptionError,
        AbstentionPolicyCorruptionError,
        RetrievalCorruptionError,
    ) as exc:
        return ArtifactCheckResult(
            name=spec.display_name,
            status="invalid",
            path=path,
            detail=str(exc),
            next_command=_format_next_command(config, spec.config_field),
        )

    return ArtifactCheckResult(
        name=spec.display_name,
        status="ok",
        path=path,
    )


def check_inference_artifacts(
    config_path: Path,
    *,
    mode: ReadinessMode = ReadinessMode.PRESENCE,
    roots: ArtifactRoots | None = None,
) -> ReadinessReport:
    """Check inference artifact readiness for a config file."""
    config = load_inference_config(config_path)
    repository = parse_repository(config.repository)
    resolved_roots = roots or ArtifactRoots()
    paths = resolve_artifact_paths(config, repository, resolved_roots)

    results: list[ArtifactCheckResult] = []
    for spec in ARTIFACT_SPECS:
        path = paths[spec.config_field]
        if mode == ReadinessMode.PRESENCE:
            results.append(_check_artifact_presence(spec=spec, path=path, config=config))
        else:
            results.append(
                _check_artifact_integrity(
                    spec=spec,
                    path=path,
                    config=config,
                    repository=repository,
                )
            )

    ready = all(result.status == "ok" for result in results)
    strict_error: str | None = None

    if ready and mode == ReadinessMode.STRICT:
        try:
            load_inference_bundle(
                config_path,
                repository=repository,
                baselines_root=resolved_roots.baselines_root,
                threshold_policies_root=resolved_roots.threshold_policies_root,
                abstention_policies_root=resolved_roots.abstention_policies_root,
                retrieval_baselines_root=resolved_roots.retrieval_baselines_root,
                model_ready_root=resolved_roots.model_ready_root,
            )
        except InferenceBundleError as exc:
            ready = False
            strict_error = str(exc)

    return ReadinessReport(
        config_path=config_path,
        repository=config.repository,
        mode=mode,
        results=tuple(results),
        ready=ready,
        strict_error=strict_error,
    )


def format_readiness_report(report: ReadinessReport) -> str:
    """Format a human-readable readiness report."""
    lines = [
        "RepoTriage artifact readiness check",
        f"Config: {report.config_path}",
        f"Repository: {report.repository}",
        f"Mode: {report.mode.value}",
        "",
    ]

    for result in report.results:
        if result.status == "ok":
            status_label = "[OK]"
        elif result.status == "missing":
            status_label = "[MISSING]"
        else:
            status_label = "[INVALID]"

        lines.append(f"{status_label} {result.name}")
        lines.append(f"     {result.path.as_posix()}")
        if result.detail:
            lines.append(f"     detail: {result.detail}")
        if result.status != "ok" and result.next_command:
            lines.append("     next:")
            for command_line in result.next_command.splitlines():
                lines.append(f"       {command_line}")
        lines.append("")

    if report.strict_error:
        lines.append("[INVALID] inference bundle compatibility")
        lines.append(f"     detail: {report.strict_error}")
        lines.append("")

    if report.ready:
        lines.extend(
            [
                "Ready for:",
                "  docker compose up --build",
                f"  repotriage serve --config {report.config_path.as_posix()}",
            ]
        )
    else:
        lines.extend(
            [
                "Not ready.",
                "See docs/demo.md for the full artifact pipeline.",
            ]
        )

    return "\n".join(lines) + "\n"


def format_readiness_report_json(report: ReadinessReport) -> str:
    """Format a machine-readable readiness report."""
    payload = {
        "config_path": report.config_path.as_posix(),
        "repository": report.repository,
        "mode": report.mode.value,
        "ready": report.ready,
        "strict_error": report.strict_error,
        "artifacts": [
            {
                "name": result.name,
                "status": result.status,
                "path": result.path.as_posix(),
                "detail": result.detail,
                "next_command": result.next_command,
            }
            for result in report.results
        ],
    }
    return json.dumps(payload, indent=2) + "\n"


__all__ = [
    "ArtifactCheckResult",
    "ArtifactRoots",
    "ArtifactSpec",
    "ARTIFACT_SPECS",
    "InferenceConfigError",
    "ReadinessMode",
    "ReadinessReport",
    "check_inference_artifacts",
    "format_readiness_report",
    "format_readiness_report_json",
    "resolve_artifact_paths",
]
