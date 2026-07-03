"""Build (or reuse) an immutable global threshold-policy artifact."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from repotriage.baseline.builder import validate_baseline_artifact_integrity
from repotriage.baseline.evaluator import compute_split_metrics
from repotriage.baseline.models import BaselineManifest, SplitMetrics, floats_consistent
from repotriage.filesystem import atomic_write_bytes, best_effort_remove_tree
from repotriage.github.models import RepositoryRef
from repotriage.paths import resolve_within_directory
from repotriage.threshold_policy.config import load_threshold_policy_config
from repotriage.threshold_policy.evaluator import evaluate_frozen_threshold
from repotriage.threshold_policy.models import (
    COMPARISON_JSON_FILE,
    CONFIG_JSON_FILE,
    MANIFEST_JSON_FILE,
    METRICS_TEST_JSON_FILE,
    METRICS_VALIDATION_JSON_FILE,
    POLICY_JSON_FILE,
    REPORT_MARKDOWN_FILE,
    SWEEP_VALIDATION_JSON_FILE,
    ComparisonDocument,
    FrozenThresholdPolicyConfig,
    PolicyDocument,
    SweepValidationDocument,
    ThresholdPolicyBuildError,
    ThresholdPolicyConfigDocument,
    ThresholdPolicyCorruptionError,
    ThresholdPolicyInputError,
    ThresholdPolicyManifest,
    ThresholdSweepRow,
    compute_policy_id,
    compute_policy_input_sha256,
)
from repotriage.threshold_policy.reader import (
    ValidationScoreBundle,
    load_test_scores,
    load_validation_scores,
)
from repotriage.threshold_policy.report import (
    build_comparison_document,
    serialize_comparison_json,
    serialize_config_json,
    serialize_manifest_json,
    serialize_metrics_json,
    serialize_policy_json,
    serialize_report_markdown,
    serialize_sweep_validation_json,
    sha256_hex,
)
from repotriage.threshold_policy.selector import (
    ThresholdSelectionResult,
    freeze_threshold_policy,
    select_threshold,
)
from repotriage.threshold_policy.sweep import (
    build_threshold_sweep,
    predictions_from_scores,
)

logger = logging.getLogger(__name__)

DEFAULT_BASELINES_ROOT = Path("data/baselines/github")
DEFAULT_THRESHOLD_POLICIES_ROOT = Path("data/threshold_policies/github")


@dataclass(frozen=True)
class ThresholdPolicyBuildResult:
    repository: RepositoryRef
    policy_dir: Path
    manifest: ThresholdPolicyManifest
    selected_threshold: float
    selected_threshold_basis_points: int
    validation_macro_f1_at_reference: float | None
    selected_validation_macro_f1: float | None
    test_macro_f1_at_reference: float | None
    selected_test_macro_f1: float | None
    cache_hit: bool


def _load_manifest(policy_dir: Path) -> ThresholdPolicyManifest:
    manifest_path = resolve_within_directory(policy_dir, MANIFEST_JSON_FILE)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ThresholdPolicyCorruptionError(
            f"Unable to read threshold-policy manifest at {manifest_path}: {exc}"
        ) from exc
    try:
        return ThresholdPolicyManifest.model_validate(payload)
    except ValidationError as exc:
        raise ThresholdPolicyCorruptionError(
            f"Invalid threshold-policy manifest at {manifest_path}: {exc}"
        ) from exc


def _verify_file_hash(path: Path, expected_sha256: str) -> bytes:
    if not path.is_file():
        raise ThresholdPolicyCorruptionError(f"Missing artifact file: {path}")
    data = path.read_bytes()
    actual = sha256_hex(data)
    if actual != expected_sha256:
        raise ThresholdPolicyCorruptionError(
            f"Hash mismatch for {path.name}: expected {expected_sha256}, got {actual}"
        )
    return data


def _metrics_close(left, right) -> None:
    for field in (
        "macro_f1",
        "micro_f1",
        "macro_precision",
        "macro_recall",
        "subset_accuracy",
        "hamming_loss",
    ):
        left_value = getattr(left.aggregate, field)
        right_value = getattr(right.aggregate, field)
        if left_value is None and right_value is None:
            continue
        if left_value is None or right_value is None:
            raise ThresholdPolicyCorruptionError(
                f"Metric {field} mismatch: {left_value} vs {right_value}"
            )
        if not floats_consistent(left_value, right_value):
            raise ThresholdPolicyCorruptionError(
                f"Metric {field} mismatch: {left_value} vs {right_value}"
            )


def _sweep_row_for_basis_points(
    rows: list[ThresholdSweepRow], basis_points: int
) -> ThresholdSweepRow:
    for row in rows:
        if row.threshold_basis_points == basis_points:
            return row
    raise ThresholdPolicyCorruptionError(
        f"Selected threshold basis points {basis_points} not found in sweep"
    )


def _validate_sweep_ordering(rows: list[ThresholdSweepRow]) -> None:
    basis_points = [row.threshold_basis_points for row in rows]
    if basis_points != sorted(basis_points):
        raise ThresholdPolicyCorruptionError("Sweep rows are not sorted by threshold_basis_points")
    if len(set(basis_points)) != len(basis_points):
        raise ThresholdPolicyCorruptionError("Duplicate threshold_basis_points in sweep")


def validate_threshold_policy_artifact_integrity(
    policy_dir: Path,
    *,
    expected_repository: RepositoryRef,
    expected_policy_id: str | None = None,
    check_dir_name: bool = True,
) -> ThresholdPolicyManifest:
    """Validate a threshold-policy artifact using only on-disk bytes."""
    if not policy_dir.is_dir():
        raise ThresholdPolicyCorruptionError(
            f"Threshold-policy directory does not exist: {policy_dir}"
        )

    manifest = _load_manifest(policy_dir)
    if check_dir_name and policy_dir.name != manifest.policy_id:
        raise ThresholdPolicyCorruptionError(
            f"Directory name {policy_dir.name!r} does not match policy_id "
            f"{manifest.policy_id!r}."
        )
    if expected_policy_id is not None and manifest.policy_id != expected_policy_id:
        raise ThresholdPolicyCorruptionError(
            f"Manifest policy_id {manifest.policy_id!r} does not match expected "
            f"{expected_policy_id!r}."
        )
    if manifest.repository != expected_repository.full_name:
        raise ThresholdPolicyCorruptionError(
            f"Manifest repository {manifest.repository!r} does not match expected "
            f"{expected_repository.full_name!r}."
        )

    config_bytes = _verify_file_hash(
        resolve_within_directory(policy_dir, manifest.config_file),
        manifest.config_sha256,
    )
    frozen_config = FrozenThresholdPolicyConfig.model_validate_json(config_bytes)
    expected_input = compute_policy_input_sha256(
        threshold_policy_version=manifest.threshold_policy_version,
        baseline_run_id=manifest.baseline_run_id,
        baseline_experiment_sha256=manifest.baseline_experiment_sha256,
        model_semantic_sha256=manifest.model_semantic_sha256,
        predictions_validation_sha256=manifest.predictions_validation_sha256,
        predictions_test_sha256=manifest.predictions_test_sha256,
        selected_candidate_id=manifest.selected_candidate_id,
        threshold_grid=frozen_config.threshold_grid,
        selection_rule_version=manifest.selection_rule_version,
        metric_contract_version=manifest.metric_contract_version,
    )
    if manifest.policy_input_sha256 != expected_input:
        raise ThresholdPolicyCorruptionError("manifest policy_input_sha256 mismatch")
    expected_id = compute_policy_id(manifest.baseline_run_id, expected_input)
    if manifest.policy_id != expected_id:
        raise ThresholdPolicyCorruptionError("manifest policy_id mismatch")

    for relative_path, expected_sha256 in (
        (manifest.policy_file, manifest.policy_sha256),
        (manifest.sweep_validation_file, manifest.sweep_validation_sha256),
        (manifest.metrics_validation_file, manifest.metrics_validation_sha256),
        (manifest.metrics_test_file, manifest.metrics_test_sha256),
        (manifest.comparison_file, manifest.comparison_sha256),
        (manifest.report_file, manifest.report_sha256),
    ):
        _verify_file_hash(resolve_within_directory(policy_dir, relative_path), expected_sha256)

    policy_bytes = _verify_file_hash(
        resolve_within_directory(policy_dir, manifest.policy_file),
        manifest.policy_sha256,
    )
    policy_document = PolicyDocument.model_validate_json(policy_bytes)
    if (
        policy_document.selection.selected_threshold_basis_points
        != manifest.selected_threshold_basis_points
    ):
        raise ThresholdPolicyCorruptionError(
            "policy.json selected threshold does not match manifest"
        )

    sweep_bytes = _verify_file_hash(
        resolve_within_directory(policy_dir, manifest.sweep_validation_file),
        manifest.sweep_validation_sha256,
    )
    sweep_document = SweepValidationDocument.model_validate_json(sweep_bytes)
    _validate_sweep_ordering(sweep_document.rows)
    if len(sweep_document.rows) != manifest.sweep_threshold_count:
        raise ThresholdPolicyCorruptionError("Sweep row count does not match manifest")

    selected_row = _sweep_row_for_basis_points(
        sweep_document.rows, manifest.selected_threshold_basis_points
    )
    metrics_validation_bytes = _verify_file_hash(
        resolve_within_directory(policy_dir, manifest.metrics_validation_file),
        manifest.metrics_validation_sha256,
    )
    stored_validation_metrics = SplitMetrics.model_validate_json(metrics_validation_bytes)
    _metrics_close(selected_row.metrics, stored_validation_metrics)

    report_text = resolve_within_directory(policy_dir, manifest.report_file).read_text(
        encoding="utf-8"
    )
    denominator = frozen_config.threshold_grid.denominator
    selected_threshold = manifest.selected_threshold_basis_points / denominator
    if f"**{selected_threshold:.2f}**" not in report_text:
        raise ThresholdPolicyCorruptionError("report.md does not reference selected threshold")

    return manifest


def validate_threshold_policy_against_baseline(
    policy_dir: Path,
    baseline_dir: Path,
    *,
    expected_repository: RepositoryRef,
    check_dir_name: bool = True,
) -> ThresholdPolicyManifest:
    """Recompute selection and metrics from baseline inputs and compare to stored artifact."""
    manifest = validate_threshold_policy_artifact_integrity(
        policy_dir,
        expected_repository=expected_repository,
        check_dir_name=check_dir_name,
    )
    baseline_manifest = validate_baseline_artifact_integrity(
        baseline_dir,
        expected_repository=expected_repository,
        expected_baseline_run_id=manifest.baseline_run_id,
    )

    if baseline_manifest.baseline_experiment_sha256 != manifest.baseline_experiment_sha256:
        raise ThresholdPolicyCorruptionError("baseline_experiment_sha256 mismatch")
    if baseline_manifest.model_semantic_sha256 != manifest.model_semantic_sha256:
        raise ThresholdPolicyCorruptionError("model_semantic_sha256 mismatch")
    if baseline_manifest.predictions_validation_sha256 != manifest.predictions_validation_sha256:
        raise ThresholdPolicyCorruptionError("predictions_validation_sha256 mismatch")
    if baseline_manifest.predictions_test_sha256 != manifest.predictions_test_sha256:
        raise ThresholdPolicyCorruptionError("predictions_test_sha256 mismatch")
    if baseline_manifest.selected_candidate_id != manifest.selected_candidate_id:
        raise ThresholdPolicyCorruptionError("selected_candidate_id mismatch")

    config_bytes = resolve_within_directory(policy_dir, manifest.config_file).read_bytes()
    frozen_config = FrozenThresholdPolicyConfig.model_validate_json(config_bytes)

    val_bundle = load_validation_scores(
        baseline_dir,
        baseline_manifest=baseline_manifest,
        candidate_id=manifest.selected_candidate_id,
        expected_repository=expected_repository,
    )
    sweep_rows = build_threshold_sweep(
        labels=val_bundle.labels,
        y_true=val_bundle.y_true,
        y_score=val_bundle.y_score,
        grid=frozen_config.threshold_grid,
    )
    selection = select_threshold(
        sweep=sweep_rows,
        reference_threshold_basis_points=frozen_config.reference_threshold_basis_points,
        selection_rule_version=frozen_config.selection_rule_version,
        denominator=frozen_config.threshold_grid.denominator,
    )
    if selection.selected_threshold_basis_points != manifest.selected_threshold_basis_points:
        raise ThresholdPolicyCorruptionError("Recomputed selected threshold mismatch")

    selected_row = _sweep_row_for_basis_points(
        sweep_rows, selection.selected_threshold_basis_points
    )
    stored_validation_metrics = SplitMetrics.model_validate_json(
        resolve_within_directory(policy_dir, manifest.metrics_validation_file).read_bytes()
    )
    _metrics_close(selected_row.metrics, stored_validation_metrics)

    frozen = freeze_threshold_policy(selection)
    test_bundle = load_test_scores(
        baseline_dir,
        baseline_manifest=baseline_manifest,
        candidate_id=manifest.selected_candidate_id,
        expected_repository=expected_repository,
    )
    recomputed_test_metrics = evaluate_frozen_threshold(frozen=frozen, bundle=test_bundle)
    stored_test_metrics = SplitMetrics.model_validate_json(
        resolve_within_directory(policy_dir, manifest.metrics_test_file).read_bytes()
    )
    _metrics_close(recomputed_test_metrics, stored_test_metrics)

    comparison = ComparisonDocument.model_validate_json(
        resolve_within_directory(policy_dir, manifest.comparison_file).read_bytes()
    )
    reference_bp = frozen_config.reference_threshold_basis_points
    reference_threshold = reference_bp / frozen_config.threshold_grid.denominator
    reference_val_metrics = _sweep_row_for_basis_points(sweep_rows, reference_bp).metrics
    reference_test_pred = predictions_from_scores(test_bundle.y_score, reference_threshold)
    reference_test_metrics = compute_split_metrics(
        split="test",
        labels=test_bundle.labels,
        y_true=test_bundle.y_true,
        y_pred=reference_test_pred,
        y_score=test_bundle.y_score,
        threshold=reference_threshold,
        score_type="probability_estimates",
    )
    recomputed_comparison = build_comparison_document(
        reference_threshold_basis_points=reference_bp,
        selected_threshold_basis_points=selection.selected_threshold_basis_points,
        denominator=frozen_config.threshold_grid.denominator,
        validation_reference_metrics=reference_val_metrics,
        validation_selected_metrics=selection.selected_validation_metrics,
        test_reference_metrics=reference_test_metrics,
        test_selected_metrics=recomputed_test_metrics,
    )
    if comparison.model_dump(mode="json") != recomputed_comparison.model_dump(mode="json"):
        raise ThresholdPolicyCorruptionError("comparison.json does not match recomputed values")

    return manifest


def publish_threshold_policy(staging_dir: Path, final_dir: Path) -> None:
    if final_dir.exists():
        raise ThresholdPolicyBuildError(
            f"Refusing to overwrite existing threshold-policy artifact at {final_dir}"
        )
    os.rename(staging_dir, final_dir)


def _run_validation_only_pipeline(
    *,
    config: ThresholdPolicyConfigDocument,
    baseline_dir: Path,
    baseline_manifest: BaselineManifest,
    expected_repository: RepositoryRef,
) -> tuple[ValidationScoreBundle, list[ThresholdSweepRow], ThresholdSelectionResult]:
    val_bundle = load_validation_scores(
        baseline_dir,
        baseline_manifest=baseline_manifest,
        candidate_id=config.selected_candidate_id,
        expected_repository=expected_repository,
    )
    sweep_rows = build_threshold_sweep(
        labels=val_bundle.labels,
        y_true=val_bundle.y_true,
        y_score=val_bundle.y_score,
        grid=config.threshold_grid,
    )
    selection = select_threshold(
        sweep=sweep_rows,
        reference_threshold_basis_points=config.reference_threshold_basis_points,
        selection_rule_version=config.selection_rule_version,
        denominator=config.threshold_grid.denominator,
    )
    return val_bundle, sweep_rows, selection


def build_threshold_policy(
    repository: RepositoryRef,
    config_path: Path,
    *,
    baselines_root: Path = DEFAULT_BASELINES_ROOT,
    threshold_policies_root: Path = DEFAULT_THRESHOLD_POLICIES_ROOT,
) -> ThresholdPolicyBuildResult:
    """Build or reuse one immutable global threshold-policy artifact."""
    config, _config_bytes, config_source_hash, config_semantic_hash = load_threshold_policy_config(
        config_path
    )
    if config.repository != repository.full_name:
        raise ThresholdPolicyInputError(
            f"Config repository {config.repository!r} does not match requested "
            f"repository {repository.full_name!r}."
        )

    baseline_dir = baselines_root / repository.slug / config.baseline_run_id
    if not baseline_dir.is_dir():
        raise ThresholdPolicyInputError(
            f"No baseline artifact found for {repository.full_name} with baseline run id "
            f"{config.baseline_run_id!r} at {baseline_dir}."
        )

    baseline_manifest = validate_baseline_artifact_integrity(
        baseline_dir,
        expected_repository=repository,
        expected_baseline_run_id=config.baseline_run_id,
    )

    if baseline_manifest.selected_candidate_id != config.selected_candidate_id:
        raise ThresholdPolicyInputError(
            f"Config selected_candidate_id {config.selected_candidate_id!r} does not match "
            f"baseline selected candidate {baseline_manifest.selected_candidate_id!r}."
        )

    policy_input_sha256 = compute_policy_input_sha256(
        threshold_policy_version=config.threshold_policy_version,
        baseline_run_id=config.baseline_run_id,
        baseline_experiment_sha256=baseline_manifest.baseline_experiment_sha256,
        model_semantic_sha256=baseline_manifest.model_semantic_sha256,
        predictions_validation_sha256=baseline_manifest.predictions_validation_sha256,
        predictions_test_sha256=baseline_manifest.predictions_test_sha256,
        selected_candidate_id=config.selected_candidate_id,
        threshold_grid=config.threshold_grid,
        selection_rule_version=config.selection_rule_version,
        metric_contract_version=config.metric_contract_version,
    )
    policy_id = compute_policy_id(config.baseline_run_id, policy_input_sha256)
    final_dir = threshold_policies_root / repository.slug / policy_id

    if final_dir.exists():
        manifest = validate_threshold_policy_against_baseline(
            final_dir,
            baseline_dir,
            expected_repository=repository,
        )
        comparison = ComparisonDocument.model_validate_json(
            (final_dir / COMPARISON_JSON_FILE).read_text(encoding="utf-8")
        )
        logger.info("Threshold-policy cache hit for %s at %s", repository.full_name, final_dir)
        return ThresholdPolicyBuildResult(
            repository=repository,
            policy_dir=final_dir,
            manifest=manifest,
            selected_threshold=manifest.selected_threshold_basis_points
            / config.threshold_grid.denominator,
            selected_threshold_basis_points=manifest.selected_threshold_basis_points,
            validation_macro_f1_at_reference=comparison.validation.reference.macro_f1,
            selected_validation_macro_f1=comparison.validation.selected.macro_f1,
            test_macro_f1_at_reference=comparison.test.reference.macro_f1,
            selected_test_macro_f1=comparison.test.selected.macro_f1,
            cache_hit=True,
        )

    val_bundle, sweep_rows, selection = _run_validation_only_pipeline(
        config=config,
        baseline_dir=baseline_dir,
        baseline_manifest=baseline_manifest,
        expected_repository=repository,
    )
    frozen = freeze_threshold_policy(selection)

    test_bundle = load_test_scores(
        baseline_dir,
        baseline_manifest=baseline_manifest,
        candidate_id=config.selected_candidate_id,
        expected_repository=repository,
    )
    test_metrics = evaluate_frozen_threshold(frozen=frozen, bundle=test_bundle)

    reference_bp = config.reference_threshold_basis_points
    reference_threshold = reference_bp / config.threshold_grid.denominator
    reference_val_row = _sweep_row_for_basis_points(sweep_rows, reference_bp)
    reference_test_pred = predictions_from_scores(test_bundle.y_score, reference_threshold)
    reference_test_metrics = compute_split_metrics(
        split="test",
        labels=test_bundle.labels,
        y_true=test_bundle.y_true,
        y_pred=reference_test_pred,
        y_score=test_bundle.y_score,
        threshold=reference_threshold,
        score_type="probability_estimates",
    )
    comparison = build_comparison_document(
        reference_threshold_basis_points=reference_bp,
        selected_threshold_basis_points=selection.selected_threshold_basis_points,
        denominator=config.threshold_grid.denominator,
        validation_reference_metrics=reference_val_row.metrics,
        validation_selected_metrics=selection.selected_validation_metrics,
        test_reference_metrics=reference_test_metrics,
        test_selected_metrics=test_metrics,
    )

    frozen_config = FrozenThresholdPolicyConfig(
        repository=config.repository,
        baseline_run_id=config.baseline_run_id,
        selected_candidate_id=config.selected_candidate_id,
        metric_contract_version=config.metric_contract_version,
        selection_rule_version=config.selection_rule_version,
        reference_threshold_basis_points=config.reference_threshold_basis_points,
        threshold_grid=config.threshold_grid,
    )
    policy_document = PolicyDocument(selection=selection.selection_audit)
    sweep_document = SweepValidationDocument(
        threshold_grid=config.threshold_grid,
        rows=sweep_rows,
    )

    config_json = serialize_config_json(frozen_config)
    policy_json = serialize_policy_json(policy_document)
    sweep_json = serialize_sweep_validation_json(sweep_document)
    metrics_validation_json = serialize_metrics_json(selection.selected_validation_metrics)
    metrics_test_json = serialize_metrics_json(test_metrics)
    comparison_json = serialize_comparison_json(comparison)
    report_md = serialize_report_markdown(
        repository=config.repository,
        baseline_run_id=config.baseline_run_id,
        selected_candidate_id=config.selected_candidate_id,
        policy_id=policy_id,
        reference_threshold_basis_points=reference_bp,
        selected_threshold_basis_points=selection.selected_threshold_basis_points,
        denominator=config.threshold_grid.denominator,
        sweep_rows=sweep_rows,
        comparison=comparison,
        grid=config.threshold_grid,
    )

    built_at = datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")
    manifest = ThresholdPolicyManifest(
        policy_id=policy_id,
        policy_input_sha256=policy_input_sha256,
        config_source_sha256=config_source_hash,
        config_semantic_sha256=config_semantic_hash,
        repository=config.repository,
        baseline_run_id=config.baseline_run_id,
        baseline_experiment_sha256=baseline_manifest.baseline_experiment_sha256,
        model_dataset_id=baseline_manifest.model_dataset_id,
        model_semantic_sha256=baseline_manifest.model_semantic_sha256,
        selected_candidate_id=config.selected_candidate_id,
        predictions_validation_sha256=baseline_manifest.predictions_validation_sha256,
        predictions_test_sha256=baseline_manifest.predictions_test_sha256,
        metric_contract_version=config.metric_contract_version,
        selection_rule_version=config.selection_rule_version,
        reference_threshold_basis_points=reference_bp,
        selected_threshold_basis_points=selection.selected_threshold_basis_points,
        validation_record_count=val_bundle.record_count,
        test_record_count=test_bundle.record_count,
        target_count=val_bundle.target_count,
        sweep_threshold_count=len(sweep_rows),
        built_at=built_at,
        config_sha256=sha256_hex(config_json),
        policy_sha256=sha256_hex(policy_json),
        sweep_validation_sha256=sha256_hex(sweep_json),
        metrics_validation_sha256=sha256_hex(metrics_validation_json),
        metrics_test_sha256=sha256_hex(metrics_test_json),
        comparison_sha256=sha256_hex(comparison_json),
        report_sha256=sha256_hex(report_md),
    )
    manifest_json = serialize_manifest_json(manifest)

    slug_dir = threshold_policies_root / repository.slug
    slug_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=".staging-", dir=str(slug_dir))
    )

    try:
        atomic_write_bytes(staging_dir / CONFIG_JSON_FILE, config_json)
        atomic_write_bytes(staging_dir / POLICY_JSON_FILE, policy_json)
        atomic_write_bytes(staging_dir / SWEEP_VALIDATION_JSON_FILE, sweep_json)
        atomic_write_bytes(staging_dir / METRICS_VALIDATION_JSON_FILE, metrics_validation_json)
        atomic_write_bytes(staging_dir / METRICS_TEST_JSON_FILE, metrics_test_json)
        atomic_write_bytes(staging_dir / COMPARISON_JSON_FILE, comparison_json)
        atomic_write_bytes(staging_dir / REPORT_MARKDOWN_FILE, report_md)
        atomic_write_bytes(staging_dir / MANIFEST_JSON_FILE, manifest_json)

        validate_threshold_policy_artifact_integrity(
            staging_dir,
            expected_repository=repository,
            expected_policy_id=policy_id,
            check_dir_name=False,
        )
        validate_threshold_policy_against_baseline(
            staging_dir,
            baseline_dir,
            expected_repository=repository,
            check_dir_name=False,
        )
        publish_threshold_policy(staging_dir, final_dir)
    except Exception:
        best_effort_remove_tree(staging_dir)
        raise

    return ThresholdPolicyBuildResult(
        repository=repository,
        policy_dir=final_dir,
        manifest=manifest,
        selected_threshold=selection.selected_threshold,
        selected_threshold_basis_points=selection.selected_threshold_basis_points,
        validation_macro_f1_at_reference=comparison.validation.reference.macro_f1,
        selected_validation_macro_f1=comparison.validation.selected.macro_f1,
        test_macro_f1_at_reference=comparison.test.reference.macro_f1,
        selected_test_macro_f1=comparison.test.selected.macro_f1,
        cache_hit=False,
    )


def format_threshold_policy_summary(result: ThresholdPolicyBuildResult) -> str:
    lines = [
        f"repository: {result.repository.full_name}",
        f"baseline_run_id: {result.manifest.baseline_run_id}",
        f"threshold_policy_id: {result.manifest.policy_id}",
        f"selected_threshold: {result.selected_threshold:.2f}",
        f"validation_macro_f1_at_0.50: "
        f"{result.validation_macro_f1_at_reference:.6f}"
        if result.validation_macro_f1_at_reference is not None
        else "validation_macro_f1_at_0.50: n/a",
        f"selected_validation_macro_f1: "
        f"{result.selected_validation_macro_f1:.6f}"
        if result.selected_validation_macro_f1 is not None
        else "selected_validation_macro_f1: n/a",
        f"test_macro_f1_at_0.50: "
        f"{result.test_macro_f1_at_reference:.6f}"
        if result.test_macro_f1_at_reference is not None
        else "test_macro_f1_at_0.50: n/a",
        f"selected_test_macro_f1: "
        f"{result.selected_test_macro_f1:.6f}"
        if result.selected_test_macro_f1 is not None
        else "selected_test_macro_f1: n/a",
        f"artifact_path: {result.policy_dir}",
        f"cache_hit: {'true' if result.cache_hit else 'false'}",
    ]
    return "\n".join(lines)
