"""Build (or reuse) an immutable abstention-policy artifact."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from repotriage.abstention_policy.confidence import (
    IssueConfidenceTable,
    build_issue_confidence_table,
)
from repotriage.abstention_policy.config import load_abstention_policy_config
from repotriage.abstention_policy.evaluator import (
    compute_confidence_bins,
    evaluate_frozen_abstention_split,
)
from repotriage.abstention_policy.models import (
    COMPARISON_JSON_FILE,
    CONFIDENCE_BINS_TEST_JSON_FILE,
    CONFIDENCE_BINS_VALIDATION_JSON_FILE,
    CONFIG_JSON_FILE,
    MANIFEST_JSON_FILE,
    METRICS_TEST_JSON_FILE,
    METRICS_VALIDATION_JSON_FILE,
    POLICY_JSON_FILE,
    REPORT_MARKDOWN_FILE,
    SWEEP_VALIDATION_JSON_FILE,
    AbstentionPolicyBuildError,
    AbstentionPolicyConfigDocument,
    AbstentionPolicyCorruptionError,
    AbstentionPolicyInputError,
    AbstentionPolicyManifest,
    AbstentionSplitMetrics,
    AbstentionSweepRow,
    ComparisonDocument,
    FrozenAbstentionPolicyConfig,
    HandledMetrics,
    PolicyDocument,
    SweepValidationDocument,
    compute_policy_id,
    compute_policy_input_sha256,
    handled_metrics_from_split_metrics,
)
from repotriage.abstention_policy.reader import (
    ThresholdPolicyInputs,
    load_test_scores,
    load_threshold_policy_inputs,
    load_validation_scores,
)
from repotriage.abstention_policy.report import (
    build_comparison_document,
    serialize_comparison_json,
    serialize_confidence_bins_json,
    serialize_config_json,
    serialize_manifest_json,
    serialize_policy_json,
    serialize_report_markdown,
    serialize_split_metrics_json,
    serialize_sweep_validation_json,
    sha256_hex,
)
from repotriage.abstention_policy.selector import (
    AbstentionSelectionResult,
    freeze_abstention_policy,
    select_abstention_threshold,
)
from repotriage.abstention_policy.sweep import build_abstention_sweep
from repotriage.baseline.builder import validate_baseline_artifact_integrity
from repotriage.baseline.evaluator import compute_split_metrics
from repotriage.baseline.models import BaselineManifest, floats_consistent
from repotriage.filesystem import atomic_write_bytes, best_effort_remove_tree
from repotriage.github.models import RepositoryRef
from repotriage.paths import resolve_within_directory
from repotriage.threshold_policy.builder import validate_threshold_policy_artifact_integrity
from repotriage.threshold_policy.models import ThresholdGridConfig
from repotriage.threshold_policy.reader import ValidationScoreBundle

logger = logging.getLogger(__name__)

DEFAULT_BASELINES_ROOT = Path("data/baselines/github")
DEFAULT_THRESHOLD_POLICIES_ROOT = Path("data/threshold_policies/github")
DEFAULT_ABSTENTION_POLICIES_ROOT = Path("data/abstention_policies/github")


@dataclass(frozen=True)
class AbstentionPolicyBuildResult:
    repository: RepositoryRef
    policy_dir: Path
    manifest: AbstentionPolicyManifest
    classification_threshold: float
    classification_threshold_basis_points: int
    selected_abstention_threshold: float
    selected_abstention_basis_points: int
    validation_coverage: float
    validation_handled_subset_accuracy: float | None
    test_coverage: float
    test_handled_subset_accuracy: float | None
    cache_hit: bool


def _load_manifest(policy_dir: Path) -> AbstentionPolicyManifest:
    manifest_path = resolve_within_directory(policy_dir, MANIFEST_JSON_FILE)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AbstentionPolicyCorruptionError(
            f"Unable to read abstention-policy manifest at {manifest_path}: {exc}"
        ) from exc
    try:
        return AbstentionPolicyManifest.model_validate(payload)
    except ValidationError as exc:
        raise AbstentionPolicyCorruptionError(
            f"Invalid abstention-policy manifest at {manifest_path}: {exc}"
        ) from exc


def _verify_file_hash(path: Path, expected_sha256: str) -> bytes:
    if not path.is_file():
        raise AbstentionPolicyCorruptionError(f"Missing artifact file: {path}")
    data = path.read_bytes()
    actual = sha256_hex(data)
    if actual != expected_sha256:
        raise AbstentionPolicyCorruptionError(
            f"Hash mismatch for {path.name}: expected {expected_sha256}, got {actual}"
        )
    return data


def _handled_metrics_close(left: HandledMetrics, right: HandledMetrics) -> None:
    for field in (
        "subset_accuracy",
        "samples_f1",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "mean_predicted_label_cardinality",
        "mean_true_label_cardinality",
    ):
        left_value = getattr(left, field)
        right_value = getattr(right, field)
        if left_value is None and right_value is None:
            continue
        if left_value is None or right_value is None:
            raise AbstentionPolicyCorruptionError(
                f"Handled metric {field} mismatch: {left_value} vs {right_value}"
            )
        if not floats_consistent(left_value, right_value):
            raise AbstentionPolicyCorruptionError(
                f"Handled metric {field} mismatch: {left_value} vs {right_value}"
            )
    for field in ("false_positive_count", "false_negative_count"):
        if getattr(left, field) != getattr(right, field):
            raise AbstentionPolicyCorruptionError(
                f"Handled metric {field} mismatch: {getattr(left, field)} vs "
                f"{getattr(right, field)}"
            )


def _sweep_row_for_basis_points(
    rows: list[AbstentionSweepRow], basis_points: int
) -> AbstentionSweepRow:
    for row in rows:
        if row.abstention_basis_points == basis_points:
            return row
    raise AbstentionPolicyCorruptionError(
        f"Selected abstention basis points {basis_points} not found in sweep"
    )


def _validate_sweep_ordering(rows: list[AbstentionSweepRow]) -> None:
    basis_points = [row.abstention_basis_points for row in rows]
    if basis_points != sorted(basis_points):
        raise AbstentionPolicyCorruptionError(
            "Sweep rows are not sorted by abstention_basis_points"
        )
    if len(set(basis_points)) != len(basis_points):
        raise AbstentionPolicyCorruptionError("Duplicate abstention_basis_points in sweep")


def _split_metrics_close(left: AbstentionSplitMetrics, right: AbstentionSplitMetrics) -> None:
    for field in (
        "total_count",
        "handled_count",
        "abstained_count",
        "forced_abstention_count",
        "coverage",
        "abstention_rate",
    ):
        if getattr(left, field) != getattr(right, field):
            raise AbstentionPolicyCorruptionError(
                f"Split metric {field} mismatch: {getattr(left, field)} vs {getattr(right, field)}"
            )
    _handled_metrics_close(left.handled_metrics, right.handled_metrics)


def publish_abstention_policy(staging_dir: Path, final_dir: Path) -> None:
    if final_dir.exists():
        raise AbstentionPolicyBuildError(
            f"Refusing to overwrite existing abstention-policy artifact at {final_dir}"
        )
    os.rename(staging_dir, final_dir)


def validate_abstention_policy_artifact_integrity(
    policy_dir: Path,
    *,
    expected_repository: RepositoryRef,
    expected_policy_id: str | None = None,
    check_dir_name: bool = True,
) -> AbstentionPolicyManifest:
    """Validate an abstention-policy artifact using only on-disk bytes."""
    if not policy_dir.is_dir():
        raise AbstentionPolicyCorruptionError(
            f"Abstention-policy directory does not exist: {policy_dir}"
        )

    manifest = _load_manifest(policy_dir)
    if check_dir_name and policy_dir.name != manifest.policy_id:
        raise AbstentionPolicyCorruptionError(
            f"Directory name {policy_dir.name!r} does not match policy_id {manifest.policy_id!r}."
        )
    if expected_policy_id is not None and manifest.policy_id != expected_policy_id:
        raise AbstentionPolicyCorruptionError(
            f"Manifest policy_id {manifest.policy_id!r} does not match expected "
            f"{expected_policy_id!r}."
        )
    if manifest.repository != expected_repository.full_name:
        raise AbstentionPolicyCorruptionError(
            f"Manifest repository {manifest.repository!r} does not match expected "
            f"{expected_repository.full_name!r}."
        )

    config_bytes = _verify_file_hash(
        resolve_within_directory(policy_dir, manifest.config_file),
        manifest.config_sha256,
    )
    frozen_config = FrozenAbstentionPolicyConfig.model_validate_json(config_bytes)
    expected_input = compute_policy_input_sha256(
        abstention_policy_version=manifest.abstention_policy_version,
        baseline_run_id=manifest.baseline_run_id,
        baseline_experiment_sha256=manifest.baseline_experiment_sha256,
        model_semantic_sha256=manifest.model_semantic_sha256,
        predictions_validation_sha256=manifest.predictions_validation_sha256,
        predictions_test_sha256=manifest.predictions_test_sha256,
        threshold_policy_id=manifest.threshold_policy_id,
        threshold_policy_sha256=manifest.threshold_policy_sha256,
        classification_threshold_basis_points=manifest.classification_threshold_basis_points,
        confidence_definition=manifest.confidence_definition,
        abstention_grid=frozen_config.abstention_grid,
        minimum_coverage=manifest.minimum_coverage,
        selection_rule_version=manifest.selection_rule_version,
        metric_contract_version=manifest.metric_contract_version,
    )
    if manifest.policy_input_sha256 != expected_input:
        raise AbstentionPolicyCorruptionError("manifest policy_input_sha256 mismatch")
    expected_id = compute_policy_id(manifest.threshold_policy_id, expected_input)
    if manifest.policy_id != expected_id:
        raise AbstentionPolicyCorruptionError("manifest policy_id mismatch")

    for relative_path, expected_sha256 in (
        (manifest.policy_file, manifest.policy_sha256),
        (manifest.sweep_validation_file, manifest.sweep_validation_sha256),
        (manifest.metrics_validation_file, manifest.metrics_validation_sha256),
        (manifest.metrics_test_file, manifest.metrics_test_sha256),
        (manifest.confidence_bins_validation_file, manifest.confidence_bins_validation_sha256),
        (manifest.confidence_bins_test_file, manifest.confidence_bins_test_sha256),
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
        policy_document.selection.selected_abstention_basis_points
        != manifest.selected_abstention_basis_points
    ):
        raise AbstentionPolicyCorruptionError(
            "policy.json selected abstention threshold does not match manifest"
        )

    sweep_bytes = _verify_file_hash(
        resolve_within_directory(policy_dir, manifest.sweep_validation_file),
        manifest.sweep_validation_sha256,
    )
    sweep_document = SweepValidationDocument.model_validate_json(sweep_bytes)
    _validate_sweep_ordering(sweep_document.rows)
    if len(sweep_document.rows) != manifest.sweep_threshold_count:
        raise AbstentionPolicyCorruptionError("Sweep row count does not match manifest")
    if sweep_document.abstention_grid != frozen_config.abstention_grid:
        raise AbstentionPolicyCorruptionError("Sweep grid does not match frozen config")

    selected_row = _sweep_row_for_basis_points(
        sweep_document.rows, manifest.selected_abstention_basis_points
    )
    metrics_validation_bytes = _verify_file_hash(
        resolve_within_directory(policy_dir, manifest.metrics_validation_file),
        manifest.metrics_validation_sha256,
    )
    stored_validation_metrics = AbstentionSplitMetrics.model_validate_json(metrics_validation_bytes)
    classification_threshold = (
        manifest.classification_threshold_basis_points / frozen_config.abstention_grid.denominator
    )
    _split_metrics_close(
        AbstentionSplitMetrics(
            split="validation",
            classification_threshold=classification_threshold,
            abstention_threshold=selected_row.abstention_threshold,
            total_count=selected_row.total_count,
            handled_count=selected_row.handled_count,
            abstained_count=selected_row.abstained_count,
            forced_abstention_count=selected_row.forced_abstention_count,
            coverage=selected_row.coverage,
            abstention_rate=selected_row.abstention_rate,
            handled_metrics=selected_row.handled_metrics,
        ),
        stored_validation_metrics,
    )

    comparison = ComparisonDocument.model_validate_json(
        _verify_file_hash(
            resolve_within_directory(policy_dir, manifest.comparison_file),
            manifest.comparison_sha256,
        )
    )
    if comparison.selected_abstention_basis_points != manifest.selected_abstention_basis_points:
        raise AbstentionPolicyCorruptionError("comparison selected threshold mismatch")
    val_sel = comparison.validation.selected_abstention_handled
    if val_sel.handled_count != selected_row.handled_count:
        raise AbstentionPolicyCorruptionError("comparison handled_count mismatch")
    if not floats_consistent(val_sel.coverage, selected_row.coverage):
        raise AbstentionPolicyCorruptionError("comparison coverage mismatch")
    if (
        val_sel.subset_accuracy is not None
        and selected_row.handled_metrics.subset_accuracy is not None
    ):
        if not floats_consistent(
            val_sel.subset_accuracy, selected_row.handled_metrics.subset_accuracy
        ):
            raise AbstentionPolicyCorruptionError("comparison subset_accuracy mismatch")

    report_text = resolve_within_directory(policy_dir, manifest.report_file).read_text(
        encoding="utf-8"
    )
    denominator = frozen_config.abstention_grid.denominator
    classification_threshold = manifest.classification_threshold_basis_points / denominator
    selected_abstention_threshold = manifest.selected_abstention_basis_points / denominator
    if f"**{classification_threshold:.2f}**" not in report_text:
        raise AbstentionPolicyCorruptionError(
            "report.md does not reference classification threshold"
        )
    if f"**{selected_abstention_threshold:.2f}**" not in report_text:
        raise AbstentionPolicyCorruptionError(
            "report.md does not reference selected abstention threshold"
        )
    if f">= {manifest.minimum_coverage:.2f}" not in report_text:
        raise AbstentionPolicyCorruptionError("report.md does not reference minimum coverage")

    return manifest


def _full_set_classification_metrics(
    *,
    split: str,
    labels: list[str],
    y_true,
    y_pred,
    y_score,
    classification_threshold: float,
) -> HandledMetrics:
    metrics = compute_split_metrics(
        split=split,
        labels=labels,
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        threshold=classification_threshold,
        score_type="probability_estimates",
    )
    return handled_metrics_from_split_metrics(metrics)


def _run_validation_only_pipeline(
    *,
    config: AbstentionPolicyConfigDocument,
    baseline_dir: Path,
    baseline_manifest: BaselineManifest,
    threshold_inputs: ThresholdPolicyInputs,
    expected_repository: RepositoryRef,
    resolved_grid: ThresholdGridConfig,
    classification_threshold_basis_points: int,
) -> tuple[
    ValidationScoreBundle, IssueConfidenceTable, list[AbstentionSweepRow], AbstentionSelectionResult
]:
    classification_threshold = classification_threshold_basis_points / resolved_grid.denominator
    val_bundle = load_validation_scores(
        baseline_dir,
        baseline_manifest=baseline_manifest,
        candidate_id=threshold_inputs.manifest.selected_candidate_id,
        expected_repository=expected_repository,
    )
    table, sweep_rows = build_abstention_sweep(
        labels=val_bundle.labels,
        y_true=val_bundle.y_true,
        y_score=val_bundle.y_score,
        issue_ids=val_bundle.issue_ids,
        classification_threshold=classification_threshold,
        grid=resolved_grid,
        confidence_definition=config.confidence_definition,
    )
    selection = select_abstention_threshold(
        sweep=sweep_rows,
        minimum_coverage=config.minimum_coverage,
        classification_threshold_basis_points=classification_threshold_basis_points,
        selection_rule_version=config.selection_rule_version,
        denominator=resolved_grid.denominator,
    )
    return val_bundle, table, sweep_rows, selection


def validate_abstention_policy_against_inputs(
    policy_dir: Path,
    baseline_dir: Path,
    threshold_policy_dir: Path,
    *,
    expected_repository: RepositoryRef,
    check_dir_name: bool = True,
) -> AbstentionPolicyManifest:
    """Recompute abstention selection and metrics from upstream inputs."""
    manifest = validate_abstention_policy_artifact_integrity(
        policy_dir,
        expected_repository=expected_repository,
        check_dir_name=check_dir_name,
    )
    baseline_manifest = validate_baseline_artifact_integrity(
        baseline_dir,
        expected_repository=expected_repository,
        expected_baseline_run_id=manifest.baseline_run_id,
    )
    threshold_manifest = validate_threshold_policy_artifact_integrity(
        threshold_policy_dir,
        expected_repository=expected_repository,
        expected_policy_id=manifest.threshold_policy_id,
    )

    if baseline_manifest.baseline_experiment_sha256 != manifest.baseline_experiment_sha256:
        raise AbstentionPolicyCorruptionError("baseline_experiment_sha256 mismatch")
    if baseline_manifest.model_semantic_sha256 != manifest.model_semantic_sha256:
        raise AbstentionPolicyCorruptionError("model_semantic_sha256 mismatch")
    if baseline_manifest.predictions_validation_sha256 != manifest.predictions_validation_sha256:
        raise AbstentionPolicyCorruptionError("predictions_validation_sha256 mismatch")
    if baseline_manifest.predictions_test_sha256 != manifest.predictions_test_sha256:
        raise AbstentionPolicyCorruptionError("predictions_test_sha256 mismatch")
    if baseline_manifest.selected_candidate_id != manifest.selected_candidate_id:
        raise AbstentionPolicyCorruptionError("selected_candidate_id mismatch")
    if threshold_manifest.policy_sha256 != manifest.threshold_policy_sha256:
        raise AbstentionPolicyCorruptionError("threshold_policy_sha256 mismatch")
    if (
        threshold_manifest.selected_threshold_basis_points
        != manifest.classification_threshold_basis_points
    ):
        raise AbstentionPolicyCorruptionError("classification threshold mismatch")

    config_bytes = resolve_within_directory(policy_dir, manifest.config_file).read_bytes()
    frozen_config = FrozenAbstentionPolicyConfig.model_validate_json(config_bytes)

    classification_threshold = (
        manifest.classification_threshold_basis_points / frozen_config.abstention_grid.denominator
    )
    val_bundle = load_validation_scores(
        baseline_dir,
        baseline_manifest=baseline_manifest,
        candidate_id=manifest.selected_candidate_id,
        expected_repository=expected_repository,
    )
    val_table, recomputed_sweep = build_abstention_sweep(
        labels=val_bundle.labels,
        y_true=val_bundle.y_true,
        y_score=val_bundle.y_score,
        issue_ids=val_bundle.issue_ids,
        classification_threshold=classification_threshold,
        grid=frozen_config.abstention_grid,
        confidence_definition=manifest.confidence_definition,
    )
    recomputed_selection = select_abstention_threshold(
        sweep=recomputed_sweep,
        minimum_coverage=manifest.minimum_coverage,
        classification_threshold_basis_points=manifest.classification_threshold_basis_points,
        selection_rule_version=manifest.selection_rule_version,
        denominator=frozen_config.abstention_grid.denominator,
    )
    if (
        recomputed_selection.selected_abstention_basis_points
        != manifest.selected_abstention_basis_points
    ):
        raise AbstentionPolicyCorruptionError("Recomputed selected abstention threshold mismatch")

    stored_sweep = SweepValidationDocument.model_validate_json(
        (policy_dir / manifest.sweep_validation_file).read_text(encoding="utf-8")
    )
    if len(stored_sweep.rows) != len(recomputed_sweep):
        raise AbstentionPolicyCorruptionError("Recomputed sweep row count mismatch")
    for stored_row, recomputed_row in zip(stored_sweep.rows, recomputed_sweep, strict=True):
        if stored_row.abstention_basis_points != recomputed_row.abstention_basis_points:
            raise AbstentionPolicyCorruptionError("Recomputed sweep ordering mismatch")
        _handled_metrics_close(stored_row.handled_metrics, recomputed_row.handled_metrics)

    test_bundle = load_test_scores(
        baseline_dir,
        baseline_manifest=baseline_manifest,
        candidate_id=manifest.selected_candidate_id,
        expected_repository=expected_repository,
    )
    test_table = build_issue_confidence_table(
        issue_ids=test_bundle.issue_ids,
        y_score=test_bundle.y_score,
        classification_threshold=classification_threshold,
        confidence_definition=manifest.confidence_definition,
    )
    frozen = freeze_abstention_policy(
        selection=recomputed_selection,
        classification_threshold_basis_points=manifest.classification_threshold_basis_points,
        denominator=frozen_config.abstention_grid.denominator,
    )
    full_set_val = _full_set_classification_metrics(
        split="validation",
        labels=val_bundle.labels,
        y_true=val_bundle.y_true,
        y_pred=val_table.y_pred,
        y_score=val_bundle.y_score,
        classification_threshold=classification_threshold,
    )
    full_set_test = _full_set_classification_metrics(
        split="test",
        labels=test_bundle.labels,
        y_true=test_bundle.y_true,
        y_pred=test_table.y_pred,
        y_score=test_bundle.y_score,
        classification_threshold=classification_threshold,
    )
    recomputed_val_metrics = evaluate_frozen_abstention_split(
        split="validation",
        labels=val_bundle.labels,
        y_true=val_bundle.y_true,
        y_score=val_bundle.y_score,
        table=val_table,
        classification_threshold=classification_threshold,
        abstention_threshold=frozen.selected_abstention_threshold,
        full_set_reference=full_set_val,
    )
    recomputed_test_metrics = evaluate_frozen_abstention_split(
        split="test",
        labels=test_bundle.labels,
        y_true=test_bundle.y_true,
        y_score=test_bundle.y_score,
        table=test_table,
        classification_threshold=classification_threshold,
        abstention_threshold=frozen.selected_abstention_threshold,
        full_set_reference=full_set_test,
    )
    stored_validation_metrics = AbstentionSplitMetrics.model_validate_json(
        (policy_dir / manifest.metrics_validation_file).read_text(encoding="utf-8")
    )
    _split_metrics_close(recomputed_val_metrics, stored_validation_metrics)
    stored_test_metrics = AbstentionSplitMetrics.model_validate_json(
        (policy_dir / manifest.metrics_test_file).read_text(encoding="utf-8")
    )
    _split_metrics_close(recomputed_test_metrics, stored_test_metrics)

    recomputed_val_bins = compute_confidence_bins(
        split="validation",
        labels=val_bundle.labels,
        y_true=val_bundle.y_true,
        y_score=val_bundle.y_score,
        issue_ids=val_bundle.issue_ids,
        classification_threshold=classification_threshold,
        confidence_definition=manifest.confidence_definition,
    )
    recomputed_test_bins = compute_confidence_bins(
        split="test",
        labels=test_bundle.labels,
        y_true=test_bundle.y_true,
        y_score=test_bundle.y_score,
        issue_ids=test_bundle.issue_ids,
        classification_threshold=classification_threshold,
        confidence_definition=manifest.confidence_definition,
    )
    stored_val_bins = json.loads(
        (policy_dir / manifest.confidence_bins_validation_file).read_text(encoding="utf-8")
    )
    stored_test_bins = json.loads(
        (policy_dir / manifest.confidence_bins_test_file).read_text(encoding="utf-8")
    )
    if stored_val_bins != recomputed_val_bins.model_dump(mode="json"):
        raise AbstentionPolicyCorruptionError("confidence_bins_validation mismatch")
    if stored_test_bins != recomputed_test_bins.model_dump(mode="json"):
        raise AbstentionPolicyCorruptionError("confidence_bins_test mismatch")

    return manifest


def build_abstention_policy(
    repository: RepositoryRef,
    config_path: Path,
    *,
    threshold_policy_id: str,
    baselines_root: Path = DEFAULT_BASELINES_ROOT,
    threshold_policies_root: Path = DEFAULT_THRESHOLD_POLICIES_ROOT,
    abstention_policies_root: Path = DEFAULT_ABSTENTION_POLICIES_ROOT,
) -> AbstentionPolicyBuildResult:
    """Build or reuse one immutable abstention-policy artifact."""
    config, _config_bytes, config_source_hash, config_semantic_hash = load_abstention_policy_config(
        config_path
    )
    if config.repository != repository.full_name:
        raise AbstentionPolicyInputError(
            f"Config repository {config.repository!r} does not match requested "
            f"repository {repository.full_name!r}."
        )
    if config.threshold_policy_id != threshold_policy_id:
        raise AbstentionPolicyInputError(
            f"Config threshold_policy_id {config.threshold_policy_id!r} does not match "
            f"requested {threshold_policy_id!r}."
        )

    threshold_policy_dir = threshold_policies_root / repository.slug / threshold_policy_id
    if not threshold_policy_dir.is_dir():
        raise AbstentionPolicyInputError(
            f"No threshold-policy artifact found at {threshold_policy_dir}."
        )

    threshold_inputs = load_threshold_policy_inputs(
        threshold_policy_dir,
        expected_policy_id=threshold_policy_id,
        expected_repository=repository,
    )
    classification_threshold_basis_points = (
        threshold_inputs.manifest.selected_threshold_basis_points
    )
    resolved_grid = config.abstention_grid.resolve_grid(
        classification_threshold_basis_points=classification_threshold_basis_points
    )

    baseline_dir = baselines_root / repository.slug / threshold_inputs.manifest.baseline_run_id
    if not baseline_dir.is_dir():
        raise AbstentionPolicyInputError(f"No baseline artifact found at {baseline_dir}.")

    baseline_manifest = validate_baseline_artifact_integrity(
        baseline_dir,
        expected_repository=repository,
        expected_baseline_run_id=threshold_inputs.manifest.baseline_run_id,
    )

    policy_input_sha256 = compute_policy_input_sha256(
        abstention_policy_version=config.abstention_policy_version,
        baseline_run_id=threshold_inputs.manifest.baseline_run_id,
        baseline_experiment_sha256=baseline_manifest.baseline_experiment_sha256,
        model_semantic_sha256=baseline_manifest.model_semantic_sha256,
        predictions_validation_sha256=baseline_manifest.predictions_validation_sha256,
        predictions_test_sha256=baseline_manifest.predictions_test_sha256,
        threshold_policy_id=threshold_policy_id,
        threshold_policy_sha256=threshold_inputs.policy_sha256,
        classification_threshold_basis_points=classification_threshold_basis_points,
        confidence_definition=config.confidence_definition,
        abstention_grid=resolved_grid,
        minimum_coverage=config.minimum_coverage,
        selection_rule_version=config.selection_rule_version,
        metric_contract_version=config.metric_contract_version,
    )
    policy_id = compute_policy_id(threshold_policy_id, policy_input_sha256)
    final_dir = abstention_policies_root / repository.slug / policy_id

    if final_dir.exists():
        manifest = validate_abstention_policy_against_inputs(
            final_dir,
            baseline_dir,
            threshold_policy_dir,
            expected_repository=repository,
        )
        metrics_validation = AbstentionSplitMetrics.model_validate_json(
            (final_dir / METRICS_VALIDATION_JSON_FILE).read_text(encoding="utf-8")
        )
        metrics_test = AbstentionSplitMetrics.model_validate_json(
            (final_dir / METRICS_TEST_JSON_FILE).read_text(encoding="utf-8")
        )
        logger.info("Abstention-policy cache hit for %s at %s", repository.full_name, final_dir)
        return AbstentionPolicyBuildResult(
            repository=repository,
            policy_dir=final_dir,
            manifest=manifest,
            classification_threshold=classification_threshold_basis_points
            / resolved_grid.denominator,
            classification_threshold_basis_points=classification_threshold_basis_points,
            selected_abstention_threshold=manifest.selected_abstention_basis_points
            / resolved_grid.denominator,
            selected_abstention_basis_points=manifest.selected_abstention_basis_points,
            validation_coverage=metrics_validation.coverage,
            validation_handled_subset_accuracy=metrics_validation.handled_metrics.subset_accuracy,
            test_coverage=metrics_test.coverage,
            test_handled_subset_accuracy=metrics_test.handled_metrics.subset_accuracy,
            cache_hit=True,
        )

    val_bundle, val_table, sweep_rows, selection = _run_validation_only_pipeline(
        config=config,
        baseline_dir=baseline_dir,
        baseline_manifest=baseline_manifest,
        threshold_inputs=threshold_inputs,
        expected_repository=repository,
        resolved_grid=resolved_grid,
        classification_threshold_basis_points=classification_threshold_basis_points,
    )
    frozen = freeze_abstention_policy(
        selection=selection,
        classification_threshold_basis_points=classification_threshold_basis_points,
        denominator=resolved_grid.denominator,
    )

    test_bundle = load_test_scores(
        baseline_dir,
        baseline_manifest=baseline_manifest,
        candidate_id=threshold_inputs.manifest.selected_candidate_id,
        expected_repository=repository,
    )
    test_table = build_issue_confidence_table(
        issue_ids=test_bundle.issue_ids,
        y_score=test_bundle.y_score,
        classification_threshold=frozen.classification_threshold,
        confidence_definition=config.confidence_definition,
    )

    full_set_val = _full_set_classification_metrics(
        split="validation",
        labels=val_bundle.labels,
        y_true=val_bundle.y_true,
        y_pred=val_table.y_pred,
        y_score=val_bundle.y_score,
        classification_threshold=frozen.classification_threshold,
    )
    full_set_test = _full_set_classification_metrics(
        split="test",
        labels=test_bundle.labels,
        y_true=test_bundle.y_true,
        y_pred=test_table.y_pred,
        y_score=test_bundle.y_score,
        classification_threshold=frozen.classification_threshold,
    )

    metrics_validation = evaluate_frozen_abstention_split(
        split="validation",
        labels=val_bundle.labels,
        y_true=val_bundle.y_true,
        y_score=val_bundle.y_score,
        table=val_table,
        classification_threshold=frozen.classification_threshold,
        abstention_threshold=frozen.selected_abstention_threshold,
        full_set_reference=full_set_val,
    )
    metrics_test = evaluate_frozen_abstention_split(
        split="test",
        labels=test_bundle.labels,
        y_true=test_bundle.y_true,
        y_score=test_bundle.y_score,
        table=test_table,
        classification_threshold=frozen.classification_threshold,
        abstention_threshold=frozen.selected_abstention_threshold,
        full_set_reference=full_set_test,
    )

    val_predicted_count = val_bundle.record_count - int(val_table.forced_abstention_mask.sum())
    test_predicted_count = test_bundle.record_count - int(test_table.forced_abstention_mask.sum())
    comparison = build_comparison_document(
        classification_threshold_basis_points=classification_threshold_basis_points,
        selected_abstention_basis_points=selection.selected_abstention_basis_points,
        denominator=resolved_grid.denominator,
        validation_full_set_metrics=full_set_val,
        validation_selected_metrics=selection.selected_validation_metrics,
        validation_full_set_coverage=val_predicted_count / val_bundle.record_count,
        validation_full_set_handled_count=val_predicted_count,
        validation_selected_coverage=selection.selected_validation_coverage,
        validation_selected_handled_count=selection.selected_validation_handled_count,
        test_full_set_metrics=full_set_test,
        test_selected_metrics=metrics_test.handled_metrics,
        test_full_set_coverage=test_predicted_count / test_bundle.record_count,
        test_full_set_handled_count=test_predicted_count,
        test_selected_coverage=metrics_test.coverage,
        test_selected_handled_count=metrics_test.handled_count,
    )

    validation_bins = compute_confidence_bins(
        split="validation",
        labels=val_bundle.labels,
        y_true=val_bundle.y_true,
        y_score=val_bundle.y_score,
        issue_ids=val_bundle.issue_ids,
        classification_threshold=frozen.classification_threshold,
        confidence_definition=config.confidence_definition,
    )
    test_bins = compute_confidence_bins(
        split="test",
        labels=test_bundle.labels,
        y_true=test_bundle.y_true,
        y_score=test_bundle.y_score,
        issue_ids=test_bundle.issue_ids,
        classification_threshold=frozen.classification_threshold,
        confidence_definition=config.confidence_definition,
    )

    frozen_config = FrozenAbstentionPolicyConfig(
        repository=config.repository,
        threshold_policy_id=threshold_policy_id,
        baseline_run_id=threshold_inputs.manifest.baseline_run_id,
        confidence_definition=config.confidence_definition,
        metric_contract_version=config.metric_contract_version,
        selection_rule_version=config.selection_rule_version,
        minimum_coverage=config.minimum_coverage,
        classification_threshold_basis_points=classification_threshold_basis_points,
        abstention_grid=resolved_grid,
    )
    policy_document = PolicyDocument(selection=selection.selection_audit)
    sweep_document = SweepValidationDocument(
        classification_threshold_basis_points=classification_threshold_basis_points,
        abstention_grid=resolved_grid,
        rows=sweep_rows,
    )

    config_json = serialize_config_json(frozen_config)
    policy_json = serialize_policy_json(policy_document)
    sweep_json = serialize_sweep_validation_json(sweep_document)
    metrics_validation_json = serialize_split_metrics_json(metrics_validation)
    metrics_test_json = serialize_split_metrics_json(metrics_test)
    confidence_bins_validation_json = serialize_confidence_bins_json(validation_bins)
    confidence_bins_test_json = serialize_confidence_bins_json(test_bins)
    comparison_json = serialize_comparison_json(comparison)
    report_md = serialize_report_markdown(
        repository=config.repository,
        baseline_run_id=threshold_inputs.manifest.baseline_run_id,
        threshold_policy_id=threshold_policy_id,
        policy_id=policy_id,
        classification_threshold_basis_points=classification_threshold_basis_points,
        selected_abstention_basis_points=selection.selected_abstention_basis_points,
        minimum_coverage=config.minimum_coverage,
        denominator=resolved_grid.denominator,
        confidence_definition=config.confidence_definition,
        sweep_rows=sweep_rows,
        comparison=comparison,
        grid=resolved_grid,
        validation_bins=validation_bins,
        test_bins=test_bins,
    )

    built_at = datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")
    manifest = AbstentionPolicyManifest(
        policy_id=policy_id,
        policy_input_sha256=policy_input_sha256,
        config_source_sha256=config_source_hash,
        config_semantic_sha256=config_semantic_hash,
        repository=config.repository,
        baseline_run_id=threshold_inputs.manifest.baseline_run_id,
        baseline_experiment_sha256=baseline_manifest.baseline_experiment_sha256,
        model_dataset_id=baseline_manifest.model_dataset_id,
        model_semantic_sha256=baseline_manifest.model_semantic_sha256,
        threshold_policy_id=threshold_policy_id,
        threshold_policy_sha256=threshold_inputs.policy_sha256,
        selected_candidate_id=threshold_inputs.manifest.selected_candidate_id,
        predictions_validation_sha256=baseline_manifest.predictions_validation_sha256,
        predictions_test_sha256=baseline_manifest.predictions_test_sha256,
        confidence_definition=config.confidence_definition,
        metric_contract_version=config.metric_contract_version,
        selection_rule_version=config.selection_rule_version,
        classification_threshold_basis_points=classification_threshold_basis_points,
        selected_abstention_basis_points=selection.selected_abstention_basis_points,
        minimum_coverage=config.minimum_coverage,
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
        confidence_bins_validation_sha256=sha256_hex(confidence_bins_validation_json),
        confidence_bins_test_sha256=sha256_hex(confidence_bins_test_json),
        comparison_sha256=sha256_hex(comparison_json),
        report_sha256=sha256_hex(report_md),
    )
    manifest_json = serialize_manifest_json(manifest)

    slug_dir = abstention_policies_root / repository.slug
    slug_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=".staging-", dir=str(slug_dir)))

    try:
        atomic_write_bytes(staging_dir / CONFIG_JSON_FILE, config_json)
        atomic_write_bytes(staging_dir / POLICY_JSON_FILE, policy_json)
        atomic_write_bytes(staging_dir / SWEEP_VALIDATION_JSON_FILE, sweep_json)
        atomic_write_bytes(staging_dir / METRICS_VALIDATION_JSON_FILE, metrics_validation_json)
        atomic_write_bytes(staging_dir / METRICS_TEST_JSON_FILE, metrics_test_json)
        atomic_write_bytes(
            staging_dir / CONFIDENCE_BINS_VALIDATION_JSON_FILE, confidence_bins_validation_json
        )
        atomic_write_bytes(staging_dir / CONFIDENCE_BINS_TEST_JSON_FILE, confidence_bins_test_json)
        atomic_write_bytes(staging_dir / COMPARISON_JSON_FILE, comparison_json)
        atomic_write_bytes(staging_dir / REPORT_MARKDOWN_FILE, report_md)
        atomic_write_bytes(staging_dir / MANIFEST_JSON_FILE, manifest_json)

        validate_abstention_policy_artifact_integrity(
            staging_dir,
            expected_repository=repository,
            expected_policy_id=policy_id,
            check_dir_name=False,
        )
        validate_abstention_policy_against_inputs(
            staging_dir,
            baseline_dir,
            threshold_policy_dir,
            expected_repository=repository,
            check_dir_name=False,
        )
        publish_abstention_policy(staging_dir, final_dir)
    except Exception:
        best_effort_remove_tree(staging_dir)
        raise

    return AbstentionPolicyBuildResult(
        repository=repository,
        policy_dir=final_dir,
        manifest=manifest,
        classification_threshold=frozen.classification_threshold,
        classification_threshold_basis_points=classification_threshold_basis_points,
        selected_abstention_threshold=frozen.selected_abstention_threshold,
        selected_abstention_basis_points=selection.selected_abstention_basis_points,
        validation_coverage=metrics_validation.coverage,
        validation_handled_subset_accuracy=metrics_validation.handled_metrics.subset_accuracy,
        test_coverage=metrics_test.coverage,
        test_handled_subset_accuracy=metrics_test.handled_metrics.subset_accuracy,
        cache_hit=False,
    )


def format_abstention_policy_summary(result: AbstentionPolicyBuildResult) -> str:
    lines = [
        f"repository: {result.repository.full_name}",
        f"baseline_run_id: {result.manifest.baseline_run_id}",
        f"threshold_policy_id: {result.manifest.threshold_policy_id}",
        f"abstention_policy_id: {result.manifest.policy_id}",
        f"classification_threshold: {result.classification_threshold:.2f}",
        f"selected_abstention_threshold: {result.selected_abstention_threshold:.2f}",
        f"validation_coverage: {result.validation_coverage:.6f}",
        f"validation_handled_subset_accuracy: {result.validation_handled_subset_accuracy:.6f}"
        if result.validation_handled_subset_accuracy is not None
        else "validation_handled_subset_accuracy: n/a",
        f"test_coverage: {result.test_coverage:.6f}",
        f"test_handled_subset_accuracy: {result.test_handled_subset_accuracy:.6f}"
        if result.test_handled_subset_accuracy is not None
        else "test_handled_subset_accuracy: n/a",
        f"artifact_path: {result.policy_dir}",
        f"cache_hit: {'true' if result.cache_hit else 'false'}",
    ]
    return "\n".join(lines)
