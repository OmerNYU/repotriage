"""Build (or reuse) an immutable multilabel baseline artifact."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
from pydantic import ValidationError

from repotriage.baseline.config import get_candidate_by_id, load_baseline_config
from repotriage.baseline.evaluator import (
    build_prediction_records,
    evaluate_frozen_candidate,
    metrics_from_predictions,
)
from repotriage.baseline.models import (
    BASELINE_MANIFEST_SCHEMA_VERSION,
    BASELINE_VERSION,
    CANDIDATE_RESULTS_JSON_FILE,
    CONFIG_JSON_FILE,
    FEATURE_SUMMARY_JSON_FILE,
    METRICS_MARKDOWN_FILE,
    METRICS_TEST_JSON_FILE,
    MODEL_JOBLIB_FILE,
    MODEL_SEMANTIC_CONTRACT_VERSION,
    PREDICTIONS_TEST_JSONL_FILE,
    PREDICTIONS_VALIDATION_JSONL_FILE,
    BaselineBuildError,
    BaselineConfigDocument,
    BaselineCorruptionError,
    BaselineInputError,
    BaselineManifest,
    CandidateResultsDocument,
    CandidateValidationResult,
    FeatureSummary,
    FrozenConfigDocument,
    PredictionRecord,
    SplitMetrics,
    compute_baseline_experiment_sha256,
    compute_baseline_run_id,
    compute_baseline_run_sha256,
    floats_consistent,
)
from repotriage.baseline.models_ml import load_model_from_bundle, model_semantic_sha256
from repotriage.baseline.reader import load_test_split, load_training_splits
from repotriage.baseline.report import (
    serialize_candidate_results_json,
    serialize_feature_summary_json,
    serialize_frozen_config_json,
    serialize_manifest_json,
    serialize_metrics_json,
    serialize_metrics_markdown,
    serialize_predictions_jsonl,
    sha256_hex,
)
from repotriage.baseline.runtime import build_environment_fingerprint
from repotriage.baseline.selector import (
    run_candidate_selection,
    to_candidate_validation_result,
)
from repotriage.filesystem import atomic_write_bytes, best_effort_remove_tree
from repotriage.github.models import RepositoryRef
from repotriage.model_dataset.models import LabelMap, ModelDatasetManifest, ModelReadyRecord
from repotriage.model_dataset.reader import read_model_ready_records
from repotriage.paths import resolve_within_directory

logger = logging.getLogger(__name__)

DEFAULT_MODEL_READY_ROOT = Path("data/model_ready/github")
DEFAULT_BASELINES_ROOT = Path("data/baselines/github")


@dataclass(frozen=True)
class BaselineBuildResult:
    repository: RepositoryRef
    baseline_dir: Path
    manifest: BaselineManifest
    selected_candidate_id: str
    cache_hit: bool


def _package_version(module_name: str) -> str | None:
    try:
        import importlib.metadata as metadata

        return metadata.version(module_name)
    except Exception:
        return None


def collect_environment_metadata_and_hash() -> tuple:
    return build_environment_fingerprint(
        numpy_version=_package_version("numpy"),
        scipy_version=_package_version("scipy"),
        scikit_learn_version=_package_version("scikit-learn"),
        joblib_version=_package_version("joblib"),
    )


def _verify_file_hash(path: Path, expected_sha256: str) -> bytes:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise BaselineCorruptionError(f"Unable to read baseline file {path}: {exc}") from exc
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise BaselineCorruptionError(
            f"Hash mismatch for {path.name}: expected {expected_sha256}, got {actual}"
        )
    return data


def _load_manifest(baseline_dir: Path) -> BaselineManifest:
    manifest_path = baseline_dir / "manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return BaselineManifest.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise BaselineCorruptionError(
            f"Invalid baseline manifest at {manifest_path}: {exc}"
        ) from exc


def _record_sort_key(record: ModelReadyRecord) -> tuple[str, int]:
    return (record.model_dump(mode="json")["created_at"], record.issue_id)


def _stream_model_ready_records(
    model_dataset_dir: Path,
    manifest: ModelDatasetManifest,
    split: str,
) -> list[ModelReadyRecord]:
    return sorted(
        read_model_ready_records(model_dataset_dir, manifest, split=split),
        key=_record_sort_key,
    )


def _assert_manifest_identity(manifest: BaselineManifest) -> None:
    if manifest.schema_version != BASELINE_MANIFEST_SCHEMA_VERSION:
        raise BaselineCorruptionError(
            f"Unsupported manifest schema_version {manifest.schema_version!r}."
        )
    if manifest.baseline_version != BASELINE_VERSION:
        raise BaselineCorruptionError(
            f"Unsupported baseline_version {manifest.baseline_version!r}."
        )

    expected_experiment = compute_baseline_experiment_sha256(
        baseline_version=manifest.baseline_version,
        model_dataset_id=manifest.model_dataset_id,
        records_sha256=manifest.records_sha256,
        label_map_sha256=manifest.label_map_sha256,
        config_semantic_sha256=manifest.config_semantic_sha256,
        candidate_set_version=manifest.candidate_set_version,
        selection_rule_version=manifest.selection_rule_version,
        metric_contract_version=manifest.metric_contract_version,
        model_semantic_contract_version=manifest.model_semantic_contract_version,
        threshold=manifest.threshold,
        score_type=manifest.score_type,
        training_protocol_version=manifest.training_protocol_version,
        random_state=manifest.random_state,
    )
    if manifest.baseline_experiment_sha256 != expected_experiment:
        raise BaselineCorruptionError(
            "baseline_experiment_sha256 is inconsistent with manifest fields"
        )

    expected_run_sha256 = compute_baseline_run_sha256(
        manifest.baseline_experiment_sha256,
        manifest.numerical_environment_sha256,
    )
    if manifest.baseline_run_sha256 != expected_run_sha256:
        raise BaselineCorruptionError("baseline_run_sha256 is inconsistent with manifest fields")

    expected_run_id = compute_baseline_run_id(manifest.model_dataset_id, expected_run_sha256)
    if manifest.baseline_run_id != expected_run_id:
        raise BaselineCorruptionError(
            f"baseline_run_id {manifest.baseline_run_id!r} is inconsistent with inputs "
            f"(expected {expected_run_id!r})."
        )


def validate_baseline_artifact_integrity(
    baseline_dir: Path,
    *,
    expected_repository: RepositoryRef,
    expected_baseline_run_id: str,
    check_dir_name: bool = True,
) -> BaselineManifest:
    """Validate a baseline artifact using only on-disk bytes (no joblib.load)."""
    if not baseline_dir.is_dir():
        raise BaselineCorruptionError(f"Baseline directory does not exist: {baseline_dir}")

    manifest = _load_manifest(baseline_dir)
    if check_dir_name and baseline_dir.name != manifest.baseline_run_id:
        raise BaselineCorruptionError(
            f"Baseline directory {baseline_dir.name!r} does not match manifest "
            f"baseline_run_id {manifest.baseline_run_id!r}."
        )
    if manifest.baseline_run_id != expected_baseline_run_id:
        raise BaselineCorruptionError(
            f"Manifest baseline_run_id {manifest.baseline_run_id!r} does not match expected "
            f"{expected_baseline_run_id!r}."
        )
    if manifest.repository != expected_repository.full_name:
        raise BaselineCorruptionError(
            f"Manifest repository {manifest.repository!r} does not match expected "
            f"{expected_repository.full_name!r}."
        )
    _assert_manifest_identity(manifest)

    for relative_path, expected_sha256 in (
        (manifest.config_file, manifest.config_sha256),
        (manifest.candidate_results_file, manifest.candidate_results_sha256),
        (manifest.metrics_test_file, manifest.metrics_test_sha256),
        (manifest.metrics_markdown_file, manifest.metrics_markdown_sha256),
        (manifest.predictions_validation_file, manifest.predictions_validation_sha256),
        (manifest.predictions_test_file, manifest.predictions_test_sha256),
        (manifest.feature_summary_file, manifest.feature_summary_sha256),
        (manifest.model_file, manifest.model_sha256),
    ):
        file_path = resolve_within_directory(baseline_dir, relative_path)
        _verify_file_hash(file_path, expected_sha256)

    candidate_bytes = _verify_file_hash(
        resolve_within_directory(baseline_dir, manifest.candidate_results_file),
        manifest.candidate_results_sha256,
    )
    candidate_results = CandidateResultsDocument.model_validate_json(candidate_bytes)
    if candidate_results.selection.winner_candidate_id != manifest.selected_candidate_id:
        raise BaselineCorruptionError(
            "candidate_results winner does not match manifest selected_candidate_id"
        )

    val_predictions_path = resolve_within_directory(
        baseline_dir, manifest.predictions_validation_file
    )
    test_predictions_path = resolve_within_directory(baseline_dir, manifest.predictions_test_file)
    val_count = sum(1 for _ in val_predictions_path.open("r", encoding="utf-8"))
    test_count = sum(1 for _ in test_predictions_path.open("r", encoding="utf-8"))
    if val_count != manifest.validation_prediction_count:
        raise BaselineCorruptionError(
            f"Validation prediction count {val_count} does not match manifest "
            f"{manifest.validation_prediction_count}"
        )
    if test_count != manifest.test_record_count:
        raise BaselineCorruptionError(
            f"Test prediction count {test_count} does not match manifest "
            f"{manifest.test_record_count}"
        )

    return manifest


def _validate_source_alignment(
    *,
    model_dataset_dir: Path,
    manifest: ModelDatasetManifest,
    label_map: LabelMap,
    records: list[PredictionRecord],
    split: str,
    declared_candidate_ids: set[str] | None = None,
) -> None:
    canonical_records = _stream_model_ready_records(model_dataset_dir, manifest, split)
    if len(records) != len(canonical_records) and declared_candidate_ids is None:
        raise BaselineCorruptionError(
            f"Prediction count {len(records)} does not match canonical {split} count "
            f"{len(canonical_records)}"
        )

    if declared_candidate_ids is not None:
        by_candidate: dict[str, list[PredictionRecord]] = defaultdict(list)
        for record in records:
            if record.candidate_id is None:
                raise BaselineCorruptionError("validation prediction missing candidate_id")
            by_candidate[record.candidate_id].append(record)
        if set(by_candidate) != declared_candidate_ids:
            raise BaselineCorruptionError("validation predictions have unexpected candidate set")
        for candidate_id, candidate_records in by_candidate.items():
            if len(candidate_records) != len(canonical_records):
                raise BaselineCorruptionError(
                    f"candidate {candidate_id!r} has {len(candidate_records)} validation "
                    f"predictions; expected {len(canonical_records)}"
                )
            sorted_records = sorted(
                candidate_records,
                key=lambda record: (
                    next(
                        canonical.model_dump(mode="json")["created_at"]
                        for canonical in canonical_records
                        if canonical.issue_id == record.issue_id
                    ),
                    record.issue_id,
                ),
            )
            _validate_source_alignment(
                model_dataset_dir=model_dataset_dir,
                manifest=manifest,
                label_map=label_map,
                records=sorted_records,
                split=split,
                declared_candidate_ids=None,
            )
        return

    for index, (stored, canonical) in enumerate(zip(records, canonical_records, strict=True)):
        if stored.issue_id != canonical.issue_id:
            raise BaselineCorruptionError(
                f"{split} prediction issue_id mismatch at index {index}"
            )
        if stored.issue_number != canonical.issue_number:
            raise BaselineCorruptionError(
                f"{split} prediction issue_number mismatch at index {index}"
            )
        if stored.true_vector != list(canonical.target_vector):
            raise BaselineCorruptionError(
                f"{split} prediction true_vector does not match canonical target_vector "
                f"at index {index}"
            )
        if len(stored.true_vector) != label_map.target_count:
            raise BaselineCorruptionError("prediction true_vector has unexpected length")


def _audit_candidate_selection(
    *,
    candidate_results: CandidateResultsDocument,
    validation_records: list[PredictionRecord],
    labels: list[str],
    config: BaselineConfigDocument,
) -> None:
    declared_ids = {candidate.candidate_id for candidate in config.candidates}
    by_candidate: dict[str, list[PredictionRecord]] = defaultdict(list)
    for record in validation_records:
        if record.candidate_id is None:
            raise BaselineCorruptionError("validation prediction missing candidate_id")
        by_candidate[record.candidate_id].append(record)

    if set(by_candidate) != declared_ids:
        raise BaselineCorruptionError("validation predictions candidate set mismatch")

    recomputed_scores = []
    for candidate in config.candidates:
        candidate_records = by_candidate[candidate.candidate_id]
        recomputed = metrics_from_predictions(
            labels=labels,
            records=candidate_records,
            split="validation",
        )
        stored = next(
            item
            for item in candidate_results.candidates
            if item.candidate_id == candidate.candidate_id
        )
        _assert_metrics_close(stored.metrics, recomputed)
        recomputed_scores.append((candidate.candidate_id, recomputed))

    ranked = sorted(
        recomputed_scores,
        key=lambda item: (
            item[1].aggregate.macro_average_precision or float("-inf"),
            item[1].aggregate.macro_f1 or float("-inf"),
            item[1].aggregate.micro_f1 or float("-inf"),
            item[0],
        ),
        reverse=True,
    )
    recomputed_winner = ranked[0][0]
    if recomputed_winner != candidate_results.selection.winner_candidate_id:
        raise BaselineCorruptionError(
            "recomputed winner does not match stored selection audit"
        )

    if candidate_results.dummy_baseline is not None:
        if candidate_results.dummy_baseline.candidate_id != "dummy_all_zero":
            raise BaselineCorruptionError("dummy baseline has unexpected candidate_id")
        if candidate_results.dummy_baseline.selectable:
            raise BaselineCorruptionError("dummy baseline must not be selectable")


def verify_baseline_model_consistency(
    baseline_dir: Path,
    *,
    expected_repository: RepositoryRef,
    model_dataset_dir: Path,
    model_dataset_manifest: ModelDatasetManifest,
    label_map: LabelMap,
    trust_model_file: bool = True,
) -> None:
    """Recompute scores from model.joblib and compare to stored predictions."""
    if not trust_model_file:
        raise ValueError("verify_baseline_model_consistency requires trust_model_file=True")

    manifest = _load_manifest(baseline_dir)
    if manifest.repository != expected_repository.full_name:
        raise BaselineCorruptionError("manifest repository mismatch")

    model_path = resolve_within_directory(baseline_dir, manifest.model_file)
    bundle = joblib.load(model_path)
    model = load_model_from_bundle(bundle)
    recomputed_semantic = model_semantic_sha256(model)
    if recomputed_semantic != manifest.model_semantic_sha256:
        raise BaselineCorruptionError(
            "model_semantic_sha256 does not match recomputed semantic fingerprint"
        )

    for relative_path, split_name in (
        (manifest.predictions_validation_file, "validation"),
        (manifest.predictions_test_file, "test"),
    ):
        path = resolve_within_directory(baseline_dir, relative_path)
        stored_records: list[PredictionRecord] = []
        for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not raw_line.strip():
                raise BaselineCorruptionError(f"Blank line in {path} at line {line_number}")
            stored_records.append(PredictionRecord.model_validate_json(raw_line))

        if split_name == "validation":
            candidate_records = [
                record
                for record in stored_records
                if record.candidate_id == manifest.selected_candidate_id
            ]
            eval_records = candidate_records
        else:
            eval_records = stored_records

        canonical_records = _stream_model_ready_records(
            model_dataset_dir, model_dataset_manifest, split_name
        )
        texts = [record.feature_text for record in canonical_records]
        recomputed_scores = model.predict_proba_matrix(texts)
        recomputed_preds = (recomputed_scores >= model.threshold).astype(int)

        for index, (stored, canonical) in enumerate(
            zip(eval_records, canonical_records, strict=True)
        ):
            if stored.score_vector is None:
                raise BaselineCorruptionError("stored prediction missing score_vector")
            for label_index, score in enumerate(stored.score_vector):
                if not floats_consistent(score, float(recomputed_scores[index, label_index])):
                    raise BaselineCorruptionError(
                        f"model score mismatch at {split_name} row {index} label {label_index}"
                    )
            expected_vector = [int(value) for value in recomputed_preds[index].tolist()]
            if stored.predicted_vector != expected_vector:
                raise BaselineCorruptionError(
                    f"model predicted_vector mismatch at {split_name} row {index}"
                )


def validate_baseline_against_inputs(
    baseline_dir: Path,
    *,
    expected_repository: RepositoryRef,
    model_dataset_dir: Path,
    model_dataset_manifest: ModelDatasetManifest,
    label_map: LabelMap,
    config: BaselineConfigDocument,
    config_semantic_sha256: str,
    expected_baseline_run_id: str,
    expected_baseline_experiment_sha256: str,
    expected_numerical_environment_sha256: str,
    check_dir_name: bool = True,
) -> BaselineManifest:
    """Validate artifact integrity and lineage to model-ready inputs (no joblib.load)."""
    manifest = validate_baseline_artifact_integrity(
        baseline_dir,
        expected_repository=expected_repository,
        expected_baseline_run_id=expected_baseline_run_id,
        check_dir_name=check_dir_name,
    )
    if manifest.model_dataset_id != model_dataset_manifest.model_dataset_id:
        raise BaselineCorruptionError("manifest model_dataset_id disagrees with model-ready input")
    if manifest.records_sha256 != model_dataset_manifest.records_sha256:
        raise BaselineCorruptionError("manifest records_sha256 disagrees with model-ready manifest")
    if manifest.label_map_sha256 != model_dataset_manifest.label_map_sha256:
        raise BaselineCorruptionError(
            "manifest label_map_sha256 disagrees with model-ready manifest"
        )
    if manifest.config_semantic_sha256 != config_semantic_sha256:
        raise BaselineCorruptionError("manifest config_semantic_sha256 disagrees with config file")
    if manifest.baseline_experiment_sha256 != expected_baseline_experiment_sha256:
        raise BaselineCorruptionError(
            "manifest baseline_experiment_sha256 disagrees with expected experiment hash"
        )
    if manifest.numerical_environment_sha256 != expected_numerical_environment_sha256:
        raise BaselineCorruptionError(
            "manifest numerical_environment_sha256 disagrees with current environment"
        )
    if config.repository != expected_repository.full_name:
        raise BaselineInputError(
            f"Config repository {config.repository!r} does not match "
            f"{expected_repository.full_name!r}"
        )
    if manifest.target_count != label_map.target_count:
        raise BaselineCorruptionError("manifest target_count disagrees with label_map")

    labels = label_map.labels
    candidate_results = CandidateResultsDocument.model_validate_json(
        _verify_file_hash(
            resolve_within_directory(baseline_dir, manifest.candidate_results_file),
            manifest.candidate_results_sha256,
        )
    )

    val_path = resolve_within_directory(baseline_dir, manifest.predictions_validation_file)
    val_records: list[PredictionRecord] = []
    for line_number, raw_line in enumerate(
        val_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            raise BaselineCorruptionError(f"Blank line in {val_path} at line {line_number}")
        val_records.append(PredictionRecord.model_validate_json(raw_line))

    _validate_source_alignment(
        model_dataset_dir=model_dataset_dir,
        manifest=model_dataset_manifest,
        label_map=label_map,
        records=val_records,
        split="validation",
        declared_candidate_ids={candidate.candidate_id for candidate in config.candidates},
    )
    _audit_candidate_selection(
        candidate_results=candidate_results,
        validation_records=val_records,
        labels=labels,
        config=config,
    )

    test_path = resolve_within_directory(baseline_dir, manifest.predictions_test_file)
    test_records: list[PredictionRecord] = []
    for line_number, raw_line in enumerate(
        test_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            raise BaselineCorruptionError(f"Blank line in {test_path} at line {line_number}")
        test_records.append(PredictionRecord.model_validate_json(raw_line))

    _validate_source_alignment(
        model_dataset_dir=model_dataset_dir,
        manifest=model_dataset_manifest,
        label_map=label_map,
        records=test_records,
        split="test",
    )

    stored_metrics = SplitMetrics.model_validate_json(
        _verify_file_hash(
            resolve_within_directory(baseline_dir, manifest.metrics_test_file),
            manifest.metrics_test_sha256,
        )
    )
    recomputed_test = metrics_from_predictions(
        labels=labels,
        records=test_records,
        split="test",
    )
    _assert_metrics_close(stored_metrics, recomputed_test)

    return manifest


def _assert_metrics_close(stored: SplitMetrics, recomputed: SplitMetrics) -> None:
    for field_name in ("subset_accuracy", "hamming_loss"):
        stored_value = getattr(stored.aggregate, field_name)
        recomputed_value = getattr(recomputed.aggregate, field_name)
        if not floats_consistent(stored_value, recomputed_value):
            raise BaselineCorruptionError(
                f"Recomputed {field_name} {recomputed_value} does not match stored {stored_value}"
            )
    for metric_name in ("macro_average_precision", "macro_f1", "micro_f1"):
        stored_value = getattr(stored.aggregate, metric_name)
        recomputed_value = getattr(recomputed.aggregate, metric_name)
        if stored_value is None and recomputed_value is None:
            continue
        if stored_value is None or recomputed_value is None:
            raise BaselineCorruptionError(f"Recomputed {metric_name} disagrees with stored metrics")
        if not floats_consistent(stored_value, recomputed_value):
            raise BaselineCorruptionError(
                f"Recomputed {metric_name} {recomputed_value} does not match stored {stored_value}"
            )


def publish_baseline(staging_dir: Path, final_dir: Path) -> None:
    if final_dir.exists():
        raise BaselineBuildError(f"Refusing to overwrite existing baseline artifact: {final_dir}")
    os.rename(staging_dir, final_dir)


def train_baseline(
    repository: RepositoryRef,
    model_dataset_id: str,
    config_path: Path,
    *,
    model_ready_root: Path = DEFAULT_MODEL_READY_ROOT,
    baselines_root: Path = DEFAULT_BASELINES_ROOT,
) -> BaselineBuildResult:
    """Train, select, evaluate, and publish one immutable baseline artifact."""
    config, _config_bytes, config_source_hash, config_semantic_hash = load_baseline_config(
        config_path
    )
    if config.repository != repository.full_name:
        raise BaselineInputError(
            f"Config repository {config.repository!r} does not match requested "
            f"repository {repository.full_name!r}."
        )

    model_dataset_dir = model_ready_root / repository.slug / model_dataset_id
    training_splits = load_training_splits(
        model_dataset_dir,
        expected_repository=repository,
        expected_model_dataset_id=model_dataset_id,
    )
    manifest = training_splits.manifest
    label_map = training_splits.label_map
    labels = label_map.labels
    threshold = config.threshold_policy.threshold

    environment, env_hash = collect_environment_metadata_and_hash()

    baseline_experiment_sha256 = compute_baseline_experiment_sha256(
        baseline_version=config.baseline_version,
        model_dataset_id=model_dataset_id,
        records_sha256=manifest.records_sha256,
        label_map_sha256=manifest.label_map_sha256,
        config_semantic_sha256=config_semantic_hash,
        candidate_set_version=config.candidate_set_version,
        selection_rule_version=config.selection_rule_version,
        metric_contract_version=config.metric_contract_version,
        model_semantic_contract_version=MODEL_SEMANTIC_CONTRACT_VERSION,
        threshold=threshold,
        score_type=config.threshold_policy.score_type,
        training_protocol_version=config.training_protocol_version,
        random_state=config.random_state,
    )
    baseline_run_sha256 = compute_baseline_run_sha256(
        baseline_experiment_sha256,
        env_hash,
    )
    baseline_run_id = compute_baseline_run_id(model_dataset_id, baseline_run_sha256)
    final_dir = baselines_root / repository.slug / baseline_run_id

    if final_dir.exists():
        published = validate_baseline_against_inputs(
            final_dir,
            expected_repository=repository,
            model_dataset_dir=model_dataset_dir,
            model_dataset_manifest=manifest,
            label_map=label_map,
            config=config,
            config_semantic_sha256=config_semantic_hash,
            expected_baseline_run_id=baseline_run_id,
            expected_baseline_experiment_sha256=baseline_experiment_sha256,
            expected_numerical_environment_sha256=env_hash,
        )
        logger.info("Baseline-cache hit for %s at %s", repository.full_name, final_dir)
        return BaselineBuildResult(
            repository=repository,
            baseline_dir=final_dir,
            manifest=published,
            selected_candidate_id=published.selected_candidate_id,
            cache_hit=True,
        )

    selection = run_candidate_selection(
        config=config,
        splits=training_splits,
        repository=repository.full_name,
        model_dataset_id=model_dataset_id,
        baseline_run_id=baseline_run_id,
    )
    winner_id = selection.winner_id
    winner_trained = selection.trained_winner
    selected_candidate = get_candidate_by_id(config, winner_id)

    test_split = load_test_split(
        model_dataset_dir,
        manifest=manifest,
        label_map=label_map,
        training_splits=training_splits,
    )
    test_metrics, test_predictions, test_scores = evaluate_frozen_candidate(
        model=winner_trained.model,
        test_records=test_split.records,
        test_texts=test_split.texts,
        test_targets=test_split.targets,
        labels=labels,
    )

    test_prediction_records = build_prediction_records(
        repository=repository.full_name,
        model_dataset_id=model_dataset_id,
        baseline_run_id=baseline_run_id,
        labels=labels,
        records=test_split.records,
        predictions=test_predictions,
        scores=test_scores,
        split="test",
        threshold=threshold,
        score_type="probability_estimates",
        candidate_id=winner_id,
    )

    candidate_results = CandidateResultsDocument(
        dummy_baseline=CandidateValidationResult(
            candidate_id="dummy_all_zero",
            selectable=False,
            metrics=selection.dummy_metrics,
        ),
        candidates=[to_candidate_validation_result(item) for item in selection.scored_candidates],
        selection=selection.selection_audit,
    )

    feature_summary = FeatureSummary(
        train_only_fit=True,
        vocabulary_size=winner_trained.training_report.vocabulary_size,
        train_record_count=len(training_splits.train.records),
        target_count=len(labels),
        convergence_warnings=winner_trained.training_report.convergence_warnings,
        label_convergence={
            report.label: report.n_iter
            for report in winner_trained.training_report.label_reports
        },
    )

    frozen_config = FrozenConfigDocument(
        repository=repository.full_name,
        model_dataset_id=model_dataset_id,
        selected_candidate_id=winner_id,
        random_state=config.random_state,
        threshold_policy=config.threshold_policy,
        selected_candidate=selected_candidate,
        all_candidates=config.candidates,
    )

    config_bytes = serialize_frozen_config_json(frozen_config)
    candidate_results_bytes = serialize_candidate_results_json(candidate_results)
    metrics_test_bytes = serialize_metrics_json(test_metrics)
    metrics_md_bytes = serialize_metrics_markdown(
        candidate_results=candidate_results,
        test_metrics=test_metrics,
        selected_candidate_id=winner_id,
    )
    predictions_validation_bytes = serialize_predictions_jsonl(
        selection.validation_prediction_records
    )
    predictions_test_bytes = serialize_predictions_jsonl(test_prediction_records)
    feature_summary_bytes = serialize_feature_summary_json(feature_summary)
    model_buffer = io.BytesIO()
    joblib.dump(winner_trained.model.to_bundle(), model_buffer)
    model_bytes = model_buffer.getvalue()
    model_semantic = model_semantic_sha256(winner_trained.model)

    config_sha256 = sha256_hex(config_bytes)
    candidate_results_sha256 = sha256_hex(candidate_results_bytes)
    metrics_test_sha256 = sha256_hex(metrics_test_bytes)
    metrics_md_sha256 = sha256_hex(metrics_md_bytes)
    predictions_validation_sha256 = sha256_hex(predictions_validation_bytes)
    predictions_test_sha256 = sha256_hex(predictions_test_bytes)
    feature_summary_sha256 = sha256_hex(feature_summary_bytes)
    model_sha256 = sha256_hex(model_bytes)

    slug_dir = baselines_root / repository.slug
    slug_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{repository.slug}.{baseline_run_id}.staging-", dir=slug_dir)
    )

    logger.info(
        "Training baseline %s for %s (selected from %d candidates)",
        baseline_run_id,
        repository.full_name,
        len(config.candidates),
    )

    try:
        atomic_write_bytes(staging_dir / CONFIG_JSON_FILE, config_bytes)
        atomic_write_bytes(staging_dir / CANDIDATE_RESULTS_JSON_FILE, candidate_results_bytes)
        atomic_write_bytes(staging_dir / METRICS_TEST_JSON_FILE, metrics_test_bytes)
        atomic_write_bytes(staging_dir / METRICS_MARKDOWN_FILE, metrics_md_bytes)
        atomic_write_bytes(
            staging_dir / PREDICTIONS_VALIDATION_JSONL_FILE,
            predictions_validation_bytes,
        )
        atomic_write_bytes(staging_dir / PREDICTIONS_TEST_JSONL_FILE, predictions_test_bytes)
        atomic_write_bytes(staging_dir / FEATURE_SUMMARY_JSON_FILE, feature_summary_bytes)
        atomic_write_bytes(staging_dir / MODEL_JOBLIB_FILE, model_bytes)

        built_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        published_manifest = BaselineManifest(
            baseline_run_id=baseline_run_id,
            baseline_experiment_sha256=baseline_experiment_sha256,
            numerical_environment_sha256=env_hash,
            baseline_run_sha256=baseline_run_sha256,
            repository=repository.full_name,
            model_dataset_id=model_dataset_id,
            records_sha256=manifest.records_sha256,
            label_map_sha256=manifest.label_map_sha256,
            config_semantic_sha256=config_semantic_hash,
            config_source_sha256=config_source_hash,
            random_state=config.random_state,
            threshold=threshold,
            selected_candidate_id=winner_id,
            built_at=built_at,
            validation_record_count=len(training_splits.validation.records),
            validation_prediction_count=len(selection.validation_prediction_records),
            test_record_count=len(test_split.records),
            target_count=len(labels),
            config_sha256=config_sha256,
            candidate_results_sha256=candidate_results_sha256,
            metrics_test_sha256=metrics_test_sha256,
            metrics_markdown_sha256=metrics_md_sha256,
            predictions_validation_sha256=predictions_validation_sha256,
            predictions_test_sha256=predictions_test_sha256,
            feature_summary_sha256=feature_summary_sha256,
            model_sha256=model_sha256,
            model_semantic_sha256=model_semantic,
            environment=environment,
        )
        atomic_write_bytes(
            staging_dir / "manifest.json",
            serialize_manifest_json(published_manifest),
        )

        validate_baseline_against_inputs(
            staging_dir,
            expected_repository=repository,
            model_dataset_dir=model_dataset_dir,
            model_dataset_manifest=manifest,
            label_map=label_map,
            config=config,
            config_semantic_sha256=config_semantic_hash,
            expected_baseline_run_id=baseline_run_id,
            expected_baseline_experiment_sha256=baseline_experiment_sha256,
            expected_numerical_environment_sha256=env_hash,
            check_dir_name=False,
        )
        verify_baseline_model_consistency(
            staging_dir,
            expected_repository=repository,
            model_dataset_dir=model_dataset_dir,
            model_dataset_manifest=manifest,
            label_map=label_map,
            trust_model_file=True,
        )
        publish_baseline(staging_dir, final_dir)
    except Exception:
        best_effort_remove_tree(staging_dir)
        raise
    else:
        best_effort_remove_tree(staging_dir)

    return BaselineBuildResult(
        repository=repository,
        baseline_dir=final_dir,
        manifest=published_manifest,
        selected_candidate_id=winner_id,
        cache_hit=False,
    )


def format_baseline_summary(result: BaselineBuildResult) -> str:
    manifest = result.manifest
    candidate_results_path = result.baseline_dir / CANDIDATE_RESULTS_JSON_FILE
    candidate_results = CandidateResultsDocument.model_validate_json(
        candidate_results_path.read_bytes()
    )
    winner = next(
        item
        for item in candidate_results.candidates
        if item.candidate_id == manifest.selected_candidate_id
    )
    test_metrics = SplitMetrics.model_validate_json(
        (result.baseline_dir / METRICS_TEST_JSON_FILE).read_bytes()
    )
    lines = [
        f"Repository: {manifest.repository}",
        f"Model-dataset ID: {manifest.model_dataset_id}",
        f"Baseline run ID: {manifest.baseline_run_id}",
        f"Experiment hash: {manifest.baseline_experiment_sha256}",
        f"Environment hash: {manifest.numerical_environment_sha256}",
        f"Selected candidate: {manifest.selected_candidate_id}",
        f"Validation macro AP: {winner.metrics.aggregate.macro_average_precision}",
        f"Test macro AP: {test_metrics.aggregate.macro_average_precision}",
        f"Artifact path: {result.baseline_dir}",
        f"Baseline-cache hit: {'yes' if result.cache_hit else 'no'}",
    ]
    return "\n".join(lines)
