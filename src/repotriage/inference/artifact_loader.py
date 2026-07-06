"""Load and validate the four-artifact inference bundle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
from pydantic import ValidationError

from repotriage.abstention_policy.builder import validate_abstention_policy_artifact_integrity
from repotriage.abstention_policy.models import (
    POLICY_JSON_FILE as ABSTENTION_POLICY_JSON_FILE,
)
from repotriage.abstention_policy.models import (
    AbstentionPolicyCorruptionError,
    AbstentionPolicyManifest,
)
from repotriage.abstention_policy.models import (
    PolicyDocument as AbstentionPolicyDocument,
)
from repotriage.abstention_policy.reader import load_threshold_policy_inputs
from repotriage.baseline.builder import DEFAULT_BASELINES_ROOT, validate_baseline_artifact_integrity
from repotriage.baseline.models import BaselineManifest
from repotriage.baseline.models_ml import (
    TfidfMultiLabelLogRegModel,
    load_model_from_bundle,
    model_semantic_sha256,
)
from repotriage.github.models import RepositoryRef
from repotriage.inference.config import InferenceConfigDocument
from repotriage.inference.models import InferenceBundleError
from repotriage.model_dataset.builder import (
    DEFAULT_MODEL_READY_ROOT,
    validate_model_dataset_artifact_integrity,
)
from repotriage.model_dataset.models import LabelMap, ModelDatasetManifest
from repotriage.paths import resolve_within_directory
from repotriage.retrieval.index import (
    RetrievalIndex,
    compute_index_semantic_sha256,
    load_corpus_matrix,
)
from repotriage.retrieval.models import (
    CORPUS_MATRIX_NPZ_FILE,
    CORPUS_RECORDS_JSONL_FILE,
    VECTORIZER_JOBLIB_FILE,
    CorpusRecord,
    RetrievalCorruptionError,
    RetrievalManifest,
)
from repotriage.retrieval.validators import validate_retrieval_artifact_integrity
from repotriage.threshold_policy.builder import (
    DEFAULT_THRESHOLD_POLICIES_ROOT,
    validate_threshold_policy_artifact_integrity,
)
from repotriage.threshold_policy.models import ThresholdPolicyManifest

DEFAULT_ABSTENTION_POLICIES_ROOT = Path("data/abstention_policies/github")
DEFAULT_RETRIEVAL_BASELINES_ROOT = Path("data/retrieval_baselines/github")


@dataclass(frozen=True)
class LoadedInferenceBundle:
    """In-memory runtime bundle combining all inference artifacts."""

    config: InferenceConfigDocument
    config_path: Path
    repository: RepositoryRef
    model_dataset_manifest: ModelDatasetManifest
    label_map: LabelMap
    label_order: list[str]
    baseline_manifest: BaselineManifest
    threshold_manifest: ThresholdPolicyManifest
    abstention_manifest: AbstentionPolicyManifest
    retrieval_manifest: RetrievalManifest
    model: TfidfMultiLabelLogRegModel
    retrieval_index: RetrievalIndex
    classification_threshold: float
    classification_threshold_basis_points: int
    abstention_threshold: float
    abstention_threshold_basis_points: int
    confidence_definition: str
    serialization_security_warning: str | None


def _read_jsonl(path: Path, model: type) -> list:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(model.model_validate_json(line))
        except ValidationError as exc:
            raise RetrievalCorruptionError(
                f"Invalid JSONL record at {path}:{line_number}: {exc}"
            ) from exc
    return records


def _load_abstention_policy_document(policy_dir: Path) -> AbstentionPolicyDocument:
    policy_path = resolve_within_directory(policy_dir, ABSTENTION_POLICY_JSON_FILE)
    try:
        return AbstentionPolicyDocument.model_validate_json(policy_path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise AbstentionPolicyCorruptionError(
            f"Invalid abstention policy document at {policy_path}: {exc}"
        ) from exc


def _load_baseline_model(
    baseline_dir: Path,
    manifest: BaselineManifest,
) -> TfidfMultiLabelLogRegModel:
    model_path = resolve_within_directory(baseline_dir, manifest.model_file)
    bundle = joblib.load(model_path)
    model = load_model_from_bundle(bundle)
    recomputed = model_semantic_sha256(model)
    if recomputed != manifest.model_semantic_sha256:
        raise InferenceBundleError(
            "Loaded model semantic fingerprint does not match baseline manifest."
        )
    return model


def _load_retrieval_index(
    retrieval_dir: Path,
    manifest: RetrievalManifest,
    *,
    label_order: list[str],
) -> RetrievalIndex:
    vectorizer = joblib.load(resolve_within_directory(retrieval_dir, VECTORIZER_JOBLIB_FILE))
    corpus_matrix = load_corpus_matrix(
        str(resolve_within_directory(retrieval_dir, CORPUS_MATRIX_NPZ_FILE))
    )
    corpus_records = _read_jsonl(
        resolve_within_directory(retrieval_dir, CORPUS_RECORDS_JSONL_FILE),
        CorpusRecord,
    )
    loaded_index = RetrievalIndex(
        vectorizer=vectorizer,
        corpus_matrix=corpus_matrix,
        corpus_records=corpus_records,
        train_issue_ids=[record.issue_id for record in corpus_records],
    )
    recomputed = compute_index_semantic_sha256(
        loaded_index,
        retrieval_baseline_version=manifest.retrieval_baseline_version,
        model_dataset_id=manifest.model_dataset_id,
        repository=manifest.repository,
        label_order=label_order,
        top_k=manifest.top_k,
        similarity_metric=manifest.similarity_metric,
        metric_contract_version=manifest.metric_contract_version,
        numerical_environment_sha256=manifest.numerical_environment_sha256,
    )
    if recomputed != manifest.index_semantic_sha256:
        raise InferenceBundleError(
            "Loaded retrieval index semantic fingerprint does not match manifest."
        )
    return loaded_index


def validate_inference_bundle_compatibility(
    *,
    config: InferenceConfigDocument,
    model_dataset_manifest: ModelDatasetManifest,
    label_map: LabelMap,
    baseline_manifest: BaselineManifest,
    threshold_manifest: ThresholdPolicyManifest,
    abstention_manifest: AbstentionPolicyManifest,
    retrieval_manifest: RetrievalManifest,
    model: TfidfMultiLabelLogRegModel,
) -> None:
    """Validate cross-artifact lineage and semantic identity."""
    if config.repository != model_dataset_manifest.repository:
        raise InferenceBundleError("Config repository does not match model-dataset manifest.")
    if config.model_dataset_id != model_dataset_manifest.model_dataset_id:
        raise InferenceBundleError("Config model_dataset_id does not match model-dataset manifest.")
    if config.text_representation_version != model_dataset_manifest.text_representation_version:
        raise InferenceBundleError(
            "Config text_representation_version does not match model-dataset manifest."
        )
    if config.baseline_run_id != baseline_manifest.baseline_run_id:
        raise InferenceBundleError("Config baseline_run_id does not match baseline manifest.")
    if config.threshold_policy_id != threshold_manifest.policy_id:
        raise InferenceBundleError("Config threshold_policy_id does not match threshold manifest.")
    if config.abstention_policy_id != abstention_manifest.policy_id:
        raise InferenceBundleError(
            "Config abstention_policy_id does not match abstention manifest."
        )
    if config.retrieval_run_id != retrieval_manifest.retrieval_run_id:
        raise InferenceBundleError("Config retrieval_run_id does not match retrieval manifest.")

    if threshold_manifest.baseline_run_id != baseline_manifest.baseline_run_id:
        raise InferenceBundleError("Threshold policy baseline_run_id does not match baseline.")
    if abstention_manifest.threshold_policy_id != threshold_manifest.policy_id:
        raise InferenceBundleError(
            "Abstention policy threshold_policy_id does not match threshold."
        )
    if abstention_manifest.baseline_run_id != baseline_manifest.baseline_run_id:
        raise InferenceBundleError("Abstention policy baseline_run_id does not match baseline.")

    if baseline_manifest.model_dataset_id != model_dataset_manifest.model_dataset_id:
        raise InferenceBundleError("Baseline model_dataset_id does not match model-dataset.")
    if retrieval_manifest.model_dataset_id != model_dataset_manifest.model_dataset_id:
        raise InferenceBundleError("Retrieval model_dataset_id does not match model-dataset.")

    model_semantic = baseline_manifest.model_semantic_sha256
    if threshold_manifest.model_semantic_sha256 != model_semantic:
        raise InferenceBundleError("Threshold policy model_semantic_sha256 mismatch.")
    if abstention_manifest.model_semantic_sha256 != model_semantic:
        raise InferenceBundleError("Abstention policy model_semantic_sha256 mismatch.")

    if (
        threshold_manifest.selected_threshold_basis_points
        != abstention_manifest.classification_threshold_basis_points
    ):
        raise InferenceBundleError(
            "Threshold and abstention classification threshold basis points disagree."
        )

    if retrieval_manifest.records_sha256 != model_dataset_manifest.records_sha256:
        raise InferenceBundleError("Retrieval records_sha256 does not match model-dataset.")
    if retrieval_manifest.label_map_sha256 != model_dataset_manifest.label_map_sha256:
        raise InferenceBundleError("Retrieval label_map_sha256 does not match model-dataset.")

    if model.labels != label_map.labels:
        raise InferenceBundleError("Model labels do not match label_map.labels order.")
    if len(label_map.labels) != label_map.target_count:
        raise InferenceBundleError("label_map target_count does not match labels length.")


def load_inference_bundle(
    config_path: Path,
    *,
    repository: RepositoryRef,
    baselines_root: Path = DEFAULT_BASELINES_ROOT,
    threshold_policies_root: Path = DEFAULT_THRESHOLD_POLICIES_ROOT,
    abstention_policies_root: Path = DEFAULT_ABSTENTION_POLICIES_ROOT,
    retrieval_baselines_root: Path = DEFAULT_RETRIEVAL_BASELINES_ROOT,
    model_ready_root: Path = DEFAULT_MODEL_READY_ROOT,
) -> LoadedInferenceBundle:
    """Load, integrity-check, and validate all artifacts for local inference."""
    from repotriage.inference.config import load_inference_config

    config = load_inference_config(config_path)
    if config.repository != repository.full_name:
        raise InferenceBundleError(
            f"Config repository {config.repository!r} does not match requested "
            f"{repository.full_name!r}."
        )

    model_dataset_dir = model_ready_root / repository.slug / config.model_dataset_id
    model_dataset_manifest, label_map, _split_report = validate_model_dataset_artifact_integrity(
        model_dataset_dir,
        expected_repository=repository,
        expected_model_dataset_id=config.model_dataset_id,
    )
    label_order = list(label_map.labels)

    baseline_dir = baselines_root / repository.slug / config.baseline_run_id
    baseline_manifest = validate_baseline_artifact_integrity(
        baseline_dir,
        expected_repository=repository,
        expected_baseline_run_id=config.baseline_run_id,
    )

    threshold_dir = threshold_policies_root / repository.slug / config.threshold_policy_id
    threshold_manifest = validate_threshold_policy_artifact_integrity(
        threshold_dir,
        expected_repository=repository,
        expected_policy_id=config.threshold_policy_id,
    )
    threshold_inputs = load_threshold_policy_inputs(
        threshold_dir,
        expected_policy_id=config.threshold_policy_id,
        expected_repository=repository,
    )
    classification_threshold_basis_points = (
        threshold_inputs.policy_document.selection.selected_threshold_basis_points
    )
    if classification_threshold_basis_points != threshold_manifest.selected_threshold_basis_points:
        raise InferenceBundleError(
            "Threshold policy document selected threshold does not match manifest."
        )
    classification_threshold = classification_threshold_basis_points / 100

    abstention_dir = abstention_policies_root / repository.slug / config.abstention_policy_id
    abstention_manifest = validate_abstention_policy_artifact_integrity(
        abstention_dir,
        expected_repository=repository,
        expected_policy_id=config.abstention_policy_id,
    )
    abstention_policy_document = _load_abstention_policy_document(abstention_dir)
    abstention_threshold_basis_points = (
        abstention_policy_document.selection.selected_abstention_basis_points
    )
    abstention_threshold = abstention_threshold_basis_points / 100

    retrieval_dir = retrieval_baselines_root / repository.slug / config.retrieval_run_id
    retrieval_manifest = validate_retrieval_artifact_integrity(
        retrieval_dir,
        expected_repository=repository,
        expected_retrieval_run_id=config.retrieval_run_id,
    )

    model = _load_baseline_model(baseline_dir, baseline_manifest)
    retrieval_index = _load_retrieval_index(
        retrieval_dir,
        retrieval_manifest,
        label_order=label_order,
    )

    validate_inference_bundle_compatibility(
        config=config,
        model_dataset_manifest=model_dataset_manifest,
        label_map=label_map,
        baseline_manifest=baseline_manifest,
        threshold_manifest=threshold_manifest,
        abstention_manifest=abstention_manifest,
        retrieval_manifest=retrieval_manifest,
        model=model,
    )

    serialization_warning = baseline_manifest.environment.serialization_security_warning

    return LoadedInferenceBundle(
        config=config,
        config_path=config_path,
        repository=repository,
        model_dataset_manifest=model_dataset_manifest,
        label_map=label_map,
        label_order=label_order,
        baseline_manifest=baseline_manifest,
        threshold_manifest=threshold_manifest,
        abstention_manifest=abstention_manifest,
        retrieval_manifest=retrieval_manifest,
        model=model,
        retrieval_index=retrieval_index,
        classification_threshold=classification_threshold,
        classification_threshold_basis_points=classification_threshold_basis_points,
        abstention_threshold=abstention_threshold,
        abstention_threshold_basis_points=abstention_threshold_basis_points,
        confidence_definition=abstention_manifest.confidence_definition,
        serialization_security_warning=serialization_warning,
    )
