"""Build (or reuse) an immutable target-label policy artifact (lp2 contract).

The policy binds itself to the normalized dataset bytes, the audit artifact bytes, the
audit id, the configuration schema, and the canonical configuration hash via a single
``policy_input_sha256``. It computes objective coverage, resolves human-authored decisions,
enforces selection criteria, then publishes ``label_policy.json``, ``label_policy.md``, and
``manifest.json`` atomically from a hidden staging directory. Immutable policy ids are
never overwritten; an existing corrupt or incompatible artifact is reported, not replaced.
The explicit ``--audit-id`` is honoured as given and must reference an explicitly supported
audit contract; the policy never follows the audit package's current version implicitly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from repotriage.audit.builder import DEFAULT_AUDITS_ROOT, validate_audit_artifact
from repotriage.audit.models import AuditDocument, AuditManifest
from repotriage.dataset.builder import (
    DEFAULT_PROCESSED_ROOT,
    validate_processed_dataset_integrity,
)
from repotriage.dataset.models import ISSUE_SCHEMA_VERSION, ProcessedManifest
from repotriage.filesystem import atomic_write_bytes, best_effort_remove_tree
from repotriage.github.models import RepositoryRef
from repotriage.label_policy.analyzer import analyze_label_policy
from repotriage.label_policy.config import load_config
from repotriage.label_policy.models import (
    LABEL_POLICY_DOCUMENT_SCHEMA_VERSION,
    LABEL_POLICY_JSON_FILE,
    LABEL_POLICY_MANIFEST_SCHEMA_VERSION,
    LABEL_POLICY_MARKDOWN_FILE,
    LABEL_POLICY_VERSION,
    SUPPORTED_AUDIT_DOCUMENT_SCHEMA_VERSIONS,
    SUPPORTED_AUDIT_VERSIONS,
    SUPPORTED_CONFIG_SCHEMA_VERSIONS,
    LabelPolicyConfigError,
    LabelPolicyCorruptionError,
    LabelPolicyDocument,
    LabelPolicyError,
    LabelPolicyIdentity,
    LabelPolicyInputError,
    LabelPolicyManifest,
    compute_policy_id,
    compute_policy_input_sha256,
)
from repotriage.label_policy.report import (
    serialize_policy_json,
    serialize_policy_markdown,
    sha256_hex,
)
from repotriage.paths import resolve_within_directory

logger = logging.getLogger(__name__)

DEFAULT_POLICIES_ROOT = Path("data/policies/github")


@dataclass(frozen=True)
class LabelPolicyBuildResult:
    """Summary of a policy build or policy-cache hit."""

    repository: RepositoryRef
    policy_dir: Path
    manifest: LabelPolicyManifest
    document: LabelPolicyDocument
    cache_hit: bool


def _load_and_validate_dataset(
    repository: RepositoryRef, dataset_id: str, processed_root: Path
) -> tuple[Path, ProcessedManifest]:
    """Resolve and integrity-validate the normalized dataset (no raw cache required)."""
    dataset_dir = processed_root / repository.slug / dataset_id
    if not dataset_dir.is_dir():
        raise LabelPolicyInputError(
            f"No normalized dataset found for {repository.full_name} with dataset id "
            f"{dataset_id!r} at {dataset_dir}."
        )
    manifest = validate_processed_dataset_integrity(
        dataset_dir,
        expected_repository=repository,
        expected_dataset_id=dataset_id,
        expected_issue_schema_version=ISSUE_SCHEMA_VERSION,
    )
    return dataset_dir, manifest


def _load_and_validate_audit(
    repository: RepositoryRef,
    processed_manifest: ProcessedManifest,
    audit_id: str,
    audits_root: Path,
) -> tuple[Path, AuditManifest, AuditDocument]:
    """Resolve and validate the explicit audit artifact and confirm it matches the dataset.

    The supplied ``audit_id`` is honoured as given; the audit is then checked against the
    explicitly supported audit-version and document-schema allowlists for this policy
    version, so a newly minted audit contract is never auto-accepted.
    """
    audit_dir = audits_root / repository.slug / audit_id
    if not audit_dir.is_dir():
        raise LabelPolicyInputError(
            f"No audit artifact found for {repository.full_name} with audit id "
            f"{audit_id!r} at {audit_dir}."
        )

    audit_manifest, audit_document = validate_audit_artifact(
        audit_dir,
        expected_repository=repository,
        expected_dataset_id=processed_manifest.dataset_id,
        expected_dataset_output_sha256=processed_manifest.output_sha256,
        expected_issue_schema_version=processed_manifest.issue_schema_version,
        expected_normalizer_version=processed_manifest.normalizer_version,
        expected_audit_id=audit_id,
    )

    if audit_manifest.audit_version not in SUPPORTED_AUDIT_VERSIONS:
        raise LabelPolicyInputError(
            f"Audit version {audit_manifest.audit_version!r} is not supported by policy "
            f"version {LABEL_POLICY_VERSION} (supported: "
            f"{sorted(SUPPORTED_AUDIT_VERSIONS)})."
        )
    if (
        audit_manifest.audit_document_schema_version
        not in SUPPORTED_AUDIT_DOCUMENT_SCHEMA_VERSIONS
    ):
        raise LabelPolicyInputError(
            f"Audit document schema version "
            f"{audit_manifest.audit_document_schema_version!r} is not supported by policy "
            f"version {LABEL_POLICY_VERSION} (supported: "
            f"{sorted(SUPPORTED_AUDIT_DOCUMENT_SCHEMA_VERSIONS)})."
        )

    if audit_manifest.issues_analyzed != processed_manifest.issues_written:
        raise LabelPolicyInputError(
            f"Audit issues_analyzed {audit_manifest.issues_analyzed} disagrees with the "
            f"processed dataset issues_written {processed_manifest.issues_written}."
        )
    if audit_document.repository_summary.total_issues != processed_manifest.issues_written:
        raise LabelPolicyInputError(
            "Audit total issue count disagrees with the processed dataset issues_written."
        )
    return audit_dir, audit_manifest, audit_document


def _verify_report_file(
    policy_dir: Path,
    *,
    relative_path: str,
    expected_sha256: str,
    parse_document: bool,
) -> LabelPolicyDocument | None:
    try:
        resolved = resolve_within_directory(policy_dir, relative_path)
    except ValueError as exc:
        raise LabelPolicyCorruptionError(
            f"Policy manifest references an unsafe path: {relative_path!r}"
        ) from exc
    if not resolved.is_file():
        raise LabelPolicyCorruptionError(f"Missing policy report file: {relative_path}")
    data = resolved.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise LabelPolicyCorruptionError(
            f"Policy report hash mismatch for {relative_path}: expected {expected_sha256}, "
            f"found {actual}."
        )
    if not parse_document:
        return None
    try:
        return LabelPolicyDocument.model_validate_json(data.decode("utf-8"))
    except (ValidationError, UnicodeDecodeError) as exc:
        raise LabelPolicyCorruptionError(
            f"Invalid policy document in {relative_path}: {exc}"
        ) from exc


def validate_label_policy_artifact(
    policy_dir: Path,
    *,
    expected_repository: RepositoryRef,
    expected_dataset_id: str,
    expected_dataset_output_sha256: str,
    expected_audit_id: str,
    expected_audit_json_sha256: str,
    expected_audit_version: str,
    expected_config_schema_version: str,
    expected_config_sha256: str,
    expected_issue_schema_version: str,
    expected_normalizer_version: str,
    expected_policy_id: str,
    expected_policy_input_sha256: str,
    check_dir_name: bool = True,
) -> tuple[LabelPolicyManifest, LabelPolicyDocument]:
    """Validate an on-disk policy artifact, raising on any corruption or mismatch."""
    manifest_path = policy_dir / "manifest.json"
    if not manifest_path.is_file():
        raise LabelPolicyCorruptionError(f"Missing policy manifest at {manifest_path}")
    try:
        manifest = LabelPolicyManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (ValidationError, json.JSONDecodeError) as exc:
        raise LabelPolicyCorruptionError(
            f"Invalid policy manifest at {manifest_path}: {exc}"
        ) from exc
    except OSError as exc:
        raise LabelPolicyCorruptionError(
            f"Unable to read policy manifest at {manifest_path}: {exc}"
        ) from exc

    if check_dir_name and policy_dir.name != manifest.policy_id:
        raise LabelPolicyCorruptionError(
            f"Policy directory {policy_dir.name!r} does not match manifest policy_id "
            f"{manifest.policy_id!r}."
        )

    expectations: list[tuple[str, object, object]] = [
        ("policy_id", manifest.policy_id, expected_policy_id),
        ("policy_input_sha256", manifest.policy_input_sha256, expected_policy_input_sha256),
        ("policy_version", manifest.policy_version, LABEL_POLICY_VERSION),
        ("schema_version", manifest.schema_version, LABEL_POLICY_MANIFEST_SCHEMA_VERSION),
        (
            "label_policy_document_schema_version",
            manifest.label_policy_document_schema_version,
            LABEL_POLICY_DOCUMENT_SCHEMA_VERSION,
        ),
        ("repository", manifest.repository, expected_repository.full_name),
        ("dataset_id", manifest.dataset_id, expected_dataset_id),
        ("dataset_output_sha256", manifest.dataset_output_sha256, expected_dataset_output_sha256),
        ("audit_id", manifest.audit_id, expected_audit_id),
        ("audit_json_sha256", manifest.audit_json_sha256, expected_audit_json_sha256),
        ("audit_version", manifest.audit_version, expected_audit_version),
        (
            "config_schema_version",
            manifest.config_schema_version,
            expected_config_schema_version,
        ),
        ("config_sha256", manifest.config_sha256, expected_config_sha256),
        ("issue_schema_version", manifest.issue_schema_version, expected_issue_schema_version),
        ("normalizer_version", manifest.normalizer_version, expected_normalizer_version),
    ]
    for field_name, actual, expected in expectations:
        if actual != expected:
            raise LabelPolicyCorruptionError(
                f"Policy manifest {field_name} {actual!r} does not match expected "
                f"{expected!r}."
            )

    document = _verify_report_file(
        policy_dir,
        relative_path=manifest.label_policy_json_file,
        expected_sha256=manifest.label_policy_json_sha256,
        parse_document=True,
    )
    _verify_report_file(
        policy_dir,
        relative_path=manifest.label_policy_markdown_file,
        expected_sha256=manifest.label_policy_markdown_sha256,
        parse_document=False,
    )
    assert document is not None
    _cross_check_document_against_manifest(manifest, document)
    return manifest, document


def _cross_check_document_against_manifest(
    manifest: LabelPolicyManifest, document: LabelPolicyDocument
) -> None:
    """Verify ``label_policy.json`` agrees semantically with ``manifest.json``."""
    identity = document.identity
    coverage = document.coverage
    checks: list[tuple[str, object, object]] = [
        ("policy_id", identity.policy_id, manifest.policy_id),
        ("policy_input_sha256", identity.policy_input_sha256, manifest.policy_input_sha256),
        ("policy_version", identity.policy_version, manifest.policy_version),
        (
            "label_policy_document_schema_version",
            document.schema_version,
            manifest.label_policy_document_schema_version,
        ),
        ("repository", identity.repository, manifest.repository),
        ("dataset_id", identity.dataset_id, manifest.dataset_id),
        ("dataset_output_sha256", identity.dataset_output_sha256, manifest.dataset_output_sha256),
        ("audit_id", identity.audit_id, manifest.audit_id),
        ("audit_json_sha256", identity.audit_json_sha256, manifest.audit_json_sha256),
        ("audit_version", identity.audit_version, manifest.audit_version),
        (
            "config_schema_version",
            identity.config_schema_version,
            manifest.config_schema_version,
        ),
        ("config_sha256", identity.config_sha256, manifest.config_sha256),
        ("issue_schema_version", identity.issue_schema_version, manifest.issue_schema_version),
        ("normalizer_version", identity.normalizer_version, manifest.normalizer_version),
        ("total_audited_labels", coverage.total_audited_labels, manifest.total_audited_labels),
        ("included_label_count", coverage.included_label_count, manifest.included_label_count),
        ("deferred_label_count", coverage.deferred_label_count, manifest.deferred_label_count),
        ("excluded_label_count", coverage.excluded_label_count, manifest.excluded_label_count),
        ("explicit_label_count", coverage.explicit_label_count, manifest.explicit_label_count),
        ("default_label_count", coverage.default_label_count, manifest.default_label_count),
        (
            "issues_with_included_target",
            coverage.issues_with_included_target,
            manifest.issues_with_included_target,
        ),
        (
            "issues_without_included_target",
            coverage.issues_without_included_target,
            manifest.issues_without_included_target,
        ),
        (
            "target_coverage_fraction",
            coverage.target_coverage_fraction,
            manifest.target_coverage_fraction,
        ),
        (
            "included_target_assignments",
            coverage.included_target_assignments,
            manifest.included_target_assignments,
        ),
        (
            "included_target_cardinality",
            coverage.included_target_cardinality,
            manifest.included_target_cardinality,
        ),
    ]
    for field_name, document_value, manifest_value in checks:
        if document_value != manifest_value:
            raise LabelPolicyCorruptionError(
                f"label_policy.json {field_name} {document_value!r} disagrees with manifest "
                f"{field_name} {manifest_value!r}."
            )


def publish_label_policy(staging_dir: Path, final_dir: Path) -> None:
    """Atomically publish a completed staging directory to its immutable final path."""
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    if final_dir.exists():
        raise LabelPolicyError(f"Refusing to overwrite existing policy directory {final_dir}.")
    os.rename(staging_dir, final_dir)


def build_label_policy(
    repository: RepositoryRef,
    dataset_id: str,
    audit_id: str,
    config_path: Path,
    *,
    processed_root: Path = DEFAULT_PROCESSED_ROOT,
    audits_root: Path = DEFAULT_AUDITS_ROOT,
    policies_root: Path = DEFAULT_POLICIES_ROOT,
) -> LabelPolicyBuildResult:
    """Build one target-label policy, publishing or reusing an immutable artifact."""
    dataset_dir, processed_manifest = _load_and_validate_dataset(
        repository, dataset_id, processed_root
    )
    audit_dir, audit_manifest, audit_document = _load_and_validate_audit(
        repository, processed_manifest, audit_id, audits_root
    )

    config, config_hash = load_config(config_path)
    if config.repository != repository.full_name:
        raise LabelPolicyConfigError(
            f"Configuration repository {config.repository!r} does not match requested "
            f"repository {repository.full_name!r}."
        )
    if config.config_schema_version not in SUPPORTED_CONFIG_SCHEMA_VERSIONS:
        raise LabelPolicyConfigError(
            f"Configuration schema version {config.config_schema_version!r} is not "
            f"supported by policy version {LABEL_POLICY_VERSION} (supported: "
            f"{sorted(SUPPORTED_CONFIG_SCHEMA_VERSIONS)})."
        )

    policy_input_sha256 = compute_policy_input_sha256(
        policy_version=LABEL_POLICY_VERSION,
        dataset_id=processed_manifest.dataset_id,
        dataset_output_sha256=processed_manifest.output_sha256,
        audit_id=audit_manifest.audit_id,
        audit_json_sha256=audit_manifest.audit_json_sha256,
        config_schema_version=config.config_schema_version,
        config_sha256=config_hash,
    )
    policy_id = compute_policy_id(dataset_id, policy_input_sha256, LABEL_POLICY_VERSION)
    final_dir = policies_root / repository.slug / policy_id

    validate_kwargs = dict(
        expected_repository=repository,
        expected_dataset_id=processed_manifest.dataset_id,
        expected_dataset_output_sha256=processed_manifest.output_sha256,
        expected_audit_id=audit_manifest.audit_id,
        expected_audit_json_sha256=audit_manifest.audit_json_sha256,
        expected_audit_version=audit_manifest.audit_version,
        expected_config_schema_version=config.config_schema_version,
        expected_config_sha256=config_hash,
        expected_issue_schema_version=processed_manifest.issue_schema_version,
        expected_normalizer_version=processed_manifest.normalizer_version,
        expected_policy_id=policy_id,
        expected_policy_input_sha256=policy_input_sha256,
    )

    if final_dir.exists():
        manifest, document = validate_label_policy_artifact(final_dir, **validate_kwargs)
        logger.info("Policy-cache hit for %s at %s", repository.full_name, final_dir)
        return LabelPolicyBuildResult(
            repository=repository,
            policy_dir=final_dir,
            manifest=manifest,
            document=document,
            cache_hit=True,
        )

    analysis = analyze_label_policy(dataset_dir, processed_manifest, audit_document, config)
    coverage = analysis.coverage

    identity = LabelPolicyIdentity(
        policy_version=LABEL_POLICY_VERSION,
        policy_id=policy_id,
        policy_input_sha256=policy_input_sha256,
        repository=processed_manifest.repository,
        dataset_id=processed_manifest.dataset_id,
        dataset_output_sha256=processed_manifest.output_sha256,
        audit_id=audit_manifest.audit_id,
        audit_json_sha256=audit_manifest.audit_json_sha256,
        audit_version=audit_manifest.audit_version,
        config_schema_version=config.config_schema_version,
        config_sha256=config_hash,
        issue_schema_version=processed_manifest.issue_schema_version,
        normalizer_version=processed_manifest.normalizer_version,
    )
    document = LabelPolicyDocument(
        identity=identity,
        selection_criteria=config.selection_criteria,
        coverage=coverage,
        decisions=analysis.decisions,
    )

    policy_json_bytes = serialize_policy_json(document)
    policy_markdown_bytes = serialize_policy_markdown(document)
    policy_json_sha256 = sha256_hex(policy_json_bytes)
    policy_markdown_sha256 = sha256_hex(policy_markdown_bytes)

    slug_dir = policies_root / repository.slug
    slug_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{repository.slug}.{policy_id}.staging-", dir=slug_dir)
    )

    logger.info(
        "Building label policy %s for %s (%d included, %d deferred, %d excluded)",
        policy_id,
        repository.full_name,
        coverage.included_label_count,
        coverage.deferred_label_count,
        coverage.excluded_label_count,
    )

    try:
        atomic_write_bytes(staging_dir / LABEL_POLICY_JSON_FILE, policy_json_bytes)
        atomic_write_bytes(staging_dir / LABEL_POLICY_MARKDOWN_FILE, policy_markdown_bytes)

        written_json_sha = sha256_hex((staging_dir / LABEL_POLICY_JSON_FILE).read_bytes())
        written_md_sha = sha256_hex((staging_dir / LABEL_POLICY_MARKDOWN_FILE).read_bytes())
        if written_json_sha != policy_json_sha256 or written_md_sha != policy_markdown_sha256:
            raise LabelPolicyError(
                "Policy report hash verification failed after writing staging files."
            )

        manifest = LabelPolicyManifest(
            policy_version=LABEL_POLICY_VERSION,
            policy_id=policy_id,
            policy_input_sha256=policy_input_sha256,
            repository=processed_manifest.repository,
            dataset_id=processed_manifest.dataset_id,
            dataset_output_sha256=processed_manifest.output_sha256,
            audit_id=audit_manifest.audit_id,
            audit_json_sha256=audit_manifest.audit_json_sha256,
            audit_version=audit_manifest.audit_version,
            config_schema_version=config.config_schema_version,
            config_sha256=config_hash,
            issue_schema_version=processed_manifest.issue_schema_version,
            normalizer_version=processed_manifest.normalizer_version,
            built_at=datetime.now(UTC),
            total_audited_labels=coverage.total_audited_labels,
            included_label_count=coverage.included_label_count,
            deferred_label_count=coverage.deferred_label_count,
            excluded_label_count=coverage.excluded_label_count,
            explicit_label_count=coverage.explicit_label_count,
            default_label_count=coverage.default_label_count,
            issues_with_included_target=coverage.issues_with_included_target,
            issues_without_included_target=coverage.issues_without_included_target,
            target_coverage_fraction=coverage.target_coverage_fraction,
            included_target_assignments=coverage.included_target_assignments,
            included_target_cardinality=coverage.included_target_cardinality,
            label_policy_json_file=LABEL_POLICY_JSON_FILE,
            label_policy_json_sha256=policy_json_sha256,
            label_policy_markdown_file=LABEL_POLICY_MARKDOWN_FILE,
            label_policy_markdown_sha256=policy_markdown_sha256,
        )
        atomic_write_bytes(
            staging_dir / "manifest.json",
            (manifest.model_dump_json() + "\n").encode("utf-8"),
        )

        validate_label_policy_artifact(staging_dir, check_dir_name=False, **validate_kwargs)

        publish_label_policy(staging_dir, final_dir)
    except BaseException:
        best_effort_remove_tree(staging_dir)
        raise

    logger.info(
        "Published label policy %s for %s at %s", policy_id, repository.full_name, final_dir
    )
    return LabelPolicyBuildResult(
        repository=repository,
        policy_dir=final_dir,
        manifest=manifest,
        document=document,
        cache_hit=False,
    )


def format_label_policy_summary(result: LabelPolicyBuildResult) -> str:
    """Build the user-facing policy summary."""
    manifest = result.manifest
    coverage = result.document.coverage
    lines = [
        f"Repository: {manifest.repository}",
        f"Dataset ID: {manifest.dataset_id}",
        f"Audit ID: {manifest.audit_id}",
        f"Policy ID: {manifest.policy_id}",
        f"Policy-input SHA-256: {manifest.policy_input_sha256}",
        f"Included labels: {manifest.included_label_count}",
        f"Deferred labels: {manifest.deferred_label_count}",
        f"Excluded labels: {manifest.excluded_label_count}",
        f"Explicitly reviewed labels: {manifest.explicit_label_count}",
        f"Default-applied labels: {manifest.default_label_count}",
        f"Target coverage: {coverage.target_coverage_fraction:.4f}",
        f"Issues with no included target: {coverage.issues_without_included_target}",
        f"Output directory: {result.policy_dir}",
        f"Policy-cache hit: {'yes' if result.cache_hit else 'no'}",
    ]
    return "\n".join(lines)
