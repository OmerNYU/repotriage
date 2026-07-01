"""Command-line interface for RepoTriage."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

from repotriage.audit.builder import (
    DEFAULT_AUDITS_ROOT,
    audit_dataset,
    format_audit_summary,
)
from repotriage.audit.models import AUDIT_ID_PATTERN, AuditError
from repotriage.dataset.builder import (
    DEFAULT_PROCESSED_ROOT,
    build_dataset,
    format_dataset_summary,
)
from repotriage.dataset.models import DATASET_ID_PATTERN, DatasetError
from repotriage.github.client import GitHubAPIError, GitHubRateLimitError
from repotriage.github.ingestion import DEFAULT_OUTPUT_ROOT, fetch_repository_issues, format_summary
from repotriage.github.models import (
    CacheConflictError,
    CacheCorruptionError,
    InvalidRepositoryError,
    parse_repository,
)
from repotriage.label_policy.builder import (
    DEFAULT_POLICIES_ROOT,
    build_label_policy,
    format_label_policy_summary,
)
from repotriage.label_policy.models import POLICY_ID_PATTERN, LabelPolicyError
from repotriage.model_dataset.builder import (
    DEFAULT_MODEL_READY_ROOT,
    build_model_dataset,
    format_model_dataset_summary,
)
from repotriage.model_dataset.models import MODEL_DATASET_ID_PATTERN, ModelDatasetError

logger = logging.getLogger(__name__)

_DATASET_ID_RE = re.compile(DATASET_ID_PATTERN)
_AUDIT_ID_RE = re.compile(AUDIT_ID_PATTERN)
_POLICY_ID_RE = re.compile(POLICY_ID_PATTERN)
_MODEL_DATASET_ID_RE = re.compile(MODEL_DATASET_ID_PATTERN)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="repotriage", description="RepoTriage CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser(
        "fetch-issues",
        help="Download GitHub repository issues into the local raw cache",
    )
    fetch_parser.add_argument(
        "--repo",
        required=True,
        help="Repository in owner/name form, for example pandas-dev/pandas",
    )
    fetch_parser.add_argument(
        "--max-pages",
        type=int,
        required=True,
        help="Maximum number of API pages to fetch (positive integer)",
    )
    fetch_parser.add_argument(
        "--refresh",
        action="store_true",
        help="Replace an existing cached import",
    )
    fetch_parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root directory for raw GitHub data (default: {DEFAULT_OUTPUT_ROOT})",
    )

    build_parser = subparsers.add_parser(
        "build-dataset",
        help="Normalize a raw GitHub snapshot into an immutable issue-only dataset",
    )
    build_parser.add_argument(
        "--repo",
        required=True,
        help="Repository in owner/name form, for example pandas-dev/pandas",
    )
    build_parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root directory for raw GitHub data (default: {DEFAULT_OUTPUT_ROOT})",
    )
    build_parser.add_argument(
        "--processed-root",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
        help=f"Root directory for processed datasets (default: {DEFAULT_PROCESSED_ROOT})",
    )

    audit_parser = subparsers.add_parser(
        "audit-dataset",
        help="Audit one explicit normalized dataset into an immutable audit artifact",
    )
    audit_parser.add_argument(
        "--repo",
        required=True,
        help="Repository in owner/name form, for example pandas-dev/pandas",
    )
    audit_parser.add_argument(
        "--dataset-id",
        required=True,
        help="Explicit normalized dataset id to audit",
    )
    audit_parser.add_argument(
        "--processed-root",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
        help=f"Root directory for processed datasets (default: {DEFAULT_PROCESSED_ROOT})",
    )
    audit_parser.add_argument(
        "--audits-root",
        type=Path,
        default=DEFAULT_AUDITS_ROOT,
        help=f"Root directory for audit artifacts (default: {DEFAULT_AUDITS_ROOT})",
    )

    policy_parser = subparsers.add_parser(
        "build-label-policy",
        help="Build an immutable target-label policy from a dataset, audit, and config",
    )
    policy_parser.add_argument(
        "--repo",
        required=True,
        help="Repository in owner/name form, for example pandas-dev/pandas",
    )
    policy_parser.add_argument(
        "--dataset-id",
        required=True,
        help="Explicit normalized dataset id",
    )
    policy_parser.add_argument(
        "--audit-id",
        required=True,
        help="Explicit audit id for the same dataset",
    )
    policy_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the human-authored label-policy configuration JSON",
    )
    policy_parser.add_argument(
        "--processed-root",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
        help=f"Root directory for processed datasets (default: {DEFAULT_PROCESSED_ROOT})",
    )
    policy_parser.add_argument(
        "--audits-root",
        type=Path,
        default=DEFAULT_AUDITS_ROOT,
        help=f"Root directory for audit artifacts (default: {DEFAULT_AUDITS_ROOT})",
    )
    policy_parser.add_argument(
        "--policies-root",
        type=Path,
        default=DEFAULT_POLICIES_ROOT,
        help=f"Root directory for policy artifacts (default: {DEFAULT_POLICIES_ROOT})",
    )

    model_dataset_parser = subparsers.add_parser(
        "build-model-dataset",
        help="Build an immutable model-ready dataset from a dataset, policy, and split config",
    )
    model_dataset_parser.add_argument(
        "--repo",
        required=True,
        help="Repository in owner/name form, for example pandas-dev/pandas",
    )
    model_dataset_parser.add_argument(
        "--dataset-id",
        required=True,
        help="Explicit normalized dataset id",
    )
    model_dataset_parser.add_argument(
        "--policy-id",
        required=True,
        help="Explicit label-policy id for the same dataset",
    )
    model_dataset_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the temporal split configuration JSON",
    )
    model_dataset_parser.add_argument(
        "--processed-root",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
        help=f"Root directory for processed datasets (default: {DEFAULT_PROCESSED_ROOT})",
    )
    model_dataset_parser.add_argument(
        "--policies-root",
        type=Path,
        default=DEFAULT_POLICIES_ROOT,
        help=f"Root directory for policy artifacts (default: {DEFAULT_POLICIES_ROOT})",
    )
    model_dataset_parser.add_argument(
        "--model-ready-root",
        type=Path,
        default=DEFAULT_MODEL_READY_ROOT,
        help=f"Root directory for model-ready artifacts (default: {DEFAULT_MODEL_READY_ROOT})",
    )
    return parser


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


def run_fetch_issues(args: argparse.Namespace) -> int:
    if args.max_pages < 1:
        print("--max-pages must be a positive integer.", file=sys.stderr)
        return 2

    try:
        repository = parse_repository(args.repo)
    except InvalidRepositoryError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        result = fetch_repository_issues(
            repository,
            max_pages=args.max_pages,
            output_root=args.output_root,
            refresh=args.refresh,
        )
    except (CacheConflictError, CacheCorruptionError, GitHubRateLimitError, GitHubAPIError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(format_summary(result))
    return 0


def run_build_dataset(args: argparse.Namespace) -> int:
    try:
        repository = parse_repository(args.repo)
    except InvalidRepositoryError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        result = build_dataset(
            repository,
            raw_root=args.raw_root,
            processed_root=args.processed_root,
        )
    except (CacheConflictError, CacheCorruptionError, DatasetError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(format_dataset_summary(result))
    return 0


def run_audit_dataset(args: argparse.Namespace) -> int:
    try:
        repository = parse_repository(args.repo)
    except InvalidRepositoryError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if not _DATASET_ID_RE.fullmatch(args.dataset_id):
        print(
            f"Invalid dataset id {args.dataset_id!r}. Expected a content-aware dataset id "
            "such as 20260628T161306010651Z-n1-074402d21505.",
            file=sys.stderr,
        )
        return 2

    try:
        result = audit_dataset(
            repository,
            args.dataset_id,
            processed_root=args.processed_root,
            audits_root=args.audits_root,
        )
    except (DatasetError, AuditError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(format_audit_summary(result))
    return 0


def run_build_label_policy(args: argparse.Namespace) -> int:
    try:
        repository = parse_repository(args.repo)
    except InvalidRepositoryError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if not _DATASET_ID_RE.fullmatch(args.dataset_id):
        print(
            f"Invalid dataset id {args.dataset_id!r}. Expected a content-aware dataset id "
            "such as 20260628T161306010651Z-n1-074402d21505.",
            file=sys.stderr,
        )
        return 2

    if not _AUDIT_ID_RE.fullmatch(args.audit_id):
        print(
            f"Invalid audit id {args.audit_id!r}. Expected an audit id such as "
            "20260628T161306010651Z-n1-074402d21505-a2.",
            file=sys.stderr,
        )
        return 2

    try:
        result = build_label_policy(
            repository,
            args.dataset_id,
            args.audit_id,
            args.config,
            processed_root=args.processed_root,
            audits_root=args.audits_root,
            policies_root=args.policies_root,
        )
    except (DatasetError, AuditError, LabelPolicyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(format_label_policy_summary(result))
    return 0


def run_build_model_dataset(args: argparse.Namespace) -> int:
    try:
        repository = parse_repository(args.repo)
    except InvalidRepositoryError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if not _DATASET_ID_RE.fullmatch(args.dataset_id):
        print(
            f"Invalid dataset id {args.dataset_id!r}. Expected a content-aware dataset id "
            "such as 20260628T161306010651Z-n1-074402d21505.",
            file=sys.stderr,
        )
        return 2

    if not _POLICY_ID_RE.fullmatch(args.policy_id):
        print(
            f"Invalid policy id {args.policy_id!r}. Expected a policy id such as "
            "20260628T161306010651Z-n1-074402d21505-lp2-95899f0f5b37.",
            file=sys.stderr,
        )
        return 2

    try:
        result = build_model_dataset(
            repository,
            args.dataset_id,
            args.policy_id,
            args.config,
            processed_root=args.processed_root,
            policies_root=args.policies_root,
            model_ready_root=args.model_ready_root,
        )
    except (DatasetError, LabelPolicyError, ModelDatasetError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(format_model_dataset_summary(result))
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "fetch-issues":
        return run_fetch_issues(args)

    if args.command == "build-dataset":
        return run_build_dataset(args)

    if args.command == "audit-dataset":
        return run_audit_dataset(args)

    if args.command == "build-label-policy":
        return run_build_label_policy(args)

    if args.command == "build-model-dataset":
        return run_build_model_dataset(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
