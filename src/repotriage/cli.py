"""Command-line interface for RepoTriage."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from repotriage.dataset.builder import (
    DEFAULT_PROCESSED_ROOT,
    build_dataset,
    format_dataset_summary,
)
from repotriage.dataset.models import DatasetError
from repotriage.github.client import GitHubAPIError, GitHubRateLimitError
from repotriage.github.ingestion import DEFAULT_OUTPUT_ROOT, fetch_repository_issues, format_summary
from repotriage.github.models import (
    CacheConflictError,
    CacheCorruptionError,
    InvalidRepositoryError,
    parse_repository,
)

logger = logging.getLogger(__name__)


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


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "fetch-issues":
        return run_fetch_issues(args)

    if args.command == "build-dataset":
        return run_build_dataset(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
