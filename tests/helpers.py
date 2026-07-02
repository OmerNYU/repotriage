"""Shared test helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from repotriage.dataset.builder import serialize_issues_jsonl
from repotriage.dataset.models import (
    NORMALIZER_VERSION,
    NormalizedIssue,
    ProcessedManifest,
    compute_dataset_id,
    source_manifest_relpath,
)
from repotriage.github.models import (
    DEFAULT_ISSUE_REQUEST_PARAMETERS,
    GITHUB_API_VERSION,
    Manifest,
    RepositoryRef,
    count_item_types,
)


def make_issue(number: int) -> dict:
    return {"id": number, "number": number, "title": f"Issue {number}"}


def make_pull_request(number: int) -> dict:
    return {
        "id": number,
        "number": number,
        "title": f"PR {number}",
        "pull_request": {"url": f"https://api.github.com/repos/o/r/pulls/{number}"},
    }


_UNSET = object()


DEFAULT_TEST_REPOSITORY = "pandas-dev/pandas"


def make_raw_issue(
    number: int,
    *,
    repository: str = DEFAULT_TEST_REPOSITORY,
    issue_id: int | None = None,
    title: str | None = None,
    body: str | None = "Body text",
    labels: list[str] | None = None,
    state: str = "open",
    user: Any = _UNSET,
    comments: int = 0,
    created_at: str = "2026-06-24T16:09:03Z",
    updated_at: str = "2026-06-24T16:13:42Z",
    closed_at: str | None = None,
    html_url: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a raw GitHub issue record close to the real API shape.

    The default ``html_url`` is derived from ``repository`` and ``number`` so the record
    is internally consistent with the repository it is normalized under.
    """
    label_names = [] if labels is None else labels
    resolved_user: Any
    if user is _UNSET:
        resolved_user = {"login": f"user{number}", "type": "User"}
    else:
        resolved_user = user
    record: dict[str, Any] = {
        "id": issue_id if issue_id is not None else number,
        "number": number,
        "title": title if title is not None else f"Issue {number}",
        "body": body,
        "labels": [{"name": name} for name in label_names],
        "state": state,
        "user": resolved_user,
        "comments": comments,
        "created_at": created_at,
        "updated_at": updated_at,
        "closed_at": closed_at,
        "html_url": html_url or f"https://github.com/{repository}/issues/{number}",
    }
    if extra:
        record.update(extra)
    return record


def make_raw_pull_request(
    number: int, *, repository: str = DEFAULT_TEST_REPOSITORY, **kwargs: Any
) -> dict[str, Any]:
    """Build a raw record carrying the pull_request key."""
    record = make_raw_issue(number, repository=repository, **kwargs)
    record["html_url"] = f"https://github.com/{repository}/pull/{number}"
    record["pull_request"] = {
        "url": f"https://api.github.com/repos/{repository}/pulls/{number}"
    }
    return record


def write_raw_snapshot(
    raw_root: Path,
    repository: RepositoryRef,
    pages: list[list[dict[str, Any]]],
    *,
    fetched_at: datetime | None = None,
    api_version: str = GITHUB_API_VERSION,
) -> Path:
    """Write a synthetic but valid raw cache snapshot and return its directory."""
    cache_dir = raw_root / repository.slug
    pages_dir = cache_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    output_files: list[str] = []
    raw_total = 0
    issues_total = 0
    prs_total = 0
    for index, items in enumerate(pages, start=1):
        relative = f"pages/page_{index:04d}.json"
        (cache_dir / relative).write_text(
            json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        output_files.append(relative)
        page_raw, page_issues, page_prs = count_item_types(items)
        raw_total += page_raw
        issues_total += page_issues
        prs_total += page_prs

    pages_fetched = len(pages)
    manifest = Manifest(
        repository=repository.full_name,
        endpoint=repository.issues_base_endpoint,
        request_parameters=DEFAULT_ISSUE_REQUEST_PARAMETERS,
        fetched_at=fetched_at or datetime.now(UTC),
        api_version=api_version,
        authenticated=False,
        requested_max_pages=max(pages_fetched, 1),
        pages_fetched=pages_fetched,
        raw_items_received=raw_total,
        issues_received=issues_total,
        pull_requests_received=prs_total,
        output_files=output_files,
    )
    (cache_dir / "manifest.json").write_text(
        manifest.model_dump_json() + "\n", encoding="utf-8"
    )
    return cache_dir


def json_response(
    payload: object,
    *,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        headers=headers or {},
        content=json.dumps(payload).encode("utf-8"),
        request=httpx.Request("GET", "https://api.github.com/test"),
    )


DEFAULT_CREATED_AT = datetime(2025, 3, 1, 12, 0, 0, tzinfo=UTC)
DEFAULT_PROCESSED_FETCHED_AT = datetime(2026, 6, 28, 16, 13, 6, 10651, tzinfo=UTC)


def make_normalized_issue(
    number: int,
    *,
    repository: str = DEFAULT_TEST_REPOSITORY,
    issue_id: int | None = None,
    title: str | None = None,
    body: str = "Body text",
    labels: list[str] | None = None,
    state: str = "open",
    author_login: str | None = "octocat",
    author_type: str | None = "User",
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
    closed_at: datetime | None = None,
    comments_count: int = 0,
    source_page: str = "pages/page_0001.json",
) -> NormalizedIssue:
    """Construct a valid :class:`NormalizedIssue` for audit tests."""
    created = created_at or DEFAULT_CREATED_AT
    updated = updated_at or created
    resolved_author_type = author_type if author_login is not None else None
    return NormalizedIssue(
        repository=repository,
        issue_id=issue_id if issue_id is not None else number,
        issue_number=number,
        title=title if title is not None else f"Issue {number}",
        body=body,
        labels=labels if labels is not None else [],
        state=state,
        author_login=author_login,
        author_type=resolved_author_type,
        created_at=created,
        updated_at=updated,
        closed_at=closed_at,
        comments_count=comments_count,
        html_url=f"https://github.com/{repository}/issues/{number}",
        source_page=source_page,
    )


def write_label_policy_config(
    path: Path,
    *,
    repository: str = DEFAULT_TEST_REPOSITORY,
    labels: list[dict[str, Any]],
    default: dict[str, Any] | None = None,
    selection_criteria: dict[str, int] | None = None,
    notes: str = "",
    config_schema_version: str = "2",
    indent: int | None = 2,
) -> Path:
    """Write an lp2 label-policy configuration JSON file and return its path."""
    resolved_default = default or {
        "decision": "exclude",
        "role": "unreviewed",
        "reason_code": "unreviewed_default",
        "leakage_risk": "high",
        "explanation": "default exclusion",
    }
    payload: dict[str, Any] = {
        "config_schema_version": config_schema_version,
        "repository": repository,
        "notes": notes,
        "selection_criteria": selection_criteria
        or {
            "min_total_support": 1,
            "min_active_months": 1,
            "min_recent_support": 1,
            "recent_window_months": 4,
        },
        "default": resolved_default,
        "labels": labels,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=indent) + "\n", encoding="utf-8")
    return path


def write_temporal_split_config(
    path: Path,
    *,
    repository: str = DEFAULT_TEST_REPOSITORY,
    validation_start: str = "2026-02-01T00:00:00Z",
    test_start: str = "2026-04-01T00:00:00Z",
    low_support_warning_threshold: int = 5,
) -> Path:
    """Write a temporal split configuration JSON file and return its path."""
    payload = {
        "config_schema_version": "1",
        "repository": repository,
        "split_strategy": "temporal_calendar",
        "validation_start": validation_start,
        "test_start": test_start,
        "boundary_semantics": {
            "train": "created_at < validation_start",
            "validation": "validation_start <= created_at < test_start",
            "test": "created_at >= test_start",
        },
        "minimum_positive_support": {"train": 1, "validation": 1, "test": 1},
        "low_support_warning_threshold": low_support_warning_threshold,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def write_processed_dataset(
    processed_root: Path,
    repository: RepositoryRef,
    issues: list[NormalizedIssue],
    *,
    fetched_at: datetime | None = None,
    normalizer_version: str = NORMALIZER_VERSION,
    source_snapshot_sha256: str | None = None,
    pull_requests_excluded: int = 0,
) -> tuple[Path, str]:
    """Write a synthetic but valid normalized dataset and return (dir, dataset_id)."""
    fetched = fetched_at or DEFAULT_PROCESSED_FETCHED_AT
    snapshot = source_snapshot_sha256 or ("a" * 64)
    manifest_sha = "b" * 64

    output_bytes = serialize_issues_jsonl(issues)
    output_sha256 = hashlib.sha256(output_bytes).hexdigest()
    dataset_id = compute_dataset_id(fetched, normalizer_version, snapshot)

    issues_written = len(issues)
    unlabelled = sum(1 for issue in issues if not issue.labels)
    empty_body = sum(1 for issue in issues if issue.body == "")

    manifest = ProcessedManifest(
        dataset_id=dataset_id,
        repository=repository.full_name,
        normalizer_version=normalizer_version,
        built_at=fetched,
        source_manifest=source_manifest_relpath(repository.slug),
        source_manifest_sha256=manifest_sha,
        source_snapshot_sha256=snapshot,
        source_manifest_schema_version="2",
        source_fetched_at=fetched,
        source_api_version=GITHUB_API_VERSION,
        source_pages_fetched=1,
        raw_records_read=issues_written + pull_requests_excluded,
        pull_requests_excluded=pull_requests_excluded,
        issues_written=issues_written,
        unlabelled_issues=unlabelled,
        empty_body_issues=empty_body,
        output_file="issues.jsonl",
        output_sha256=output_sha256,
    )

    dataset_dir = processed_root / repository.slug / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "issues.jsonl").write_bytes(output_bytes)
    (dataset_dir / "manifest.json").write_text(
        manifest.model_dump_json() + "\n", encoding="utf-8"
    )
    return dataset_dir, dataset_id


def write_baseline_config(path: Path, *, repository: str = DEFAULT_TEST_REPOSITORY) -> Path:
    """Write a minimal baseline configuration JSON file and return its path."""
    payload = {
        "config_schema_version": "1",
        "baseline_version": "4",
        "repository": repository,
        "candidate_set_version": "1",
        "selection_rule_version": "1",
        "metric_contract_version": "2",
        "training_protocol_version": "train_only_v1",
        "random_state": 42,
        "threshold_policy": {"threshold": 0.5, "score_type": "probability_estimates"},
        "candidates": [
            {
                "candidate_id": "c1_unigram",
                "tfidf": {
                    "analyzer": "word",
                    "ngram_range": [1, 1],
                    "lowercase": True,
                    "min_df": 1,
                    "sublinear_tf": True,
                    "norm": "l2",
                },
                "logreg": {
                    "C": 1.0,
                    "solver": "lbfgs",
                    "max_iter": 2000,
                    "class_weight": None,
                },
            },
            {
                "candidate_id": "c2_bigram",
                "tfidf": {
                    "analyzer": "word",
                    "ngram_range": [1, 2],
                    "lowercase": True,
                    "min_df": 1,
                    "sublinear_tf": True,
                    "norm": "l2",
                },
                "logreg": {
                    "C": 1.0,
                    "solver": "lbfgs",
                    "max_iter": 2000,
                    "class_weight": None,
                },
            },
            {
                "candidate_id": "c3_bigram_balanced",
                "tfidf": {
                    "analyzer": "word",
                    "ngram_range": [1, 2],
                    "lowercase": True,
                    "min_df": 1,
                    "sublinear_tf": True,
                    "norm": "l2",
                },
                "logreg": {
                    "C": 1.0,
                    "solver": "lbfgs",
                    "max_iter": 2000,
                    "class_weight": "balanced",
                },
            },
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
