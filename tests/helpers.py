"""Shared test helpers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

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
