"""Pure transformation of raw GitHub issue records into normalized issues."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from repotriage.dataset.models import (
    MalformedIssueError,
    NormalizedIssue,
    RawIssueRecord,
    RawLabel,
)


def normalize_labels(raw_labels: list[RawLabel]) -> list[str]:
    """Extract label names with exact duplicates removed and a deterministic sort.

    Original spelling and casing are preserved. Exact duplicates are removed via a
    set, then names are sorted ascending by Unicode code point (case-sensitive, so
    uppercase sorts before lowercase). Empty label lists are returned unchanged.
    """
    names = {label.name for label in raw_labels}
    return sorted(names)


def _describe_source(source_page: str, position: int, raw_item: dict[str, Any]) -> str:
    identifier = raw_item.get("number")
    if identifier is None:
        identifier = raw_item.get("id")
    identifier_text = f" issue {identifier}" if identifier is not None else ""
    return f"{source_page} position {position}{identifier_text}"


def normalize_issue(
    raw_item: dict[str, Any],
    *,
    repository: str,
    source_page: str,
    position: int,
) -> NormalizedIssue:
    """Normalize a single raw issue record into a :class:`NormalizedIssue`.

    ``raw_item`` must already have been confirmed not to be a pull request. On any
    validation failure a :class:`MalformedIssueError` is raised carrying the source
    page, position, and issue identifier when available.
    """
    try:
        record = RawIssueRecord.model_validate(raw_item)
    except ValidationError as exc:
        location = _describe_source(source_page, position, raw_item)
        raise MalformedIssueError(f"Malformed issue record at {location}: {exc}") from exc

    author_login = record.user.login if record.user is not None else None
    author_type = record.user.type if record.user is not None else None

    try:
        return NormalizedIssue(
            repository=repository,
            issue_id=record.id,
            issue_number=record.number,
            title=record.title,
            body=record.body if record.body is not None else "",
            labels=normalize_labels(record.labels),
            state=record.state,
            author_login=author_login,
            author_type=author_type,
            created_at=record.created_at,
            updated_at=record.updated_at,
            closed_at=record.closed_at,
            comments_count=record.comments,
            html_url=record.html_url,
            source_page=source_page,
        )
    except ValidationError as exc:
        location = _describe_source(source_page, position, raw_item)
        raise MalformedIssueError(f"Malformed issue record at {location}: {exc}") from exc
