"""Pydantic models, validation, and domain exceptions for dataset normalization."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_serializer,
    field_validator,
    model_validator,
)

from repotriage.paths import resolve_within_directory

ISSUE_SCHEMA_VERSION: Literal["1"] = "1"
NORMALIZER_VERSION: Literal["1"] = "1"
PROCESSED_MANIFEST_SCHEMA_VERSION: Literal["1"] = "1"

DEFAULT_OUTPUT_FILE = "issues.jsonl"

SHA256_PATTERN = r"^[0-9a-f]{64}$"
# <YYYYMMDD>T<HHMMSS+microseconds = 12 digits>Z-n<version>-<12 lowercase hex>
DATASET_ID_PATTERN = r"^\d{8}T\d{12}Z-n[1-9]\d*-[0-9a-f]{12}$"

_SHA256_RE = re.compile(SHA256_PATTERN)
_DATASET_ID_RE = re.compile(DATASET_ID_PATTERN)
# Canonical issue path: exactly /<owner>/<repo>/issues/<number>. No trailing slash is
# accepted (the canonical GitHub html_url has none), and extra path components and
# pull-request paths (/pull/<number>) are rejected by the anchored, fixed-segment shape.
_ISSUE_URL_PATH_RE = re.compile(r"^/([^/]+)/([^/]+)/issues/(\d+)$")

Sha256Hex = Annotated[str, StringConstraints(pattern=SHA256_PATTERN)]
DatasetId = Annotated[str, StringConstraints(pattern=DATASET_ID_PATTERN)]


class DatasetError(RuntimeError):
    """Base class for dataset-normalization domain errors."""


class DatasetBuildError(DatasetError):
    """Raised when a dataset build fails an invariant outside record validation."""


class MalformedIssueError(DatasetError):
    """Raised when a raw issue record cannot be normalized under strict mode."""


class DuplicateIssueError(DatasetError):
    """Raised when normalized issues contain a duplicate identity."""


class DatasetCorruptionError(DatasetError):
    """Raised when an existing processed dataset snapshot is corrupt or mismatched."""


class DatasetReadError(DatasetError):
    """Raised when streaming or validating normalized issue JSONL fails."""


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def format_utc_datetime(value: datetime) -> str:
    """Serialize a datetime as a canonical UTC ISO-8601 string ending in ``Z``.

    Naive datetimes are assumed to be UTC (consistent with the ingestion manifest
    policy). Microseconds are omitted when zero and emitted as six digits otherwise,
    matching :meth:`datetime.isoformat`.
    """
    normalized = _ensure_utc(value)
    return normalized.isoformat().replace("+00:00", "Z")


def compute_dataset_id(
    fetched_at: datetime,
    normalizer_version: str,
    source_snapshot_sha256: str,
) -> str:
    """Derive a deterministic, content-aware, immutable dataset id.

    Identity is ``source contents + transformation version``: a change to the raw
    source bytes (snapshot hash) or the normalizer version yields a different id.
    """
    timestamp = _ensure_utc(fetched_at).strftime("%Y%m%dT%H%M%S%fZ")
    short_hash = source_snapshot_sha256[:12]
    return f"{timestamp}-n{normalizer_version}-{short_hash}"


def _is_safe_relative_path(value: str) -> bool:
    if not value:
        return False
    try:
        resolve_within_directory(Path("/__lineage_anchor__"), value)
    except ValueError:
        return False
    return True


class RawUser(BaseModel):
    """Subset of a raw GitHub user object needed for normalization."""

    model_config = ConfigDict(extra="ignore")

    login: str
    type: str


class RawLabel(BaseModel):
    """Subset of a raw GitHub label object needed for normalization."""

    model_config = ConfigDict(extra="ignore")

    name: str

    @field_validator("name")
    @classmethod
    def name_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("label name must not be empty or whitespace-only")
        return value


class RawIssueRecord(BaseModel):
    """Validated view of a raw GitHub issue record (pull requests excluded upstream)."""

    model_config = ConfigDict(extra="ignore")

    id: int = Field(gt=0)
    number: int = Field(gt=0)
    title: str
    body: str | None = None
    labels: list[RawLabel] = Field(default_factory=list)
    state: Literal["open", "closed"]
    user: RawUser | None = None
    comments: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime
    closed_at: datetime | None = None
    html_url: str

    @field_validator("title")
    @classmethod
    def title_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("title must not be empty or whitespace-only")
        return value


def _parse_github_issue_url(value: str) -> tuple[str, str, int]:
    """Parse and structurally validate a GitHub issue ``html_url``.

    Returns ``(owner, repo, issue_number)`` on success. Performs no cross-field check
    against the owning record; it only enforces the canonical issue-URL shape:

    - scheme exactly ``https`` and host exactly ``github.com`` (no embedded credentials,
      no non-standard port);
    - path exactly ``/<owner>/<repo>/issues/<number>`` (no trailing slash, no extra path
      components, and pull-request paths such as ``/pull/<number>`` are rejected);
    - no query string and no fragment, since the canonical API ``html_url`` carries none.
    """
    parsed = urlparse(value)
    if parsed.scheme != "https":
        raise ValueError(f"html_url must use https, got {parsed.scheme!r}")
    if parsed.username or parsed.password:
        raise ValueError("html_url must not contain embedded credentials")
    if parsed.hostname != "github.com":
        raise ValueError(f"html_url host must be github.com, got {parsed.hostname!r}")
    if parsed.port not in (None, 443):
        raise ValueError(f"html_url must not specify a non-standard port: {parsed.port}")
    if parsed.query:
        raise ValueError("html_url must not contain a query string")
    if parsed.fragment:
        raise ValueError("html_url must not contain a fragment")
    match = _ISSUE_URL_PATH_RE.fullmatch(parsed.path)
    if match is None:
        raise ValueError(f"html_url path is not a GitHub issue path: {parsed.path!r}")
    owner, repo, number = match.group(1), match.group(2), int(match.group(3))
    return owner, repo, number


class NormalizedIssue(BaseModel):
    """Immutable, normalized representation of a single GitHub issue."""

    schema_version: Literal["1"] = ISSUE_SCHEMA_VERSION
    repository: str
    issue_id: int = Field(gt=0)
    issue_number: int = Field(gt=0)
    title: str
    body: str
    labels: list[str] = Field(default_factory=list)
    state: Literal["open", "closed"]
    author_login: str | None = None
    author_type: str | None = None
    created_at: datetime
    updated_at: datetime
    closed_at: datetime | None = None
    comments_count: int = Field(ge=0)
    html_url: str
    source_page: str

    @field_validator("title")
    @classmethod
    def title_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("title must not be empty or whitespace-only")
        return value

    @field_validator("created_at", "updated_at", "closed_at")
    @classmethod
    def ensure_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return _ensure_utc(value)

    @field_serializer("created_at", "updated_at", "closed_at", when_used="json")
    def serialize_datetimes(self, value: datetime | None) -> str | None:
        if value is None:
            return None
        return format_utc_datetime(value)

    @model_validator(mode="after")
    def html_url_matches_record(self) -> NormalizedIssue:
        """Cross-check that ``html_url`` identifies this record's repository and number.

        Owner/repository names are compared case-insensitively via ``str.casefold()``
        because GitHub treats them as case-insensitive; the issue number must match
        exactly. Errors describe only the inconsistency and never echo unrelated record
        content (title, body, author, etc.).
        """
        url_owner, url_repo, url_number = _parse_github_issue_url(self.html_url)

        parts = self.repository.split("/")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(
                f"repository must be in owner/name form, got {self.repository!r}"
            )
        repo_owner, repo_name = parts

        if url_owner.casefold() != repo_owner.casefold():
            raise ValueError(
                f"html_url owner {url_owner!r} does not match repository owner "
                f"{repo_owner!r}"
            )
        if url_repo.casefold() != repo_name.casefold():
            raise ValueError(
                f"html_url repository {url_repo!r} does not match repository name "
                f"{repo_name!r}"
            )
        if url_number != self.issue_number:
            raise ValueError(
                f"html_url issue number {url_number} does not match issue_number "
                f"{self.issue_number}"
            )
        return self


class ProcessedManifest(BaseModel):
    """Lineage manifest describing one immutable normalized dataset snapshot."""

    schema_version: Literal["1"] = PROCESSED_MANIFEST_SCHEMA_VERSION
    issue_schema_version: Literal["1"] = ISSUE_SCHEMA_VERSION
    dataset_id: DatasetId
    repository: str
    normalizer_version: str
    built_at: datetime
    source_manifest: str
    source_manifest_sha256: Sha256Hex
    source_snapshot_sha256: Sha256Hex
    source_manifest_schema_version: str
    source_fetched_at: datetime
    source_api_version: str
    source_pages_fetched: int = Field(ge=0)
    raw_records_read: int = Field(ge=0)
    pull_requests_excluded: int = Field(ge=0)
    issues_written: int = Field(ge=0)
    unlabelled_issues: int = Field(ge=0)
    empty_body_issues: int = Field(ge=0)
    output_file: str = DEFAULT_OUTPUT_FILE
    output_sha256: Sha256Hex

    @field_validator("built_at", "source_fetched_at")
    @classmethod
    def ensure_utc(cls, value: datetime) -> datetime:
        return _ensure_utc(value)

    @field_serializer("built_at", "source_fetched_at", when_used="json")
    def serialize_datetimes(self, value: datetime) -> str:
        return format_utc_datetime(value)

    @model_validator(mode="after")
    def validate_invariants(self) -> ProcessedManifest:
        if self.raw_records_read != self.pull_requests_excluded + self.issues_written:
            raise ValueError(
                "raw_records_read must equal pull_requests_excluded + issues_written"
            )
        if self.unlabelled_issues > self.issues_written:
            raise ValueError("unlabelled_issues must not exceed issues_written")
        if self.empty_body_issues > self.issues_written:
            raise ValueError("empty_body_issues must not exceed issues_written")

        if not _is_safe_relative_path(self.output_file):
            raise ValueError(f"output_file must be a safe relative path: {self.output_file!r}")
        if not _is_safe_relative_path(self.source_manifest):
            raise ValueError(
                f"source_manifest must be a safe relative path: {self.source_manifest!r}"
            )

        expected_id = compute_dataset_id(
            self.source_fetched_at,
            self.normalizer_version,
            self.source_snapshot_sha256,
        )
        if self.dataset_id != expected_id:
            raise ValueError(
                "dataset_id is inconsistent with source_fetched_at, normalizer_version, "
                f"and source_snapshot_sha256 (expected {expected_id!r}, got {self.dataset_id!r})"
            )
        return self

    def model_dump_json(self, **kwargs: Any) -> str:
        kwargs.setdefault("indent", 2)
        return super().model_dump_json(**kwargs)


def source_manifest_relpath(repository_slug: str) -> str:
    """Logical, portable location of the raw manifest relative to the raw root."""
    return PurePosixPath(repository_slug, "manifest.json").as_posix()
