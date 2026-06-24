"""Pydantic models and validation helpers for GitHub issue ingestion."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import urlencode

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

REPO_PATTERN = re.compile(r"^[^/\s]+/[^/\s]+$")
REPOSITORY_COMPONENT_INVALID = re.compile(r"[\s/\\?#\x00-\x1f]")

GITHUB_API_VERSION = "2026-03-10"
MANIFEST_SCHEMA_VERSION: Literal["2"] = "2"


class InvalidRepositoryError(ValueError):
    """Raised when a repository string is not in owner/name form."""


class CacheConflictError(RuntimeError):
    """Raised when an existing cache does not match the current request."""


class CacheCorruptionError(RuntimeError):
    """Raised when a cache directory is incomplete or unsafe to use."""


class CacheRecoveryError(RuntimeError):
    """Raised when cache publication and automatic rollback both fail."""


class IssueRequestParameters(BaseModel):
    """GitHub issues endpoint query parameters."""

    model_config = ConfigDict(frozen=True)

    state: Literal["all"] = "all"
    sort: Literal["created"] = "created"
    direction: Literal["desc"] = "desc"
    per_page: Literal[100] = 100


DEFAULT_ISSUE_REQUEST_PARAMETERS = IssueRequestParameters()


def _validate_repository_component(value: str, field_name: str) -> str:
    if not value:
        raise ValueError(f"Repository {field_name} must not be empty.")
    if value != value.strip():
        raise ValueError(
            f"Repository {field_name} must not contain leading or trailing whitespace."
        )
    if value in {".", ".."}:
        raise ValueError(f"Repository {field_name} must not be '.' or '..'.")
    if REPOSITORY_COMPONENT_INVALID.search(value):
        raise ValueError(
            f"Repository {field_name} contains invalid characters "
            "(whitespace, slash, backslash, '?', '#', or control characters are not allowed)."
        )
    return value


def parse_repository(value: str) -> RepositoryRef:
    """Parse and validate a repository string in owner/name form."""
    if not isinstance(value, str) or not value.strip():
        raise InvalidRepositoryError(
            "Repository must be in owner/name form, for example pandas-dev/pandas."
        )

    candidate = value.strip()
    if candidate != value:
        raise InvalidRepositoryError(
            "Repository must be in owner/name form without leading or trailing whitespace."
        )

    if not REPO_PATTERN.fullmatch(candidate):
        raise InvalidRepositoryError(
            f"Invalid repository {value!r}. Expected owner/name with exactly one slash, "
            "for example pandas-dev/pandas."
        )

    owner, name = candidate.split("/", 1)
    try:
        return RepositoryRef(owner=owner, name=name)
    except ValidationError as exc:
        raise InvalidRepositoryError(
            f"Invalid repository {value!r}. {exc.errors()[0]['msg']}"
        ) from exc


class RepositoryRef(BaseModel):
    """Validated GitHub repository reference."""

    owner: str
    name: str

    @field_validator("owner", "name")
    @classmethod
    def validate_component(cls, value: str, info: Any) -> str:
        field_name = "owner" if info.field_name == "owner" else "name"
        return _validate_repository_component(value, field_name)

    @property
    def slug(self) -> str:
        return f"{self.owner}__{self.name}"

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def issues_base_endpoint(self) -> str:
        return f"https://api.github.com/repos/{self.owner}/{self.name}/issues"

    def issues_request_url(
        self,
        parameters: IssueRequestParameters = DEFAULT_ISSUE_REQUEST_PARAMETERS,
    ) -> str:
        query = urlencode(parameters.model_dump())
        return f"{self.issues_base_endpoint}?{query}"


def count_item_types(items: list[dict[str, Any]]) -> tuple[int, int, int]:
    """Return (raw_total, issues, pull_requests) for a list of API items."""
    issues = sum(1 for item in items if "pull_request" not in item)
    pull_requests = sum(1 for item in items if "pull_request" in item)
    raw_total = len(items)
    return raw_total, issues, pull_requests


class Manifest(BaseModel):
    """Metadata describing a cached GitHub issues import."""

    schema_version: Literal["2"] = MANIFEST_SCHEMA_VERSION
    repository: str
    endpoint: str
    request_parameters: IssueRequestParameters
    fetched_at: datetime
    api_version: str = GITHUB_API_VERSION
    authenticated: bool
    requested_max_pages: int = Field(gt=0)
    pages_fetched: int = Field(ge=0)
    raw_items_received: int = Field(ge=0)
    issues_received: int = Field(ge=0)
    pull_requests_received: int = Field(ge=0)
    output_files: list[str]
    rate_limit_limit: int | None = None
    rate_limit_remaining: int | None = None
    rate_limit_reset: int | None = None

    @field_validator("fetched_at")
    @classmethod
    def ensure_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_invariants(self) -> Manifest:
        if self.raw_items_received != self.issues_received + self.pull_requests_received:
            raise ValueError(
                "raw_items_received must equal issues_received + pull_requests_received"
            )
        if self.pages_fetched > self.requested_max_pages:
            raise ValueError("pages_fetched must not exceed requested_max_pages")
        if self.pages_fetched == 0 and self.output_files:
            raise ValueError("output_files must be empty when pages_fetched is 0")
        if len(self.output_files) != self.pages_fetched:
            raise ValueError("output_files length must equal pages_fetched")
        if len(set(self.output_files)) != len(self.output_files):
            raise ValueError("output_files must be unique")
        return self

    def model_dump_json(self, **kwargs: Any) -> str:
        kwargs.setdefault("indent", 2)
        return super().model_dump_json(**kwargs)
