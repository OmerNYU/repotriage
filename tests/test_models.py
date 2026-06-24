"""Tests for repository and manifest models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from repotriage.github.models import (
    DEFAULT_ISSUE_REQUEST_PARAMETERS,
    InvalidRepositoryError,
    Manifest,
    RepositoryRef,
    count_item_types,
    parse_repository,
)
from tests.helpers import make_issue, make_pull_request


@pytest.mark.parametrize(
    ("value", "owner", "name"),
    [
        ("pandas-dev/pandas", "pandas-dev", "pandas"),
        ("octo-org/hello-world", "octo-org", "hello-world"),
        ("scikit-learn/scikit-learn", "scikit-learn", "scikit-learn"),
        ("huggingface/transformers", "huggingface", "transformers"),
        ("owner/repo.name", "owner", "repo.name"),
        ("owner/repo_name", "owner", "repo_name"),
    ],
)
def test_parse_repository_accepts_valid_values(value: str, owner: str, name: str) -> None:
    repo = parse_repository(value)
    assert repo.owner == owner
    assert repo.name == name
    assert repo.slug == f"{owner}__{name}"
    assert repo.full_name == value


@pytest.mark.parametrize(
    "value",
    [
        "pandas",
        "/pandas",
        "pandas-dev/",
        "pandas-dev/pandas/extra",
        " pandas-dev/pandas",
        "pandas-dev/pandas ",
        "pandas dev/pandas",
    ],
)
def test_parse_repository_rejects_invalid_values(value: str) -> None:
    with pytest.raises(InvalidRepositoryError, match="owner/name"):
        parse_repository(value)


@pytest.mark.parametrize(
    ("owner", "name"),
    [
        ("pandas-dev", "pandas"),
        ("owner", "repo_name"),
        ("owner", "repo.name"),
    ],
)
def test_repository_ref_direct_construction_accepts_valid_values(owner: str, name: str) -> None:
    repo = RepositoryRef(owner=owner, name=name)
    assert repo.full_name == f"{owner}/{name}"


@pytest.mark.parametrize(
    ("owner", "name"),
    [
        ("", "repo"),
        ("owner", ""),
        ("owner/name", "repo"),
        ("owner", "repo/name"),
        ("owner", "repo name"),
        ("owner", r"repo\name"),
        ("owner", "repo?name"),
        ("owner", "repo#name"),
        ("owner", "."),
        ("owner", ".."),
        (" owner", "repo"),
        ("owner", "repo "),
    ],
)
def test_repository_ref_direct_construction_rejects_invalid_values(owner: str, name: str) -> None:
    with pytest.raises(ValidationError):
        RepositoryRef(owner=owner, name=name)


def test_parse_repository_wraps_model_validation_error() -> None:
    with pytest.raises(InvalidRepositoryError, match="Invalid repository"):
        parse_repository("bad/name?")


def test_count_item_types_separates_issues_and_pull_requests() -> None:
    items = [make_issue(1), make_pull_request(2), make_issue(3)]
    raw_total, issues, pull_requests = count_item_types(items)
    assert raw_total == 3
    assert issues == 2
    assert pull_requests == 1


def _base_manifest_kwargs() -> dict:
    from datetime import UTC, datetime

    return {
        "repository": "o/r",
        "endpoint": "https://api.github.com/repos/o/r/issues",
        "request_parameters": DEFAULT_ISSUE_REQUEST_PARAMETERS,
        "fetched_at": datetime.now(UTC),
        "authenticated": False,
        "requested_max_pages": 2,
        "pages_fetched": 1,
        "raw_items_received": 1,
        "issues_received": 1,
        "pull_requests_received": 0,
        "output_files": ["pages/page_0001.json"],
    }


def test_manifest_requires_consistent_totals() -> None:
    with pytest.raises(ValueError, match="raw_items_received"):
        Manifest(**{**_base_manifest_kwargs(), "raw_items_received": 3, "issues_received": 2})


def test_manifest_rejects_pages_fetched_above_requested_max() -> None:
    with pytest.raises(ValueError, match="pages_fetched"):
        Manifest(**{**_base_manifest_kwargs(), "pages_fetched": 3})


def test_manifest_rejects_output_file_count_mismatch() -> None:
    kwargs = _base_manifest_kwargs()
    kwargs["output_files"] = ["pages/page_0001.json", "pages/page_0002.json"]
    with pytest.raises(ValueError, match="output_files length"):
        Manifest(**kwargs)


def test_manifest_rejects_duplicate_output_files() -> None:
    kwargs = _base_manifest_kwargs()
    kwargs["pages_fetched"] = 2
    kwargs["output_files"] = ["pages/page_0001.json", "pages/page_0001.json"]
    with pytest.raises(ValidationError, match="unique"):
        Manifest(**kwargs)


def test_manifest_requires_empty_output_files_when_no_pages() -> None:
    kwargs = _base_manifest_kwargs()
    kwargs.update(
        {
            "pages_fetched": 0,
            "raw_items_received": 0,
            "issues_received": 0,
            "output_files": ["pages/page_0001.json"],
        }
    )
    manifest = Manifest.model_construct(**kwargs)
    with pytest.raises(ValidationError, match="empty when pages_fetched is 0"):
        Manifest.model_validate(manifest.model_dump(mode="json"))


def test_issue_request_parameters_are_immutable() -> None:
    with pytest.raises(ValidationError):
        DEFAULT_ISSUE_REQUEST_PARAMETERS.state = "open"  # type: ignore[misc]


def test_issues_request_url_uses_request_parameters() -> None:
    repo = parse_repository("o/r")
    url = repo.issues_request_url(DEFAULT_ISSUE_REQUEST_PARAMETERS)
    assert url == (
        "https://api.github.com/repos/o/r/issues?"
        "state=all&sort=created&direction=desc&per_page=100"
    )
