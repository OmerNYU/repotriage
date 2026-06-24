"""Tests for issue ingestion and caching."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from repotriage.github.client import GitHubAPIError, GitHubClient
from repotriage.github.ingestion import (
    cache_dir_for,
    fetch_repository_issues,
    publish_staging_directory,
    validate_cache,
)
from repotriage.github.models import (
    DEFAULT_ISSUE_REQUEST_PARAMETERS,
    GITHUB_API_VERSION,
    CacheConflictError,
    CacheCorruptionError,
    CacheRecoveryError,
    IssueRequestParameters,
    Manifest,
    RepositoryRef,
)
from tests.helpers import json_response, make_issue, make_pull_request


def build_mock_transport(
    pages: list[list[dict]],
    *,
    call_counter: list[int] | None = None,
) -> httpx.MockTransport:
    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        if call_counter is not None:
            call_counter[0] = state["calls"]

        page_index = min(state["calls"] - 1, len(pages) - 1)
        headers: dict[str, str] = {
            "x-ratelimit-limit": "5000",
            "x-ratelimit-remaining": "4999",
            "x-ratelimit-reset": "1700000000",
        }
        if page_index < len(pages) - 1:
            headers["link"] = (
                f"<https://api.github.com/repos/o/r/issues?page={page_index + 2}>; rel=\"next\""
            )
        return json_response(pages[page_index], headers=headers)

    return httpx.MockTransport(handler)


def write_manifest_to_cache(
    cache_dir: Path,
    repository: RepositoryRef,
    *,
    max_pages: int = 1,
    pages_fetched: int = 1,
    output_files: list[str] | None = None,
    api_version: str = GITHUB_API_VERSION,
    request_parameters: IssueRequestParameters = DEFAULT_ISSUE_REQUEST_PARAMETERS,
) -> Manifest:
    cache_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = cache_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    files = output_files or [
        f"pages/page_{index:04d}.json" for index in range(1, pages_fetched + 1)
    ]
    for relative_path in files:
        (cache_dir / relative_path).write_text("[]\n", encoding="utf-8")

    manifest = Manifest(
        repository=repository.full_name,
        endpoint=repository.issues_base_endpoint,
        request_parameters=request_parameters,
        fetched_at=datetime.now(UTC),
        api_version=api_version,
        authenticated=False,
        requested_max_pages=max_pages,
        pages_fetched=pages_fetched,
        raw_items_received=0,
        issues_received=0,
        pull_requests_received=0,
        output_files=files,
    )
    (cache_dir / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    return manifest


def test_manifest_contains_request_parameters(
    repository: RepositoryRef,
    output_root: Path,
) -> None:
    transport = build_mock_transport([[make_issue(1)]])
    client = GitHubClient(transport=transport)

    result = fetch_repository_issues(
        repository,
        max_pages=1,
        output_root=output_root,
        client=client,
    )

    assert result.manifest.request_parameters == DEFAULT_ISSUE_REQUEST_PARAMETERS
    assert result.manifest.endpoint == repository.issues_base_endpoint
    assert result.manifest.schema_version == "2"


def test_manifest_totals_are_internally_consistent(
    repository: RepositoryRef,
    output_root: Path,
) -> None:
    transport = build_mock_transport(
        [
            [make_issue(1), make_pull_request(2)],
            [make_issue(3)],
        ]
    )
    client = GitHubClient(transport=transport)

    result = fetch_repository_issues(
        repository,
        max_pages=2,
        output_root=output_root,
        client=client,
    )

    manifest = result.manifest
    assert manifest.raw_items_received == 3
    assert manifest.issues_received == 2
    assert manifest.pull_requests_received == 1
    assert manifest.raw_items_received == manifest.issues_received + manifest.pull_requests_received
    assert manifest.pages_fetched == 2
    assert manifest.output_files == ["pages/page_0001.json", "pages/page_0002.json"]

    page_one = json.loads((result.cache_dir / "pages/page_0001.json").read_text(encoding="utf-8"))
    assert "pull_request" in page_one[1]


def test_exact_matching_cache_causes_no_network_request(
    repository: RepositoryRef,
    output_root: Path,
) -> None:
    call_counter = [0]
    transport = build_mock_transport([[make_issue(1)]], call_counter=call_counter)
    client = GitHubClient(transport=transport)

    first = fetch_repository_issues(
        repository,
        max_pages=1,
        output_root=output_root,
        client=client,
    )
    assert first.cache_hit is False
    assert call_counter[0] == 1

    call_counter[0] = 0
    second = fetch_repository_issues(
        repository,
        max_pages=1,
        output_root=output_root,
        client=client,
    )

    assert second.cache_hit is True
    assert call_counter[0] == 0


def test_different_max_pages_is_not_cache_hit(
    repository: RepositoryRef,
    output_root: Path,
) -> None:
    cache_dir = cache_dir_for(repository, output_root)
    write_manifest_to_cache(cache_dir, repository, max_pages=2, pages_fetched=2)

    with pytest.raises(CacheConflictError, match="max_pages=2"):
        validate_cache(
            cache_dir,
            expected_repository=repository,
            expected_max_pages=20,
            expected_api_version=GITHUB_API_VERSION,
            expected_request_parameters=DEFAULT_ISSUE_REQUEST_PARAMETERS,
        )


def test_different_api_version_is_not_cache_hit(
    repository: RepositoryRef,
    output_root: Path,
) -> None:
    cache_dir = cache_dir_for(repository, output_root)
    write_manifest_to_cache(cache_dir, repository, api_version="2022-11-28")

    with pytest.raises(CacheConflictError, match="api_version"):
        validate_cache(
            cache_dir,
            expected_repository=repository,
            expected_max_pages=1,
            expected_api_version=GITHUB_API_VERSION,
            expected_request_parameters=DEFAULT_ISSUE_REQUEST_PARAMETERS,
        )


def test_missing_output_page_invalidates_cache(
    repository: RepositoryRef,
    output_root: Path,
) -> None:
    cache_dir = cache_dir_for(repository, output_root)
    write_manifest_to_cache(
        cache_dir,
        repository,
        output_files=["pages/page_0001.json", "pages/page_0002.json"],
        pages_fetched=2,
        max_pages=2,
    )
    (cache_dir / "pages/page_0002.json").unlink()

    with pytest.raises(CacheCorruptionError, match="Missing cached output file"):
        validate_cache(
            cache_dir,
            expected_repository=repository,
            expected_max_pages=2,
            expected_api_version=GITHUB_API_VERSION,
            expected_request_parameters=DEFAULT_ISSUE_REQUEST_PARAMETERS,
        )


def test_output_file_count_mismatch_invalidates_cache(
    repository: RepositoryRef,
    output_root: Path,
) -> None:
    cache_dir = cache_dir_for(repository, output_root)
    write_manifest_to_cache(cache_dir, repository, max_pages=2, pages_fetched=1)
    manifest_path = cache_dir / "manifest.json"
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_data["pages_fetched"] = 2
    manifest_path.write_text(json.dumps(manifest_data, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(CacheCorruptionError, match="output_files length"):
        validate_cache(
            cache_dir,
            expected_repository=repository,
            expected_max_pages=2,
            expected_api_version=GITHUB_API_VERSION,
            expected_request_parameters=DEFAULT_ISSUE_REQUEST_PARAMETERS,
        )


def test_output_path_escape_is_rejected(
    repository: RepositoryRef,
    output_root: Path,
) -> None:
    cache_dir = cache_dir_for(repository, output_root)
    write_manifest_to_cache(
        cache_dir,
        repository,
        output_files=["../escape.json"],
        pages_fetched=1,
        max_pages=1,
    )

    with pytest.raises(CacheCorruptionError, match="escapes cache directory"):
        validate_cache(
            cache_dir,
            expected_repository=repository,
            expected_max_pages=1,
            expected_api_version=GITHUB_API_VERSION,
            expected_request_parameters=DEFAULT_ISSUE_REQUEST_PARAMETERS,
        )


def test_different_request_parameters_is_not_reused(
    repository: RepositoryRef,
    output_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = cache_dir_for(repository, output_root)
    write_manifest_to_cache(cache_dir, repository)

    conflicting_manifest = Manifest.model_construct(
        schema_version="2",
        repository=repository.full_name,
        endpoint=repository.issues_base_endpoint,
        request_parameters=IssueRequestParameters.model_construct(state="open"),
        fetched_at=datetime.now(UTC),
        api_version=GITHUB_API_VERSION,
        authenticated=False,
        requested_max_pages=1,
        pages_fetched=1,
        raw_items_received=0,
        issues_received=0,
        pull_requests_received=0,
        output_files=["pages/page_0001.json"],
    )
    monkeypatch.setattr(
        "repotriage.github.ingestion._read_manifest_file",
        lambda _cache_dir: conflicting_manifest,
    )

    with pytest.raises(CacheConflictError, match="request_parameters"):
        fetch_repository_issues(
            repository,
            max_pages=1,
            output_root=output_root,
            request_parameters=DEFAULT_ISSUE_REQUEST_PARAMETERS,
        )


def test_refresh_replaces_cache(
    repository: RepositoryRef,
    output_root: Path,
) -> None:
    transport = build_mock_transport([[make_issue(1)]])
    client = GitHubClient(transport=transport)

    fetch_repository_issues(
        repository,
        max_pages=1,
        output_root=output_root,
        client=client,
    )

    transport = build_mock_transport([[make_issue(1), make_issue(2)]])
    client = GitHubClient(transport=transport)
    refreshed = fetch_repository_issues(
        repository,
        max_pages=1,
        output_root=output_root,
        refresh=True,
        client=client,
    )

    assert refreshed.cache_hit is False
    assert refreshed.manifest.raw_items_received == 2
    page = json.loads((refreshed.cache_dir / "pages/page_0001.json").read_text(encoding="utf-8"))
    assert len(page) == 2


def test_failed_refresh_preserves_previous_valid_cache(
    repository: RepositoryRef,
    output_root: Path,
) -> None:
    transport = build_mock_transport([[make_issue(1)]])
    client = GitHubClient(transport=transport)
    first = fetch_repository_issues(
        repository,
        max_pages=1,
        output_root=output_root,
        client=client,
    )

    call_count = 0

    def failing_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return json_response(
                [make_issue(99)],
                headers={
                    "link": '<https://api.github.com/repos/o/r/issues?page=2>; rel="next"',
                },
            )
        return json_response([], status_code=404)

    failing_client = GitHubClient(transport=httpx.MockTransport(failing_handler))

    with pytest.raises(GitHubAPIError):
        fetch_repository_issues(
            repository,
            max_pages=2,
            output_root=output_root,
            refresh=True,
            client=failing_client,
        )

    restored = validate_cache(
        first.cache_dir,
        expected_repository=repository,
        expected_max_pages=1,
        expected_api_version=GITHUB_API_VERSION,
        expected_request_parameters=DEFAULT_ISSUE_REQUEST_PARAMETERS,
    )
    assert restored == first.manifest
    page = json.loads((first.cache_dir / "pages/page_0001.json").read_text(encoding="utf-8"))
    assert page[0]["number"] == 1


def test_initial_fetch_failure_leaves_no_completed_cache(
    repository: RepositoryRef,
    output_root: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response([], status_code=404)

    client = GitHubClient(transport=httpx.MockTransport(handler))
    cache_dir = cache_dir_for(repository, output_root)

    with pytest.raises(GitHubAPIError):
        fetch_repository_issues(
            repository,
            max_pages=1,
            output_root=output_root,
            client=client,
        )

    assert not cache_dir.exists()


def test_publish_failure_restores_backup(
    repository: RepositoryRef,
    output_root: Path,
    tmp_path: Path,
) -> None:
    cache_dir = cache_dir_for(repository, output_root)
    write_manifest_to_cache(cache_dir, repository)
    original_manifest = (cache_dir / "manifest.json").read_text(encoding="utf-8")

    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    (staging_dir / "manifest.json").write_text('{"new": true}\n', encoding="utf-8")
    backup_dir = tmp_path / "backup"

    def failing_rename(source: Path, target: Path) -> None:
        if source == staging_dir and target == cache_dir:
            raise OSError("publish failed")
        source.rename(target)

    with pytest.raises(OSError, match="publish failed"):
        publish_staging_directory(
            staging_dir,
            cache_dir,
            backup_dir=backup_dir,
            rename=failing_rename,
        )

    assert cache_dir.exists()
    assert (cache_dir / "manifest.json").read_text(encoding="utf-8") == original_manifest


def test_failed_refresh_does_not_publish_new_manifest(
    repository: RepositoryRef,
    output_root: Path,
) -> None:
    transport = build_mock_transport([[make_issue(1)]])
    client = GitHubClient(transport=transport)
    first = fetch_repository_issues(
        repository,
        max_pages=1,
        output_root=output_root,
        client=client,
    )
    original_manifest_text = (first.cache_dir / "manifest.json").read_text(encoding="utf-8")

    def failing_rename(source: Path, target: Path) -> None:
        if target == first.cache_dir and ".staging-" in source.name:
            raise OSError("publish failed during refresh")
        source.rename(target)

    transport = build_mock_transport([[make_issue(1), make_issue(2)]])
    client = GitHubClient(transport=transport)

    with pytest.raises(OSError, match="publish failed during refresh"):
        fetch_repository_issues(
            repository,
            max_pages=1,
            output_root=output_root,
            refresh=True,
            client=client,
            rename=failing_rename,
        )

    assert (first.cache_dir / "manifest.json").read_text(encoding="utf-8") == original_manifest_text
    page = json.loads((first.cache_dir / "pages/page_0001.json").read_text(encoding="utf-8"))
    assert page[0]["number"] == 1


def test_keyboard_interrupt_during_publication_restores_backup(
    repository: RepositoryRef,
    output_root: Path,
    tmp_path: Path,
) -> None:
    cache_dir = cache_dir_for(repository, output_root)
    write_manifest_to_cache(cache_dir, repository)
    original_manifest = (cache_dir / "manifest.json").read_text(encoding="utf-8")

    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    (staging_dir / "manifest.json").write_text('{"new": true}\n', encoding="utf-8")
    backup_dir = tmp_path / "backup"

    def interrupting_rename(source: Path, target: Path) -> None:
        if source == staging_dir and target == cache_dir:
            raise KeyboardInterrupt
        source.rename(target)

    with pytest.raises(KeyboardInterrupt):
        publish_staging_directory(
            staging_dir,
            cache_dir,
            backup_dir=backup_dir,
            rename=interrupting_rename,
        )

    assert cache_dir.exists()
    assert (cache_dir / "manifest.json").read_text(encoding="utf-8") == original_manifest


def test_system_exit_during_publication_restores_backup(
    repository: RepositoryRef,
    output_root: Path,
    tmp_path: Path,
) -> None:
    cache_dir = cache_dir_for(repository, output_root)
    write_manifest_to_cache(cache_dir, repository)
    original_manifest = (cache_dir / "manifest.json").read_text(encoding="utf-8")

    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    backup_dir = tmp_path / "backup"

    def exiting_rename(source: Path, target: Path) -> None:
        if source == staging_dir and target == cache_dir:
            raise SystemExit(1)
        source.rename(target)

    with pytest.raises(SystemExit):
        publish_staging_directory(
            staging_dir,
            cache_dir,
            backup_dir=backup_dir,
            rename=exiting_rename,
        )

    assert cache_dir.exists()
    assert (cache_dir / "manifest.json").read_text(encoding="utf-8") == original_manifest


def test_rollback_failure_raises_cache_recovery_error(
    repository: RepositoryRef,
    output_root: Path,
    tmp_path: Path,
) -> None:
    cache_dir = cache_dir_for(repository, output_root)
    write_manifest_to_cache(cache_dir, repository)

    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    backup_dir = tmp_path / "backup"
    rename_calls = 0

    def failing_rename(source: Path, target: Path) -> None:
        nonlocal rename_calls
        rename_calls += 1
        if rename_calls == 1:
            source.rename(target)
            return
        if rename_calls == 2:
            raise OSError("publish failed")
        raise OSError("rollback failed")

    with pytest.raises(CacheRecoveryError, match="automatic rollback also failed"):
        publish_staging_directory(
            staging_dir,
            cache_dir,
            backup_dir=backup_dir,
            rename=failing_rename,
        )

    assert backup_dir.exists()


def test_successful_refresh_leaves_no_staging_or_backup_directories(
    repository: RepositoryRef,
    output_root: Path,
) -> None:
    transport = build_mock_transport([[make_issue(1)]])
    client = GitHubClient(transport=transport)
    fetch_repository_issues(
        repository,
        max_pages=1,
        output_root=output_root,
        client=client,
    )

    transport = build_mock_transport([[make_issue(1), make_issue(2)]])
    client = GitHubClient(transport=transport)
    fetch_repository_issues(
        repository,
        max_pages=1,
        output_root=output_root,
        refresh=True,
        client=client,
    )

    cache_dir = cache_dir_for(repository, output_root)
    leftovers = list(output_root.glob(f".{repository.slug}.*"))
    assert leftovers == []
    assert cache_dir.exists()
    assert (cache_dir / "manifest.json").is_file()
    assert len(list((cache_dir / "pages").glob("page_*.json"))) == 1


def test_internally_owned_client_is_closed_on_fetch_failure(
    repository: RepositoryRef,
    output_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = False
    original_close = GitHubClient.close

    def spy_close(self: GitHubClient) -> None:
        nonlocal closed
        closed = True
        original_close(self)

    monkeypatch.setattr(GitHubClient, "close", spy_close)

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response([], status_code=404)

    with pytest.raises(GitHubAPIError):
        fetch_repository_issues(
            repository,
            max_pages=1,
            output_root=output_root,
        )

    assert closed is True


def test_keyboard_interrupt_during_fetch_cleans_staging_without_live_cache(
    repository: RepositoryRef,
    output_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = cache_dir_for(repository, output_root)

    def interrupt_fetch(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(
        "repotriage.github.ingestion.GitHubClient.fetch_issues_pages",
        interrupt_fetch,
    )

    with pytest.raises(KeyboardInterrupt):
        fetch_repository_issues(
            repository,
            max_pages=1,
            output_root=output_root,
        )

    assert not cache_dir.exists()
    leftovers = list(output_root.glob(f".{repository.slug}.staging-*"))
    assert leftovers == []
