"""Tests for the GitHub HTTP client."""

from __future__ import annotations

import httpx
import pytest

from repotriage.github.client import (
    GitHubAPIError,
    GitHubClient,
    GitHubRateLimitError,
    parse_next_link,
    validate_github_api_url,
)
from repotriage.github.models import RepositoryRef
from tests.helpers import json_response, make_issue


def test_parse_next_link_extracts_next_url() -> None:
    header = (
        '<https://api.github.com/repos/o/r/issues?page=2>; rel="next", '
        '<https://api.github.com/repos/o/r/issues?page=5>; rel="last"'
    )
    assert parse_next_link(header) == "https://api.github.com/repos/o/r/issues?page=2"
    assert parse_next_link(None) is None


def test_token_header_present_when_env_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "test-token-value")
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.headers))
        return json_response([])

    client = GitHubClient(transport=httpx.MockTransport(handler))
    repo = RepositoryRef(owner="o", name="r")
    list(client.fetch_issues_pages(repo, max_pages=1))
    client.close()

    assert captured["authorization"] == "Bearer test-token-value"


def test_token_header_absent_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.headers))
        return json_response([])

    client = GitHubClient(transport=httpx.MockTransport(handler))
    repo = RepositoryRef(owner="o", name="r")
    list(client.fetch_issues_pages(repo, max_pages=1))
    client.close()

    assert "authorization" not in captured


def test_first_request_uses_request_parameters() -> None:
    captured_url = ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_url
        captured_url = str(request.url)
        return json_response([])

    client = GitHubClient(transport=httpx.MockTransport(handler))
    repo = RepositoryRef(owner="o", name="r")
    list(client.fetch_issues_pages(repo, max_pages=1))
    client.close()

    assert "state=all" in captured_url
    assert "sort=created" in captured_url
    assert "direction=desc" in captured_url
    assert "per_page=100" in captured_url


def test_pagination_follows_rel_next() -> None:
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        if "page=2" in str(request.url):
            return json_response([make_issue(3)])
        return json_response(
            [make_issue(1), make_issue(2)],
            headers={
                "link": '<https://api.github.com/repos/o/r/issues?page=2>; rel="next"',
            },
        )

    client = GitHubClient(transport=httpx.MockTransport(handler))
    repo = RepositoryRef(owner="o", name="r")
    pages = list(client.fetch_issues_pages(repo, max_pages=5))
    client.close()

    assert len(pages) == 2
    assert len(requests) == 2
    assert "page=2" in requests[1]


def test_pagination_stops_at_max_pages() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(
            [make_issue(1)],
            headers={
                "link": '<https://api.github.com/repos/o/r/issues?page=99>; rel="next"',
            },
        )

    client = GitHubClient(transport=httpx.MockTransport(handler))
    repo = RepositoryRef(owner="o", name="r")
    pages = list(client.fetch_issues_pages(repo, max_pages=2))
    client.close()

    assert len(pages) == 2


def test_pagination_stops_when_no_next_link() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return json_response([make_issue(1)])

    client = GitHubClient(transport=httpx.MockTransport(handler))
    repo = RepositoryRef(owner="o", name="r")
    pages = list(client.fetch_issues_pages(repo, max_pages=10))
    client.close()

    assert len(pages) == 1
    assert call_count == 1


def test_transient_failure_is_retried() -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return json_response([], status_code=503)
        return json_response([make_issue(1)])

    client = GitHubClient(
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
    )
    repo = RepositoryRef(owner="o", name="r")
    pages = list(client.fetch_issues_pages(repo, max_pages=1))
    client.close()

    assert attempts == 2
    assert len(sleeps) == 1
    assert len(pages) == 1


def test_permanent_http_error_is_not_retried() -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return json_response([], status_code=404)

    client = GitHubClient(
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
    )
    repo = RepositoryRef(owner="o", name="r")

    with pytest.raises(GitHubAPIError, match="status 404"):
        list(client.fetch_issues_pages(repo, max_pages=1))

    client.close()
    assert attempts == 1
    assert sleeps == []


def test_rate_limited_403_with_zero_remaining_is_retried() -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return json_response(
                [],
                status_code=403,
                headers={"x-ratelimit-remaining": "0"},
            )
        return json_response([make_issue(1)])

    client = GitHubClient(transport=httpx.MockTransport(handler), sleep=sleeps.append)
    repo = RepositoryRef(owner="o", name="r")
    pages = list(client.fetch_issues_pages(repo, max_pages=1))
    client.close()

    assert attempts == 2
    assert len(sleeps) == 1
    assert len(pages) == 1


def test_rate_limited_403_with_retry_after_is_retried() -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return json_response([], status_code=403, headers={"retry-after": "1"})
        return json_response([make_issue(1)])

    client = GitHubClient(transport=httpx.MockTransport(handler), sleep=sleeps.append)
    repo = RepositoryRef(owner="o", name="r")
    pages = list(client.fetch_issues_pages(repo, max_pages=1))
    client.close()

    assert attempts == 2
    assert sleeps == [1.0]
    assert len(pages) == 1


def test_ordinary_403_is_not_retried() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return json_response([], status_code=403)

    client = GitHubClient(transport=httpx.MockTransport(handler), sleep=lambda _: None)
    repo = RepositoryRef(owner="o", name="r")

    with pytest.raises(GitHubAPIError, match="status 403"):
        list(client.fetch_issues_pages(repo, max_pages=1))

    client.close()
    assert attempts == 1


def test_http_429_is_retried() -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return json_response([], status_code=429, headers={"retry-after": "1"})
        return json_response([make_issue(1)])

    client = GitHubClient(transport=httpx.MockTransport(handler), sleep=sleeps.append)
    repo = RepositoryRef(owner="o", name="r")
    pages = list(client.fetch_issues_pages(repo, max_pages=1))
    client.close()

    assert attempts == 2
    assert sleeps == [1.0]
    assert len(pages) == 1


def test_excessive_retry_delay_raises_rate_limit_error() -> None:
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(
            [],
            status_code=429,
            headers={
                "retry-after": "120",
                "x-ratelimit-reset": "1700000000",
            },
        )

    client = GitHubClient(transport=httpx.MockTransport(handler), sleep=sleeps.append)
    repo = RepositoryRef(owner="o", name="r")

    with pytest.raises(GitHubRateLimitError, match="longer than 60 seconds"):
        list(client.fetch_issues_pages(repo, max_pages=1))

    client.close()
    assert sleeps == []


def test_validate_github_api_url_accepts_normal_url() -> None:
    url = validate_github_api_url(
        "https://api.github.com/repos/o/r/issues?page=2",
        repository="o/r",
    )
    assert url.startswith("https://api.github.com/")


@pytest.mark.parametrize(
    "url",
    [
        "http://api.github.com/repos/o/r/issues",
        "https://example.com/repos/o/r/issues",
        "https://api.github.com.evil.com/repos/o/r/issues",
        "https://user:pass@api.github.com/repos/o/r/issues",
        "https://api.github.com:444/repos/o/r/issues",
    ],
)
def test_validate_github_api_url_rejects_untrusted_urls(url: str) -> None:
    with pytest.raises(GitHubAPIError, match="Rejected"):
        validate_github_api_url(url, repository="o/r")


def test_malicious_pagination_url_is_rejected_before_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "super-secret-token-value")
    requested_hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_hosts.append(request.url.host or "")
        return json_response(
            [make_issue(1)],
            headers={
                "link": '<https://evil.example.com/steal>; rel="next"',
            },
        )

    client = GitHubClient(transport=httpx.MockTransport(handler))
    repo = RepositoryRef(owner="o", name="r")

    with pytest.raises(GitHubAPIError, match="evil.example.com"):
        list(client.fetch_issues_pages(repo, max_pages=5))

    client.close()
    assert requested_hosts == ["api.github.com"]


def test_malicious_pagination_error_does_not_expose_token(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "super-secret-token-value")

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(
            [make_issue(1)],
            headers={
                "link": '<https://evil.example.com/steal>; rel="next"',
            },
        )

    client = GitHubClient(transport=httpx.MockTransport(handler))
    repo = RepositoryRef(owner="o", name="r")

    with pytest.raises(GitHubAPIError):
        list(client.fetch_issues_pages(repo, max_pages=5))

    client.close()
    combined = caplog.text
    assert "super-secret-token-value" not in combined


def test_empty_page_is_not_yielded() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return json_response([])

    client = GitHubClient(transport=httpx.MockTransport(handler))
    repo = RepositoryRef(owner="o", name="r")
    pages = list(client.fetch_issues_pages(repo, max_pages=1))
    client.close()

    assert pages == []
    assert call_count == 1
