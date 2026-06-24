"""HTTP client for GitHub's repository issues endpoint."""

from __future__ import annotations

import email.utils
import logging
import os
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

import httpx

from repotriage import __version__
from repotriage.github.models import (
    DEFAULT_ISSUE_REQUEST_PARAMETERS,
    GITHUB_API_VERSION,
    IssueRequestParameters,
    RepositoryRef,
)

logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"
GITHUB_API_HOST = "api.github.com"
DEFAULT_CONNECT_TIMEOUT = 10.0
DEFAULT_READ_TIMEOUT = 30.0
# Up to MAX_RETRIES additional attempts after the first request (4 total attempts).
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_MAX_SECONDS = 30.0
MAX_RETRY_WAIT_SECONDS = 60.0

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

LINK_NEXT_PATTERN = re.compile(r'<([^>]+)>;\s*rel="next"')


class GitHubAPIError(RuntimeError):
    """Raised when GitHub returns a non-retryable HTTP error."""

    def __init__(
        self,
        message: str,
        *,
        repository: str,
        status_code: int | None = None,
    ) -> None:
        self.repository = repository
        self.status_code = status_code
        super().__init__(message)


class GitHubRateLimitError(GitHubAPIError):
    """Raised when GitHub rate limiting requires waiting longer than allowed."""


@dataclass(frozen=True)
class RateLimitInfo:
    """Subset of GitHub rate-limit response headers."""

    limit: int | None
    remaining: int | None
    reset: int | None


@dataclass(frozen=True)
class IssuesPage:
    """One page of issues returned by the GitHub API."""

    page_number: int
    items: list[dict[str, Any]]
    rate_limit: RateLimitInfo


def validate_github_api_url(url: str, *, repository: str) -> str:
    """Validate that a request URL targets the trusted GitHub API host."""
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise GitHubAPIError(
            f"Rejected non-HTTPS GitHub API URL for {repository}: {parsed.scheme!r}.",
            repository=repository,
        )
    if parsed.username or parsed.password:
        raise GitHubAPIError(
            f"Rejected GitHub API URL with credentials for {repository}.",
            repository=repository,
        )
    hostname = parsed.hostname
    if hostname != GITHUB_API_HOST:
        host_label = hostname or "missing hostname"
        raise GitHubAPIError(
            f"Rejected untrusted GitHub API host for {repository}: {host_label}.",
            repository=repository,
        )
    if parsed.port not in (None, 443):
        raise GitHubAPIError(
            f"Rejected GitHub API URL with unsupported port for {repository}: {parsed.port}.",
            repository=repository,
        )
    return url


def parse_next_link(link_header: str | None) -> str | None:
    """Extract the URL for rel=\"next\" from a GitHub Link header."""
    if not link_header:
        return None

    for part in link_header.split(","):
        match = LINK_NEXT_PATTERN.search(part.strip())
        if match:
            return match.group(1)
    return None


def parse_retry_after(header_value: str | None) -> float | None:
    """Parse a Retry-After header value as seconds to wait."""
    if not header_value:
        return None

    stripped = header_value.strip()
    if stripped.isdigit():
        return float(stripped)

    try:
        retry_at = email.utils.parsedate_to_datetime(stripped)
    except (TypeError, ValueError):
        return None

    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=UTC)
    return max((retry_at - datetime.now(UTC)).total_seconds(), 0.0)


def extract_rate_limit(headers: httpx.Headers) -> RateLimitInfo:
    """Read rate-limit headers from a GitHub response."""

    def _to_int(name: str) -> int | None:
        value = headers.get(name)
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            return None

    return RateLimitInfo(
        limit=_to_int("x-ratelimit-limit"),
        remaining=_to_int("x-ratelimit-remaining"),
        reset=_to_int("x-ratelimit-reset"),
    )


def is_rate_limited_response(response: httpx.Response) -> bool:
    """Return True when response headers indicate GitHub rate limiting."""
    if response.status_code == 429:
        return True
    if response.status_code != 403:
        return False
    if response.headers.get("retry-after"):
        return True
    return response.headers.get("x-ratelimit-remaining") == "0"


def is_retryable_response(response: httpx.Response) -> bool:
    """Return True when the response should be retried."""
    return response.status_code in RETRYABLE_STATUS_CODES or is_rate_limited_response(response)


def format_rate_limit_reset_message(rate_limit: RateLimitInfo) -> str:
    if rate_limit.reset is not None:
        reset_at = datetime.fromtimestamp(rate_limit.reset, tz=UTC).isoformat()
        return f"Rate limit resets at {reset_at} UTC."
    return "Rate limit reset time is unavailable."


def compute_retry_delay(
    response: httpx.Response,
    *,
    attempt: int,
) -> float:
    """Compute retry delay in seconds before applying the caller's wait cap."""
    retry_after = parse_retry_after(response.headers.get("retry-after"))
    if retry_after is not None:
        return retry_after
    return min(BACKOFF_BASE_SECONDS * (2**attempt), BACKOFF_MAX_SECONDS)


def raise_rate_limit_wait_error(response: httpx.Response, *, repository: str) -> None:
    rate_limit = extract_rate_limit(response.headers)
    reset_message = format_rate_limit_reset_message(rate_limit)
    raise GitHubRateLimitError(
        f"GitHub rate limit for {repository} requires waiting longer than "
        f"{int(MAX_RETRY_WAIT_SECONDS)} seconds. {reset_message}",
        repository=repository,
        status_code=response.status_code,
    )


class GitHubClient:
    """Minimal GitHub REST client for repository issue pages."""

    def __init__(
        self,
        *,
        token: str | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        read_timeout: float = DEFAULT_READ_TIMEOUT,
    ) -> None:
        self._token = token if token is not None else os.environ.get("GITHUB_TOKEN")
        self._sleep = sleep
        self._timeout = httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=30.0,
            pool=30.0,
        )
        self._client = httpx.Client(
            base_url=GITHUB_API_BASE,
            transport=transport,
            timeout=self._timeout,
            headers=self._build_headers(),
        )

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
            "User-Agent": f"repotriage/{__version__}",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    @property
    def authenticated(self) -> bool:
        return bool(self._token)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> GitHubClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def fetch_issues_pages(
        self,
        repository: RepositoryRef,
        *,
        max_pages: int,
        request_parameters: IssueRequestParameters = DEFAULT_ISSUE_REQUEST_PARAMETERS,
    ) -> Iterator[IssuesPage]:
        """Yield issue pages until limits or pagination end."""
        if max_pages < 1:
            raise ValueError("max_pages must be a positive integer")

        next_url: str | None = validate_github_api_url(
            repository.issues_request_url(request_parameters),
            repository=repository.full_name,
        )
        page_number = 0

        while next_url is not None and page_number < max_pages:
            page_number += 1
            response = self._request_with_retries(next_url, repository=repository)
            items = response.json()
            if not isinstance(items, list):
                raise GitHubAPIError(
                    f"Expected a JSON array for {repository.full_name}, "
                    f"got {type(items).__name__}.",
                    repository=repository.full_name,
                    status_code=response.status_code,
                )

            if len(items) == 0:
                break

            rate_limit = extract_rate_limit(response.headers)
            logger.info(
                "Fetched page %d for %s with %d raw entries",
                page_number,
                repository.full_name,
                len(items),
            )
            yield IssuesPage(
                page_number=page_number,
                items=items,
                rate_limit=rate_limit,
            )

            raw_next_url = parse_next_link(response.headers.get("link"))
            next_url = (
                validate_github_api_url(raw_next_url, repository=repository.full_name)
                if raw_next_url is not None
                else None
            )

    def _request_with_retries(self, url: str, *, repository: RepositoryRef) -> httpx.Response:
        validated_url = validate_github_api_url(url, repository=repository.full_name)
        attempt = 0
        while True:
            try:
                response = self._client.get(validated_url)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt >= MAX_RETRIES:
                    raise GitHubAPIError(
                        f"Network error while fetching {repository.full_name}: {exc}",
                        repository=repository.full_name,
                    ) from exc
                delay = min(BACKOFF_BASE_SECONDS * (2**attempt), BACKOFF_MAX_SECONDS)
                logger.warning(
                    "Retrying %s after network error (attempt %d/%d, sleeping %.1fs)",
                    repository.full_name,
                    attempt + 1,
                    MAX_RETRIES,
                    delay,
                )
                attempt += 1
                self._sleep(delay)
                continue

            if is_retryable_response(response):
                if attempt >= MAX_RETRIES:
                    if is_rate_limited_response(response):
                        raise_rate_limit_wait_error(response, repository=repository.full_name)
                    response.raise_for_status()
                delay = compute_retry_delay(response, attempt=attempt)
                if delay > MAX_RETRY_WAIT_SECONDS:
                    raise_rate_limit_wait_error(response, repository=repository.full_name)
                logger.warning(
                    "Retrying %s after HTTP %d (attempt %d/%d, sleeping %.1fs)",
                    repository.full_name,
                    response.status_code,
                    attempt + 1,
                    MAX_RETRIES,
                    delay,
                )
                attempt += 1
                self._sleep(delay)
                continue

            if response.is_error:
                raise GitHubAPIError(
                    f"GitHub API request failed for {repository.full_name} "
                    f"with status {response.status_code}.",
                    repository=repository.full_name,
                    status_code=response.status_code,
                )

            return response
