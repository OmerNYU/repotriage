"""Shared test helpers."""

from __future__ import annotations

import json

import httpx


def make_issue(number: int) -> dict:
    return {"id": number, "number": number, "title": f"Issue {number}"}


def make_pull_request(number: int) -> dict:
    return {
        "id": number,
        "number": number,
        "title": f"PR {number}",
        "pull_request": {"url": f"https://api.github.com/repos/o/r/pulls/{number}"},
    }


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
