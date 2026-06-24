# RepoTriage

RepoTriage is an ML-assisted GitHub issue-triage platform. This repository currently implements the first vertical slice: downloading GitHub repository issues and caching the original API responses locally.

## Installation

Install the project in editable mode with development dependencies:

```bash
python -m pip install -e ".[dev]"
```

## Running tests and lint checks

```bash
pytest
ruff check .
```

## Fetching issues

Download up to two pages of issues for `pandas-dev/pandas`:

```bash
repotriage fetch-issues --repo pandas-dev/pandas --max-pages 2
```

Optional flags:

- `--refresh` replaces an existing cached import
- `--output-root PATH` overrides the default raw-data root

## GitHub authentication

GitHub rate limits are much lower for unauthenticated requests. You can optionally provide a personal access token through the environment:

```bash
export GITHUB_TOKEN=your_token_here
repotriage fetch-issues --repo pandas-dev/pandas --max-pages 2
```

Do not commit `.env` files, tokens, or downloaded raw data.

## API version and raw page files

The client pins GitHub REST API version `2026-03-10`.

Each `page_XXXX.json` file stores raw decoded API records with no fields removed or transformed. These are not byte-for-byte copies of GitHub's HTTP response bodies. The files intentionally include both issues and pull requests because the API returns both item types together. The manifest counts issues and pull requests separately.

## Cache behavior

Raw files are written under:

```text
data/raw/github/<owner>__<repo>/
```

For example:

```text
data/raw/github/pandas-dev__pandas/
├── manifest.json
└── pages/
    ├── page_0001.json
    └── page_0002.json
```

A cache is reused only when the on-disk manifest and page files match the current request configuration, including:

- repository
- pinned API version
- issue request parameters (`state`, `sort`, `direction`, `per_page`)
- `--max-pages`

If an existing cache does not match, the CLI reports a cache conflict and instructs you to run with `--refresh`. It does not silently reuse a partial cache or automatically fetch additional pages.

`--refresh` downloads into a unique staging directory and publishes the new snapshot with rollback support. If publication fails, the previous valid cache is restored when possible.

Concurrent fetches or refreshes for the same repository are not supported yet.

Do not commit raw downloaded data or secrets.
