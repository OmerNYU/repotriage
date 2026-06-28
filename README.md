# RepoTriage

RepoTriage is an ML-assisted GitHub issue-triage platform. This repository currently implements three vertical slices: downloading GitHub repository issues and caching the original API responses locally (raw ingestion), normalizing one raw snapshot into an immutable, issue-only dataset (dataset normalization), and auditing one normalized dataset into an immutable, deterministic audit artifact (dataset audit).

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

## Building a normalized dataset

Once a raw snapshot exists, normalize it into an immutable, issue-only dataset:

```bash
repotriage build-dataset --repo pandas-dev/pandas
```

Optional flags:

- `--raw-root PATH` overrides the raw-data root (default `data/raw/github`)
- `--processed-root PATH` overrides the processed-data root (default `data/processed/github`)

### Raw versus normalized data

Raw pages under `data/raw/` are verbatim decoded GitHub API records and intentionally
contain both issues and pull requests. The normalized dataset is a derived, reduced
view containing only issues, with a small stable schema per issue. Raw data is the
source of truth; normalized data can always be rebuilt from it.

### Output layout and JSONL format

A build writes an immutable, versioned snapshot:

```text
data/processed/github/pandas-dev__pandas/<dataset-id>/
├── issues.jsonl
└── manifest.json
```

`issues.jsonl` contains one normalized issue as a JSON object per line, UTF-8 encoded
with non-ASCII characters preserved. Output is deterministic for a given raw snapshot
and normalizer version: issues are sorted by `issue_number` ascending, JSON keys are
sorted, every line ends with a newline, and a SHA-256 of the exact output bytes is
recorded in the manifest.

Datetime fields are serialized with an explicit canonical UTC format: aware timestamps
are converted to UTC and rendered as ISO-8601 ending in `Z` (for example
`2026-06-24T16:09:03Z`). Microseconds are omitted when zero and emitted as six digits
otherwise; null `closed_at` remains JSON `null`. This makes the output bytes independent
of any library default spelling.

Each normalized issue includes `schema_version`, `repository`, `issue_id`,
`issue_number`, `title`, `body` (null bodies become `""`), `labels` (label names with
exact duplicates removed and sorted by Unicode code point, case-sensitively, with
original casing preserved), `state`, `author_login`/`author_type` (both null for
deleted accounts), UTC `created_at`/`updated_at`/`closed_at`, `comments_count`,
`html_url`, and `source_page`.

### Immutable, content-aware snapshot IDs

The dataset id is derived deterministically from the raw snapshot's `fetched_at`
timestamp (microsecond precision), the normalizer version, and the first 12 hexadecimal
characters of the full raw-snapshot hash, for example
`20260624T162950093080Z-n1-2e3424600185`. Dataset identity therefore represents
"source contents + transformation version": a change to the raw source bytes or the
normalizer version produces a different id. Re-running against the same valid raw
snapshot validates and reuses the existing processed snapshot (a processed-cache hit)
instead of overwriting it. There is no `--refresh` for processed datasets because ids
are immutable.

### Lineage hashes

Two SHA-256 hashes bind a processed dataset to its raw source:

- `source_manifest_sha256` covers only the raw `manifest.json` bytes.
- `source_snapshot_sha256` covers the complete raw snapshot: the `manifest.json` bytes
  plus every raw page listed in the manifest, fed in manifest order with a fixed version
  marker and length-prefixed paths and contents. This detects raw-page changes even when
  `manifest.json` is unchanged.

On a repeated build both hashes are recomputed from the current raw cache and compared
to the processed manifest. A mismatch is reported as a lineage error, never a cache hit.

### Strict validation behavior

Version 1 is strict. If any issue record is malformed, the build fails and reports the
source page and record position (and issue identifier when available); records are never
silently dropped and no partial dataset is published. Duplicate `issue_id` or
`issue_number` values also fail the build rather than being deduplicated. Builds publish
atomically from a hidden staging directory; a failure or interruption leaves no published
dataset and never alters the raw cache. If an existing processed snapshot is corrupt
(missing output file, hash mismatch, or invalid manifest), the build reports a clear
error instead of silently rebuilding over it. Concurrent builds of the same dataset id
are not supported.

### Versioning

Four version concepts are tracked separately in the processed manifest:

- `schema_version` - shape of the processed `manifest.json`.
- `issue_schema_version` - shape of each normalized JSONL issue record.
- `normalizer_version` - transformation behavior (label handling, null handling,
  included fields, ordering); changing these requires bumping it.
- `source_manifest_schema_version` - the ingestion-manifest schema used as input,
  populated from the validated raw manifest.

### Lineage manifest

`manifest.json` records provenance and reconciliation: `schema_version`,
`issue_schema_version`, `dataset_id`, `repository`, `normalizer_version`, `built_at`,
`source_manifest` (a portable path relative to the raw root, for example
`pandas-dev__pandas/manifest.json`), `source_manifest_sha256`, `source_snapshot_sha256`,
`source_manifest_schema_version`, `source_fetched_at`, `source_api_version`,
`source_pages_fetched`, the counts `raw_records_read`, `pull_requests_excluded`,
`issues_written`, `unlabelled_issues`, `empty_body_issues`, plus `output_file` and
`output_sha256`. SHA-256 fields are validated as 64 lowercase hex characters, the
`dataset_id` is format-validated and checked for consistency with its inputs, and the
manifest enforces `raw_records_read == pull_requests_excluded + issues_written`.

Processed data is local and git-ignored; do not commit `data/processed/`.

## Auditing a normalized dataset

Once a normalized dataset exists, audit one explicit dataset into an immutable audit
artifact:

```bash
repotriage audit-dataset \
  --repo pandas-dev/pandas \
  --dataset-id 20260628T161306010651Z-n1-074402d21505
```

Both `--repo` and `--dataset-id` are required. The command audits exactly one explicitly
named dataset; there is no implicit "latest" selection. Optional flags:

- `--processed-root PATH` overrides the processed-data root (default `data/processed/github`)
- `--audits-root PATH` overrides the audit-artifact root (default `data/audits/github`)

### Normalized data versus audit artifacts

The normalized dataset is the trusted input contract for an audit: the audit subsystem
reads only the validated `issues.jsonl` and its `manifest.json` and never parses raw
GitHub pages. The dataset subsystem does not depend on the audit subsystem. An audit is
a derived, read-only report; it never modifies the normalized dataset.

Before analysis, the audit validates the integrity of the normalized dataset
(`validate_processed_dataset_integrity`): manifest parsing and invariants, directory-name
and id consistency, the requested repository and dataset id, a supported issue schema, safe
paths, and an output SHA-256 check. This integrity check is deliberately separate from the
builder's raw-source compatibility checks, so an audit depends only on the local processed
artifact and never requires the (mutable) raw GitHub cache. A corrupt dataset is reported
rather than audited, and a dataset containing zero issues is rejected before anything is
written.

### Output layout

An audit writes an immutable, versioned artifact:

```text
data/audits/github/pandas-dev__pandas/<dataset-id>-a2/
├── audit.json
├── audit.md
└── manifest.json
```

`audit.json` is the full, machine-readable audit. `audit.md` is a concise human-readable
report with sections for dataset identity, repository overview, text quality, label
distribution, the rare-label summary, top labels, top co-occurring label pairs, temporal
coverage, suitability warnings, and an interpretation caveat. The complete label and
label-pair lists live in `audit.json`; the Markdown shows only a deterministic top subset.

### Objective metrics versus policy warnings

The audit strictly separates two concerns:

- Objective statistics describe the dataset without judgement: issue/label/state counts
  and fractions, title/body/total-text character-length distributions (`total_text_chars`
  is the per-issue sum of title and body lengths) and structural indicators (fenced code
  blocks, URLs, Markdown headings, empty/short/long bodies), label frequencies, cardinality
  and density, rare-label support buckets, label co-occurrence pairs, and monthly temporal
  coverage in UTC. Temporal coverage distinguishes the active month count (distinct months
  that actually contain issues) from the calendar span in months (the inclusive number of
  months between the earliest and latest issue), which can differ for a sparse dataset.
  Percentiles and the median use a single explicit rule (type-7 linear interpolation) so
  results never depend on a library default.
- Suitability warnings are heuristics with a stable code, severity, measured value,
  threshold, and explanation (for example `INSUFFICIENT_LABELLED_ISSUES`,
  `HIGH_UNLABELLED_RATE`, `LIMITED_TEMPORAL_COVERAGE`, `SEVERE_LABEL_LONG_TAIL`, and
  `LOW_TEXT_COMPLETENESS`). These thresholds are versioned heuristics tied to the audit
  version, not universal scientific rules, and there is deliberately no single aggregate
  "quality score". Label-role classification (workflow, type, or component labels) is
  still manual and out of scope.

### Immutable, content-aware audit IDs

The audit id is the dataset id plus an audit-version suffix, for example
`20260628T161306010651Z-n1-074402d21505-a2` with audit version `2`. Because the analysis
and policy for a given audit version are fixed, no configuration hash is part of the
identity. Re-running an audit against the same valid dataset validates and reuses the
existing artifact (an audit-cache hit) instead of overwriting it; immutable audit ids are
never overwritten. An existing corrupt or incompatible audit is reported as an error and
left untouched. Because the version suffix is part of the path, artifacts from different
audit versions coexist: a previously published `-a1` artifact is never read or written by
v2 code and remains intact alongside new `-a2` artifacts. If thresholds later become
configurable, audit identity would need to additionally incorporate that configuration
identity.

### Deterministic JSON and Markdown

`audit.json` and `audit.md` are deterministic for the same normalized dataset bytes and
audit version. `audit.json` is UTF-8, with sorted keys, two-space indentation, `\n`
newlines, a trailing newline, and full-precision numbers; `audit.md` renders counts as
integers, fractions to four decimals, character-length means/percentiles to one decimal,
and datetimes in canonical UTC. Neither file contains a build timestamp. SHA-256 hashes of
both files are recorded in the audit manifest, and the only changing field across rebuilds
is the manifest's `built_at`.

### Lineage manifest

The audit `manifest.json` binds the artifact to its source: `schema_version` (manifest
schema), `audit_document_schema_version` (the `audit.json` document schema, mirrored by the
top-level `schema_version` field inside `audit.json`), `audit_version` (the analysis/policy
version), `audit_id`, `repository`, `dataset_id`, `dataset_output_sha256`,
`issue_schema_version`, `normalizer_version`, `built_at`, `issues_analyzed`, and the file
names and SHA-256 hashes of `audit.json` and `audit.md`. Validation recomputes both report
hashes and then semantically cross-checks the parsed `audit.json` against the manifest
(audit id, repository, dataset id and output hash, versions, and `issues_analyzed` versus
the document's total issue count). These checks detect accidental local corruption and
inconsistency; they are not designed to resist a coordinated rewrite of every file and its
recorded hash together.

### Audit limitations

An audit describes only the bytes of the normalized dataset it was given; it cannot detect
sampling bias in how the raw issues were originally fetched, and it does not interpret
label semantics. Audit artifacts are local and git-ignored; do not commit `data/audits/`.

## Limitations: mutable raw history vs immutable processed history

- The raw cache stores one mutable latest snapshot per repository.
- `fetch-issues --refresh` replaces that raw cache in place.
- Processed datasets, by contrast, use immutable versioned directories; old processed
  datasets remain available after a raw refresh.
- The exact raw page bytes that produced an older processed dataset may no longer exist
  on disk after a raw refresh. `source_snapshot_sha256` can prove whether the currently
  available raw bytes match a processed dataset, but it cannot recreate deleted raw bytes.
- Versioned raw snapshots are a possible future improvement and are intentionally out of
  scope for this milestone (scoped technical debt).

### Legacy processed datasets

Processed datasets created before this hardening pass used the old id format
(`<TIMESTAMP>-n<version>`, with no trailing `-<12 hex>` suffix) and an older,
incompatible manifest schema (no `source_snapshot_sha256`, `issue_schema_version`, or
`source_manifest_schema_version`, and a non-portable absolute `source_manifest`). These
are treated as older incompatible local artifacts: they are never migrated or overwritten
automatically. Because all data under `data/processed/` is git-ignored and can be
regenerated from the validated raw cache, delete any legacy snapshot manually and rebuild.

Inspect `data/processed/github/<owner>__<repo>/` for any directory whose name lacks the
`-<12 hex>` suffix (legacy format), then remove only that snapshot. For example, for the
local pandas dataset:

```bash
# Inspect first
ls -d data/processed/github/pandas-dev__pandas/*Z-n[0-9]* 2>/dev/null

# Remove only the legacy pandas snapshot (old format had no -<hash> suffix)
rm -rf "data/processed/github/pandas-dev__pandas/20260624T162950093080Z-n1"
```

After removal, the next `repotriage build-dataset --repo pandas-dev/pandas` run creates
the new content-aware immutable snapshot
(`20260624T162950093080Z-n1-<source-hash-prefix>`).
