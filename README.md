# RepoTriage

RepoTriage is an ML-assisted GitHub issue-triage platform. This repository currently implements three vertical slices: downloading GitHub repository issues and caching the original API responses locally (raw ingestion), normalizing one raw snapshot into an immutable, issue-only dataset (dataset normalization), and auditing one normalized dataset into an immutable, deterministic audit artifact (dataset audit).

## Installation

Install the project in editable mode with development dependencies:

```bash
python -m pip install -e ".[dev,ml]"
```

The `ml` optional dependency group is required for baseline training (`scikit-learn`, `numpy`, `scipy`, `joblib`).

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

## Building a target-label policy

```bash
repotriage build-label-policy \
  --repo pandas-dev/pandas \
  --dataset-id 20260628T161306010651Z-n1-074402d21505 \
  --audit-id 20260628T161306010651Z-n1-074402d21505-a2 \
  --config configs/label_policies/pandas-dev__pandas/policy-v2.json
```

A target-label policy decides which repository labels are first-model classification
targets. It combines three immutable or tracked inputs: the normalized dataset, its audit
artifact, and a version-controlled, human-authored decision configuration. The current
contract is policy version 2 (`lp2`), with document, manifest, and configuration schemas
all at version 2.

### Why not every label is a model target

A repository's label set mixes several distinct kinds of labels. Many are maintainer
workflow states (for example `Needs Triage`, `Needs Info`) or post-investigation outcomes
(for example `Closing Candidate`, `Duplicate Report`) that are decided during triage and
are not derivable from the initial issue title and body. Training a model to predict such
labels from the initial text leaks the answer or learns the maintainer's process rather
than the issue's content. Other labels are semantically legitimate but too rare or too
inactive recently to support a reliable first model. The policy makes these distinctions
explicit instead of treating every label as a target.

### Objective audit versus human policy

The audit is purely objective: it counts supports, fractions, and monthly coverage with no
judgement. The policy layer is where human, semantic decisions live. The policy never
re-derives roles or reasons from label names; every reviewed label carries an explicit
decision (`include`, `defer`, or `exclude`), a controlled `role`, a controlled
`leakage_risk`, a controlled `reason_code`, and a free-form explanation. Every audited
label that is not explicitly reviewed receives one safe default decision
(`exclude` / `unreviewed` / `unreviewed_default`), so the generated policy always contains
exactly one decision per audited label. Each decision also records its
`decision_source` (`explicit` or `default`), and the policy reports explicit-versus-default
label counts; the Markdown groups are derived from `decision` and `decision_source`, never
from explanation strings.

### Reason-code vocabulary and consistency

Reason codes are a single controlled vocabulary, validated against the decision and source:
`include` requires `selected_target`; default-applied exclusions require
`unreviewed_default`; explicit exclusions require `workflow_label` or
`post_investigation_outcome`; deferrals require `insufficient_total_support`,
`insufficient_active_months`, `insufficient_recent_support`, or `manual_deferral`
(`manual_deferral` requires a non-blank explanation).

### Semantic suitability, leakage risk, and enforced selection criteria

A label is a good target when it is semantically predictable from the initial issue text,
is not a workflow or post-investigation outcome, and has enough support to learn from.
`leakage_risk` records how likely a label is to be assigned only after investigation
(workflow and resolution labels are high risk). The configuration carries a strict
`selection_criteria` model (`min_total_support`, `min_active_months`, `min_recent_support`,
`recent_window_months`, all strictly positive). These thresholds are enforced with
inclusive boundaries: every included label must satisfy all three (`value >= threshold`
passes), or the build fails. A future manually approved exception must use an explicit
`criteria_override_explanation` on that label; criteria violations are never silently
permitted. The exact criteria appear in both `label_policy.json` and `label_policy.md`.

### Temporal support and the recent active-month window

The policy enriches each label decision with objective facts derived by re-streaming the
normalized dataset (the audit document does not carry per-label monthly support): total
support, issue fraction, active-month count, first and last month, and recent-window
support. The recent active-month window is formed by sorting the dataset's distinct active
`%Y-%m` keys and taking the final `recent_window_months` of them (all of them when fewer
exist); a label's recent-window support is the number of its issues created in those
months. This is a policy-selection heuristic, not an ML data split. Dataset-derived counts
are cross-checked against the audit so the two artifacts can never silently disagree.

### Weak supervision

Labels are applied by maintainers and are incomplete. The policy treats an issue without a
given label as a negative example for that label (a weak-supervision assumption). Absent
labels therefore reflect what maintainers applied, not ground truth, and coverage counts
describe applied labels in this dataset only. Issues with no included target are kept, not
dropped.

### Immutable, content-addressed policy artifacts

The policy id is `<dataset-id>-lp2-<12-hex-policy-input-hash-prefix>`. Identity is bound to
every output-affecting input through a canonical `policy_input_sha256` over the policy
version, dataset id, dataset output SHA-256, audit id, audit JSON SHA-256, configuration
schema version, and configuration SHA-256. Changing any of these (dataset bytes, audit
bytes, audit id, configuration semantics, configuration schema, or policy version) changes
the policy id. The configuration hash itself is computed from the parsed, validated,
label-sorted canonical configuration (sorted-key compact JSON), so reformatting or
reordering label entries does not change identity, while changing any decision, role,
leakage risk, reason, explanation, criteria, default, notes, or label inventory does. The
policy never follows the audit package's current version implicitly: the explicit
`--audit-id` is honoured as given and must reference an explicitly supported audit contract
(audit version and document schema on the policy's allowlist), so a newly minted `a3` audit
is not auto-accepted. An artifact is published under
`data/policies/github/<slug>/<policy-id>/` as `label_policy.json`, `label_policy.md`, and
`manifest.json` using the same deterministic, staged, rename-only, never-overwrite
publication as audits. Re-running with the same dataset, audit, validated configuration,
and policy version yields the same policy id and a cache hit. Existing `lp1` artifacts are
immutable and coexist untouched with `lp2`. The full 125-label inventory
lives in `label_policy.json`; the Markdown report uses concise deterministic sections. The
human-authored configuration under `configs/label_policies/` is tracked in git; generated
policy artifacts are git-ignored (do not commit `data/policies/`).

## Building a model-ready dataset

```bash
repotriage build-model-dataset \
  --repo pandas-dev/pandas \
  --dataset-id 20260628T161306010651Z-n1-074402d21505 \
  --policy-id 20260628T161306010651Z-n1-074402d21505-lp2-95899f0f5b37 \
  --config configs/model_datasets/pandas-dev__pandas/temporal-v1.json
```

A model-ready dataset is the first ML-facing artifact. It consumes only the validated
normalized dataset and the validated label-policy artifact; it does not read the mutable raw
cache or require the audit artifact at build time. For each issue it retains stable identity
fields, snapshot title and body (as fetched from the normalized dataset), a deterministic
classifier feature text, canonical target labels and a fixed-order binary target vector, and a
temporal train/validation/test split assignment.

The current contract is model-dataset version 1 (`md1`). Feature text uses
`TEXT_REPRESENTATION_VERSION` 1: `[TITLE]\\n<title>\\n\\n[BODY]\\n<body>` with CRLF/CR
normalized to LF and no other cleaning. Target order comes from
`policy_document.coverage.included_labels` exactly. Records are sorted by
`(created_at, issue_id)`.

Temporal splits are controlled by a tracked configuration under `configs/model_datasets/`
(not CLI defaults). The pandas `temporal-v1` config uses calendar boundaries:
train before `2026-02-01T00:00:00Z`, validation from February–March 2026, test from
April 2026 onward. Zero positives for any included label in train, validation, or test is a
hard build error; 1–4 positives in validation or test produces structured warnings in
`split_report.json`. All-zero target vectors are retained.

Identity binds the normalized dataset output hash, policy JSON hash, split-config hash,
text-representation version, and temporal-splitter version via `model_dataset_input_sha256`.
An artifact is published under `data/model_ready/github/<slug>/<model-dataset-id>/` as
`records.jsonl`, `label_map.json`, `split_report.json`, `split_report.md`, and
`manifest.json` using the same staged, rename-only, never-overwrite publication pattern as
other subsystems. Generated model-ready artifacts are git-ignored (do not commit
`data/model_ready/`).

### Evaluation limitations

**Snapshot-text leakage risk.** Records retain the title and body from the normalized
dataset snapshot, not a guaranteed historical version at issue creation time. GitHub issue
text can be edited after creation; this pipeline does not fetch or store prior body versions.
Models trained on `feature_text` may therefore see post-hoc edits present in the processed
snapshot.

**Recent-label right-censoring risk.** Labels reflect the policy snapshot applied at build
time. Issues created near the end of the dataset window may not yet have received their
final label set when the snapshot was taken, so tail-period issues can be under-labeled
relative to their eventual state.

Any change to record, label-map, or split-report schema versions or to support-validation
semantics requires bumping `MODEL_DATASET_VERSION` (and thus the model-dataset id). Output
contract schema versions are listed in `OUTPUT_CONTRACT_VERSIONS` in
`src/repotriage/model_dataset/models.py`.

## Training a multilabel baseline

Train the first transparent TF-IDF + per-label logistic regression baseline from a validated
model-ready artifact:

```bash
repotriage train-baseline \
  --repo pandas-dev/pandas \
  --model-dataset-id 20260628T161306010651Z-n1-074402d21505-md1-14a9768bded7 \
  --config configs/baselines/pandas-dev__pandas/tfidf-logreg-v1.json
```

Optional flags:

- `--model-ready-root PATH` overrides the default model-ready root
- `--baselines-root PATH` overrides the default baseline artifact root

Baseline artifacts are written under:

```text
data/baselines/github/<owner>__<repo>/<baseline-run-id>/
```

Each artifact includes frozen configuration, validation candidate comparison, test metrics,
validation/test prediction JSONL files, a feature summary, and a serialized model bundle
(`model.joblib`). Generated baseline artifacts are git-ignored (do not commit
`data/baselines/`).

### Baseline protocol

1. **Train only on train split** — the TF-IDF vectorizer and per-label logistic regressions
   are fit on train `feature_text` and train targets only.
2. **Validation selection** — a small predeclared candidate set is compared on validation
   metrics; macro average precision is the primary selection metric. Test data is not loaded
   until after the winner is frozen.
3. **Frozen test evaluation** — the selected train-only model is evaluated on test once.

The fixed decision threshold is `0.5` on `predict_proba` outputs. Stored scores are
**probability estimates**, not calibrated confidence.

### Baseline identity (bl4)

Baseline run IDs use the form `<model-dataset-id>-bl4-<12-hex>` and bind three hashes:

- `baseline_experiment_sha256` — model-dataset lineage, canonical config semantics, protocol
  versions (including `model_semantic_contract_version`), threshold, and random seed (not raw
  config file bytes).
- `numerical_environment_sha256` — Python implementation/version, OS system, machine
  architecture, numpy/scipy/scikit-learn/joblib/threadpoolctl versions, a canonical
  numerical-backend fingerprint (exact BLAS/LAPACK/OpenMP backend versions, threading layer,
  and backend architecture from `threadpoolctl.threadpool_info()`), and the controlled
  `numerical_thread_limit`.
- `baseline_run_sha256` — `sha256(experiment_hash + environment_hash)`.

Numerical fitting and scoring run under `threadpoolctl.threadpool_limits(limits=1)` to remove
machine-default BLAS thread-count nondeterminism. The numerical-backend fingerprint excludes
volatile fields (absolute paths, install dirs, hostnames, PIDs, live thread counts), so it
changes when a backend *version* changes but not when file paths differ.

Cache hits require an exact environment fingerprint match. Identical experiments may produce
different run IDs across library or numerical-backend versions. The bl4 contract supersedes
bl1/bl2/bl3; older artifacts remain on disk as historical local artifacts and are never
overwritten.

### Model serialization contract (bl4)

`model_semantic_sha256` is the authoritative identity for the learned model state. It covers
inference-relevant fitted fields only: sorted vocabulary mapping, `idf_` array bytes, per-label
estimator parameters, `classes_`, `coef_`, `intercept_`, and `n_iter_`. Volatile pickle
metadata (for example scikit-learn's private `_stop_words_id`, which stores `id(stop_words)`)
is excluded structurally.

`model_sha256` remains a local file-integrity checksum for the serialized `model.joblib`
bytes on disk. **Byte-identical model serialization is not guaranteed** across processes even
when predictions, metrics, and `model_semantic_sha256` match. Same-run equivalence is based on
the semantic fingerprint plus identical stored predictions and metrics.

### Metric contract v2

Macro precision/recall/F1 use **zero-filled** averaging over all labels (undefined per-label
metrics count as `0.0`). Defined-only means are recorded separately as
`macro_*_defined_only`. Per-label F1 is `0.0` when precision and recall are both defined
and zero. `samples_f1` uses the `empty_empty -> 1.0` policy (`samples_f1_empty_empty_policy:
"one"`).

Validation predictions store **all candidates** (`candidate_id` on each row); test
predictions store the frozen winner only.

### Reproducibility and trust boundary

`validate_baseline_artifact_integrity` and `validate_baseline_against_inputs` verify hashes,
schemas, source alignment, and candidate-selection audit **without** unpickling
`model.joblib`. `verify_baseline_model_consistency(trust_model_file=True)` is the only
validator path that loads the serialized model and recomputes scores.

Metrics and predictions are reproducible within the recorded numerical environment
documented in each artifact manifest. They are **not** guaranteed byte-identical across
operating systems, BLAS implementations, or library patch versions.

`model.joblib` uses pickle-based serialization. **Do not load model files from untrusted
sources.**

## Global threshold policy (tp1)

Session 6 publishes one **global probability threshold** per frozen baseline artifact.
The policy consumes stored validation and test score vectors from the baseline artifact
(no model loading), sweeps an integer basis-point grid on validation only, and freezes
the winning threshold before evaluating test metrics.

```bash
repotriage build-threshold-policy \
  --repo pandas-dev/pandas \
  --baseline-run-id 20260628T161306010651Z-n1-074402d21505-md1-14a9768bded7-bl4-46227a0ec602 \
  --config configs/threshold_policies/pandas-dev__pandas/global-v1.json
```

Artifacts are written under `data/threshold_policies/github/<owner>__<repo>/<policy-id>/`
with ids of the form `<baseline-run-id>-tp1-<12-hex>`. Selection uses validation macro F1,
then micro F1, then proximity to the 0.50 reference threshold (in basis points), then
higher threshold. Test metrics are informational only and never influence selection.

## Abstention policy (ap1)

Session 7 publishes one **issue-level abstention threshold** per frozen threshold-policy
artifact. The policy consumes baseline score vectors and the Session 6 classification
threshold (no model loading), sweeps an integer basis-point abstention grid on validation
only, and freezes the winning abstention threshold before evaluating test metrics and
confidence-bin diagnostics.

```bash
repotriage build-abstention-policy \
  --repo pandas-dev/pandas \
  --threshold-policy-id 20260628T161306010651Z-n1-074402d21505-md1-14a9768bded7-bl4-46227a0ec602-tp1-ccaab0996458 \
  --config configs/abstention_policies/pandas-dev__pandas/issue-confidence-v1.json
```

Artifacts are written under `data/abstention_policies/github/<owner>__<repo>/<policy-id>/`
with ids of the form `<threshold-policy-id>-ap1-<12-hex>`. Issue confidence is the maximum
score among predicted labels at the classification threshold. Selection requires validation
coverage >= 0.25, then ranks by handled subset accuracy, handled samples F1, coverage, and
lower abstention threshold. Test metrics and confidence bins are informational only.

## Similar-issue retrieval baseline (rb1)

Session 8 publishes one **TF-IDF cosine-similarity retrieval baseline** per model-ready
dataset. The artifact indexes train-split issues only, retrieves nearest neighbors for
validation and test queries, and evaluates label-overlap retrieval metrics.

```bash
repotriage build-retrieval-baseline \
  --repo pandas-dev/pandas \
  --model-dataset-id 20260628T161306010651Z-n1-074402d21505-md1-14a9768bded7 \
  --config configs/retrieval_baselines/pandas-dev__pandas/tfidf-cosine-v1.json
```

Artifacts are written under
`data/retrieval_baselines/github/<owner>__<repo>/<retrieval-run-id>/` with ids of the form
`<model-dataset-id>-rb1-<12-hex>`. Retrieved issues are **similar historical issues**
and nearest neighbors under the selected TF-IDF representation — not guaranteed duplicates and not evidence of
semantic understanding. The vectorizer is fit on train records only; validation and test
records are queries only. Test metrics are informational only.

`vectorizer.joblib` uses pickle-based serialization. **Do not load serialized index files
from untrusted sources.** Joblib bytes are not guaranteed deterministic across environments;
`index_semantic_sha256` in each manifest is the authoritative fitted-state identity.

## Local issue inference (Session 9)

Session 9 adds a **local inference pathway** that scores a new issue-like input (title and
body) and returns one combined JSON response with label scores, thresholded predictions,
abstention decision, similar historical train-corpus issues, artifact IDs, and
reproducibility metadata.

The inference bundle is **config-only**: `configs/inference/<owner>__<repo>/local-v1.json`
binds the four canonical artifacts (baseline classifier, threshold policy, abstention
policy, retrieval baseline). No separate inference artifact is published under `data/`.

Feature text uses the same v1 contract as the model-ready dataset (`[TITLE]` / `[BODY]`
markers with CRLF/CR normalization only). Inference imports
`build_feature_text_v1` from the model-ready builder path — do not format title/body
differently at inference time.

```bash
repotriage infer-issue \
  --repo pandas-dev/pandas \
  --config configs/inference/pandas-dev__pandas/local-v1.json \
  --title "BUG: loc indexing returns unexpected result" \
  --body "When using .loc with a list indexer, result dtype is wrong." \
  --top-k 5 \
  --pretty
```

The response includes:

- `classification`: per-label scores in canonical label order, threshold `0.39`, and
  predicted labels;
- `abstention`: issue confidence (`max_predicted_label_score` among predicted labels) vs
  abstention threshold `0.84`; issues with no predicted labels are forced to abstain;
- `retrieval`: top-k similar train-corpus issues with `predicted_label_overlap` (overlap
  between **predicted** labels and each neighbor's historical selected labels — not
  true-label overlap);
- `artifacts` and `reproducibility`: lineage IDs and semantic fingerprints.

`model.joblib` and `vectorizer.joblib` use pickle-based serialization via joblib.
**Load only trusted local artifacts** (`trust_serialized_models: true` in the inference
config). Integrity checks run before unpickling; semantic fingerprints are verified after
load.

Prerequisites: Sessions 5–8 artifacts for the bound repository must exist locally under
`data/baselines/`, `data/threshold_policies/`, `data/abstention_policies/`, and
`data/retrieval_baselines/`, plus the model-ready dataset under `data/model_ready/`.

## FastAPI inference backend (Session 10)

Session 10 adds a thin FastAPI HTTP wrapper over the Session 9 local inference pathway.
This is a **local/dev backend contract**, not a production-hardened deployment. The server
loads one inference bundle at startup and exposes two endpoints:

- `GET /health` — liveness plus loaded repository and config path;
- `POST /api/v1/infer` — score a new issue title/body and return the same JSON response
  shape as `repotriage infer-issue`.

Install dependencies (inference still requires the `[ml]` extra):

```bash
pip install -e ".[ml,dev]"
```

Start the server with the canonical pandas inference config:

```bash
repotriage serve \
  --config configs/inference/pandas-dev__pandas/local-v1.json \
  --host 127.0.0.1 \
  --port 8000
```

Alternative startup via uvicorn factory (requires `REPOTRIAGE_INFERENCE_CONFIG`):

```bash
export REPOTRIAGE_INFERENCE_CONFIG=configs/inference/pandas-dev__pandas/local-v1.json
uvicorn repotriage.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl -sS http://127.0.0.1:8000/health | python -m json.tool
```

Infer request:

```bash
curl -sS -X POST http://127.0.0.1:8000/api/v1/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "title": "BUG: loc indexing returns unexpected result",
    "body": "When using .loc with a list indexer, result dtype is wrong.",
    "top_k": 5
  }' | python -m json.tool
```

Validation error (missing `title`):

```bash
curl -sS -X POST http://127.0.0.1:8000/api/v1/infer \
  -H 'Content-Type: application/json' \
  -d '{"body": "no title"}' -w '\nHTTP %{http_code}\n'
```

The POST body accepts the same fields as `InferenceIssueInput`: `title` (required),
`body` (optional, default empty string), `top_k` (optional), and optional
`issue_number` / `issue_id` for forward compatibility. Repository and artifact binding
come from the server config, not the request body.

OpenAPI documentation is available at `/docs` when the server is running.

Prerequisites are identical to Session 9: Sessions 5–8 artifacts for the bound
repository must exist locally.

### Session 10 limitations

- **Single-repo only** — one inference config per server process; no repo selector in the
  API.
- **No authentication** — any client with network access can call the inference endpoint.
- **No rate limiting** beyond server defaults.
- **No request-size hardening** beyond framework defaults.
- **Synchronous CPU-bound inference** — concurrent requests share one process.
- **In-memory bundle** — model, retrieval index, and corpus matrix stay loaded for the
  process lifetime.
- **Pickle/joblib security** — same trust model as Session 9; load only trusted local
  artifacts.
- **No hot reload** — config or artifact changes require a server restart.
- **No GitHub write-back or deployment** in this milestone. For Docker Compose,
  see Session 12.
- **`generated_at` timestamps** differ between separate CLI and API calls for the same
  input.
- **`issue_number` / `issue_id`** are accepted in the request body but are not reflected
  in the response today.
- **Retrieval results** are similar historical train-corpus issues and nearest neighbors
  under the TF-IDF representation — not guaranteed duplicates and not evidence of semantic
  understanding.

## Maintainer feedback persistence (Session 11)

Session 11 adds the first durable maintainer-feedback persistence layer. After a maintainer
reviews an inference prediction, the backend can store a review event with predicted and
corrected labels, review action, optional note, issue metadata, and inference artifact IDs.
This is a write-only local/dev foundation for human-in-the-loop feedback — not production
deployment.

Install dependencies (PostgreSQL driver is optional and only needed for `postgresql://`
URLs):

```bash
pip install -e ".[ml,dev,db]"
```

### Database setup

By default the server stores feedback in a local SQLite file:

```text
./data/repotriage_feedback.db
```

For PostgreSQL, set `DATABASE_URL` (requires the `[db]` extra):

```bash
createdb repotriage
export DATABASE_URL="postgresql+psycopg://repotriage:repotriage@localhost:5432/repotriage"
```

### Start the server

```bash
repotriage serve \
  --config configs/inference/pandas-dev__pandas/local-v1.json \
  --host 127.0.0.1 \
  --port 8000 \
  --database-url "${DATABASE_URL:-sqlite:///./data/repotriage_feedback.db}"
```

The `--database-url` flag overrides the `DATABASE_URL` environment variable. When neither
is set, the default SQLite path above is used.

`GET /health` reports inference-bundle status only (repository and config path). It does
not expose feedback-database connectivity; an unreachable feedback database causes startup
to fail before the server accepts requests.

### Workflow: infer, then submit feedback

```bash
# 1. Score an issue
curl -sS -X POST http://127.0.0.1:8000/api/v1/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "title": "BUG: loc indexing returns unexpected result",
    "body": "When using .loc with a list indexer, result dtype is wrong.",
    "top_k": 5
  }' | python -m json.tool

# 2. Submit maintainer feedback using labels and artifact IDs from the infer response
curl -sS -X POST http://127.0.0.1:8000/api/v1/feedback \
  -H 'Content-Type: application/json' \
  -d '{
    "repository": "pandas-dev/pandas",
    "issue_number": 12345,
    "issue_title": "BUG: loc indexing returns unexpected result",
    "issue_body_preview": "When using .loc...",
    "predicted_labels": ["Indexing"],
    "accepted_labels": ["Bug", "Indexing"],
    "rejected_labels": [],
    "review_action": "corrected",
    "reviewer_note": "Should also include Bug.",
    "inference_artifacts": {
      "model_dataset_id": "<from infer response artifacts>",
      "baseline_run_id": "<from infer response artifacts>",
      "threshold_policy_id": "<from infer response artifacts>",
      "abstention_policy_id": "<from infer response artifacts>",
      "retrieval_run_id": "<from infer response artifacts>"
    }
  }' -w '\nHTTP %{http_code}\n'
```

Validation error (wrong repository):

```bash
curl -sS -X POST http://127.0.0.1:8000/api/v1/feedback \
  -H 'Content-Type: application/json' \
  -d '{
    "repository": "other/repo",
    "issue_number": 12345,
    "issue_title": "BUG: example",
    "predicted_labels": ["Indexing"],
    "accepted_labels": ["Indexing"],
    "rejected_labels": [],
    "review_action": "accepted",
    "inference_artifacts": {
      "model_dataset_id": "<from infer response artifacts>",
      "baseline_run_id": "<from infer response artifacts>",
      "threshold_policy_id": "<from infer response artifacts>",
      "abstention_policy_id": "<from infer response artifacts>",
      "retrieval_run_id": "<from infer response artifacts>"
    }
  }' -w '\nHTTP %{http_code}\n'
```

A successful response returns HTTP 201:

```json
{
  "feedback_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "created_at": "2026-07-07T13:00:00Z",
  "status": "stored"
}
```

### Feedback request fields

| Field | Required | Description |
|-------|----------|-------------|
| `repository` | yes | Must match the server-bound repository from inference config |
| `issue_number` | yes | GitHub issue number (`> 0`) |
| `issue_title` | yes | Issue title at review time |
| `issue_body_preview` | no | Up to 200 characters of issue body |
| `predicted_labels` | yes | Labels the model predicted |
| `accepted_labels` | yes | Labels the maintainer accepts as correct |
| `rejected_labels` | no | Labels explicitly rejected (default `[]`) |
| `review_action` | yes | `accepted`, `corrected`, or `rejected` |
| `reviewer_note` | no | Optional free-text note (max 4000 chars) |
| `inference_artifacts` | yes | Five artifact IDs from the inference response |

### Validation rules

- **Repository** must match the loaded inference bundle (same single-repo binding as
  `POST /api/v1/infer`).
- **Labels** must belong to the bundle's canonical `label_order`; no duplicates within a
  list; `accepted_labels` and `rejected_labels` must be disjoint.
- **Artifact IDs** must match the format regexes and exactly match the loaded bundle's
  config IDs.
- **`review_action` coherence**:
  - `accepted`: `accepted_labels == predicted_labels` and `rejected_labels` is empty
  - `corrected`: `accepted_labels != predicted_labels`
  - `rejected`: `accepted_labels` is empty and `rejected_labels` equals `predicted_labels`

The endpoint does not verify that `issue_number` exists in the dataset or that
`predicted_labels` match a fresh inference call.

### Session 11 limitations

- **No authentication** — any client with network access can write feedback.
- **Write-only** — no feedback read/list API in this milestone.
- **No deduplication, edit, or delete** — multiple reviews per issue are allowed.
- **No linkage to a server-side inference request ID** — only artifact IDs and issue
  metadata are stored.
- **Schema bootstrap via `create_all` only** — no Alembic migrations yet.
- **SQLite default** is a dev convenience; production-oriented setups should use
  PostgreSQL via `DATABASE_URL`.
- **No automatic retraining, GitHub labeling/commenting, multi-repo support, frontend,
  or production deployment**. For local Docker Compose, see Session 12.

Prerequisites are identical to Session 10: Sessions 5–8 artifacts for the bound repository
must exist locally.

## Docker Compose backend (Session 12)

Session 12 adds a **local Docker Compose runtime** for the Session 10–11 backend. A developer
can start the FastAPI inference API and a PostgreSQL feedback database with one command.
This is for **cloneability and local deployment readiness**, not production cloud deployment.

### Prerequisites

Inference artifacts are **not committed** to git. Before starting Compose, Sessions 5–8
artifacts for `pandas-dev/pandas` must exist on the host under:

- `data/model_ready/github/`
- `data/baselines/github/`
- `data/threshold_policies/github/`
- `data/abstention_policies/github/`
- `data/retrieval_baselines/github/`

See Sessions 5–8 above for how to generate these directories. A fresh `git clone` alone is
not sufficient — copy artifacts from another machine or run the local build pipeline first.

Create host directories if needed:

```bash
mkdir -p data/{model_ready,baselines,threshold_policies,abstention_policies,retrieval_baselines}/github
```

### Quick start

```bash
cp .env.example .env
docker compose up --build
```

The backend listens on `http://localhost:8000` by default (`BACKEND_PORT` in `.env`).

Health check:

```bash
curl -sS http://localhost:8000/health | python -m json.tool
```

Infer request:

```bash
curl -sS -X POST http://localhost:8000/api/v1/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "title": "BUG: loc indexing returns unexpected result",
    "body": "When using .loc with a list indexer, result dtype is wrong.",
    "top_k": 5
  }' | python -m json.tool
```

Feedback request (use artifact IDs from the infer response or from `local-v1.json`):

```bash
curl -sS -X POST http://localhost:8000/api/v1/feedback \
  -H 'Content-Type: application/json' \
  -d '{
    "repository": "pandas-dev/pandas",
    "issue_number": 12345,
    "issue_title": "BUG: loc indexing returns unexpected result",
    "predicted_labels": ["Indexing"],
    "accepted_labels": ["Bug", "Indexing"],
    "rejected_labels": [],
    "review_action": "corrected",
    "inference_artifacts": {
      "model_dataset_id": "20260628T161306010651Z-n1-074402d21505-md1-14a9768bded7",
      "baseline_run_id": "20260628T161306010651Z-n1-074402d21505-md1-14a9768bded7-bl4-46227a0ec602",
      "threshold_policy_id": "20260628T161306010651Z-n1-074402d21505-md1-14a9768bded7-bl4-46227a0ec602-tp1-ccaab0996458",
      "abstention_policy_id": "20260628T161306010651Z-n1-074402d21505-md1-14a9768bded7-bl4-46227a0ec602-tp1-ccaab0996458-ap1-9c3c140e7ccb",
      "retrieval_run_id": "20260628T161306010651Z-n1-074402d21505-md1-14a9768bded7-rb1-deb29b6da4eb"
    }
  }' -w '\nHTTP %{http_code}\n'
```

### Architecture

- **`backend`** — builds from [`Dockerfile`](Dockerfile); runs `repotriage serve` on
  `0.0.0.0:8000` with ML artifacts bind-mounted read-only from `./data/`.
- **`postgres`** — `postgres:16-alpine` with a named volume (`postgres_data`) for durable
  feedback storage.
- **Configs** — baked into the backend image from `configs/` (committed).
- **Artifacts** — host-mounted at runtime; not copied into the image.

`GET /health` reports inference-bundle status only. It does not probe PostgreSQL; an
unreachable database causes startup to fail before requests are accepted.

### Environment variables

Copy [`.env.example`](.env.example) to `.env` before the first run:

| Variable | Default | Purpose |
|----------|---------|---------|
| `POSTGRES_USER` | `repotriage` | PostgreSQL role |
| `POSTGRES_PASSWORD` | `repotriage` | PostgreSQL password (local dev only) |
| `POSTGRES_DB` | `repotriage` | PostgreSQL database name |
| `BACKEND_PORT` | `8000` | Host port mapped to the backend container |
| `REPOTRIAGE_INFERENCE_CONFIG` | `configs/inference/pandas-dev__pandas/local-v1.json` | Inference bundle config path |
| `DATABASE_URL` | `postgresql+psycopg://repotriage:repotriage@postgres:5432/repotriage` | SQLAlchemy URL for feedback persistence |

### Verify PostgreSQL persistence

```bash
docker compose exec postgres \
  psql -U repotriage -d repotriage \
  -c "SELECT id, repository, issue_number, review_action, created_at FROM feedback_events ORDER BY created_at DESC LIMIT 5;"
```

### Smoke test

```bash
./scripts/docker-smoke.sh
```

This script builds and starts the stack, calls `/health`, `/api/v1/infer`, and
`/api/v1/feedback`, then verifies at least one row exists in `feedback_events`.

### Stopping and resetting

```bash
docker compose down          # stop containers, keep postgres volume
docker compose down -v       # stop containers and delete postgres data
```

### Non-Docker local development

Docker Compose is optional. Outside Docker, `repotriage serve` still defaults to SQLite
(`./data/repotriage_feedback.db`) when `DATABASE_URL` is unset. See Session 11 above.

### Session 12 limitations

- **Local dev only** — default `.env` credentials are not suitable for production.
- **Artifacts required on host** — git clone alone does not include ML artifacts.
- **No artifact distribution** — no download service or release image with baked artifacts.
- **No authentication, frontend, cloud deployment, or Kubernetes**.
- **No hot reload** — config or artifact changes require `docker compose restart backend`.
- **Synchronous CPU-bound inference** — same as Session 10.
- **Pickle/joblib trust model** — mount only trusted local artifacts.
- **`/health` does not check Postgres** after startup.
- **Schema bootstrap via `create_all` only** — no Alembic migrations.
- **Postgres port not published** to the host by default (backend connects internally).

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
