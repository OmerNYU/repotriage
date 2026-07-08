# Development notes

This file is a short orientation for contributors. For running the local demo and
preparing artifacts, use [`demo.md`](demo.md).

## Subsystem order

RepoTriage is organized as an immutable artifact pipeline plus a local inference
and review surface:

```text
fetch-issues
  → build-dataset
  → audit-dataset
  → build-label-policy
  → build-model-dataset
  → train-baseline
  → build-threshold-policy
  → build-abstention-policy
  → build-retrieval-baseline
  → inference config (binds artifacts)
  → FastAPI serve / Docker Compose
  → React maintainer UI + feedback persistence
```

Each ML stage publishes versioned, content-bound artifacts under `data/` (git-ignored).
The inference bundle is config-only: `configs/inference/<owner>__<repo>/...` points at
the classifier, threshold policy, abstention policy, and retrieval baseline.

## ML engine at a glance

See the README **Architecture** and **How the ML engine works** sections for full diagrams.

At runtime, the React UI calls FastAPI (`/api/v1/infer`), which builds feature text,
scores labels with the frozen TF-IDF + logistic regression artifact, applies threshold
and abstention policies, and retrieves similar train-split issues. Feedback writes go to
the feedback DB via `/api/v1/feedback` and are not used for online learning yet.

The artifact pipeline is separate from serving: historical GitHub data is turned into a
label policy, temporal model-ready split, classifier, threshold, abstention, and
retrieval index before inference can run.

## Local surfaces

| Surface | Role |
|---------|------|
| CLI (`repotriage …`) | Fetch, build, train, check artifacts, `infer-issue`, `serve` |
| FastAPI | `GET /health`, `POST /api/v1/infer`, `POST /api/v1/feedback` |
| React UI | Maintainer score / review / feedback loop |
| Compose | Backend + PostgreSQL (+ optional frontend) for local demos |

## Current scope

The shipped demo targets **pandas-dev/pandas** with a fixed 15-label policy.
Config and artifact layout are repository-oriented, but multi-repo runtime support
is not shipped. Feedback is write-only persistence for human review — not a
retraining loop.
