# RepoTriage local demo guide

This guide walks through running the RepoTriage maintainer demo on a fresh clone. The demo
scores a GitHub issue title/body, shows predicted labels, abstention, similar historical
issues, and persists maintainer feedback.

## Overview

| Component | URL | Purpose |
|-----------|-----|---------|
| Backend API | `http://localhost:8000` | Inference + feedback API |
| Maintainer UI | `http://localhost:5173` | Local review workflow |
| PostgreSQL | internal (Compose) | Feedback persistence |

**Important:** ML inference artifacts are **not committed to git**. A fresh `git clone` alone
is not enough to run the demo. You must either copy artifacts from another machine or build
them locally.

The canonical inference-bound artifact set for `pandas-dev/pandas` is about **13 MB**.

## Prerequisites

- Python 3.11+
- Docker and Docker Compose (for the recommended demo path)
- Node.js 18+ and npm (only if running the Vite dev UI instead of the Compose frontend)
- Optional: `GITHUB_TOKEN` if you generate artifacts from scratch via `fetch-issues`

Install the Python package:

```bash
python -m pip install -e ".[dev,ml,db]"
```

## Artifact readiness check

Before starting the backend or Docker Compose, verify artifacts:

```bash
repotriage check-artifacts \
  --config configs/inference/pandas-dev__pandas/local-v1.json
```

Verification modes:

| Mode | Flag | What it checks |
|------|------|----------------|
| Presence (default) | (none) | Config valid; each artifact directory exists; `manifest.json` present |
| Integrity | `--integrity` | Above + SHA256 hashes and manifest identity (no model loading) |
| Strict | `--strict` | Full bundle load including joblib and cross-artifact compatibility |

After copying artifacts from another machine, prefer `--integrity`:

```bash
repotriage check-artifacts --integrity \
  --config configs/inference/pandas-dev__pandas/local-v1.json
```

Machine-readable output:

```bash
repotriage check-artifacts --json \
  --config configs/inference/pandas-dev__pandas/local-v1.json
```

Thin wrapper script:

```bash
./scripts/check-demo-ready.sh
```

## Option A: Copy artifacts (fastest)

Copy these five directory trees from a machine that already built the canonical pipeline.
Preserve the repository slug and artifact IDs bound in
`configs/inference/pandas-dev__pandas/local-v1.json`:

```text
data/model_ready/github/pandas-dev__pandas/
data/baselines/github/pandas-dev__pandas/
data/threshold_policies/github/pandas-dev__pandas/
data/abstention_policies/github/pandas-dev__pandas/
data/retrieval_baselines/github/pandas-dev__pandas/
```

Create parent directories if needed:

```bash
mkdir -p data/{model_ready,baselines,threshold_policies,abstention_policies,retrieval_baselines}/github
```

Verify after copy:

```bash
repotriage check-artifacts --integrity \
  --config configs/inference/pandas-dev__pandas/local-v1.json
```

## Option B: Generate artifacts locally

This runs the documented Sessions 1–8 pipeline for `pandas-dev/pandas`. It requires a raw
GitHub issue cache and can take significant time. Use a `GITHUB_TOKEN` for larger fetches.

### Upstream pipeline (Sessions 1–4)

```bash
repotriage fetch-issues --repo pandas-dev/pandas --max-pages 2

repotriage build-dataset --repo pandas-dev/pandas

repotriage audit-dataset \
  --repo pandas-dev/pandas \
  --dataset-id 20260628T161306010651Z-n1-074402d21505

repotriage build-label-policy \
  --repo pandas-dev/pandas \
  --dataset-id 20260628T161306010651Z-n1-074402d21505 \
  --audit-id 20260628T161306010651Z-n1-074402d21505-a2 \
  --config configs/label_policies/pandas-dev__pandas/policy-v2.json
```

### Inference artifacts (Sessions 5–8)

```bash
repotriage build-model-dataset \
  --repo pandas-dev/pandas \
  --dataset-id 20260628T161306010651Z-n1-074402d21505 \
  --policy-id 20260628T161306010651Z-n1-074402d21505-lp2-95899f0f5b37 \
  --config configs/model_datasets/pandas-dev__pandas/temporal-v1.json

repotriage train-baseline \
  --repo pandas-dev/pandas \
  --model-dataset-id 20260628T161306010651Z-n1-074402d21505-md1-14a9768bded7 \
  --config configs/baselines/pandas-dev__pandas/tfidf-logreg-v1.json

repotriage build-threshold-policy \
  --repo pandas-dev/pandas \
  --baseline-run-id 20260628T161306010651Z-n1-074402d21505-md1-14a9768bded7-bl4-46227a0ec602 \
  --config configs/threshold_policies/pandas-dev__pandas/global-v1.json

repotriage build-abstention-policy \
  --repo pandas-dev/pandas \
  --threshold-policy-id 20260628T161306010651Z-n1-074402d21505-md1-14a9768bded7-bl4-46227a0ec602-tp1-ccaab0996458 \
  --config configs/abstention_policies/pandas-dev__pandas/issue-confidence-v1.json

repotriage build-retrieval-baseline \
  --repo pandas-dev/pandas \
  --model-dataset-id 20260628T161306010651Z-n1-074402d21505-md1-14a9768bded7 \
  --config configs/retrieval_baselines/pandas-dev__pandas/tfidf-cosine-v1.json
```

Confirm readiness:

```bash
repotriage check-artifacts --strict \
  --config configs/inference/pandas-dev__pandas/local-v1.json
```

## Full artifact pipeline

The commands in Option B are the complete ordered pipeline from raw GitHub ingestion through
inference-bound artifacts. If `check-artifacts` reports a missing artifact, run the matching
`next:` command from the checker output, or follow the section above in order.

## Run Docker Compose demo

```bash
cp .env.example .env
docker compose up --build
```

- Backend: `http://localhost:8000`
- Frontend (optional Compose service): `http://localhost:5173`

Health check:

```bash
curl -sS http://localhost:8000/health | python -m json.tool
```

End-to-end smoke test (includes artifact preflight):

```bash
./scripts/docker-smoke.sh
```

## Run maintainer UI

### With Docker Compose frontend

`docker compose up --build` starts the nginx-served frontend on port 5173. The browser uses
relative URLs proxied to the backend container.

### With Vite dev server

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

Open `http://localhost:5173`. Vite proxies `/api` and `/health` to `http://127.0.0.1:8000`.

### Demo workflow

1. Confirm the header health badge shows **Connected · pandas-dev/pandas**.
2. Enter an issue **title**, optional **body**, and optional **top_k** (default 5).
3. Click **Score issue**.
4. Review predicted labels, abstention, and similar issues.
5. Enter a demo **issue number** (required for feedback).
6. Submit maintainer feedback and confirm the stored `feedback_id`.

## Manual verification

```bash
# Artifact preflight
repotriage check-artifacts --strict \
  --config configs/inference/pandas-dev__pandas/local-v1.json

# CLI inference
repotriage infer-issue \
  --repo pandas-dev/pandas \
  --config configs/inference/pandas-dev__pandas/local-v1.json \
  --title "BUG: loc indexing returns unexpected result" \
  --body "When using .loc with a list indexer, result dtype is wrong."

# API infer
curl -sS -X POST http://localhost:8000/api/v1/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "title": "BUG: loc indexing returns unexpected result",
    "body": "When using .loc with a list indexer, result dtype is wrong.",
    "top_k": 5
  }' | python -m json.tool
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Backend container exits on startup | Missing or invalid artifacts | Run `repotriage check-artifacts`; copy or build artifacts |
| `check-artifacts` passes presence but infer fails | Corrupt partial tree | Re-run with `--integrity` or `--strict` |
| Empty bind mount in container | Host directories missing | `mkdir -p data/{model_ready,baselines,threshold_policies,abstention_policies,retrieval_baselines}/github` |
| UI shows **Disconnected** | Backend not running or wrong port | Start backend; confirm `http://localhost:8000/health` |
| Smoke test fails before Compose starts | Artifacts not ready | `./scripts/check-demo-ready.sh` |
| Smoke test fails on feedback | Postgres not ready | Retry; check `docker compose logs backend postgres` |
| `fetch-issues` rate limited | No token | `export GITHUB_TOKEN=...` |

## Limitations

- **No artifact distribution** — no download service or release image with baked artifacts.
- **Pandas MVP only** — bootstrap hints target `pandas-dev/pandas`.
- **Pickle/joblib trust model** — load only trusted local artifacts (`--strict` unpickles models).
- **Presence check is shallow** — default mode does not verify file hashes; use `--integrity` after copy.
- **Checker does not verify upstream artifacts** — only the five inference-bound artifact families.
- **Full pipeline is slow** — not suitable for instant recruiter demos without artifact copy.
- **No authentication, GitHub import, or feedback dashboard** in this milestone.
- **Canonical artifact IDs are pinned** in `local-v1.json`; regenerating with different raw data
  produces different IDs and requires updating the inference config.
