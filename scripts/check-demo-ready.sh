#!/usr/bin/env bash
# Verify inference artifacts are ready for the local demo.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${REPOTRIAGE_INFERENCE_CONFIG:-configs/inference/pandas-dev__pandas/local-v1.json}"

repotriage check-artifacts --config "$CONFIG"
