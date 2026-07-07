#!/usr/bin/env bash
# End-to-end smoke test for the Docker Compose backend stack.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BACKEND_PORT="${BACKEND_PORT:-8000}"
BASE_URL="http://127.0.0.1:${BACKEND_PORT}"

MODEL_DATASET_ID="20260628T161306010651Z-n1-074402d21505-md1-14a9768bded7"
BASELINE_RUN_ID="${MODEL_DATASET_ID}-bl4-46227a0ec602"
THRESHOLD_POLICY_ID="${BASELINE_RUN_ID}-tp1-ccaab0996458"
ABSTENTION_POLICY_ID="${THRESHOLD_POLICY_ID}-ap1-9c3c140e7ccb"
RETRIEVAL_RUN_ID="${MODEL_DATASET_ID}-rb1-deb29b6da4eb"

echo "==> Building and starting Docker Compose stack"
docker compose up -d --build --wait

echo "==> GET /health"
health_json="$(curl -fsS "${BASE_URL}/health")"
python3 -c "
import json, sys
payload = json.loads(sys.argv[1])
assert payload.get('status') == 'ok', payload
assert payload.get('repository') == 'pandas-dev/pandas', payload
print('health ok:', payload.get('repository'))
" "$health_json"

echo "==> POST /api/v1/infer"
infer_json="$(curl -fsS -X POST "${BASE_URL}/api/v1/infer" \
  -H 'Content-Type: application/json' \
  -d '{
    "title": "BUG: loc indexing returns unexpected result",
    "body": "When using .loc with a list indexer, result dtype is wrong.",
    "top_k": 5
  }')"
python3 -c "
import json, sys
payload = json.loads(sys.argv[1])
assert payload.get('repository') == 'pandas-dev/pandas', payload
assert 'classification' in payload, payload
print('infer ok: predicted labels =', payload['classification'].get('predicted_labels'))
" "$infer_json"

echo "==> POST /api/v1/feedback"
feedback_status="$(
  curl -fsS -o /tmp/repotriage-smoke-feedback.json -w '%{http_code}' \
    -X POST "${BASE_URL}/api/v1/feedback" \
    -H 'Content-Type: application/json' \
    -d "{
      \"repository\": \"pandas-dev/pandas\",
      \"issue_number\": 12345,
      \"issue_title\": \"BUG: loc indexing returns unexpected result\",
      \"predicted_labels\": [\"Indexing\"],
      \"accepted_labels\": [\"Bug\", \"Indexing\"],
      \"rejected_labels\": [],
      \"review_action\": \"corrected\",
      \"inference_artifacts\": {
        \"model_dataset_id\": \"${MODEL_DATASET_ID}\",
        \"baseline_run_id\": \"${BASELINE_RUN_ID}\",
        \"threshold_policy_id\": \"${THRESHOLD_POLICY_ID}\",
        \"abstention_policy_id\": \"${ABSTENTION_POLICY_ID}\",
        \"retrieval_run_id\": \"${RETRIEVAL_RUN_ID}\"
      }
    }"
)"
if [[ "$feedback_status" != "201" ]]; then
  echo "feedback failed with HTTP ${feedback_status}" >&2
  cat /tmp/repotriage-smoke-feedback.json >&2
  exit 1
fi
echo "feedback ok: HTTP 201"

echo "==> Verify feedback row in PostgreSQL"
row_count="$(
  docker compose exec -T postgres \
    psql -U "${POSTGRES_USER:-repotriage}" -d "${POSTGRES_DB:-repotriage}" -tAc \
    "SELECT COUNT(*) FROM feedback_events;"
)"
row_count="${row_count//[[:space:]]/}"
if [[ "$row_count" -lt 1 ]]; then
  echo "expected at least one feedback_events row, got ${row_count}" >&2
  exit 1
fi
echo "postgres ok: feedback_events count = ${row_count}"

echo "==> Smoke test passed"
