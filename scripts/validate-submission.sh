#!/usr/bin/env bash

set -euo pipefail

PING_URL="${1:-}"
REPO_DIR="${2:-.}"
IMAGE_NAME="${IMAGE_NAME:-support-triage-openenv}"

if [ -z "$PING_URL" ]; then
  echo "Usage: $0 <ping_url> [repo_dir]"
  exit 1
fi

cd "$REPO_DIR"

echo "[1/5] Checking required files"
test -f inference.py
test -f openenv.yaml
test -f pyproject.toml
test -f Dockerfile
test -f server/app.py
test -f client.py
test -f models.py

echo "[2/5] Building Docker image"
docker build -t "$IMAGE_NAME" .

echo "[3/5] Starting local container"
CID="$(docker run -d -p 8000:8000 "$IMAGE_NAME")"
cleanup() {
  docker rm -f "$CID" >/dev/null 2>&1 || true
}
trap cleanup EXIT
sleep 5

echo "[4/5] Probing local endpoints"
curl -fsS http://127.0.0.1:8000/health >/dev/null
curl -fsS http://127.0.0.1:8000/metadata >/dev/null
curl -fsS http://127.0.0.1:8000/schema >/dev/null

echo "[5/5] Probing deployed space"
curl -fsS "$PING_URL" >/dev/null

if command -v openenv >/dev/null 2>&1; then
  echo "[extra] Running openenv validate"
  openenv validate . || openenv validate "$PING_URL"
else
  echo "[extra] openenv CLI not found; skipping openenv validate"
fi

echo "Submission validation checks passed."
