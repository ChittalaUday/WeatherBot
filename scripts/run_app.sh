#!/usr/bin/env bash
# Runs both halves of the chat app. Ctrl-C stops them together.
#   ./scripts/run_app.sh
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f models/nlu_pipeline.joblib ] || { echo "no model bundle - run: python src/nlu.py --export"; exit 1; }

.venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8787 &
BACKEND=$!
trap 'kill $BACKEND 2>/dev/null || true' EXIT

cd frontend
[ -d node_modules ] || npm install
npm run dev
