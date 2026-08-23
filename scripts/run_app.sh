#!/usr/bin/env bash
# Runs both halves of the app. Ctrl-C stops them together.
#   ./scripts/run_app.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# Gate on the bundle that will actually answer, not on whichever one used to. Guarding on the
# v3 bundle let uvicorn start and then die inside the startup hook loading v4 - the check
# passed and the app was still down.
MODEL="${DEFAULT_MODEL:-v4}"
[ -f "models/nlu_${MODEL}.joblib" ] || {
  echo "no ${MODEL} bundle at models/nlu_${MODEL}.joblib - build it with:"
  echo "  python -m src.${MODEL}.model --export"
  exit 1
}

.venv/bin/uvicorn backend.api:app --host 0.0.0.0 --port 8787 &
BACKEND=$!
trap 'kill $BACKEND 2>/dev/null || true' EXIT

cd frontend
[ -d node_modules ] || npm install
npm run dev
