#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv"
FRONTEND="$ROOT/clipify-frontend"

# ── Preflight checks ──────────────────────────────────────────────────────────
if [ ! -f "$VENV/bin/uvicorn" ]; then
  echo "ERROR: .venv not found. Run:"
  echo "  python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

if [ ! -d "$FRONTEND/node_modules" ]; then
  echo "ERROR: node_modules not found. Run:"
  echo "  cd clipify-frontend && npm install"
  exit 1
fi

# ── Start backend ─────────────────────────────────────────────────────────────
echo "Starting backend  →  http://127.0.0.1:8000"
cd "$ROOT"
"$VENV/bin/uvicorn" api:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

# ── Start frontend ────────────────────────────────────────────────────────────
echo "Starting frontend →  http://localhost:5173"
cd "$FRONTEND"
npm run dev &
FRONTEND_PID=$!

# ── Cleanup on Ctrl+C ─────────────────────────────────────────────────────────
cleanup() {
  echo ""
  echo "Shutting down..."
  kill "$BACKEND_PID"  2>/dev/null || true
  kill "$FRONTEND_PID" 2>/dev/null || true
  wait "$BACKEND_PID"  2>/dev/null || true
  wait "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup INT TERM

echo ""
echo "  App running at http://localhost:5173"
echo "  Press Ctrl+C to stop."
echo ""

wait
