#!/usr/bin/env bash
# Linux launcher for the R2R Live SysID Dashboard (Windows equivalent: start_version_2.ps1).
# Starts the FastAPI backend and the Vite dev server, waits on both, and stops
# both on Ctrl-C. Override ports with BACKEND_PORT / FRONTEND_PORT.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# These defaults must match the ports PROGRESS.md and docs/INDEX.md quote, or a
# reader follows the documentation to a port nothing is listening on. They used
# to be 8014/5198 while everything else said 8024/5298.
# JAX must see the venv's own CUDA/cuDNN libraries or its GPU plugin aborts the
# interpreter on first use ("Invalid handle. Cannot load symbol cudnnGetVersion",
# core dump) -- /health keeps answering, the first retune kills the backend.
_NVLIBS="$(ls -d "$ROOT"/.venv/lib/python3*/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')"
export LD_LIBRARY_PATH="${_NVLIBS}${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

BACKEND_PORT="${BACKEND_PORT:-8024}"
FRONTEND_PORT="${FRONTEND_PORT:-5298}"

# A dashboard may already be running on these ports (vite uses --strictPort and
# would otherwise fail with a bare "Port is already in use"). Say so clearly.
for _p in "$BACKEND_PORT" "$FRONTEND_PORT"; do
    if ss -ltn "sport = :$_p" 2>/dev/null | grep -q ":$_p"; then
        echo "Port $_p is already in use - a dashboard is probably already running."
        echo "  open it:  http://127.0.0.1:${FRONTEND_PORT}/"
        echo "  or stop it first:  ss -lptn 'sport = :$_p'   then  kill <pid>"
        exit 1
    fi
done

PYTHON="$ROOT/.venv/bin/python"
VITE="$ROOT/frontend/node_modules/.bin/vite"

if [[ ! -x "$PYTHON" ]]; then
    echo "No .venv found. Create it first:" >&2
    echo "  python3.11 -m venv .venv" >&2
    echo "  .venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt" >&2
    exit 1
fi
if [[ ! -x "$VITE" ]]; then
    echo "frontend deps missing. Run: (cd frontend && npm ci)" >&2
    exit 1
fi

mkdir -p "$ROOT/logs"

PIDS=()
cleanup() {
    trap - EXIT INT TERM
    for pid in "${PIDS[@]:-}"; do
        [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Both services are exec'd directly rather than through a wrapper (npm run dev),
# so $! is the real process and the trap above can actually stop it.
"$PYTHON" -m uvicorn backend.api.main:app \
    --host 127.0.0.1 --port "$BACKEND_PORT" \
    >"$ROOT/logs/backend.log" 2>&1 &
PIDS+=($!)

sleep 2

# vite takes its root as a positional argument (there is no --root flag), so
# run it from the frontend directory. `exec` makes the subshell become vite,
# which keeps $! pointing at the real process for cleanup().
(
    cd "$ROOT/frontend"
    VITE_API_BASE_URL="http://127.0.0.1:${BACKEND_PORT}" \
        exec "$VITE" --host 127.0.0.1 --port "$FRONTEND_PORT" --strictPort
) >"$ROOT/logs/frontend.log" 2>&1 &
PIDS+=($!)

echo "Dashboard : http://127.0.0.1:${FRONTEND_PORT}/"
echo "Backend   : http://127.0.0.1:${BACKEND_PORT}/  (health: /health, docs: /docs)"
echo "Logs      : logs/backend.log  logs/frontend.log"
echo "Ctrl-C to stop both."
wait
