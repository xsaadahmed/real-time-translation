#!/bin/sh
set -e

PORT="${RTT_API_PORT:-8765}"
HOST="${RTT_API_HOST:-0.0.0.0}"

if [ "${RTT_SKIP_MODEL_WARMUP:-0}" != "1" ]; then
  echo "[entrypoint] warming translation models (set RTT_SKIP_MODEL_WARMUP=1 to skip)..."
  python - <<'PY'
from rtt.ui.gradio_app import get_store
get_store()
print("[entrypoint] models ready")
PY
fi

exec "$@"
