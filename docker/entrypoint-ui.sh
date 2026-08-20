#!/bin/sh
set -e

CONFIG_PATH="/app/public/runtime-config.json"

if [ -n "${RTT_PUBLIC_WS_URL:-}" ]; then
  mkdir -p /app/public
  printf '{"wsUrl":"%s"}\n' "$RTT_PUBLIC_WS_URL" > "$CONFIG_PATH"
  echo "[entrypoint] wrote runtime-config.json wsUrl=$RTT_PUBLIC_WS_URL"
fi

exec "$@"
