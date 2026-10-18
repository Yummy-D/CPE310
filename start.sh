#!/usr/bin/env bash
# Production startup for StoolCheck AI (Linux/macOS - gunicorn needs POSIX).
#
# Usage:
#   chmod +x start.sh      # once
#   ./start.sh
#
# Reads configuration from the environment and the project-root .env file
# (copy .env.example -> .env and fill it in first). Real environment
# variables always win over .env values.
set -euo pipefail
cd "$(dirname "$0")/backend"

export ENVIRONMENT="${ENVIRONMENT:-production}"   # never auto-reload, /docs off
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"

# Each gunicorn worker loads its own copy of the YOLO/MobileNet models, so keep
# this small (1-2 per CPU is plenty for an inference API).
WORKERS="${WEB_WORKERS:-1}"

exec gunicorn \
  --workers "$WORKERS" \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind "${HOST}:${PORT}" \
  --timeout "${GUNICORN_TIMEOUT:-120}" \
  --access-logfile - \
  --error-logfile - \
  main:app
