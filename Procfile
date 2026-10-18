# Production process definition (Heroku / Render / Railway style).
# Commands run from the repo root, so cd into backend/ where main:app lives.
# $PORT is injected by the platform; WEB_CONCURRENCY sets the worker count.
# NOTE: gunicorn requires Linux/macOS (or WSL on Windows).
web: cd backend && gunicorn -w ${WEB_CONCURRENCY:-2} -k uvicorn.workers.UvicornWorker -b 0.0.0.0:${PORT:-8000} --timeout 120 --access-logfile - --error-logfile - main:app
