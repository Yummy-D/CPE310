"""Central configuration for StoolCheck AI — environment-driven, production-ready.

Every sensitive value is read from environment variables (optionally declared
in a `.env` file at the project root, loaded here with python-dotenv):

    ENVIRONMENT              development | production  (default: development)
    MYSQL_HOST/PORT/USER/PASSWORD/DATABASE
    STOOLCHECK_DATABASE_URL full SQLAlchemy URL, optional OVERRIDE of the
                            individual MYSQL_* values above
    SECRET_KEY               JWT signing secret (legacy alias:
                             STOOLCHECK_SECRET_KEY); REQUIRED in production
    JWT_EXPIRE_MINUTES       token lifetime (default: 720 = 12h)
    HOST / PORT              uvicorn bind address for `python main.py`
    CORS_ORIGINS             comma-separated allowed browser origins
    ENABLE_DOCS              expose /docs + /redoc in production (default: no)

Rules honoured here:
- Real process environment ALWAYS wins over `.env` values, so a production
  server can simply export the variables (or use a secrets manager).
- The app still boots if python-dotenv is not installed — in that case only
  real environment variables are used (fail-soft for minimal images).
"""

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

# Load <project root>/.env if present; never overrides real environment vars.
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:  # pragma: no cover - dotenv optional at boot
    pass


def env(name: str, default: str | None = None) -> str | None:
    """Read an environment variable (after .env has been loaded)."""
    return os.environ.get(name, default)


def env_bool(name: str, default: bool = False) -> bool:
    """Parse an environment variable as a boolean (1/true/yes/on)."""
    value = env(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# --- Application environment --------------------------------------------------
ENVIRONMENT = (env("ENVIRONMENT", "development") or "development").strip().lower()
IS_PRODUCTION = ENVIRONMENT == "production"
