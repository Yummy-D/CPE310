"""SQLAlchemy setup — MySQL via environment configuration (production-ready).

Connection settings come from environment variables (see backend/config.py and
the project-root `.env.example`):

    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE

Development defaults below match XAMPP: user 'root', no password, port 3306,
database 'stoolcheck_db'.  A full URL can still be forced with:

    STOOLCHECK_DATABASE_URL=sqlite:///./stoolcheck_dev.db
"""

from urllib.parse import quote_plus

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from config import IS_PRODUCTION, env  # importing config loads <project-root>/.env via dotenv

MYSQL_HOST = env("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(env("MYSQL_PORT", "3306") or 3306)
MYSQL_USER = env("MYSQL_USER", "root")
MYSQL_PASSWORD = env("MYSQL_PASSWORD", "") or ""  # XAMPP default: no password
MYSQL_DB = env("MYSQL_DATABASE", "stoolcheck_db")

# quote_plus: production passwords often contain @ : / + characters that would
# otherwise break the SQLAlchemy URL.
DEFAULT_MYSQL_URL = (
    f"mysql+pymysql://{quote_plus(MYSQL_USER)}:{quote_plus(MYSQL_PASSWORD)}"
    f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
)

SQLALCHEMY_DATABASE_URL = env("STOOLCHECK_DATABASE_URL", DEFAULT_MYSQL_URL)

_IS_SQLITE = SQLALCHEMY_DATABASE_URL.startswith("sqlite")

# Connection arguments per database/environment:
#   - SQLite: allow FastAPI's worker threads to share connections.
#   - MySQL in production (e.g. Aiven cloud): REQUIRES an SSL/TLS connection,
#     so pass {"ssl": {}} to pymysql or the server rejects the handshake.
#   - Local MySQL (development): no extra arguments.
if _IS_SQLITE:
    _CONNECT_ARGS = {"check_same_thread": False}
elif IS_PRODUCTION:  # MySQL + ENVIRONMENT=production (Aiven requires SSL)
    _CONNECT_ARGS = {"ssl": {}}
else:
    _CONNECT_ARGS = {}

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args=_CONNECT_ARGS,
    pool_pre_ping=True,    # transparently reconnect after dropped cloud links
    pool_recycle=300,      # recycle idle links every 5 min; cloud DBs (Aiven)
                           # terminate idle connections much faster than 1h
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency: provides a DB session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
