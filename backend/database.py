"""SQLAlchemy setup — Phase 4: MySQL via XAMPP.

Default connection (XAMPP defaults): user 'root', no password, port 3306,
database 'stoolcheck_db'.  Override with the STOOLCHECK_DATABASE_URL
environment variable, e.g. to run tests against SQLite:

    set STOOLCHECK_DATABASE_URL=sqlite:///./stoolcheck_dev.db
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

MYSQL_HOST = "127.0.0.1"
MYSQL_PORT = 3306
MYSQL_USER = "root"
MYSQL_PASSWORD = ""  # XAMPP default: no password
MYSQL_DB = "stoolcheck_db"

DEFAULT_MYSQL_URL = (
    f"mysql+pymysql://{MYSQL_USER}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
)

SQLALCHEMY_DATABASE_URL = os.environ.get(
    "STOOLCHECK_DATABASE_URL", DEFAULT_MYSQL_URL
)

_IS_SQLITE = SQLALCHEMY_DATABASE_URL.startswith("sqlite")

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    # check_same_thread is SQLite-only; MySQL must not receive it.
    connect_args={"check_same_thread": False} if _IS_SQLITE else {},
    pool_pre_ping=True,   # transparently reconnect after XAMPP restarts
    pool_recycle=3600,
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
