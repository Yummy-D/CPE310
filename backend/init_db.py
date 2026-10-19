"""Create the MySQL database and tables if they don't exist.

Run once (and whenever the schema changes):
    python backend/init_db.py        # or: cd backend && python init_db.py

- Connection settings come from the environment / project-root .env
  (MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE).
- When ENVIRONMENT=production, EVERY connection - including the raw
  pymysql.connect() below - forces SSL/TLS ({}) because cloud providers
  (Aiven) reject non-encrypted clients with MySQL error 2013.
- Creates the target database (utf8mb4) and the 'users'/'analyses' tables
  if missing.  Idempotent: safe to run repeatedly.
"""

import sys

from sqlalchemy import inspect, text

import models  # noqa: F401  (registers Analysis on Base.metadata)
from database import (
    MYSQL_CONNECT_ARGS,
    MYSQL_DB,
    MYSQL_HOST,
    MYSQL_PASSWORD,
    MYSQL_PORT,
    MYSQL_USER,
    SQLALCHEMY_DATABASE_URL,
    Base,
    engine,
)


def _preflight_greeting(timeout: float = 5.0) -> str:
    """Probe the endpoint before pymysql: does a MySQL greeting arrive?

    Returns "ok" | "no-greeting" | "unreachable".  Distinguishes a
    stopped/wrong-port service (TCP opens but MySQL never speaks - the
    misleading error 2013) from genuine SSL/auth problems.
    """
    import socket

    try:
        sock = socket.create_connection((MYSQL_HOST, MYSQL_PORT), timeout=timeout)
    except OSError:
        return "unreachable"
    try:
        sock.settimeout(timeout)
        head = sock.recv(4)  # MySQL always speaks first
        return "ok" if head else "no-greeting"
    except OSError:
        return "no-greeting"
    finally:
        sock.close()


def create_database() -> None:
    """Create the schema itself (MySQL only; SQLite needs no schema step)."""
    if not SQLALCHEMY_DATABASE_URL.startswith("mysql"):
        print(f"[SKIP] non-MySQL URL in use ({SQLALCHEMY_DATABASE_URL.split(':')[0]}), no schema to create")
        return
    import pymysql

    # Same SSL decision as the SQLAlchemy engine (database.MYSQL_CONNECT_ARGS):
    # {"ssl": {}} in production (Aiven etc.), None locally.
    ssl_arg = MYSQL_CONNECT_ARGS.get("ssl")
    print(
        f"[..] connecting to {MYSQL_HOST}:{MYSQL_PORT} as {MYSQL_USER} "
        f"(ssl={'ON' if ssl_arg is not None else 'OFF'})"
    )

    # Preflight: does this endpoint actually speak MySQL?  A silent endpoint
    # (TCP opens, greeting never arrives) is what produced the misleading
    # "error 2013 Lost connection during query" - it is NOT an SSL problem,
    # so detect it and say exactly what to check in the Aiven Console.
    preflight = _preflight_greeting()
    if preflight == "unreachable":
        sys.exit(
            f"[ERROR] Cannot reach {MYSQL_HOST}:{MYSQL_PORT} (TCP refused/timeout).\n"
            "        - Local XAMPP: start MySQL in the XAMPP Control Panel.\n"
            "        - Cloud (Aiven): check the host/port and your network.\n"
        )
    if preflight == "no-greeting":
        sys.exit(
            f"[ERROR] {MYSQL_HOST}:{MYSQL_PORT} accepts TCP but never sends the\n"
            "        MySQL greeting - so this is NOT an SSL/authentication problem.\n"
            "        The MySQL service behind this endpoint is not answering.\n"
            "        Check in the Aiven Console:\n"
            "          1) the service status is RUNNING (stopped/paused services\n"
            "             accept TCP but never speak), and\n"
            "          2) MYSQL_HOST / MYSQL_PORT in .env match the Overview page.\n"
            "        (Symptom before this check: pymysql error 2013.)\n"
        )
    print("[OK] MySQL greeting received -> starting TLS/pymysql handshake")

    try:
        conn = pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            charset="utf8mb4",
            connect_timeout=15,  # fail fast instead of hanging on blocked ports
            read_timeout=15,     # Aiven's documented Python timeouts
            write_timeout=15,
            ssl=ssl_arg,         # {} enables TLS with PyMySQL defaults; None = plain
        )
    except pymysql.MySQLError as exc:
        sys.exit(
            f"[ERROR] MySQL greeting OK, but handshake failed at {MYSQL_HOST}:{MYSQL_PORT} -> {exc}\n"
            "        - Cloud (Aiven): set ENVIRONMENT=production (forces SSL) and\n"
            "          check MYSQL_USER / MYSQL_PASSWORD in .env.\n"
            "        - Local XAMPP: start MySQL in the XAMPP Control Panel.\n"
        )
    with conn:  # closes the connection on exit
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DB}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
    print(f"[OK] database '{MYSQL_DB}' exists (created if missing)")


def create_tables() -> None:
    """Create tables via SQLAlchemy metadata (idempotent)."""
    Base.metadata.create_all(bind=engine)
    inspector = inspect(engine)
    print(f"[OK] tables: {inspector.get_table_names()}")
    with engine.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM analyses")).scalar()
    print(f"[OK] rows in analyses: {n}")


def migrate_legacy_analyses() -> None:
    """Phase 5: rebuild 'analyses' with the users FK, PRESERVING existing rows.

    Legacy schema: user_id VARCHAR(username).  New schema: user_id INT FK.
    Existing usernames become placeholder accounts (their password hash is an
    invalid sentinel, so those accounts can never log in).
    """
    from sqlalchemy import inspect as sa_inspect

    inspector = sa_inspect(engine)
    if "analyses" not in inspector.get_table_names():
        return  # fresh database; create_all will build the final schema
    cols = {c["name"]: c for c in inspector.get_columns("analyses")}
    if "INT" in str(cols.get("user_id", {}).get("type", "")).upper():
        return  # already migrated

    print("[MIGRATION] legacy 'analyses' (VARCHAR user_id) detected - rebuilding with users FK")
    Base.metadata.create_all(bind=engine)  # make sure 'users' exists first

    with engine.begin() as conn:
        legacy_rows = conn.execute(
            text("SELECT id, user_id, timestamp, bristol_type FROM analyses")
        ).fetchall()
        usernames = sorted({r[1] for r in legacy_rows})
        uid_map: dict[str, int] = {}
        for name in usernames:
            existing = conn.execute(
                text("SELECT id FROM users WHERE username = :u"), {"u": name}
            ).first()
            if existing:
                uid_map[name] = existing[0]
            else:
                cur = conn.execute(
                    text(
                        "INSERT INTO users (username, hashed_password) VALUES (:u, :h)"
                    ),
                    {"u": name, "h": "!migrated-no-login"},
                )
                uid_map[name] = int(cur.lastrowid)
        conn.execute(text("DROP TABLE analyses"))

    Base.metadata.create_all(bind=engine)  # rebuild with the FK

    with engine.begin() as conn:
        for row in legacy_rows:
            conn.execute(
                text(
                    "INSERT INTO analyses (id, user_id, timestamp, bristol_type) "
                    "VALUES (:id, :uid, :ts, :bt)"
                ),
                {"id": row[0], "uid": uid_map[row[1]], "ts": row[2], "bt": row[3]},
            )
    print(
        f"[MIGRATION] preserved {len(legacy_rows)} analyses "
        f"across {len(uid_map)} username(s): {', '.join(sorted(uid_map))}"
    )
    print("[MIGRATION] NOTE: placeholder accounts cannot log in - register a fresh account.")


def migrate_users_email() -> None:
    """Add the unique 'email' column (email-based login), backfilling old rows.

    Legacy accounts (created before email login) receive a synthetic
    '<username>@migrated.local' address; those with sentinel hashes
    remain unable to log in, by design.
    """
    from sqlalchemy import inspect as sa_inspect

    inspector = sa_inspect(engine)
    if "users" not in inspector.get_table_names():
        return  # fresh database; create_all builds the final schema
    cols = {c["name"] for c in inspector.get_columns("users")}
    if "email" in cols:
        return  # already migrated

    print("[MIGRATION] adding unique 'email' column to users (backfilling legacy accounts)")
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE users ADD COLUMN email VARCHAR(255) NULL"))
        conn.execute(
            text(
                "UPDATE users SET email = CONCAT(username, '@migrated.local') "
                "WHERE email IS NULL"
            )
        )
        conn.execute(text("ALTER TABLE users MODIFY email VARCHAR(255) NOT NULL"))
        conn.execute(text("CREATE UNIQUE INDEX ix_users_email ON users (email)"))
    print("[MIGRATION] backfilled '<username>@migrated.local' for existing accounts")


if __name__ == "__main__":
    create_database()
    migrate_legacy_analyses()
    migrate_users_email()
    create_tables()
    print("\nDatabase ready.")
