"""Phase 5 tests: JWT authentication + protected endpoints + schema.

Run:  cd backend && python test_phase5.py

Covers:
  1. /register (201, duplicate 409, validation 422) and /login (401s)
  2. /analyze and /history require a valid JWT (401 + WWW-Authenticate)
  3. user_id comes from the TOKEN - a forged user_id payload is ignored
  4. Cross-user isolation: each user only sees their own history
  5. Schema: users table + analyses.user_id is an INT Foreign Key -> users.id
  6. Privacy-first image deletion still holds under auth
  7. Frontend: auth gate present, no legacy #userId input, Bearer wiring
"""

import io
import os
import re
from pathlib import Path

import jwt as pyjwt
from fastapi.testclient import TestClient

import main
from auth import SECRET_KEY
from test_helpers import PASSWORD, auth, email_for, login, register

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PNG = None


def _png() -> bytes:
    from io import BytesIO

    from PIL import Image

    global PNG
    if PNG is None:
        buf = BytesIO()
        Image.new("RGB", (64, 64), (139, 90, 43)).save(buf, format="PNG")
        PNG = buf.getvalue()
    return PNG


client = TestClient(main.app)


def test_register_and_login() -> None:
    # Fresh unique user so the test is repeatable
    import time

    name = f"p5_{int(time.time() % 100000)}"
    r = register(client, name)
    assert r.status_code == 201, f"{r.status_code} {r.text}"
    data = r.json()
    assert data["token_type"] == "bearer" and data["access_token"]
    assert data["user"]["username"] == name and data["user"]["id"] > 0
    assert data["user"]["email"] == email_for(name)

    # JWT claims: sub=email (login id), uid=user id
    claims = pyjwt.decode(
        data["access_token"], options={"verify_signature": False}, algorithms=["HS256"]
    )
    assert claims["sub"] == email_for(name) and claims["uid"] == data["user"]["id"]
    assert claims["exp"] > claims["iat"]

    # Duplicate email/username -> 409
    r = register(client, name)
    assert r.status_code == 409, r.status_code

    # Login with correct email + password -> 200 + token
    r = login(client, name)
    assert r.status_code == 200 and r.json()["access_token"]

    # Wrong password and unknown email -> identical 401 (no enumeration)
    r = client.post("/login", json={"email": email_for(name), "password": "wrong-pass"})
    assert r.status_code == 401, r.status_code
    r2 = client.post(
        "/login",
        json={"email": "definitely_missing_user@example.com", "password": "wrong-pass"},
    )
    assert r2.status_code == 401
    assert r.json() == r2.json(), "login errors must be identical (no user enumeration)"
    assert r.headers.get("www-authenticate") == "Bearer"

    # Validation: bad username / bad email / short password -> 422
    assert client.post(
        "/register",
        json={"username": "ab", "email": "ok@example.com", "password": "longenough"},
    ).status_code == 422
    assert client.post(
        "/register",
        json={"username": "valid_name", "email": "not-an-email", "password": "longenough"},
    ).status_code == 422
    assert client.post(
        "/register",
        json={"username": "valid_name", "email": "ok@example.com", "password": "12345"},
    ).status_code == 422
    print(
        f"[PASS] register(email+username)/login/JWT sub=email/409/401/422 "
        f"(user {name}, id {data['user']['id']})"
    )


def test_protected_endpoints_require_jwt() -> None:
    # No token at all
    r = client.get("/history")
    assert r.status_code == 401, r.status_code
    assert r.headers.get("www-authenticate") == "Bearer"
    r = client.post(
        "/analyze",
        files={"file": ("x.png", io.BytesIO(_png()), "image/png")},
    )
    assert r.status_code == 401, r.status_code
    assert r.headers.get("www-authenticate") == "Bearer"

    # Garbage / tampered tokens
    for bad in ("not-a-jwt", "a.b.c", "Bearer" ):
        r = client.get("/history", headers={"Authorization": bad})
        assert r.status_code == 401, (bad, r.status_code)
    # Valid token signed with the WRONG key
    forged = __import__("jwt").encode(
        {"sub": "p5_alice", "uid": 1, "exp": 9999999999}, "wrong-secret", algorithm="HS256"
    )
    r = client.get("/history", headers={"Authorization": f"Bearer {forged}"})
    assert r.status_code == 401, r.status_code
    print("[PASS] /analyze + /history reject missing/garbage/forged tokens (401 + WWW-Authenticate)")


def test_user_id_from_token_and_isolation() -> None:
    headers_a, user_a = auth(client, "p5_alice")
    headers_b, user_b = auth(client, "p5_bob")
    assert user_a["id"] != user_b["id"]

    def history(h) -> list:
        r = client.get("/history", headers=h)
        assert r.status_code == 200, r.text
        return r.json()

    a_before = history(headers_a)
    b_before = history(headers_b)

    # Alice analyzes BUT tries to forge Bob's identity via a form field.
    # The server must ignore it and use the JWT's uid.
    r = client.post(
        "/analyze",
        files={"file": ("forged.png", io.BytesIO(_png()), "image/png")},
        data={"user_id": str(user_b["id"])},  # hostile payload - must be ignored
        headers=headers_a,
    )
    assert r.status_code == 200, f"{r.status_code} {r.text}"
    body = r.json()
    assert body["user_id"] == user_a["id"], "PRIVACY/AUTH BUG: forged user_id accepted!"
    assert body["username"] == user_a["username"], "display name must match the token's user"
    assert body["image_deleted"] is True, "PRIVACY: image retained"

    a_after = history(headers_a)
    b_after = history(headers_b)
    assert len(a_after) == len(a_before) + 1, "Alice's history must grow"
    assert len(b_after) == len(b_before), "Bob's history must NOT change"
    print(
        f"[PASS] user_id from JWT (forged field ignored) + isolation "
        f"(alice {len(a_after)} rows, bob {len(b_after)} rows) + image deleted"
    )


def test_schema_user_fk() -> None:
    from sqlalchemy import inspect

    from database import SessionLocal, engine
    from models import User

    inspector = inspect(engine)
    tables = inspector.get_table_names()
    assert "users" in tables and "analyses" in tables, tables

    # analyses: exactly the 4 columns, user_id is INT
    cols = {c["name"]: c for c in inspector.get_columns("analyses")}
    assert set(cols) == {"id", "user_id", "timestamp", "bristol_type"}, set(cols)
    assert "INT" in str(cols["user_id"]["type"]).upper(), cols["user_id"]["type"]

    # ...and it is a real Foreign Key -> users.id
    fks = inspector.get_foreign_keys("analyses")
    assert any(
        fk["constrained_columns"] == ["user_id"]
        and fk["referred_table"] == "users"
        and fk["referred_columns"] == ["id"]
        for fk in fks
    ), fks

    user_cols = {c["name"] for c in inspector.get_columns("users")}
    assert user_cols == {"id", "username", "email", "hashed_password"}, user_cols
    unique_entries = list(inspector.get_unique_constraints("users"))
    unique_entries += [
        {"column_names": idx.get("column_names", [])}
        for idx in inspector.get_indexes("users")
        if idx.get("unique")
    ]
    assert any(set(e["column_names"]) == {"username"} for e in unique_entries), (
        unique_entries
    )
    assert any(set(e["column_names"]) == {"email"} for e in unique_entries), (
        unique_entries
    )

    # Password must be stored bcrypt-hashed, never in plaintext
    db = SessionLocal()
    try:
        u = db.query(User).filter(User.username == "p5_alice").first()
        assert u is not None, "p5_alice should exist from the isolation test"
        assert u.hashed_password.startswith("$2"), "password not bcrypt-hashed!"
        assert u.hashed_password != PASSWORD
    finally:
        db.close()
    print(f"[PASS] schema: users{sorted(user_cols)}; analyses.user_id INT FK -> users.id")


def test_frontend_auth_gate() -> None:
    html = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    for needle in (
        'id="authSection"',
        'id="appContent" class="hidden"',
        'id="tabLogin"',
        'id="tabRegister"',
        'id="logoutBtn"',
        "TOKEN_KEY",
        "Bearer ${t}",
        "Authorization",
        'authMode === "login" ? "login" : "register"',
        "localStorage.setItem(TOKEN_KEY",
        'id="authEmail"',
        'id="usernameField"',
        "{ email, password }",
        "{ email, username, password }",
    ):
        assert needle in html, f"missing in index.html: {needle}"

    # Legacy per-request User ID must be fully gone
    assert 'id="userId"' not in html, "legacy #userId input still present"
    assert "userIdInput" not in html, "legacy userIdInput reference still present"
    assert 'form.append("user_id"' not in html, "user_id still sent in FormData!"

    js_ids = set(re.findall(r'\$\("([A-Za-z][\w-]*)"\)', html))
    html_ids = set(re.findall(r'id="([\w-]+)"', html))
    missing = js_ids - html_ids
    assert not missing, f"JS references missing element IDs: {missing}"
    print(f"[PASS] frontend auth gate wired ({len(js_ids)} JS ids present, no legacy userId)")


if __name__ == "__main__":
    test_register_and_login()
    test_protected_endpoints_require_jwt()
    test_user_id_from_token_and_isolation()
    test_schema_user_fk()
    test_frontend_auth_gate()
    print("\nAll Phase 5 tests passed.")
