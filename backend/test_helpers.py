"""Shared auth helpers for Phase 5+ test suites.

Email-based login (Phase 5b): helpers derive a unique email per username
(<username>@example.com) so existing suites keep their simple signatures.

Works with both fastapi.testclient.TestClient and httpx.Client
(any object with .post(path, json=...) semantics).
"""

PASSWORD = "Passw0rd!123"


def email_for(username: str) -> str:
    """Deterministic unique email derived from a username."""
    return f"{username}@example.com"


def register(client, username: str, password: str = PASSWORD):
    return client.post(
        "/register",
        json={
            "username": username,
            "email": email_for(username),
            "password": password,
        },
    )


def login(client, username: str, password: str = PASSWORD):
    # Email-based login: the address is derived from the username.
    return client.post(
        "/login", json={"email": email_for(username), "password": password}
    )


_resolved: dict[str, str] = {}  # requested name -> working name in this process


def auth(client, username: str, password: str = PASSWORD):
    """Register (or log in) and return (headers, user_dict).

    Accounts that exist but cannot authenticate (migrated placeholders with
    sentinel hashes) get a unique fallback name, which is remembered for the
    rest of the process so every test shares the same resolved account.
    """
    requested = username
    username = _resolved.get(requested, requested)

    r = register(client, username, password)
    if r.status_code == 409:
        r = login(client, username, password)
    if r.status_code not in (200, 201):
        import uuid

        username = f"{requested}_{uuid.uuid4().hex[:8]}"
        r = register(client, username, password)

    assert r.status_code in (200, 201), f"auth failed: {r.status_code} {r.text}"
    _resolved[requested] = username
    data = r.json()
    return {"Authorization": f"Bearer {data['access_token']}"}, data["user"]
