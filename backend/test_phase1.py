"""Phase 1 validation tests for StoolCheck AI backend.

Run:  cd backend && python test_phase1.py

Verifies:
  1. /health responds
  2. /analyze runs mock AI, saves ONLY (user_id, timestamp, bristol_type)
  3. PRIVACY: the temporary image file is deleted after the request
     (even on the happy path)
  4. Unsupported file types are rejected
  5. /history returns the saved rows for the dashboard
"""

import io
import os
import tempfile

from fastapi.testclient import TestClient

import main
from database import SessionLocal
from models import Analysis
from test_helpers import auth

client = TestClient(main.app)

def _make_test_png() -> bytes:
    """A valid PNG that real YOLOv8 inference can decode (64x64 brown square)."""
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (64, 64), (139, 90, 43)).save(buf, format="PNG")
    return buf.getvalue()


PNG_1X1 = _make_test_png()  # legacy name; valid 64x64 PNG (old 1x1 hex had a bad CRC)


def tmp_image_files() -> set[str]:
    """Existing temp files that look like our uploads."""
    d = tempfile.gettempdir()
    return {
        os.path.join(d, f)
        for f in os.listdir(d)
        if f.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"))
    }


def test_health() -> None:
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    print("[PASS] /health")


def test_analyze_privacy_first() -> None:
    seen_paths: list[str] = []
    original = main.analyze_stool_image

    def spy(path: str) -> dict:
        seen_paths.append(path)
        assert os.path.exists(path), "temp image should exist DURING inference"
        return original(path)

    headers, user = auth(client, "test_user")
    main.analyze_stool_image = spy  # type: ignore[assignment]
    before = tmp_image_files()
    try:
        r = client.post(
            "/analyze",
            files={"file": ("stool.png", io.BytesIO(PNG_1X1), "image/png")},
            headers=headers,
        )
    finally:
        main.analyze_stool_image = original  # type: ignore[assignment]

    assert r.status_code == 200, f"analyze failed: {r.status_code} {r.text}"
    body = r.json()

    # Identity comes from the JWT (Phase 5), not from any client payload
    assert body["user_id"] == user["id"] and body["username"] == user["username"], body

    # Mock AI contract
    assert 1 <= body["bristol_type"] <= 7, body
    assert body["label"].startswith("Type")
    assert body["recommendations"]
    assert body["image_deleted"] is True, "PRIVACY: response says image retained!"

    # Temp file must be gone from disk
    assert seen_paths, "spy never saw an image path"
    leftover = [p for p in seen_paths if os.path.exists(p)]
    assert not leftover, f"PRIVACY VIOLATION: images still on disk: {leftover}"
    after = tmp_image_files()
    new_files = after - before
    assert not new_files, f"PRIVACY VIOLATION: new temp images remain: {new_files}"

    # DB: ONLY user_id, timestamp, bristol_type (plus PK id)
    db = SessionLocal()
    try:
        cols = {c.name for c in Analysis.__table__.columns}
        assert cols == {"id", "user_id", "timestamp", "bristol_type"}, cols
        # Idempotent: fetch the row this run just created (latest for this user).
        row = (
            db.query(Analysis)
            .filter(Analysis.user_id == user["id"])
            .order_by(Analysis.id.desc())
            .first()
        )
        assert row is not None, "no DB row written"
        assert row.bristol_type == body["bristol_type"]
        assert row.timestamp is not None
    finally:
        db.close()

    print(
        f"[PASS] /analyze -> type {body['bristol_type']}, "
        "temp image deleted, DB row saved (user_id, timestamp, bristol_type only)"
    )


def test_reject_non_image() -> None:
    headers, _ = auth(client, "test_user")
    r = client.post(
        "/analyze",
        files={"file": ("evil.exe", io.BytesIO(b"MZ\x90\x00"), "application/octet-stream")},
        headers=headers,
    )
    assert r.status_code == 415, f"expected 415, got {r.status_code}"
    print("[PASS] /analyze rejects unsupported media type (415)")


def test_history() -> None:
    headers, _ = auth(client, "test_user")
    r = client.get("/history", headers=headers)
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) >= 1, "expected at least one history row"
    assert all({"timestamp", "bristol_type"} <= set(row) for row in rows)
    print(f"[PASS] /history -> {len(rows)} row(s) for dashboard")


if __name__ == "__main__":
    test_health()
    test_analyze_privacy_first()
    test_reject_non_image()
    test_history()
    print("\nAll Phase 1 tests passed.")
