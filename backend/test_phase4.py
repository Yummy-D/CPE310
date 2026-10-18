"""Phase 4 tests: MySQL migration + Model Evaluation section.

Run:  cd backend && python test_phase4.py

Covers:
  1. engine.url points at XAMPP MySQL (root@127.0.0.1:3306/stoolcheck_db)
  2. init_db is idempotent (re-run safe)
  3. GET /evaluation returns real YOLOv8 training metrics from results.csv
  4. /analyze with a REAL dataset image runs REAL YOLOv8-cls (best.pt)
     and still deletes the image (privacy regression)
  5. Frontend Model Evaluation section exists and its JS ids are wired
"""

import os
import re
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

import database
import evaluation
import init_db
import main
from test_helpers import auth

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

_CT_BY_EXT = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".webp": "image/webp", ".bmp": "image/bmp", ".gif": "image/gif",
}


def test_db_is_mysql() -> None:
    url = database.engine.url
    assert url.drivername == "mysql+pymysql", url
    assert url.host == "127.0.0.1" and url.port == 3306, url
    assert url.username == "root" and (url.password or "") == "", url
    assert url.database == "stoolcheck_db", url
    print(f"[PASS] engine -> {url.render_as_string(hide_password=True)}")


def test_init_db_idempotent() -> None:
    init_db.create_database()
    init_db.create_tables()
    with database.engine.connect() as conn:
        from sqlalchemy import text
        n = conn.execute(text("SELECT COUNT(*) FROM analyses")).scalar()
    assert isinstance(n, int)
    print(f"[PASS] init_db re-run is idempotent (analyses rows: {n})")


def test_evaluation_endpoint() -> None:
    client = TestClient(main.app)
    r = client.get("/evaluation")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["available"] is True, data
    assert data["model"] == "yolov8n-cls", data
    assert data["epochs"] >= 1, data
    assert data["top1"] is not None and 0 <= data["top1"] <= 1, data
    assert data["top5"] is not None and 0 <= data["top5"] <= 1, data
    print(
        f"[PASS] /evaluation -> top1={data['top1']}, top5={data['top5']}, "
        f"epochs={data['epochs']}, val_loss={data['val_loss']}, run={data['run']}"
    )


def test_real_model_analyze_with_privacy() -> None:
    client = TestClient(main.app)
    images = [p for p in sorted((PROJECT_ROOT / "test").glob("Type */*")) if p.is_file()]
    assert images, "no images found in test/ split"
    img = images[0]
    ct = _CT_BY_EXT.get(img.suffix.lower(), "image/jpeg")

    seen: list[str] = []
    original = main.analyze_stool_image

    def spy(path: str) -> dict:
        seen.append(path)
        return original(path)

    main.analyze_stool_image = spy  # type: ignore[assignment]
    try:
        headers, _ = auth(client, "phase4_test")
        with open(img, "rb") as f:
            r = client.post(
                "/analyze",
                files={"file": (img.name, f, ct)},
                headers=headers,
            )
    finally:
        main.analyze_stool_image = original  # type: ignore[assignment]

    assert r.status_code == 200, f"{r.status_code} {r.text}"
    body = r.json()
    assert body["model"] == "yolov8n-cls", f"expected REAL model, got: {body}"
    assert 1 <= body["bristol_type"] <= 7, body
    assert body["image_deleted"] is True, body
    assert seen and not os.path.exists(seen[0]), "PRIVACY VIOLATION: image kept"
    print(
        f"[PASS] REAL YOLOv8 inference on {img.name} -> type {body['bristol_type']} "
        f"(conf {body['confidence']}), image deleted"
    )


def test_frontend_eval_section() -> None:
    html = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    for needle in [
        "Model Evaluation",
        'id="evalBadge"',
        'id="evalTop1"',
        "loadEvaluation",
        "`${API_BASE}/evaluation`",
    ]:
        assert needle in html, f"missing in index.html: {needle}"

    js_ids = set(re.findall(r'\$\("([A-Za-z][\w-]*)"\)', html))
    html_ids = set(re.findall(r'id="([\w-]+)"', html))
    missing = js_ids - html_ids
    assert not missing, f"JS references missing element IDs: {missing}"
    print(f"[PASS] frontend Model Evaluation section wired ({len(js_ids)} JS ids all present)")


if __name__ == "__main__":
    test_db_is_mysql()
    test_init_db_idempotent()
    test_evaluation_endpoint()
    test_real_model_analyze_with_privacy()
    test_frontend_eval_section()
    print("\nAll Phase 4 tests passed.")
