"""Phase 2 validation: frontend served by FastAPI + API integration intact.

Run:  cd backend && python test_phase2.py

Checks:
  1. GET / serves frontend/index.html (static mount does NOT shadow API routes)
  2. Every element ID referenced by the page's JS actually exists in the HTML
  3. External CDNs (Tailwind, Chart.js) are referenced correctly
  4. POST /analyze and GET /history still work with the mount in place
  5. API routes keep precedence (e.g. /history returns JSON, not HTML)
  6. Privacy: temp image still deleted after /analyze
"""

import io
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

from test_helpers import auth

BACKEND_DIR = Path(__file__).resolve().parent
FRONTEND_HTML = BACKEND_DIR.parent / "frontend" / "index.html"
BASE = "http://127.0.0.1:8002"  # separate port for the test server

def _make_test_png() -> bytes:
    """A valid PNG that real YOLOv8 inference can decode (64x64 brown square)."""
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (64, 64), (139, 90, 43)).save(buf, format="PNG")
    return buf.getvalue()


PNG_1X1 = _make_test_png()  # legacy name; valid 64x64 PNG (old 1x1 hex had a bad CRC)


def start_server() -> subprocess.Popen:
    kwargs: dict = {}
    if os.name == "nt":
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        )
    log = open(BACKEND_DIR / "phase2_server.log", "w", encoding="utf-8")
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--port", "8002"],
        cwd=str(BACKEND_DIR),
        stdout=log,
        stderr=subprocess.STDOUT,
        **kwargs,
    )


def wait_healthy(timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    last: Exception | None = None
    while time.time() < deadline:
        try:
            if httpx.get(f"{BASE}/health", timeout=2).status_code == 200:
                return
        except httpx.HTTPError as e:
            last = e
        time.sleep(0.3)
    raise SystemExit(f"[FAIL] server not healthy: {last}")


def test_index_served() -> str:
    r = httpx.get(f"{BASE}/", timeout=10)
    assert r.status_code == 200, f"GET / -> {r.status_code}"
    assert "text/html" in r.headers.get("content-type", ""), r.headers
    html = r.text
    for needle in [
        "StoolCheck AI",
        "cdn.tailwindcss.com",          # Tailwind CDN
        "chart.umd.min.js",             # Chart.js CDN
        "historyChart",                 # canvas for the line chart
        "analyzeBtn",
        "dropZone",
        "resultCard",
        "refreshBtn",
        'apiFetch(`${API_BASE}/analyze`', # frontend calls POST /analyze (auth-aware)
        'fetch(',                        # uses fetch at all
        "/history",                      # frontend calls GET /history
        "Model Evaluation",              # Phase 4 evaluation section
        "authSection",                   # Phase 5 login/register gate
        "Bearer ${t}",                   # JWT attached to protected fetches
    ]:
        assert needle in html, f"missing in served HTML: {needle}"
    print("[PASS] GET / serves index.html with Tailwind, Chart.js and fetch wiring")
    return html


def test_all_js_referenced_ids_exist(html: str) -> None:
    # IDs referenced from JS:  $("someId")  and  getElementById("someId")
    js_ids = set(re.findall(r'\$\("([A-Za-z][\w-]*)"\)', html))
    js_ids |= set(re.findall(r'getElementById\("([\w-]+)"\)', html))
    html_ids = set(re.findall(r'id="([\w-]+)"', html))
    missing = js_ids - html_ids
    assert not missing, f"JS references missing element IDs: {missing}"
    assert len(js_ids) >= 10, f"suspiciously few IDs found: {js_ids}"
    print(f"[PASS] all {len(js_ids)} JS-referenced element IDs exist in the HTML")


def test_api_routes_not_shadowed() -> None:
    # Unauthenticated: JSON 401 proves /history matched the API route
    # (the static mount would answer HTML instead).
    r = httpx.get(f"{BASE}/history", timeout=10)
    assert r.status_code in (200, 401), r.status_code
    assert "application/json" in r.headers.get("content-type", ""), (
        f"/history should be JSON, got {r.headers.get('content-type')}"
    )
    # With a valid JWT: 200 + a JSON list.
    client = httpx.Client(base_url=BASE, timeout=10)
    headers, _ = auth(client, "phase2_test")
    r = client.get("/history", headers=headers)
    assert r.status_code == 200, r.status_code
    assert isinstance(r.json(), list)
    print("[PASS] API routes keep precedence over the static mount (JSON, auth-protected)")


def test_analyze_still_works_and_deletes_image() -> None:
    client = httpx.Client(base_url=BASE, timeout=30)
    headers, _ = auth(client, "phase2_test")
    before = set(os.listdir(tempfile.gettempdir()))
    r = client.post(
        "/analyze",
        files={"file": ("p2.png", io.BytesIO(PNG_1X1), "image/png")},
        headers=headers,
    )
    assert r.status_code == 200, f"{r.status_code} {r.text}"
    body = r.json()
    assert 1 <= body["bristol_type"] <= 7, body
    assert body["image_deleted"] is True, body

    after = set(os.listdir(tempfile.gettempdir()))
    new_imgs = {
        n for n in (after - before)
        if n.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"))
    }
    assert not new_imgs, f"PRIVACY VIOLATION: {new_imgs}"

    hist = client.get("/history", headers=headers).json()
    assert hist, "history should contain the new row"
    assert {"timestamp", "bristol_type"} <= set(hist[-1])
    print(
        f"[PASS] /analyze via integrated server -> type {body['bristol_type']}, "
        f"image deleted, /history has {len(hist)} row(s)"
    )


def test_file_fallback_route() -> None:
    # Direct asset path also works (StaticFiles html=True)
    r = httpx.get(f"{BASE}/index.html", timeout=10)
    assert r.status_code == 200 and "StoolCheck AI" in r.text
    print("[PASS] GET /index.html serves the same page")


if __name__ == "__main__":
    assert FRONTEND_HTML.exists(), f"frontend not found: {FRONTEND_HTML}"
    proc = start_server()
    try:
        wait_healthy()
        html = test_index_served()
        test_all_js_referenced_ids_exist(html)
        test_api_routes_not_shadowed()
        test_analyze_still_works_and_deletes_image()
        test_file_fallback_route()
        print("\nAll Phase 2 tests passed.")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        # Best-effort log cleanup: a lingering child handle must not fail the test.
        try:
            (BACKEND_DIR / "phase2_server.log").unlink(missing_ok=True)
        except OSError:
            pass
        print("[CLEANUP] test server stopped")
