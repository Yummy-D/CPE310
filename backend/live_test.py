"""Live end-to-end test for Phase 1 (self-contained).

Starts its own uvicorn server as a detached subprocess, runs HTTP checks
against it (including the privacy-first image deletion), then shuts it down.

Run:  cd backend && python live_test.py
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

from test_helpers import auth

BASE = "http://127.0.0.1:8001"  # separate port to avoid clashing with dev server
def _make_test_png() -> bytes:
    """A valid PNG that real YOLOv8 inference can decode (64x64 brown square)."""
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (64, 64), (139, 90, 43)).save(buf, format="PNG")
    return buf.getvalue()


PNG_1X1 = _make_test_png()  # legacy name; valid 64x64 PNG (old 1x1 hex had a bad CRC)

BACKEND_DIR = Path(__file__).resolve().parent


def start_server() -> subprocess.Popen:
    kwargs: dict = {}
    if os.name == "nt":
        # Fully detached so closing/ restarting the console doesn't kill it
        # before we're done... we kill it ourselves at the end.
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        )
    log = open(BACKEND_DIR / "live_server.log", "w", encoding="utf-8")
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--port", "8001"],
        cwd=str(BACKEND_DIR),
        stdout=log,
        stderr=subprocess.STDOUT,
        **kwargs,
    )


def wait_healthy(timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            r = httpx.get(f"{BASE}/health", timeout=2)
            if r.status_code == 200:
                print(f"[PASS] live /health -> {r.json()}")
                return
        except httpx.HTTPError as e:
            last_err = e
        time.sleep(0.3)
    raise SystemExit(f"[FAIL] server not healthy within {timeout}s: {last_err}")


def main() -> None:
    proc = start_server()
    try:
        wait_healthy()

        img_path = Path(tempfile.gettempdir()) / "live_stool_test.png"
        img_path.write_bytes(PNG_1X1)
        before = set(os.listdir(tempfile.gettempdir()))

        # Phase 5: /analyze requires a JWT - register (or log in) first.
        client = httpx.Client(base_url=BASE, timeout=30)
        headers, _ = auth(client, "live_e2e")

        with open(img_path, "rb") as f:
            r = client.post(
                "/analyze",
                files={"file": ("live_stool_test.png", f, "image/png")},
                headers=headers,
            )
        assert r.status_code == 200, f"analyze: {r.status_code} {r.text}"
        body = r.json()
        assert 1 <= body["bristol_type"] <= 7, body
        assert body["image_deleted"] is True, body
        print(
            f"[PASS] live /analyze -> type {body['bristol_type']} "
            f"({body['label']}), image_deleted={body['image_deleted']}"
        )
        print(f"       advice: {body['recommendations'][:100]}...")

        after = set(os.listdir(tempfile.gettempdir()))
        new_imgs = {
            n
            for n in (after - before)
            if n.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"))
        }
        assert not new_imgs, f"PRIVACY VIOLATION: {new_imgs}"
        print("[PASS] privacy: no new image files left in temp dir")

        img_path.unlink(missing_ok=True)

        r = client.get("/history", headers=headers, timeout=10)
        assert r.status_code == 200, r.text
        rows = r.json()
        assert rows, "history empty"
        print(f"[PASS] live /history -> {len(rows)} row(s), latest={rows[-1]}")

        print("\nAll live E2E checks passed.")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("[CLEANUP] test server stopped")


if __name__ == "__main__":
    main()
