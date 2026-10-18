"""One-shot final smoke: start the REAL server on port 8000 (the user's URL),
verify page / evaluation / malformed-path handling, then stop it.

Run:  cd backend && python final_smoke.py
"""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

BACKEND = Path(__file__).resolve().parent
PORT = 8000
log_path = BACKEND / "smoke_server.log"


def raw_status(target: bytes) -> int:
    with socket.create_connection(("127.0.0.1", PORT), timeout=5) as s:
        s.sendall(
            b"GET " + target + b" HTTP/1.1\r\nHost: h\r\nConnection: close\r\n\r\n"
        )
        buf = b""
        while True:
            d = s.recv(4096)
            if not d:
                break
            buf += d
    return int(buf.split(b" ")[1])


def main() -> None:
    kwargs: dict = {}
    if os.name == "nt":
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        )
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--port", str(PORT)],
        cwd=str(BACKEND),
        stdout=log,
        stderr=subprocess.STDOUT,
        **kwargs,
    )
    try:
        for _ in range(50):
            try:
                if httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.3)
        else:
            raise SystemExit("[FAIL] server did not start on :8000")

        r = httpx.get(f"http://127.0.0.1:{PORT}/")
        assert r.status_code == 200, r.status_code
        assert "StoolCheck AI" in r.text and "Model Evaluation" in r.text
        print("[PASS] GET http://127.0.0.1:8000/ -> 200 (page + Model Evaluation section)")

        r = httpx.get(f"http://127.0.0.1:{PORT}/evaluation")
        data = r.json()
        assert data["available"] and data["top1"] is not None, data
        print(f"[PASS] GET /evaluation -> top1={data['top1']}, top5={data['top5']}, epochs={data['epochs']}")

        # The two historical offenders, including the user's exact logged request
        for target in (
            b"/%5D%28http%3A//127.0.0.1%3A8000/",
            b"/)http://127.0.0.1:8000",
        ):
            st = raw_status(target)
            assert st in (400, 404), (target, st)
            print(f"[PASS] malformed {target.decode('latin-1')!r} -> {st} (no crash)")

        log_txt = log_path.read_text(encoding="utf-8", errors="replace")
        assert "WinError 123" not in log_txt, "OSError leaked!"
        assert "Traceback" not in log_txt, log_txt[-1000:]
        print("[PASS] server log clean on :8000 (no WinError 123, no traceback)")
        print("\nFINAL SMOKE PASSED - http://127.0.0.1:8000/ is ready.")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        try:
            log_path.unlink(missing_ok=True)
        except OSError:
            pass
        print("[CLEANUP] smoke server stopped")


if __name__ == "__main__":
    main()
