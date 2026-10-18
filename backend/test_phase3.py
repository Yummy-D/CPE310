"""Phase 3 tests: real YOLOv8 inference wiring + Windows-safe static serving.

Run:  cd backend && python test_phase3.py

Covers:
  1. Class-name -> Bristol type mapping (real model path)
  2. Stubbed YOLO model produces the full analysis contract (no GPU needed)
  3. Graceful mock fallback when best.pt / ultralytics is not available
  4. /analyze endpoint still deletes the image (privacy regression)
  5. THE USER'S BUG: malformed request path (GET /)http://127.0.0.1:8000)
     must return 400/404 and NOT crash the server with OSError/WinError 123
"""

import io
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

import inference
import main
from fastapi.testclient import TestClient
from test_helpers import auth

BACKEND_DIR = Path(__file__).resolve().parent

def _make_test_png() -> bytes:
    """A valid PNG that real YOLOv8 inference can decode (64x64 brown square)."""
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (64, 64), (139, 90, 43)).save(buf, format="PNG")
    return buf.getvalue()


PNG_1X1 = _make_test_png()  # legacy name; valid 64x64 PNG (old 1x1 hex had a bad CRC)


def test_class_name_mapping() -> None:
    assert inference.bristol_from_class("Type 4", 3, 7) == 4
    assert inference.bristol_from_class("Type 1", 0, 7) == 1
    assert inference.bristol_from_class("7", 6, 7) == 7
    # No digits in the name -> fall back to index (7-class dataset)
    assert inference.bristol_from_class("watery", 6, 7) == 7
    try:
        inference.bristol_from_class("watery", 0, 3)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unmappable class")
    print("[PASS] class-name -> Bristol type mapping")


class _StubProbs:
    def __init__(self, top1: int, conf: float) -> None:
        self.top1 = top1
        self.top1conf = conf


class _StubResult:
    def __init__(self, top1: int, conf: float, names: dict) -> None:
        self.probs = _StubProbs(top1, conf)
        self.names = names


class _StubModel:
    """Mimics ultralytics YOLO.predict() output without needing weights."""

    def __init__(self, result: _StubResult) -> None:
        self._result = result

    def predict(self, **kwargs) -> list:
        assert Path(kwargs["source"]).exists(), "predict must receive the temp image"
        return [self._result]


def test_stubbed_real_inference() -> None:
    names = {i: f"Type {i + 1}" for i in range(7)}
    saved = inference._model
    saved_path = inference.MODEL_PATH
    try:
        inference._model = _StubModel(_StubResult(top1=2, conf=0.91, names=names))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(PNG_1X1)
            tmp = f.name
        try:
            result = inference.analyze_stool_image(tmp)
        finally:
            os.unlink(tmp)
    finally:
        inference._model = saved
        inference.MODEL_PATH = saved_path

    assert result["bristol_type"] == 3, result
    assert result["model"] == "yolov8n-cls", result
    assert result["confidence"] == 0.91, result
    assert result["label"].startswith("Type 3"), result
    assert result["recommendations"], result
    print("[PASS] stubbed YOLO model -> full contract (type 3, conf 0.91)")


def test_mock_fallback_when_model_missing() -> None:
    saved_path = inference.MODEL_PATH
    saved_model = inference._model
    try:
        inference.MODEL_PATH = Path("Z:/definitely/not/there/best.pt")
        inference._model = None
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(PNG_1X1)
            tmp = f.name
        try:
            result = inference.analyze_stool_image(tmp)
        finally:
            os.unlink(tmp)
    finally:
        inference.MODEL_PATH = saved_path
        inference._model = saved_model

    assert result["model"] == "mock-fallback", result
    assert 1 <= result["bristol_type"] <= 7, result
    assert result["recommendations"], result
    print("[PASS] missing best.pt -> graceful mock-fallback with same contract")


def test_endpoint_privacy_regression() -> None:
    client = TestClient(main.app)
    seen: list[str] = []
    original = main.analyze_stool_image

    def spy(path: str) -> dict:
        seen.append(path)
        assert os.path.exists(path), "temp image must exist during inference"
        return original(path)

    main.analyze_stool_image = spy  # type: ignore[assignment]
    try:
        headers, _ = auth(client, "phase3_test")
        r = client.post(
            "/analyze",
            files={"file": ("p3.png", io.BytesIO(PNG_1X1), "image/png")},
            headers=headers,
        )
    finally:
        main.analyze_stool_image = original  # type: ignore[assignment]

    assert r.status_code == 200, f"{r.status_code} {r.text}"
    body = r.json()
    assert body["image_deleted"] is True, body
    assert 1 <= body["bristol_type"] <= 7, body
    assert body["model"] in {"yolov8n-cls", "mock-fallback", "mock-v1"}, body
    assert seen and not os.path.exists(seen[0]), "PRIVACY VIOLATION: image kept"
    print(
        f"[PASS] /analyze via Phase 3 pipeline -> type {body['bristol_type']}, "
        f"model={body['model']}, image deleted"
    )


def _spawn_server(port: int) -> subprocess.Popen:
    kwargs: dict = {}
    if os.name == "nt":
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        )
    log = open(BACKEND_DIR / "p3_server.log", "w", encoding="utf-8")
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--port", str(port)],
        cwd=str(BACKEND_DIR),
        stdout=log,
        stderr=subprocess.STDOUT,
        **kwargs,
    )


def _raw_get(port: int, target: bytes) -> tuple[int, str]:
    """Send a raw HTTP request line (bypasses client-side URL normalization)."""
    with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
        s.sendall(
            b"GET " + target + b" HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"
        )
        chunks = []
        while True:
            data = s.recv(4096)
            if not data:
                break
            chunks.append(data)
    raw = b"".join(chunks)
    status_line = raw.split(b"\r\n", 1)[0].decode("latin-1")
    return int(status_line.split()[1]), status_line


def test_malformed_static_path_no_crash() -> None:
    """The reported WinError 123 scenario: must 4xx cleanly, server stays up."""
    port = 8004
    proc = _spawn_server(port)
    try:
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                if httpx.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.3)
        else:
            raise AssertionError("server did not start")

        # Exactly the user's failing request shape
        status, line = _raw_get(port, b"/)http://127.0.0.1:8000")
        assert status in (400, 404), f"expected 400/404, got: {line}"
        print(f"[PASS] malformed path ')http://127.0.0.1:8000' -> {status} (no crash)")

        # A few more hostile shapes
        for target in (b"/foo:bar", b"/..%5C..%5Cwindows", b"/%29http://x"):
            status, line = _raw_get(port, target)
            assert status in (200, 400, 404), f"{target!r} -> {line}"
        print("[PASS] other malformed paths handled without 500")

        # Server still fully alive afterwards
        r = httpx.get(f"http://127.0.0.1:{port}/", timeout=10)
        assert r.status_code == 200 and "StoolCheck AI" in r.text
        r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=5)
        assert r.status_code == 200
        print("[PASS] server still serves / and /health after hostile requests")

        log = (BACKEND_DIR / "p3_server.log").read_text(encoding="utf-8", errors="replace")
        assert "WinError 123" not in log, "OSError still leaking into server log"
        assert "Traceback" not in log, f"unexpected traceback in server log:\n{log[-1500:]}"
        print("[PASS] server log clean: no WinError 123, no traceback")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        # Best-effort log cleanup: a lingering child handle must not fail the test.
        try:
            (BACKEND_DIR / "p3_server.log").unlink(missing_ok=True)
        except OSError:
            pass


if __name__ == "__main__":
    test_class_name_mapping()
    test_stubbed_real_inference()
    test_mock_fallback_when_model_missing()
    test_endpoint_privacy_regression()
    test_malformed_static_path_no_crash()
    print("\nAll Phase 3 tests passed.")
