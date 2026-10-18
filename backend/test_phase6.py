"""Phase 6 tests: multi-model upload (YOLOv8 + MobileNetV3).

Run:  cd backend && python test_phase6.py

Covers:
  1. models.build_mobilenet_v3 -> torchvision MobileNetV3-Small with a 7-class
     head (Type 1-7) + the Resize/CenterCrop/ToTensor/Normalize preprocessing
  2. inference.analyze_stool_image routes by model_choice (yolov8 default,
     mobilenetv3 with graceful MOCK fallback while untrained)
  3. unknown model_choice -> ValueError (the API maps it to HTTP 422)
  4. POST /upload alias + POST /analyze with model_choice -> 200, correct echo,
     and PRIVACY: the temp image is always deleted
  5. frontend ships the model dropdown, loading overlay and comparison chart
"""

import io
import os
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

import inference
import main
import models
from test_helpers import auth

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent


def _png() -> bytes:
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (64, 64), (139, 90, 43)).save(buf, format="PNG")
    return buf.getvalue()


PNG = _png()


def test_mobilenet_architecture() -> None:
    model = models.build_mobilenet_v3(num_classes=models.NUM_CLASSES)
    head = model.classifier[3]
    assert head.out_features == 7, f"expected 7-class head, got {head}"
    assert models.CLASS_NAMES == [f"Type {i}" for i in range(1, 8)]

    steps = [type(t).__name__ for t in models.mobilenet_preprocess().transforms]
    assert steps == ["Resize", "CenterCrop", "ToTensor", "Normalize"], steps
    assert models.MOBILENET_INPUT_SIZE == 224
    print(
        "[PASS] MobileNetV3-Small head -> 7 classes; "
        "preprocessing = Resize224/CenterCrop224/ToTensor/Normalize"
    )


def test_model_routing_and_mock_fallback() -> None:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(PNG)
        tmp = f.name
    try:
        # Default call shape (legacy contract) -> YOLOv8
        r_yolo = inference.analyze_stool_image(tmp)
        assert r_yolo["model"] in {"yolov8n-cls", "mock-fallback"}, r_yolo
        assert 1 <= r_yolo["bristol_type"] <= 7

        # MobileNetV3: real model when trained, MOCK fallback while untrained
        r_mob = inference.analyze_stool_image(tmp, model_choice="mobilenetv3")
        assert r_mob["model"] in {"mobilenetv3-cls", "mobilenetv3-mock-fallback"}, r_mob
        assert 1 <= r_mob["bristol_type"] <= 7, r_mob
        assert r_mob["recommendations"], r_mob
        state = "trained" if models.mobilenet_is_trained() else "mock fallback"
        print(
            f"[PASS] routing: yolov8 -> {r_yolo['model']}, "
            f"mobilenetv3 -> {r_mob['model']} ({state})"
        )

        # Unknown engine must be rejected loudly (API turns this into 422)
        try:
            inference.analyze_stool_image(tmp, model_choice="gpt-9")
        except ValueError as exc:
            assert "model_choice" in str(exc)
        else:
            raise AssertionError("expected ValueError for unknown model_choice")
        print("[PASS] unknown model_choice raises ValueError")
    finally:
        os.unlink(tmp)


def test_upload_alias_and_privacy() -> None:
    client = TestClient(main.app)
    headers, _ = auth(client, "phase6_test")

    before = set(os.listdir(tempfile.gettempdir()))
    r = client.post(
        "/upload",
        files={"file": ("p6.png", io.BytesIO(PNG), "image/png")},
        data={"model_choice": "mobilenetv3"},
        headers=headers,
    )
    assert r.status_code == 200, f"{r.status_code} {r.text}"
    body = r.json()
    assert body["model_choice"] == "mobilenetv3", body
    assert body["model"] in {"mobilenetv3-cls", "mobilenetv3-mock-fallback"}, body
    assert 1 <= body["bristol_type"] <= 7, body
    assert body["image_deleted"] is True, body

    after = set(os.listdir(tempfile.gettempdir()))
    leaked = {
        n for n in (after - before)
        if n.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"))
    }
    assert not leaked, f"PRIVACY VIOLATION: {leaked}"
    print(f"[PASS] POST /upload (mobilenetv3) -> {body['model']}, image deleted")


def test_analyze_model_choice_and_validation() -> None:
    client = TestClient(main.app)
    headers, _ = auth(client, "phase6_test")

    # /analyze honours model_choice too
    r = client.post(
        "/analyze",
        files={"file": ("a.png", io.BytesIO(PNG), "image/png")},
        data={"model_choice": "mobilenetv3"},
        headers=headers,
    )
    assert r.status_code == 200, f"{r.status_code} {r.text}"
    assert r.json()["image_deleted"] is True

    # Legacy call without model_choice keeps working (defaults to yolov8)
    r = client.post(
        "/analyze",
        files={"file": ("b.png", io.BytesIO(PNG), "image/png")},
        headers=headers,
    )
    assert r.status_code == 200, f"{r.status_code} {r.text}"
    assert r.json()["model_choice"] == "yolov8"

    # Invalid engine -> 422 (rejected before the file touches disk)
    r = client.post(
        "/analyze",
        files={"file": ("c.png", io.BytesIO(PNG), "image/png")},
        data={"model_choice": "not-a-model"},
        headers=headers,
    )
    assert r.status_code == 422, f"{r.status_code} {r.text}"
    assert "not-a-model" in r.text
    print("[PASS] /analyze: model_choice routing + legacy default + 422 validation")


def test_frontend_multi_model_ui() -> None:
    import re

    html = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    for needle in [
        'id="modelSelect"',
        "YOLOv8 (Robust)",
        "MobileNetV3 (Lightweight)",
        'id="loadingOverlay"',
        'form.append("model_choice"',
        'id="modelCompareChart"',
        "YOLOv8 vs MobileNetV3",
        "chart.umd.min.js",
    ]:
        assert needle in html, f"missing in index.html: {needle}"

    js_ids = set(re.findall(r'\$\("([A-Za-z][\w-]*)"\)', html))
    html_ids = set(re.findall(r'id="([\w-]+)"', html))
    missing = js_ids - html_ids
    assert not missing, f"JS references missing element IDs: {missing}"
    print(f"[PASS] frontend multi-model UI wired ({len(js_ids)} JS ids all present)")


if __name__ == "__main__":
    test_mobilenet_architecture()
    test_model_routing_and_mock_fallback()
    test_upload_alias_and_privacy()
    test_analyze_model_choice_and_validation()
    test_frontend_multi_model_ui()
    print("\nAll Phase 6 tests passed.")

