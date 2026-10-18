"""Real YOLOv8n-cls inference for StoolCheck AI (Phase 3).

Drop-in replacement for ai_mock.analyze_stool_image with the SAME contract:
    analyze_stool_image(image_path: str) -> {
        "bristol_type": int (1-7), "label": str, "description": str,
        "recommendations": str, "confidence": float, "model": str,
    }

Design notes:
- backend/best.pt is loaded lazily and only once (thread-safe).
- `ultralytics` is imported lazily so the API server can boot even before
  `pip install ultralytics` finishes.
- If best.pt or ultralytics is not available yet, we fall back to the mock
  engine and report it in the `model` field ("mock-fallback") so the API
  keeps working while training is pending.
"""

import logging
import re
import threading
from pathlib import Path

from ai_mock import BRISTOL_SCALE
from ai_mock import analyze_stool_image as _mock_analyze

logger = logging.getLogger("stoolcheck.inference")

MODEL_PATH = Path(__file__).resolve().parent / "best.pt"
IMG_SIZE = 224

_model = None
_lock = threading.Lock()


class ModelNotReady(RuntimeError):
    """Raised when best.pt or ultralytics is not available yet."""


def _load_model():
    """Load (once) and return the YOLO model. Raises ModelNotReady/ImportError."""
    global _model
    if _model is not None:
        return _model
    if not MODEL_PATH.exists():
        raise ModelNotReady(
            f"Trained model not found: {MODEL_PATH} (run train_yolo.py first)"
        )
    from ultralytics import YOLO  # deferred heavy import

    _model = YOLO(str(MODEL_PATH))
    logger.info("Loaded YOLOv8-cls model %s (classes=%s)", MODEL_PATH, _model.names)
    return _model


def bristol_from_class(class_name: str, index: int, num_classes: int) -> int:
    """Map a dataset class name (e.g. 'Type 3') or index to a Bristol type 1-7."""
    m = re.search(r"(\d+)", str(class_name))
    if m:
        t = int(m.group(1))
        if 1 <= t <= 7:
            return t
    if num_classes == 7:
        return index + 1
    raise ValueError(
        f"Cannot map class '{class_name}' (index {index}) to a Bristol type"
    )


def analyze_stool_image(image_path: str) -> dict:
    """Run REAL YOLOv8-cls inference on the temp image (mock fallback if not ready)."""
    with _lock:  # serialize model load + predict (server may call concurrently)
        try:
            model = _load_model()
        except Exception as exc:  # ModelNotReady, missing ultralytics, bad file...
            logger.warning("YOLO inference unavailable (%s); using MOCK fallback.", exc)
            result = _mock_analyze(image_path)
            result["model"] = "mock-fallback"
            return result

        results = model.predict(source=image_path, imgsz=IMG_SIZE, verbose=False)
        if not results:
            raise ValueError(
                "The image could not be read (corrupt or unsupported format)."
            )
        res = results[0]
        if getattr(res, "probs", None) is None:
            raise ValueError("Model returned no classification probabilities.")
        top1 = int(res.probs.top1)
        confidence = float(res.probs.top1conf)
        class_name = res.names[top1]
        bristol_type = bristol_from_class(class_name, top1, len(res.names))

    entry = BRISTOL_SCALE[bristol_type]
    return {
        "bristol_type": bristol_type,
        "label": entry["label"],
        "description": entry["description"],
        "recommendations": entry["recommendation"],
        "confidence": round(confidence, 2),
        "model": "yolov8n-cls",
    }
