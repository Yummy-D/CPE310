"""Multi-model (YOLOv8 + MobileNetV3) inference for StoolCheck AI.

Drop-in replacement for ai_mock.analyze_stool_image with the SAME contract:
    analyze_stool_image(image_path: str, model_choice: str = "yolov8") -> {
        "bristol_type": int (1-7), "label": str, "description": str,
        "recommendations": str, "confidence": float, "model": str,
    }

model_choice routing:
- "yolov8"      -> backend/best.pt (YOLOv8n-cls, trained via train_yolo.py).
- "mobilenetv3" -> backend/mobilenetv3_best.pt (MobileNetV3-Small, 7 classes,
                   Resize/CenterCrop 224 + ImageNet normalization). Until the
                   MobileNet checkpoint exists this branch serves a MOCK
                   prediction and reports "mobilenetv3-mock-fallback", so the
                   API keeps working while MobileNet training is pending.

Design notes (unchanged from Phase 3):
- Checkpoints are loaded lazily and only once (thread-safe, one shared lock).
- Heavy frameworks (torch / ultralytics / torchvision) are imported lazily so
  the API server can boot even before `pip install -r requirements.txt` ends.
- The default call shape analyze_stool_image(path) keeps the original YOLOv8
  behaviour byte-for-byte: same fallbacks, same "model" tags.
"""

import logging
import re
import threading
from pathlib import Path

from ai_mock import BRISTOL_SCALE
from ai_mock import analyze_stool_image as _mock_analyze

logger = logging.getLogger("stoolcheck.inference")

# --- YOLOv8 --------------------------------------------------------------
MODEL_PATH = Path(__file__).resolve().parent / "best.pt"  # tests may patch this
IMG_SIZE = 224

# --- MobileNetV3 ---------------------------------------------------------
MOBILENET_PATH = Path(__file__).resolve().parent / "mobilenetv3_best.pt"

MODEL_CHOICES: tuple[str, ...] = ("yolov8", "mobilenetv3")
DEFAULT_MODEL = "yolov8"

_model = None            # YOLOv8 engine (unit tests swap this for a stub)
_mobilenet_model = None  # MobileNetV3 engine (None until trained/loaded)
_lock = threading.Lock()  # shared: serialize model load + predict per request


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


def _load_mobilenet():
    """Build MobileNetV3-Small (7 classes) and load its checkpoint ONCE.

    Raises ModelNotReady while mobilenetv3_best.pt is missing so the caller
    can fall back to the mock engine instead of serving a random-head
    prediction with fake confidence.
    """
    global _mobilenet_model
    if _mobilenet_model is not None:
        return _mobilenet_model
    if not MOBILENET_PATH.exists():
        raise ModelNotReady(
            f"MobileNetV3 weights not found: {MOBILENET_PATH} "
            "(serving a mock prediction until the model is trained)."
        )
    import torch  # deferred heavy import

    from models import NUM_CLASSES, build_mobilenet_v3

    model = build_mobilenet_v3(num_classes=NUM_CLASSES)
    try:
        checkpoint = torch.load(MOBILENET_PATH, map_location="cpu", weights_only=True)
    except Exception:
        # Whole-module / legacy checkpoints need pickle loading (our own file).
        checkpoint = torch.load(MOBILENET_PATH, map_location="cpu", weights_only=False)
    state = (
        checkpoint.get("state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    if not isinstance(state, dict) or not state:
        raise ModelNotReady(f"Unusable MobileNetV3 checkpoint: {MOBILENET_PATH}")
    model.load_state_dict(state)
    model.eval()
    _mobilenet_model = model
    logger.info("Loaded MobileNetV3 model %s (classes=%d)", MOBILENET_PATH, NUM_CLASSES)
    return _mobilenet_model


def _predict_mobilenet(image_path: str, model) -> tuple[int, float]:
    """One MobileNetV3 forward pass -> (class index 0-6, confidence 0-1)."""
    from PIL import Image

    import torch

    from models import mobilenet_preprocess

    with Image.open(image_path) as img:
        batch = mobilenet_preprocess()(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)[0]
    top1 = int(torch.argmax(probs).item())
    return top1, float(probs[top1].item())


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


def _analyze_yolov8(image_path: str) -> dict:
    """YOLOv8 path: the original Phase 3 behaviour (mock fallback if not ready)."""
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


def _analyze_mobilenetv3(image_path: str) -> dict:
    """MobileNetV3 path: real inference when trained, MOCK fallback otherwise."""
    from models import CLASS_NAMES, NUM_CLASSES

    with _lock:
        try:
            model = _load_mobilenet()
        except Exception as exc:  # not trained yet / torchvision missing / bad ckpt
            logger.warning(
                "MobileNetV3 inference unavailable (%s); using MOCK fallback.", exc
            )
            result = _mock_analyze(image_path)
            result["model"] = "mobilenetv3-mock-fallback"
            return result

        try:
            top1, confidence = _predict_mobilenet(image_path, model)
        except Exception as exc:  # unreadable/corrupt image
            raise ValueError(f"The image could not be read: {exc}") from exc
        bristol_type = bristol_from_class(CLASS_NAMES[top1], top1, NUM_CLASSES)

    entry = BRISTOL_SCALE[bristol_type]
    return {
        "bristol_type": bristol_type,
        "label": entry["label"],
        "description": entry["description"],
        "recommendations": entry["recommendation"],
        "confidence": round(confidence, 2),
        "model": "mobilenetv3-cls",
    }


def analyze_stool_image(image_path: str, model_choice: str = DEFAULT_MODEL) -> dict:
    """Run REAL inference on the temp image with the selected engine.

    - model_choice="yolov8" (default): identical to the original single-argument
      contract, so existing callers and tests keep working unchanged.
    - model_choice="mobilenetv3": MobileNetV3-Small, mock fallback until trained.
    - anything else: ValueError (main.py maps it to HTTP 422).
    """
    choice = (model_choice or DEFAULT_MODEL).strip().lower()
    if choice == "mobilenetv3":
        return _analyze_mobilenetv3(image_path)
    if choice != "yolov8":
        raise ValueError(
            f"Unknown model_choice '{model_choice}'. "
            f"Valid options: {', '.join(MODEL_CHOICES)}."
        )
    return _analyze_yolov8(image_path)
