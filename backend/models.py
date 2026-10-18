"""ORM models for StoolCheck AI.

PRIVACY: only non-sensitive metadata is persisted.
No image paths, no image bytes, no health content beyond the Bristol type.
Phase 5: user accounts + analyses linked to users via a real Foreign Key.
"""

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from database import Base


def _utcnow() -> datetime:
    """Timezone-naive UTC timestamp (SQLite stores no tz info)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class User(Base):
    """A registered account (Phase 5).

    email  = unique login identifier (used for authentication)
    username = unique display name shown in the UI
    password is stored bcrypt-hashed.
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(
        String(32), unique=True, index=True, nullable=False
    )
    email: Mapped[str] = mapped_column(
        String(255), unique=True, index=True, nullable=False
    )
    hashed_password: Mapped[str] = mapped_column(String(128), nullable=False)


class Analysis(Base):
    """One stool-image analysis result, owned by a user."""

    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), index=True, nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=_utcnow
    )
    bristol_type: Mapped[int] = mapped_column(Integer, nullable=False)


# =============================================================================
# Phase 6 - Multi-model AI architecture (MobileNetV3 alongside YOLOv8)
# -----------------------------------------------------------------------------
# NOTE: this module is imported at server boot for the ORM classes above, so
# torch / torchvision are imported LAZILY inside the builders below. MySQL
# connectivity and JWT auth therefore keep working even if the AI stack is
# mid-install, and FastAPI boot stays fast.
# =============================================================================

# Bristol Stool Scale - 7 classes (Type 1 - Type 7), in the same order as the
# dataset class folders (ImageFolder sorts them alphabetically).
CLASS_NAMES: list[str] = [f"Type {i}" for i in range(1, 8)]
NUM_CLASSES: int = 7

# MobileNetV3 preprocessing contract: Resize 224 -> CenterCrop 224 ->
# ToTensor -> Normalize (ImageNet statistics).
MOBILENET_INPUT_SIZE: int = 224
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]

# Trained checkpoint written by the MobileNet training pipeline. While this
# file is missing, inference.py serves a MOCK prediction for the
# "mobilenetv3" model_choice instead of a fake-confident real one.
MOBILENET_WEIGHTS_PATH = Path(__file__).resolve().parent / "mobilenetv3_best.pt"


def build_mobilenet_v3(num_classes: int = NUM_CLASSES, pretrained: bool = False):
    """MobileNetV3-Small with its classifier head replaced for Type 1-7.

    torchvision.models.mobilenet_v3_small's classifier is
    Sequential(Linear, Hardswish, Dropout, Linear) - index 3 is the final
    classification layer, so we swap it for a `num_classes`-way head.
    """
    import torch.nn as nn
    from torchvision.models import mobilenet_v3_small

    weights = None
    if pretrained:
        try:
            from torchvision.models import MobileNet_V3_Small_Weights

            weights = MobileNet_V3_Small_Weights.DEFAULT
        except Exception:  # offline / older torchvision -> random init
            weights = None

    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def mobilenet_preprocess():
    """Inference preprocessing: Resize 224, CenterCrop 224, ToTensor, Normalize."""
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((MOBILENET_INPUT_SIZE, MOBILENET_INPUT_SIZE)),
            transforms.CenterCrop(MOBILENET_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def mobilenet_is_trained() -> bool:
    """True once mobilenetv3_best.pt exists (otherwise inference uses the mock)."""
    return MOBILENET_WEIGHTS_PATH.is_file()
