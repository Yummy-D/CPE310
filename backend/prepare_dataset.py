"""Prepare a clean, balanced dataset for StoolCheck AI training.

Merges one or more source datasets, remaps class folders to 'Type N',
removes duplicates (exact SHA-1 + perceptual dHash), and writes a fresh
balanced split:

    dataset/
      train/Type 1 ... Type 7
      val/Type 1   ... Type 7
      test/Type 1  ... Type 7

Run (example):
    cd backend
    python prepare_dataset.py --sources "C:/path/to/new_dataset" --balance
    python train_yolo.py --data "../dataset" --epochs 50

The perceptual dedup step is what prevents train/test leakage when the
"new" dataset secretly shares photos with the old one.
"""

import argparse
import hashlib
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}


def bristol_from_name(name: str) -> int | None:
    """Extract a Bristol type (1-7) from names like 'Type 3', 'type_3', '3'."""
    m = re.search(r"(\d{1,2})", name)
    if m and 1 <= int(m.group(1)) <= 7:
        return int(m.group(1))
    return None


def collect_images(source: Path):
    """Yield (image_path, bristol_type) for every labeled image under source."""
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMG_EXTS:
            continue
        # Label = nearest ancestor folder that looks like 'Type N' (up to 3 levels).
        node = path.parent
        for _ in range(3):
            b = bristol_from_name(node.name)
            if b is not None:
                yield path, b
                break
            if node == source or node.parent == node:
                break
            node = node.parent


def file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(path: Path, size: int = 16) -> int | None:
    """16x16 perceptual difference hash; None if the image cannot be read."""
    try:
        with Image.open(path) as im:
            g = im.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
            pixels = list(g.getdata())
        bits = 0
        for row in range(size):
            for col in range(size):
                left = pixels[row * (size + 1) + col]
                right = pixels[row * (size + 1) + col + 1]
                bits = (bits << 1) | (1 if left > right else 0)
        return bits
    except Exception:
        return None


# PART2
