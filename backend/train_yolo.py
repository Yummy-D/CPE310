"""Train a YOLOv8n-cls model on the local Bristol Stool dataset (Phase 3).

Dataset (YOLO classification layout, already in the project root):
    CPE310/
      train/Type 1/ ... train/Type 7/   <- class folders with images
      valid/Type 1/ ...                 (also accepted as 'val')
      test/Type 1/  ...

Usage:
    cd backend
    python train_yolo.py                      # 10 epochs, auto device
    python train_yolo.py --epochs 25 --batch 64 --device 0

After training, best.pt is copied to backend/best.pt, which is exactly what
inference.py / main.py load at runtime.
"""

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv8n-cls (Bristol Stool Scale)")
    p.add_argument(
        "--data", type=Path, default=PROJECT_ROOT,
        help="dataset root containing the train/valid/test split folders",
    )
    p.add_argument(
        "--model", default="yolov8n-cls.pt",
        help="base weights (auto-downloaded on first run)",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default=None, help="cpu | 0 | 0,1 ... (auto if omitted)")
    p.add_argument("--project", default=str(BACKEND_DIR / "runs"), help="output directory")
    p.add_argument("--name", default="stoolcheck", help="run name")
    p.add_argument(
        "--no-install", action="store_true",
        help="do not copy the resulting best.pt into backend/",
    )
    return p.parse_args()


def validate_dataset(data_dir: Path) -> list[str]:
    """Check the expected classification layout before spending time training."""
    train = data_dir / "train"
    if not train.is_dir():
        raise SystemExit(
            f"[ERROR] 'train' folder not found at: {train}\n"
            "        Expected layout: <data>/train/Type 1/*.jpg ... train/Type 7/..."
        )
    classes = sorted(d.name for d in train.iterdir() if d.is_dir())
    if not classes:
        raise SystemExit(f"[ERROR] no class folders inside {train}")
    if len(classes) != 7:
        print(f"[WARN] expected 7 classes (Type 1-7), found {len(classes)}: {classes}")
    else:
        print(f"[OK] {len(classes)} classes: {classes}")

    splits = [s for s in ("train", "val", "valid", "test") if (data_dir / s).is_dir()]
    print(f"[OK] splits found: {splits}")
    if "val" not in splits and "valid" in splits:
        print(
            "[OK] validation split 'valid/' found - ultralytics accepts "
            "'val', 'validation' or 'valid' for classification datasets."
        )
    if "train" not in splits:
        raise SystemExit("[ERROR] no 'train' split")
    return classes


def find_best_weights(model, results, args) -> Path | None:
    """Locate best.pt wherever ultralytics saved the run."""
    candidates: list[Path] = []
    save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    if save_dir:
        candidates.append(Path(save_dir))
    save_dir = getattr(results, "save_dir", None)
    if save_dir:
        candidates.append(Path(save_dir))
    candidates.append(Path(args.project) / args.name)

    for c in candidates:
        best = c / "weights" / "best.pt"
        if best.is_file():
            return best
    print(f"[ERROR] best.pt not found. Checked: {[str(c / 'weights') for c in candidates]}")
    return None


def main() -> None:
    args = parse_args()
    data_dir = args.data.resolve()
    validate_dataset(data_dir)

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "[ERROR] ultralytics is not installed. Run:\n"
            "        pip install -r requirements.txt"
        )

    print(f"[..] loading base model: {args.model}")
    model = YOLO(args.model)

    print(f"[..] training {args.epochs} epochs on {data_dir}")
    results = model.train(
        data=str(data_dir),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        verbose=True,
    )

    best = find_best_weights(model, results, args)
    if best is None:
        raise SystemExit(1)
    print(f"[OK] training complete. best weights: {best}")

    if not args.no_install:
        dest = BACKEND_DIR / "best.pt"
        shutil.copy2(best, dest)
        print(f"[OK] installed model for the API -> {dest}")

    # Quick sanity evaluation on the held-out test split.
    try:
        metrics = model.val(data=str(data_dir), split="test", verbose=False)
        acc = getattr(metrics, "top1", None)
        if acc is not None:
            print(f"[OK] test top-1 accuracy: {acc:.3f}")
    except Exception as exc:
        print(f"[WARN] validation skipped: {exc}")


if __name__ == "__main__":
    main()
