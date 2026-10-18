"""Evaluate the trained YOLOv8n-cls model on a held-out split (default: test).

Run:
    cd backend
    python eval_model.py                 # evaluate on the test split
    python eval_model.py --split valid   # evaluate on the validation split

Reports:
  1. Ultralytics val metrics (overall top-1 / top-5 accuracy)
  2. Per-class accuracy table
  3. +/-1 Bristol-type tolerance accuracy (predicting an adjacent type is
     counted as correct - on the Bristol scale such mistakes are clinically
     far less severe than predicting the opposite end)
  4. Saves a confusion-matrix plot under backend/runs/val*/
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent


def bristol_number(label: str) -> int | None:
    """Extract the Bristol type (1-7) from a class name like 'Type 3'."""
    m = re.search(r"(\d+)", str(label))
    if m and 1 <= int(m.group(1)) <= 7:
        return int(m.group(1))
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the StoolCheck AI model")
    p.add_argument(
        "--weights", type=Path, default=BACKEND_DIR / "best.pt",
        help="model weights (default: backend/best.pt)",
    )
    p.add_argument(
        "--data", type=Path, default=PROJECT_ROOT,
        help="dataset root containing train/valid/test",
    )
    p.add_argument("--split", default="test", choices=["test", "valid", "train"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    weights = args.weights
    if not weights.is_file():
        alt = BACKEND_DIR / "runs" / "stoolcheck" / "weights" / "best.pt"
        if alt.is_file():
            weights = alt
        else:
            raise SystemExit(
                f"[ERROR] weights not found: {args.weights}\n"
                "        Train first:  python train_yolo.py"
            )

    split_dir = args.data / args.split
    class_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
    if not class_dirs:
        raise SystemExit(f"[ERROR] no class folders found in {split_dir}")

    from ultralytics import YOLO  # heavy import - only after validation

    model = YOLO(str(weights))
    print(f"[OK] weights: {weights}")
    print(f"[OK] split:   {args.split} ({len(class_dirs)} classes)")

    # ---- 1) canonical validation pass (overall metrics + plots) -------------
    results = model.val(
        data=str(args.data),
        split=args.split,
        verbose=True,
        plots=True,  # confusion matrix PNG saved under runs/val*
    )
    rd = dict(getattr(results, "results_dict", {}) or {})
    top1 = rd.get("metrics/accuracy_top1", getattr(results, "top1", None))
    top5 = rd.get("metrics/accuracy_top5", getattr(results, "top5", None))
    if top1 is not None and top5 is not None:
        print(f"\n[RESULT] {args.split} split -> top-1: {top1:.1%}   top-5: {top5:.1%}")
    else:
        print(f"\n[RESULT] raw metrics: {rd}")
    save_dir = getattr(results, "save_dir", None)
    if save_dir:
        print(f"[PLOT] confusion matrix saved under: {save_dir}")

    # ---- 2) per-class accuracy + +/-1 tolerance pass -------------------------
    names = model.names  # {index: "Type N"}
    images = [p for d in class_dirs for p in sorted(d.iterdir()) if p.is_file()]
    total = exact = within1 = 0
    per_total: dict[str, int] = defaultdict(int)
    per_correct: dict[str, int] = defaultdict(int)
    confusions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for res in model.predict(source=[str(p) for p in images], verbose=False):
        true_name = Path(res.path).parent.name
        pred_name = names[int(res.probs.top1)]
        true_b, pred_b = bristol_number(true_name), bristol_number(pred_name)
        is_exact = (
            true_b == pred_b if (true_b and pred_b) else true_name == pred_name
        )

        total += 1
        per_total[true_name] += 1
        confusions[true_name][pred_name] += 1
        if is_exact:
            exact += 1
            per_correct[true_name] += 1
        if (
            true_b and pred_b and abs(true_b - pred_b) <= 1
        ) or is_exact:
            within1 += 1

    print("\n[PER-CLASS] accuracy on the held-out split")
    print(f"  {'class':<10} {'correct':>8} {'total':>6} {'accuracy':>10}")
    for d in class_dirs:
        name = d.name
        c, t = per_correct[name], per_total[name]
        acc = f"{c / t:.0%}" if t else "n/a"
        print(f"  {name:<10} {c:>8} {t:>6} {acc:>10}")

    print(f"\n[SUMMARY] images evaluated: {total}")
    print(f"[SUMMARY] exact  top-1 accuracy:      {exact / total:.1%}  ({exact}/{total})")
    print(f"[SUMMARY] +/-1 tolerance accuracy:    {within1 / total:.1%}  ({within1}/{total})")
    print(f"[SUMMARY] random baseline (1/7):       {1 / 7:.1%}")

    # Most common mistakes (where the model gets confused)
    mistakes = [
        (t, p, n)
        for t, preds in confusions.items()
        for p, n in preds.items()
        if t != p and n > 0
    ]
    mistakes.sort(key=lambda x: -x[2])
    if mistakes:
        print("\n[CONFUSIONS] most common wrong predictions (true -> predicted xN):")
        for t, p, n in mistakes[:8]:
            print(f"  {t} -> {p}  x{n}")


if __name__ == "__main__":
    main()
