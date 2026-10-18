"""Read YOLOv8 training metrics for the Model Evaluation section (Phase 4).

Parses the results.csv written by ultralytics (runs/<name>/results.csv)
and returns the metrics of the most recent run.
"""

import csv
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
RUNS_DIR = BACKEND_DIR / "runs"
MODEL_NAME = "yolov8n-cls"


def _find_results_csv() -> Path | None:
    """Prefer the default run name, else the most recently modified run."""
    preferred = RUNS_DIR / "stoolcheck" / "results.csv"
    if preferred.is_file():
        return preferred
    candidates = sorted(
        RUNS_DIR.glob("*/results.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _num(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_evaluation() -> dict:
    """Return training metrics for the latest run (or available=False)."""
    csv_path = _find_results_csv()
    if csv_path is None:
        return {"available": False, "model": MODEL_NAME}

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"available": False, "model": MODEL_NAME}

    last = rows[-1]  # final epoch (kept for reference)

    def pick(row, *fragments: str) -> float | None:
        for key, value in row.items():
            if all(fr in key for fr in fragments):
                return _num(value)
        return None

    def _score(row) -> float:
        v = pick(row, "accuracy_top1")
        return v if v is not None else -1.0

    # Show the BEST epoch by top-1 accuracy, not the final one - with
    # overfitting the last epoch is rarely the best.
    best = max(rows, key=_score)

    return {
        "available": True,
        "model": MODEL_NAME,
        "epochs": len(rows),
        "best_epoch": int(_num(best.get("epoch")) or 0),
        "top1": _score(best) if _score(best) >= 0 else None,
        "top5": pick(best, "accuracy_top5"),
        "train_loss": pick(best, "train", "loss"),
        "val_loss": pick(best, "val", "loss"),
        "final_top1": _score(last) if _score(last) >= 0 else None,
        "run": csv_path.parent.name,
    }
