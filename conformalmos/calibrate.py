"""Calibrate on a separate labeled prediction CSV: path,predicted_mos,mos."""
import argparse
import csv
import json
import math
from pathlib import Path
from .calibration import quantile
from .predict import digest

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--evaluation-paths", required=True, type=Path,
                   help="Text file of evaluation audio paths, one per line, to check overlap")
    p.add_argument("--alpha", type=float, default=.1)
    p.add_argument("--method", choices=["finite_sample", "historical"], default="finite_sample")
    a = p.parse_args()
    with a.csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    paths = [str(Path(r["path"]).resolve()) for r in rows]
    evaluation = {str(Path(s.strip()).resolve()) for s in a.evaluation_paths.read_text().splitlines() if s.strip()}
    if not evaluation:
        p.error("Evaluation path list is empty")
    if len(paths) != len(set(paths)) or set(paths) & evaluation:
        p.error("Duplicate calibration paths or calibration/evaluation overlap")
    scores = [float(r["mos"]) for r in rows]
    preds = [float(r["predicted_mos"]) for r in rows]
    if any(not math.isfinite(v) or not 1 <= v <= 5 for v in scores+preds):
        p.error("MOS and predictions must be finite and in [1,5]")
    q, rank = quantile([abs(y-f) for y,f in zip(scores,preds)], a.alpha, a.method)
    result = dict(q_hat=q if math.isfinite(q) else "Infinity", alpha=a.alpha, method=a.method,
        n_calibration=len(rows), rank=rank, checkpoint_sha256=digest(a.checkpoint),
        note="Path disjointness does not verify training/group/content independence; audit these separately.")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(f"Saved calibration to {a.output}; rank={rank}, q_hat={q}")

if __name__ == "__main__":
    main()
