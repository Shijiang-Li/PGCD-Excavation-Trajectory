from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from _common import DEFAULT_RESULTS_DIR, DEFAULT_RUN_DIR, write_csv
from collect_table2 import read_rows

HERE = Path(__file__).resolve().parent
GENERATIVE = {"CVAE": "cvae", "CondDiff": "conddiff", "PGDiff": "pgcd"}
METRICS = ["succ", "viol_rate", "gap", "div", "best_J", "lat"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-evaluate the released CVAE / CD / PGCD checkpoints with several fixed sampling seeds "
        "and report mean, std and range of the Table 2 metrics next to the reported values.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--draws", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_DIR / "table2_resampling.csv")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    reported = {(r["test_set"], r["method"]): r for r in read_rows(DEFAULT_RESULTS_DIR / "split_results.csv")}

    draws: dict[tuple[str, str], list[dict[str, str]]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for draw in range(args.draws):
            for method, key in GENERATIVE.items():
                work = Path(tmp) / f"draw_{draw}"
                (work / key).mkdir(parents=True, exist_ok=True)
                shutil.copytree(run_dir / key / "checkpoints", work / key / "checkpoints", dirs_exist_ok=True)
                shutil.copytree(run_dir / "pso", work / "pso", dirs_exist_ok=True)  # reuse the stored PSO reference
                cmd = [sys.executable, str(HERE / f"test_{key}.py"), "--run-dir", str(work), "--sample-seed", str(draw)]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
                for row in read_rows(work / key / "results" / "summary.csv"):
                    draws.setdefault((row["test_set"], method), []).append(row)
            print(f"[resample_table2] draw {draw + 1}/{args.draws} done", flush=True)

    out_rows = []
    print(f"\n{'test set':11s} {'method':9s} {'metric':9s} {'reported':>9s} {'mean':>8s} {'std':>7s} {'min':>8s} {'max':>8s}")
    for (test_set, method), rows in draws.items():
        for metric in METRICS:
            vals = np.array([float(r[metric]) for r in rows])
            rep = float(reported[(test_set, method)][metric])
            out_rows.append({"test_set": test_set, "method": method, "metric": metric, "reported": rep,
                             "mean": vals.mean(), "std": vals.std(ddof=1), "min": vals.min(), "max": vals.max(),
                             "draws": len(vals)})
            print(f"{test_set:11s} {method:9s} {metric:9s} {rep:9.4f} {vals.mean():8.4f} {vals.std(ddof=1):7.4f} "
                  f"{vals.min():8.4f} {vals.max():8.4f}")
    write_csv(args.output, out_rows, ["test_set", "method", "metric", "reported", "mean", "std", "min", "max", "draws"])
    print(f"\n[resample_table2] saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
