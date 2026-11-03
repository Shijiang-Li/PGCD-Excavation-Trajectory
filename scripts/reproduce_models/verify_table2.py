from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from _common import DEFAULT_RESULTS_DIR, DEFAULT_RUN_DIR, TABLE2_KEYS, TABLE2_METHODS, finite_mean
from collect_table2 import MODEL_KEY_BY_METHOD, read_rows


def summarize(rows: list[dict[str, str]]) -> dict[str, float]:
    """Same aggregation as _common.evaluate_method, applied to stored per-condition rows."""
    best = np.array([float(r["best_J"]) for r in rows])
    feas = float(np.mean([float(r["feas_rate"]) for r in rows]))
    return {
        "succ": float(np.isfinite(best).mean()),
        "best_J": finite_mean(best),
        "feas_rate": feas,
        "viol_rate": 1.0 - feas,
        "gap": finite_mean(float(r["gap"]) for r in rows),
        "div": float(np.mean([float(r["div"]) for r in rows])),
        "lat": finite_mean(float(r["lat"]) for r in rows),
    }


def recompute(run_dir: Path, method: str) -> dict[str, dict[str, float]]:
    key = MODEL_KEY_BY_METHOD[method]
    if method == "PSO":  # PSO is its own reference: one run per condition, always feasible
        rows = read_rows(run_dir / "pso" / "condition_reference.csv")
        out = {}
        for test_set in dict.fromkeys(r["test_set"] for r in rows):
            sub = [r for r in rows if r["test_set"] == test_set]
            best = np.array([float(r["best_J"]) for r in sub])
            out[test_set] = {"succ": float(np.isfinite(best).mean()), "best_J": finite_mean(best), "feas_rate": 1.0,
                             "viol_rate": 0.0, "gap": 0.0, "div": 0.0, "lat": finite_mean(float(r["lat"]) for r in sub)}
        return out
    per_seed = []
    for path in sorted((run_dir / key / "results").glob("seed_*/per_condition.csv")):
        rows = read_rows(path)
        per_seed.append({t: summarize([r for r in rows if r["test_set"] == t]) for t in dict.fromkeys(r["test_set"] for r in rows)})
    return {t: {k: finite_mean(s[t][k] for s in per_seed) for k in TABLE2_KEYS} for t in per_seed[0]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute Table 2 from the stored per-condition outputs and compare with the reported table.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--table", type=Path, default=DEFAULT_RESULTS_DIR / "split_results.csv")
    args = parser.parse_args()

    reported = {(r["test_set"], r["method"]): r for r in read_rows(args.table)}
    worst = 0.0
    print(f"{'test set':11s} {'method':9s} {'succ%':>6s} {'viol%':>6s} {'gap%':>6s} {'div':>6s} {'time s':>7s}  max|diff|")
    for test_set in ("familiar", "unfamiliar"):
        for method in TABLE2_METHODS:
            got = recompute(args.run_dir.resolve(), method)[test_set]
            ref = reported[(test_set, method)]
            diff = max(abs(got[k] - float(ref[k])) for k in TABLE2_KEYS)
            worst = max(worst, diff)
            print(f"{test_set:11s} {method:9s} {100 * got['succ']:6.1f} {100 * got['viol_rate']:6.1f} "
                  f"{100 * got['gap']:6.1f} {got['div']:6.3f} {got['lat']:7.3f}  {diff:.1e}")
    ok = worst < 1e-9
    print(f"\n[verify_table2] {'PASS' if ok else 'FAIL'}: max |recomputed - reported| = {worst:.1e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
