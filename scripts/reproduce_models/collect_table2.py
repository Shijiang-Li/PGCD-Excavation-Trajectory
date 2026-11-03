from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from _common import DEFAULT_RESULTS_DIR, DEFAULT_RUN_DIR, TABLE2_KEYS, TABLE2_METHODS, setup, write_csv, write_json


MODEL_KEY_BY_METHOD = {
    "PSO": "pso",
    "FNN": "fnn",
    "PINN": "pinn",
    "PC-STOMP": "pc_stomp",
    "CVAE": "cvae",
    "CondDiff": "conddiff",
    "PGDiff": "pgcd",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect independently tested model results into Table 2 files.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--methods", nargs="*", default=TABLE2_METHODS)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    results_dir = setup(args.results_dir)

    rows: list[dict] = []
    missing: list[Path] = []
    for method in args.methods:
        model_key = MODEL_KEY_BY_METHOD[method]
        path = args.run_dir.resolve() / model_key / "results" / "summary.csv"
        if not path.exists():
            missing.append(path)
            continue
        rows.extend(read_rows(path))
    if missing and not args.allow_missing:
        raise SystemExit("missing model summaries:\n" + "\n".join(str(path) for path in missing))
    if not rows:
        raise SystemExit("no model summaries found")

    method_order = {method: i for i, method in enumerate(args.methods)}
    rows.sort(key=lambda row: (row["test_set"], method_order.get(row["method"], 999)))
    out_csv = results_dir / "split_results.csv"
    write_csv(out_csv, rows, ["test_set", "method", *TABLE2_KEYS])

    test_sets = []
    methods = []
    for row in rows:
        if row["test_set"] not in test_sets:
            test_sets.append(row["test_set"])
        if row["method"] not in methods:
            methods.append(row["method"])
    table = np.full((len(test_sets), len(methods), len(TABLE2_KEYS)), np.nan, dtype=float)
    for row in rows:
        i = test_sets.index(row["test_set"])
        j = methods.index(row["method"])
        table[i, j, :] = [float(row[key]) for key in TABLE2_KEYS]
    np.savez(results_dir / "split_results.npz", methods=np.array(methods), test_sets=np.array(test_sets), keys=np.array(TABLE2_KEYS), table=table)
    write_json(
        results_dir / "split_results_manifest.json",
        {
            "source_run_dir": str(args.run_dir.resolve()),
            "methods": methods,
            "test_sets": test_sets,
            "csv": str(out_csv),
            "npz": str(results_dir / "split_results.npz"),
        },
    )
    print(f"[collect_table2] saved {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

