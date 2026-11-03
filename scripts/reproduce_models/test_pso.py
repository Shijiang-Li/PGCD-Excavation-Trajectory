from __future__ import annotations

import argparse

import numpy as np

from _common import (
    TABLE2_KEYS,
    add_common_test_args,
    apply_quick_data_args,
    finite_mean,
    load_pso_reference,
    load_table2_data,
    pso_reference_for_condition,
    pso_reference_path,
    setup,
    write_csv,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PSO reference baseline for Table 2.")
    add_common_test_args(parser)
    args = parser.parse_args()
    apply_quick_data_args(args)
    setup(args.results_dir)
    data, _, test_sets = load_table2_data(args)
    # Reuse the stored references; PSO only runs for conditions that are not in the file yet.
    pso_cache = load_pso_reference(args.pso_reference or pso_reference_path(args.run_dir))
    ref_rows: list[dict] = []
    summary_rows: list[dict] = []
    for test_set, condition_ids in test_sets.items():
        best, lat = [], []
        for condition_id in condition_ids:
            ref = pso_reference_for_condition(data, int(condition_id), pso_cache, pso_pop=args.pso_pop, pso_iter=args.pso_iter)
            best.append(ref["best_J"])
            lat.append(ref["lat"])
            ref_rows.append(
                {
                    "test_set": test_set,
                    "condition_id": int(condition_id),
                    "best_J": ref["best_J"],
                    "lat": ref["lat"],
                }
            )
        summary_rows.append(
            {
                "test_set": test_set,
                "method": "PSO",
                "succ": float(np.isfinite(best).mean()),
                "best_J": finite_mean(best),
                "feas_rate": 1.0,
                "viol_rate": 0.0,
                "gap": 0.0,
                "div": 0.0,
                "lat": finite_mean(lat),
            }
        )
    out_ref = pso_reference_path(args.run_dir)
    write_csv(out_ref, ref_rows, ["test_set", "condition_id", "best_J", "lat"])
    out_summary = args.run_dir.resolve() / "pso" / "results" / "summary.csv"
    write_csv(out_summary, summary_rows, ["test_set", "method", *TABLE2_KEYS])
    write_json(out_summary.with_suffix(".json"), {"method": "PSO", "rows": summary_rows}, force=True)
    print(f"[test_pso] reference {out_ref}")
    print(f"[test_pso] summary {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

