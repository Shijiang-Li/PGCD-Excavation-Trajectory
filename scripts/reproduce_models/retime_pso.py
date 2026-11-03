from __future__ import annotations

import argparse
import csv

import numpy as np

from _common import add_common_test_args, load_table2_data, pso_reference_path, setup, write_csv


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-measure the PSO planning time of every reference condition on the current machine. "
        "The stored reference objectives are kept; only the lat column of condition_reference.csv is replaced.")
    add_common_test_args(parser)
    args = parser.parse_args()
    setup(args.results_dir)
    data, _, _ = load_table2_data(args)
    from pgcd_paper import cond_data as CD
    from pgcd_paper.baselines import run_pso

    path = args.pso_reference or pso_reference_path(args.run_dir)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for i, row in enumerate(rows):
        cid = int(row["condition_id"])
        scene = CD.reconstruct_scene(data["cond_meta"][cid])
        np.random.seed(cid)
        _, _, latency = run_pso(scene, pop=args.pso_pop, max_iter=args.pso_iter, n_runs=1)
        row["lat"] = latency
        if (i + 1) % 20 == 0:
            print(f"[retime_pso] {i + 1}/{len(rows)}", flush=True)
    write_csv(path, rows, ["test_set", "condition_id", "best_J", "lat"])
    print(f"[retime_pso] updated {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
