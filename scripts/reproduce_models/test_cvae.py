from __future__ import annotations

import argparse
import time

import numpy as np

from _common import (
    add_common_test_args,
    aggregate_seed_summaries,
    apply_quick_data_args,
    checkpoint_path,
    evaluate_method,
    generative_eval_tau_norm,
    load_checkpoint,
    load_pso_reference,
    load_table2_data,
    pso_reference_path,
    setup,
)


def load_cvae(path):
    from pgcd_paper.cvae_baseline import CVAEModel

    payload = load_checkpoint(path)
    model = CVAEModel(tau_dim=payload["tau_dim"], cond_dim=payload["cond_dim"], z_dim=payload["z_dim"])
    model.net.load_state_dict(payload["state_dict"], strict=True)
    model.net.eval()
    model.cm = np.asarray(payload["cm"], dtype=float)
    model.cs = np.asarray(payload["cs"], dtype=float)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description="Test CVAE baseline for Table 2.")
    add_common_test_args(parser)
    args = parser.parse_args()
    apply_quick_data_args(args)
    setup(args.results_dir)
    data, _, test_sets = load_table2_data(args)
    pso_cache = load_pso_reference(args.pso_reference or pso_reference_path(args.run_dir))

    for seed in args.seeds:
        model = load_cvae(checkpoint_path(args.run_dir, "cvae", seed))

        def evaluator(scene, cond_raw, n, eval_seed, condition_id):
            start = time.time()
            tau = model.sample(cond_raw, n=n)
            latency = time.time() - start
            best_J, feas_rate, div = generative_eval_tau_norm(tau, scene)
            return best_J, feas_rate, div, latency

        evaluate_method(
            data=data,
            test_sets=test_sets,
            method="CVAE",
            seed=seed,
            run_dir=args.run_dir,
            candidate_count=args.candidate_count,
            pso_cache=pso_cache,
            pso_pop=args.pso_pop,
            pso_iter=args.pso_iter,
            evaluator=evaluator,
            force=args.force,
        )
        print(f"[test_cvae] seed {seed} done")
    out = aggregate_seed_summaries(args.run_dir, "cvae", "CVAE", args.seeds)
    print(f"[test_cvae] summary {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

