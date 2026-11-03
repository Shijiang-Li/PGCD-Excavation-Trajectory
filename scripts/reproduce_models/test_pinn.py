from __future__ import annotations

import argparse
import time

from _common import (
    add_common_test_args,
    aggregate_seed_summaries,
    apply_quick_data_args,
    checkpoint_path,
    deterministic_eval_tau_norm,
    evaluate_method,
    load_checkpoint,
    load_pso_reference,
    load_table2_data,
    pso_reference_path,
    setup,
)


def load_pinn(path):
    from pgcd_paper.fnn_baseline import FNN, predict_fnn

    payload = load_checkpoint(path)
    model = FNN(cond_dim=payload["cond_dim"], tau_dim=payload["tau_dim"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload["stats"], predict_fnn


def main() -> int:
    parser = argparse.ArgumentParser(description="Test PINN baseline for Table 2.")
    add_common_test_args(parser)
    args = parser.parse_args()
    apply_quick_data_args(args)
    setup(args.results_dir)
    data, _, test_sets = load_table2_data(args)
    pso_cache = load_pso_reference(args.pso_reference or pso_reference_path(args.run_dir))

    for seed in args.seeds:
        model, stats, predict_pinn = load_pinn(checkpoint_path(args.run_dir, "pinn", seed))

        def evaluator(scene, cond_raw, n, eval_seed, condition_id):
            start = time.time()
            tau = predict_pinn(model, stats, cond_raw)
            latency = time.time() - start
            best_J, feas_rate, div = deterministic_eval_tau_norm(tau, scene)
            return best_J, feas_rate, div, latency

        evaluate_method(
            data=data,
            test_sets=test_sets,
            method="PINN",
            seed=seed,
            run_dir=args.run_dir,
            candidate_count=args.candidate_count,
            pso_cache=pso_cache,
            pso_pop=args.pso_pop,
            pso_iter=args.pso_iter,
            evaluator=evaluator,
            force=args.force,
        )
        print(f"[test_pinn] seed {seed} done")
    out = aggregate_seed_summaries(args.run_dir, "pinn", "PINN", args.seeds)
    print(f"[test_pinn] summary {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

