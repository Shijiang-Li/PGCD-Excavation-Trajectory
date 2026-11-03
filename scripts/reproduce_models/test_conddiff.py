from __future__ import annotations

import argparse
import time

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


def load_conddiff(path):
    from pgcd_paper.cond_diffusion import CondDiffusion

    payload = load_checkpoint(path)
    model = CondDiffusion(tau_dim=payload["tau_dim"], cond_dim=payload["cond_dim"])
    model.model.load_state_dict(payload["state_dict"], strict=True)
    model.model.eval()
    model.stats = payload["stats"]
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Conditional Diffusion baseline for Table 2.")
    add_common_test_args(parser)
    args = parser.parse_args()
    apply_quick_data_args(args)
    setup(args.results_dir)
    data, _, test_sets = load_table2_data(args)
    pso_cache = load_pso_reference(args.pso_reference or pso_reference_path(args.run_dir))

    for seed in args.seeds:
        model = load_conddiff(checkpoint_path(args.run_dir, "conddiff", seed))

        def evaluator(scene, cond_raw, n, eval_seed, condition_id):
            start = time.time()
            tau = model.sample(cond_raw, n=n)
            latency = time.time() - start
            best_J, feas_rate, div = generative_eval_tau_norm(tau, scene)
            return best_J, feas_rate, div, latency

        evaluate_method(
            data=data,
            test_sets=test_sets,
            method="CondDiff",
            seed=seed,
            run_dir=args.run_dir,
            candidate_count=args.candidate_count,
            pso_cache=pso_cache,
            pso_pop=args.pso_pop,
            pso_iter=args.pso_iter,
            evaluator=evaluator,
            force=args.force,
        )
        print(f"[test_conddiff] seed {seed} done")
    out = aggregate_seed_summaries(args.run_dir, "conddiff", "CondDiff", args.seeds)
    print(f"[test_conddiff] summary {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

