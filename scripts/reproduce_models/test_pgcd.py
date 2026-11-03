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


def load_pgcd(path):
    from pgcd_paper.cond_diffusion import CondDiffusion
    from pgcd_paper.physics_cost import CondSurrogate, make_cond_guidance

    payload = load_checkpoint(path)
    diff = CondDiffusion(tau_dim=payload["tau_dim"], cond_dim=payload["cond_dim"])
    diff.model.load_state_dict(payload["diff_state_dict"], strict=True)
    diff.model.eval()
    diff.stats = payload["diff_stats"]
    surrogate = CondSurrogate(tau_dim=payload["tau_dim"], cond_dim=payload["cond_dim"])
    surrogate.load_state_dict(payload["surrogate_state_dict"], strict=True)
    surrogate.eval()
    return diff, make_cond_guidance(surrogate), float(payload.get("eta", 0.12))


def main() -> int:
    parser = argparse.ArgumentParser(description="Test PGCD for Table 2.")
    add_common_test_args(parser)
    parser.add_argument("--eta", type=float, default=None, help="override checkpoint eta")
    args = parser.parse_args()
    apply_quick_data_args(args)
    setup(args.results_dir)
    data, _, test_sets = load_table2_data(args)
    pso_cache = load_pso_reference(args.pso_reference or pso_reference_path(args.run_dir))

    for seed in args.seeds:
        model, guidance, ckpt_eta = load_pgcd(checkpoint_path(args.run_dir, "pgcd", seed))
        eta = ckpt_eta if args.eta is None else args.eta

        def evaluator(scene, cond_raw, n, eval_seed, condition_id):
            start = time.time()
            tau = model.sample(cond_raw, n=n, guidance=guidance, eta=eta)
            latency = time.time() - start
            best_J, feas_rate, div = generative_eval_tau_norm(tau, scene)
            return best_J, feas_rate, div, latency

        evaluate_method(
            data=data,
            test_sets=test_sets,
            method="PGDiff",
            model_key="pgcd",
            seed=seed,
            run_dir=args.run_dir,
            candidate_count=args.candidate_count,
            pso_cache=pso_cache,
            pso_pop=args.pso_pop,
            pso_iter=args.pso_iter,
            evaluator=evaluator,
            force=args.force,
        )
        print(f"[test_pgcd] seed {seed} done")
    out = aggregate_seed_summaries(args.run_dir, "pgcd", "PGDiff", args.seeds)
    print(f"[test_pgcd] summary {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
