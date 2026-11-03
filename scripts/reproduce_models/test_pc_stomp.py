from __future__ import annotations

import argparse
import time

from _common import (
    add_common_test_args,
    aggregate_seed_summaries,
    apply_quick_data_args,
    checkpoint_path,
    evaluate_method,
    load_checkpoint,
    load_pso_reference,
    load_table2_data,
    pso_reference_path,
    setup,
)


def load_initializer(path):
    from pgcd_paper.fnn_baseline import FNN, predict_fnn

    payload = load_checkpoint(path)
    model = FNN(cond_dim=payload["cond_dim"], tau_dim=payload["tau_dim"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload["stats"], predict_fnn


def main() -> int:
    parser = argparse.ArgumentParser(description="Test PC-STOMP two-stage baseline for Table 2.")
    add_common_test_args(parser)
    parser.add_argument("--fnn-model-key", default="fnn", help="directory containing the trained FNN checkpoints")
    args = parser.parse_args()
    apply_quick_data_args(args)
    setup(args.results_dir)
    data, _, test_sets = load_table2_data(args)
    pso_cache = load_pso_reference(args.pso_reference or pso_reference_path(args.run_dir))

    for seed in args.seeds:
        model, stats, predict_fnn = load_initializer(checkpoint_path(args.run_dir, args.fnn_model_key, seed))

        def evaluator(scene, cond_raw, n, eval_seed, condition_id):
            from pgcd_paper.pc_stomp import pc_stomp

            start = time.time()
            tau_norm = predict_fnn(model, stats, cond_raw)
            tau0 = scene.denormalize(tau_norm)[0]
            _, J = pc_stomp(scene, tau0, scene.lb, scene.ub, seed=eval_seed)
            latency = time.time() - start
            ok = bool(J == J and abs(J) != float("inf"))
            return float(J) if ok else float("inf"), float(ok), 0.0, latency

        evaluate_method(
            data=data,
            test_sets=test_sets,
            method="PC-STOMP",
            seed=seed,
            run_dir=args.run_dir,
            candidate_count=args.candidate_count,
            pso_cache=pso_cache,
            pso_pop=args.pso_pop,
            pso_iter=args.pso_iter,
            evaluator=evaluator,
            force=args.force,
        )
        print(f"[test_pc_stomp] seed {seed} done")
    out = aggregate_seed_summaries(args.run_dir, "pc_stomp", "PC-STOMP", args.seeds)
    print(f"[test_pc_stomp] summary {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
