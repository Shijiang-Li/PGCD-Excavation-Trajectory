from __future__ import annotations

import argparse

import numpy as np

from _common import (
    add_common_train_args,
    apply_quick_data_args,
    checkpoint_path,
    load_table2_data,
    save_checkpoint,
    setup,
    stats_from_training_feasible,
    train_arrays,
    training_config,
    write_csv,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train PINN baseline for Table 2.")
    add_common_train_args(parser)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--surrogate-steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-phys", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=500, help="Print and save one loss record every N steps.")
    parser.add_argument("--quiet", action="store_true", help="Do not print per-step loss; the CSV log is still saved.")
    args = parser.parse_args()
    apply_quick_data_args(args)
    if args.quick:
        args.steps = 80
        args.surrogate_steps = 80
        args.log_every = min(args.log_every, 20)
    setup(args.results_dir)

    from pgcd_paper.fnn_baseline import best_per_condition
    from pgcd_paper.physics_cost import penalize, train_cond_surrogate
    from pgcd_paper.pinn_baseline import train_pinn
    from pgcd_paper.randomness import set_seed

    data, split, _ = load_table2_data(args)
    tau, cond, J, train_mask, feasible, train_feasible = train_arrays(data, split["train"])
    tau_stats = stats_from_training_feasible(tau, cond, train_feasible)[:2]
    cond_stats = stats_from_training_feasible(tau, cond, train_feasible)[2:]
    cfg = training_config(
        args,
        "pinn",
        {
            "steps": args.steps,
            "surrogate_steps": args.surrogate_steps,
            "lr": args.lr,
            "lambda_phys": args.lambda_phys,
            "log_every": args.log_every,
        },
    )
    model_dir = args.run_dir.resolve() / "pinn"
    write_json(model_dir / "resolved_config.json", cfg, force=True)
    cond_best, tau_best = best_per_condition(data, split["train"])
    print(
        "[train_pinn] "
        f"PSO samples={len(data['samp_J'])}, train_conditions={len(split['train'])}, "
        f"supervised_pairs={len(cond_best)}, surrogate_samples={int(train_mask.sum())}, "
        f"steps={args.steps}, surrogate_steps={args.surrogate_steps}, lr={args.lr}",
        flush=True,
    )

    for seed in args.seeds:
        ckpt = checkpoint_path(args.run_dir, "pinn", seed)
        if ckpt.exists() and not args.force:
            raise FileExistsError(f"refusing to overwrite {ckpt}; pass --force")
        set_seed(seed)
        tau_std = (tau[train_mask] - tau_stats[0]) / tau_stats[1]
        cond_std = (cond[train_mask] - cond_stats[0]) / cond_stats[1]
        print(f"[train_pinn] seed={seed} surrogate training start", flush=True)
        surrogate, surrogate_stats, surrogate_history = train_cond_surrogate(
            tau_std,
            cond_std,
            penalize(J[train_mask]),
            steps=args.surrogate_steps,
            verbose=not args.quiet,
            log_every=args.log_every,
            return_history=True,
        )
        surrogate_log_path = model_dir / "training_logs" / f"seed_{seed}_surrogate.csv"
        write_csv(surrogate_log_path, surrogate_history, ["step", "loss", "lr", "r2"])
        print(f"[train_pinn] seed={seed} PINN training start", flush=True)
        model, stats, pinn_history = train_pinn(
            cond_best,
            tau_best,
            surrogate,
            tau_stats,
            cond_stats,
            lam_phys=args.lambda_phys,
            steps=args.steps,
            lr=args.lr,
            verbose=not args.quiet,
            log_every=args.log_every,
            return_history=True,
        )
        pinn_log_path = model_dir / "training_logs" / f"seed_{seed}_pinn.csv"
        write_csv(pinn_log_path, pinn_history, ["step", "loss", "mse", "phys", "lr"])
        save_checkpoint(
            ckpt,
            {
                "schema_version": 1,
                "model_key": "pinn",
                "method": "PINN",
                "seed": seed,
                "config": cfg,
                "cond_dim": int(cond_best.shape[1]),
                "tau_dim": int(tau_best.shape[1]),
                "state_dict": model.state_dict(),
                "stats": tuple(np.asarray(v, dtype=float) for v in stats),
                "surrogate_stats": tuple(np.asarray(v, dtype=float) for v in surrogate_stats),
                "tau_stats": tuple(np.asarray(v, dtype=float) for v in tau_stats),
                "cond_stats": tuple(np.asarray(v, dtype=float) for v in cond_stats),
            },
            force=args.force,
        )
        print(f"[train_pinn] logs {surrogate_log_path}, {pinn_log_path}")
        print(f"[train_pinn] saved {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
