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
    training_config,
    write_csv,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train FNN baseline for Table 2.")
    add_common_train_args(parser)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=500, help="Print and save one loss record every N steps.")
    parser.add_argument("--quiet", action="store_true", help="Do not print per-step loss; the CSV log is still saved.")
    args = parser.parse_args()
    apply_quick_data_args(args)
    if args.quick:
        args.steps = 80
        args.log_every = min(args.log_every, 20)
    setup(args.results_dir)

    from pgcd_paper.fnn_baseline import best_per_condition, train_fnn
    from pgcd_paper.randomness import set_seed

    data, split, test_sets = load_table2_data(args)
    cfg = training_config(
        args,
        "fnn",
        {
            "steps": args.steps,
            "lr": args.lr,
            "log_every": args.log_every,
            "target": "best feasible tau per training condition",
        },
    )
    model_dir = args.run_dir.resolve() / "fnn"
    write_json(model_dir / "resolved_config.json", cfg, force=True)

    cond_best, tau_best = best_per_condition(data, split["train"])
    print(
        "[train_fnn] "
        f"PSO samples={len(data['samp_J'])}, train_conditions={len(split['train'])}, "
        f"supervised_pairs={len(cond_best)}, steps={args.steps}, lr={args.lr}",
        flush=True,
    )
    for seed in args.seeds:
        ckpt = checkpoint_path(args.run_dir, "fnn", seed)
        if ckpt.exists() and not args.force:
            raise FileExistsError(f"refusing to overwrite {ckpt}; pass --force")
        set_seed(seed)
        print(f"[train_fnn] seed={seed} training start", flush=True)
        model, stats, history = train_fnn(
            cond_best,
            tau_best,
            steps=args.steps,
            lr=args.lr,
            verbose=not args.quiet,
            log_every=args.log_every,
            return_history=True,
        )
        log_path = model_dir / "training_logs" / f"seed_{seed}.csv"
        write_csv(log_path, history, ["step", "loss", "lr"])
        save_checkpoint(
            ckpt,
            {
                "schema_version": 1,
                "model_key": "fnn",
                "method": "FNN",
                "seed": seed,
                "config": cfg,
                "cond_dim": int(cond_best.shape[1]),
                "tau_dim": int(tau_best.shape[1]),
                "state_dict": model.state_dict(),
                "stats": tuple(np.asarray(v, dtype=float) for v in stats),
            },
            force=args.force,
        )
        print(f"[train_fnn] log {log_path}")
        print(f"[train_fnn] saved {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
