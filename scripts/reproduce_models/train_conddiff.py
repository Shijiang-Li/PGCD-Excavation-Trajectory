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
    train_arrays,
    training_config,
    write_csv,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Conditional Diffusion baseline for Table 2.")
    add_common_train_args(parser)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--bc-temp", type=float, default=0.3)
    parser.add_argument("--log-every", type=int, default=500, help="Print and save one loss record every N steps.")
    parser.add_argument("--quiet", action="store_true", help="Do not print per-step loss; the CSV log is still saved.")
    args = parser.parse_args()
    apply_quick_data_args(args)
    if args.quick:
        args.steps = 80
        args.log_every = min(args.log_every, 20)
    setup(args.results_dir)

    from pgcd_paper.cond_diffusion import CondDiffusion
    from pgcd_paper.randomness import set_seed

    data, split, _ = load_table2_data(args)
    tau, cond, J, train_mask, feasible, train_feasible = train_arrays(data, split["train"])
    cfg = training_config(args, "conddiff", {"steps": args.steps, "lr": args.lr, "bc_temp": args.bc_temp, "log_every": args.log_every})
    model_dir = args.run_dir.resolve() / "conddiff"
    write_json(model_dir / "resolved_config.json", cfg, force=True)
    print(
        "[train_conddiff] "
        f"PSO samples={len(data['samp_J'])}, train_feasible_samples={int(train_feasible.sum())}, "
        f"steps={args.steps}, lr={args.lr}",
        flush=True,
    )

    for seed in args.seeds:
        ckpt = checkpoint_path(args.run_dir, "conddiff", seed)
        if ckpt.exists() and not args.force:
            raise FileExistsError(f"refusing to overwrite {ckpt}; pass --force")
        set_seed(seed)
        print(f"[train_conddiff] seed={seed} training start", flush=True)
        model, history = CondDiffusion().fit(
            tau[train_feasible],
            cond[train_feasible],
            J[train_feasible],
            steps=args.steps,
            lr=args.lr,
            bc_temp=args.bc_temp,
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
                "model_key": "conddiff",
                "method": "CondDiff",
                "seed": seed,
                "config": cfg,
                "tau_dim": int(model.tau_dim),
                "cond_dim": int(model.cond_dim),
                "state_dict": model.model.state_dict(),
                "stats": tuple(np.asarray(v, dtype=float) for v in model.stats),
            },
            force=args.force,
        )
        print(f"[train_conddiff] log {log_path}")
        print(f"[train_conddiff] saved {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
