from __future__ import annotations

import argparse

import numpy as np

from _common import (
    ETA_DEFAULT,
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
    parser = argparse.ArgumentParser(description="Train PGCD model for Table 2.")
    add_common_train_args(parser)
    parser.add_argument("--diffusion-steps", type=int, default=5000)
    parser.add_argument("--surrogate-steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--surrogate-lr", type=float, default=1e-3)
    parser.add_argument("--bc-temp", type=float, default=0.3)
    parser.add_argument("--eta", type=float, default=ETA_DEFAULT)
    parser.add_argument("--log-every", type=int, default=500, help="Print and save one loss record every N steps.")
    parser.add_argument("--quiet", action="store_true", help="Do not print per-step loss; the CSV log is still saved.")
    args = parser.parse_args()
    apply_quick_data_args(args)
    if args.quick:
        args.diffusion_steps = 80
        args.surrogate_steps = 80
        args.log_every = min(args.log_every, 20)
    setup(args.results_dir)

    from pgcd_paper.cond_diffusion import CondDiffusion
    from pgcd_paper.physics_cost import penalize, train_cond_surrogate
    from pgcd_paper.randomness import set_seed

    data, split, _ = load_table2_data(args)
    tau, cond, J, train_mask, feasible, train_feasible = train_arrays(data, split["train"])
    cfg = training_config(
        args,
        "pgcd",
        {
            "diffusion_steps": args.diffusion_steps,
            "surrogate_steps": args.surrogate_steps,
            "diffusion_lr": args.lr,
            "surrogate_lr": args.surrogate_lr,
            "bc_temp": args.bc_temp,
            "eta": args.eta,
            "log_every": args.log_every,
        },
    )
    model_dir = args.run_dir.resolve() / "pgcd"
    write_json(model_dir / "resolved_config.json", cfg, force=True)
    print(
        "[train_pgcd] "
        f"PSO samples={len(data['samp_J'])}, diffusion_train_feasible_samples={int(train_feasible.sum())}, "
        f"surrogate_samples={int(train_mask.sum())}, diffusion_steps={args.diffusion_steps}, "
        f"surrogate_steps={args.surrogate_steps}, diffusion_lr={args.lr}, surrogate_lr={args.surrogate_lr}",
        flush=True,
    )

    for seed in args.seeds:
        ckpt = checkpoint_path(args.run_dir, "pgcd", seed)
        if ckpt.exists() and not args.force:
            raise FileExistsError(f"refusing to overwrite {ckpt}; pass --force")
        set_seed(seed)
        print(f"[train_pgcd] seed={seed} diffusion training start", flush=True)
        diff, diffusion_history = CondDiffusion().fit(
            tau[train_feasible],
            cond[train_feasible],
            J[train_feasible],
            steps=args.diffusion_steps,
            lr=args.lr,
            bc_temp=args.bc_temp,
            verbose=not args.quiet,
            log_every=args.log_every,
            return_history=True,
        )
        diffusion_log_path = model_dir / "training_logs" / f"seed_{seed}_diffusion.csv"
        write_csv(diffusion_log_path, diffusion_history, ["step", "loss", "lr"])
        print(f"[train_pgcd] seed={seed} surrogate training start", flush=True)
        surrogate, surrogate_stats, surrogate_history = train_cond_surrogate(
            diff.std_tau(tau[train_mask]),
            diff.std_cond(cond[train_mask]),
            penalize(J[train_mask]),
            steps=args.surrogate_steps,
            lr=args.surrogate_lr,
            verbose=not args.quiet,
            log_every=args.log_every,
            return_history=True,
        )
        surrogate_log_path = model_dir / "training_logs" / f"seed_{seed}_surrogate.csv"
        write_csv(surrogate_log_path, surrogate_history, ["step", "loss", "lr", "r2"])
        save_checkpoint(
            ckpt,
            {
                "schema_version": 1,
                "model_key": "pgcd",
                "method": "PGDiff",
                "seed": seed,
                "config": cfg,
                "tau_dim": int(diff.tau_dim),
                "cond_dim": int(diff.cond_dim),
                "eta": float(args.eta),
                "diff_state_dict": diff.model.state_dict(),
                "diff_stats": tuple(np.asarray(v, dtype=float) for v in diff.stats),
                "surrogate_state_dict": surrogate.state_dict(),
                "surrogate_stats": tuple(np.asarray(v, dtype=float) for v in surrogate_stats),
            },
            force=args.force,
        )
        print(f"[train_pgcd] logs {diffusion_log_path}, {surrogate_log_path}")
        print(f"[train_pgcd] saved {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
