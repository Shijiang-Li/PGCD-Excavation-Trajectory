from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _common import (
    DEFAULT_RESULTS_DIR,
    add_common_test_args,
    checkpoint_path,
    generative_eval_tau_norm,
    load_checkpoint,
    load_table2_data,
    save_checkpoint,
    setup,
    train_arrays,
)
from test_pgcd import load_pgcd

ETAS = [0.0, 0.04, 0.08, 0.12, 0.20, 0.30, 0.45]
GUIDE_STEPS = [0, 10, 20, 40, 70, 100]


def load_feas_guidance(run_dir: Path, seed: int, diff, data, split, steps: int):
    """Feasibility-only surrogate for the ablation, trained once per seed on the Table 2 training split."""
    from pgcd_paper.physics_cost import FeasSurrogate, make_feas_guidance, train_feas_surrogate
    from pgcd_paper.randomness import set_seed

    path = run_dir.resolve() / "pgcd" / "feas_surrogate" / f"seed_{seed}.pt"
    if path.exists():
        payload = load_checkpoint(path)
        model = FeasSurrogate()
        model.load_state_dict(payload["state_dict"], strict=True)
    else:
        tau, cond, J, train_mask, feasible, _ = train_arrays(data, split["train"])
        set_seed(seed)
        model = train_feas_surrogate(diff.std_tau(tau[train_mask]), diff.std_cond(cond[train_mask]),
                                     (~feasible[train_mask]).astype(float), steps=steps, verbose=False)
        save_checkpoint(path, {"schema_version": 1, "seed": seed, "steps": steps, "state_dict": model.state_dict()})
    model.eval()
    return make_feas_guidance(model)


def evaluate(diff, data, cids, guidance, eta, n_cand, guide_last, sample_seed):
    from pgcd_paper import cond_data as CD
    from pgcd_paper.randomness import set_seed

    # Same sampling noise for every setting, so identical settings (eta = 0, K = 0, no guidance) agree exactly.
    set_seed(sample_seed)
    feas, best, div = [], [], []
    for cid in cids:
        scene = CD.reconstruct_scene(data["cond_meta"][cid])
        tau = diff.sample(data["conds"][cid], n=n_cand, guidance=guidance, eta=eta, guide_last=guide_last)
        # Same metrics as Table 2 (feasibility, best feasible J, Euclidean diversity of feasible candidates).
        best_J, feas_rate, d = generative_eval_tau_norm(tau, scene)
        feas.append(feas_rate); div.append(d)
        best.append(best_J if np.isfinite(best_J) else np.nan)
    return float(np.mean(feas)), float(np.nanmean(best)), float(np.mean(div))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Guidance-strength (eta) and guided-step (K) sweeps and guidance-term ablation (Figs. 7, 8) "
        "with the Table 2 PGCD checkpoints on the validation conditions.")
    add_common_test_args(parser)
    parser.set_defaults(candidate_count=96)  # candidates per validation condition (--n-cand)
    parser.add_argument("--feas-steps", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_DIR / "guidance_analysis.npz")
    args = parser.parse_args()
    if args.sample_seed is None:
        args.sample_seed = 0
    setup(args.results_dir)
    data, split, _ = load_table2_data(args)
    val = [int(c) for c in split["val"]]
    print(f"[guidance] {len(val)} validation conditions, {args.candidate_count} candidates each, seeds {args.seeds}", flush=True)

    rec = {k: [] for k in ("sweep", "steps", "modes")}
    for seed in args.seeds:
        diff, guide_full, eta_main = load_pgcd(checkpoint_path(args.run_dir, "pgcd", seed))
        guide_feas = load_feas_guidance(args.run_dir, seed, diff, data, split, args.feas_steps)
        run = lambda g, e, k: evaluate(diff, data, val, g, e, args.candidate_count, k, args.sample_seed)
        rec["sweep"].append([run(guide_full if e > 0 else None, e, 40) for e in ETAS])
        rec["steps"].append([run(guide_full if k > 0 else None, eta_main, k) for k in GUIDE_STEPS])
        rec["modes"].append([run(g, e, 40) for g, e in ((None, 0.0), (guide_feas, eta_main), (guide_full, eta_main))])
        for name, rows in rec.items():
            print(f"[guidance] seed {seed} {name}: " + "  ".join(f"{f * 100:.1f}/{b:.3f}/{d:.3f}" for f, b, d in rows[-1]),
                  flush=True)

    # Average over the checkpoint seeds, as in Table 2: [..., (feas, best_J, div)].
    sweep, steps, modes = (np.mean(rec[k], axis=0) for k in ("sweep", "steps", "modes"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output,
             eta=np.array(ETAS), sweep_feas=sweep[:, 0], sweep_best=sweep[:, 1], sweep_div=sweep[:, 2],
             guide_steps=np.array(GUIDE_STEPS), guide_step_feas=steps[:, 0], guide_step_best=steps[:, 1],
             guide_step_div=steps[:, 2],
             mode_names=np.array(["none", "feasibility", "feas+perf(full)"]),
             mode_feas=modes[:, 0], mode_best=modes[:, 1], mode_div=modes[:, 2],
             seeds=np.array(args.seeds), n_val=len(val), n_cand=args.candidate_count)
    print(f"[guidance] saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
