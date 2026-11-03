from __future__ import annotations

import argparse
from math import comb
from pathlib import Path

import numpy as np

from _common import (
    DEFAULT_RESULTS_DIR,
    add_common_test_args,
    checkpoint_path,
    load_pso_reference,
    load_table2_data,
    pso_reference_for_condition,
    pso_reference_path,
    setup,
)
from test_conddiff import load_conddiff
from test_pgcd import load_pgcd

NS = [1, 2, 4, 8, 16, 32, 64, 128]


def budget_curves(J: np.ndarray, pso_best: float, rng: np.random.RandomState, draws: int):
    """Success and best-of-N gap for N candidates drawn without replacement from a pool of M."""
    M = len(J)
    ok = np.isfinite(J)
    k = int(ok.sum())
    succ, gap = [], []
    for N in NS:
        # Exact: probability that at least one of N candidates is feasible.
        succ.append(1.0 - comb(M - k, N) / comb(M, N))
        mins = []
        for _ in range(draws):
            js = J[rng.choice(M, N, replace=False)]
            if np.isfinite(js).any():
                mins.append(js[np.isfinite(js)].min())
        gap.append((np.mean(mins) - pso_best) / abs(pso_best) if mins else np.nan)
    return np.array(succ), np.array(gap)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Candidate-budget sweep (Fig. 6) with the Table 2 checkpoints on the Table 2 test sets.")
    add_common_test_args(parser)
    parser.add_argument("--M", type=int, default=256, help="candidate pool per condition")
    parser.add_argument("--draws", type=int, default=200, help="random subsets per N for the gap")
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_DIR / "selection_budget.npz")
    args = parser.parse_args()
    if args.sample_seed is None:
        args.sample_seed = 0
    setup(args.results_dir)
    data, _, test_sets = load_table2_data(args)
    pso_cache = load_pso_reference(args.pso_reference or pso_reference_path(args.run_dir))
    rng = np.random.RandomState(args.sample_seed)

    out = {"NS": np.array(NS), "seeds": np.array(args.seeds), "M": args.M}
    for method, key, tag in (("CondDiff", "conddiff", "CD"), ("PGDiff", "pgcd", "PG")):
        per_seed = {t: {"succ": [], "gap": [], "feas": []} for t in test_sets}
        for seed in args.seeds:
            if key == "pgcd":
                model, guidance, eta = load_pgcd(checkpoint_path(args.run_dir, key, seed))
                sample = lambda c: model.sample(c, n=args.M, guidance=guidance, eta=eta)
            else:
                model = load_conddiff(checkpoint_path(args.run_dir, key, seed))
                sample = lambda c: model.sample(c, n=args.M)
            for test_set, cids in test_sets.items():
                S, G, F = [], [], []
                for cid in cids:
                    ref = pso_reference_for_condition(data, int(cid), pso_cache, pso_pop=args.pso_pop, pso_iter=args.pso_iter)
                    scene = ref["scene"]
                    J = scene.J_chunked(scene.denormalize(np.asarray(sample(data["conds"][cid]), float)))
                    s, g = budget_curves(J, ref["best_J"], rng, args.draws)
                    S.append(s); G.append(g); F.append(np.isfinite(J).mean())
                per_seed[test_set]["succ"].append(np.mean(S, 0))
                per_seed[test_set]["gap"].append(np.nanmean(G, 0))
                per_seed[test_set]["feas"].append(np.mean(F))
                print(f"[selection_budget] {method} seed {seed} {test_set}: feas {np.mean(F) * 100:.1f}%  "
                      f"succ(N=1,8) {np.mean(S, 0)[0] * 100:.1f}/{np.mean(S, 0)[3] * 100:.1f}", flush=True)
        for test_set, rec in per_seed.items():
            t = "fam" if test_set == "familiar" else "ood"
            # Average over checkpoint seeds, as in Table 2.
            out[f"sel_{t}_{tag}_succ"] = np.mean(rec["succ"], 0)
            out[f"sel_{t}_{tag}_gap"] = np.mean(rec["gap"], 0)
            out[f"sel_{t}_{tag}_feas"] = float(np.mean(rec["feas"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **out)
    print(f"[selection_budget] saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
