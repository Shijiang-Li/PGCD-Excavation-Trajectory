from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = ROOT / "paper_reproduction_results"
DEFAULT_RUN_DIR = ROOT / "artifacts" / "model_runs" / "table2_n8"
TABLE2_METHODS = ["PSO", "FNN", "PINN", "PC-STOMP", "CVAE", "CondDiff", "PGDiff"]
TABLE2_KEYS = ["succ", "best_J", "feas_rate", "viol_rate", "gap", "div", "lat"]
ZERO = np.zeros(4)
ONE = np.ones(4)
ETA_DEFAULT = 0.12


def ensure_project_root_on_path() -> None:
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def setup(results_dir: Path = DEFAULT_RESULTS_DIR) -> Path:
    ensure_project_root_on_path()
    results_dir = results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PGCD_RESULTS_DIR"] = str(results_dir)
    pass  # platform root is resolved by pgcd_paper.config
    return results_dir


def normalize_paths(args) -> None:
    if hasattr(args, "results_dir"):
        args.results_dir = args.results_dir.resolve()
    if hasattr(args, "run_dir"):
        args.run_dir = args.run_dir.resolve()
    if hasattr(args, "pso_reference") and args.pso_reference is not None:
        args.pso_reference = args.pso_reference.resolve()


def parse_seed_list(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("seed list cannot be empty")
    return seeds


def write_json(path: Path, payload: dict, *, force: bool = True) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {path}; pass --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_table2_data(args):
    normalize_paths(args)
    setup(args.results_dir)
    from pgcd_paper import cond_data as CD
    from pgcd_paper.evaluation.selection import select_condition_subset

    data = CD.build_loo(n_idx=args.n_idx, pop=args.data_pop, iters=args.data_iters, cache=True)
    split = CD.split_iood(data, ood_material=args.ood)
    meta_codes = data["cond_meta"][:, 0].astype(int)
    subset_count_fam = None if args.subset_policy == "full" else args.n_fam
    subset_count_ood = None if args.subset_policy == "full" else args.n_ood
    policy = "stratified" if args.subset_policy == "full" else args.subset_policy
    fam = list(select_condition_subset(split["fam_test"], meta_codes[split["fam_test"]], subset_count_fam, policy, seed=0))
    ood = list(select_condition_subset(split["ood_test"], meta_codes[split["ood_test"]], subset_count_ood, policy, seed=0))
    if getattr(args, "sample_seed", None) is not None:
        from pgcd_paper.randomness import set_seed

        set_seed(args.sample_seed)
    return data, split, {"familiar": fam, "unfamiliar": ood}


def add_common_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--seeds", type=parse_seed_list, default=parse_seed_list("0,1"))
    parser.add_argument("--n-idx", type=int, default=100)
    parser.add_argument("--data-pop", type=int, default=40)
    parser.add_argument("--data-iters", type=int, default=15)
    parser.add_argument("--ood", default="soil")
    parser.add_argument("--n-fam", type=int, default=48)
    parser.add_argument("--n-ood", type=int, default=60)
    parser.add_argument("--subset-policy", choices=["legacy-prefix", "stratified", "full"], default="full")
    parser.add_argument("--quick", action="store_true", help="small run for entry-point validation only")
    parser.add_argument("--force", action="store_true")


def add_common_test_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--seeds", type=parse_seed_list, default=parse_seed_list("0,1"))
    parser.add_argument("--candidate-count", "--n-cand", dest="candidate_count", type=int, default=8)
    parser.add_argument("--n-idx", type=int, default=100)
    parser.add_argument("--data-pop", type=int, default=40)
    parser.add_argument("--data-iters", type=int, default=15)
    parser.add_argument("--ood", default="soil")
    parser.add_argument("--n-fam", type=int, default=48)
    parser.add_argument("--n-ood", type=int, default=60)
    parser.add_argument("--subset-policy", choices=["legacy-prefix", "stratified", "full"], default="full")
    parser.add_argument("--pso-pop", type=int, default=40)
    parser.add_argument("--pso-iter", type=int, default=15)
    parser.add_argument("--pso-reference", type=Path, default=None)
    parser.add_argument("--sample-seed", type=int, default=None, help="fix the candidate-sampling RNG (default: unseeded)")
    parser.add_argument("--quick", action="store_true", help="small run for entry-point validation only")
    parser.add_argument("--force", action="store_true")


def apply_quick_data_args(args) -> None:
    if args.quick:
        args.n_idx = 4
        args.data_pop = 8
        args.data_iters = 3
        args.n_fam = 2
        args.n_ood = 2
        if hasattr(args, "pso_pop"):
            args.pso_pop = 8
        if hasattr(args, "pso_iter"):
            args.pso_iter = 3


def finite_mean(values) -> float:
    arr = np.asarray(list(values), dtype=float)
    ok = np.isfinite(arr)
    return float(arr[ok].mean()) if ok.any() else float("nan")


def checkpoint_path(run_dir: Path, model_key: str, seed: int) -> Path:
    return run_dir.resolve() / model_key / "checkpoints" / f"seed_{seed}.pt"


def save_checkpoint(path: Path, payload: dict, *, force: bool = False) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {path}; pass --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported checkpoint schema: {path}")
    return payload


def training_config(args, model_key: str, extra: dict) -> dict:
    return {
        "model": model_key,
        "dataset": {"n_idx": args.n_idx, "pso_pop": args.data_pop, "pso_iters": args.data_iters},
        "split": {
            "ood": args.ood,
            "n_fam": args.n_fam,
            "n_ood": args.n_ood,
            "subset_policy": args.subset_policy,
        },
        "hyperparameters": extra,
        "hyperparameter_source": (
            "Defaults follow the PGCD paper reproduction code: Table 2 candidate count n=8, "
            "diffusion/surrogate/CVAE 5000 steps, FNN/PINN 4000 steps, eta=0.12. "
            "Quick mode intentionally reduces these values for script validation only."
        ),
    }


def train_arrays(data, train_cids):
    tau = np.asarray(data["samp_tau"], dtype=float)
    cond = np.asarray(data["conds"][data["samp_cond"]], dtype=float)
    J = np.asarray(data["samp_J"], dtype=float)
    train_mask = np.isin(data["samp_cond"], train_cids)
    feasible = np.isfinite(J)
    return tau, cond, J, train_mask, feasible, train_mask & feasible


def stats_from_training_feasible(tau: np.ndarray, cond: np.ndarray, train_feasible: np.ndarray):
    tm = tau[train_feasible].mean(0)
    ts = tau[train_feasible].std(0) + 1e-9
    cm = cond[train_feasible].mean(0)
    cs = cond[train_feasible].std(0) + 1e-9
    return tm, ts, cm, cs


def load_pso_reference(path: Path | None) -> dict[int, dict]:
    if path is None or not path.exists():
        return {}
    rows = {}
    with path.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows[int(row["condition_id"])] = {
                "best_J": float(row["best_J"]),
                "lat": float(row.get("lat", 0.0)),
                "test_set": row.get("test_set", ""),
            }
    return rows


def pso_reference_path(run_dir: Path) -> Path:
    return run_dir.resolve() / "pso" / "condition_reference.csv"


def pso_reference_for_condition(data, condition_id: int, cache: dict[int, dict], *, pso_pop: int, pso_iter: int):
    from pgcd_paper import cond_data as CD
    from pgcd_paper.baselines import run_pso

    if condition_id not in cache:
        scene = CD.reconstruct_scene(data["cond_meta"][condition_id])
        np.random.seed(condition_id)  # reproducible reference for conditions not yet in condition_reference.csv
        xs, values, latency = run_pso(scene, pop=pso_pop, max_iter=pso_iter, n_runs=1)
        cache[condition_id] = {
            "best_J": float(np.min(values)),
            "lat": float(latency),
            "scene": scene,
            "best_tau": np.asarray(xs[0], dtype=float),
        }
    elif "scene" not in cache[condition_id]:
        cache[condition_id]["scene"] = CD.reconstruct_scene(data["cond_meta"][condition_id])
    return cache[condition_id]


def evaluate_method(
    *,
    data,
    test_sets: dict[str, list[int]],
    method: str,
    model_key: str | None = None,
    seed: int,
    run_dir: Path,
    candidate_count: int,
    pso_cache: dict[int, dict],
    pso_pop: int,
    pso_iter: int,
    evaluator: Callable,
    force: bool,
) -> dict[str, dict[str, float]]:
    from pgcd_paper import metrics as MM

    per_condition_rows: list[dict] = []
    summaries: dict[str, dict[str, float]] = {}
    for test_set, condition_ids in test_sets.items():
        rec = {"best_J": [], "feas_rate": [], "div": [], "lat": [], "gap": []}
        for condition_id in condition_ids:
            ref = pso_reference_for_condition(data, int(condition_id), pso_cache, pso_pop=pso_pop, pso_iter=pso_iter)
            scene = ref["scene"]
            cond_raw = data["conds"][condition_id]
            best_J, feas_rate, div, latency = evaluator(scene, cond_raw, candidate_count, seed, int(condition_id))
            gap = (best_J - ref["best_J"]) / abs(ref["best_J"]) if np.isfinite(best_J) else float("nan")
            rec["best_J"].append(best_J)
            rec["feas_rate"].append(feas_rate)
            rec["div"].append(div)
            rec["lat"].append(latency)
            rec["gap"].append(gap)
            per_condition_rows.append(
                {
                    "test_set": test_set,
                    "condition_id": int(condition_id),
                    "method": method,
                    "seed": seed,
                    "best_J": best_J,
                    "pso_best_J": ref["best_J"],
                    "feas_rate": feas_rate,
                    "viol_rate": 1.0 - feas_rate,
                    "gap": gap,
                    "div": div,
                    "lat": latency,
                    "success": int(np.isfinite(best_J)),
                }
            )
        best_arr = np.asarray(rec["best_J"], dtype=float)
        feas = float(np.mean(rec["feas_rate"])) if rec["feas_rate"] else float("nan")
        summaries[test_set] = {
            "succ": float(np.isfinite(best_arr).mean()) if len(best_arr) else float("nan"),
            "best_J": finite_mean(rec["best_J"]),
            "feas_rate": feas,
            "viol_rate": 1.0 - feas if np.isfinite(feas) else float("nan"),
            "gap": finite_mean(rec["gap"]),
            "div": float(np.mean(rec["div"])) if rec["div"] else float("nan"),
            "lat": finite_mean(rec["lat"]),
        }

    result_key = model_key or method.lower().replace("-", "_")
    seed_dir = run_dir.resolve() / result_key / "results" / f"seed_{seed}"
    write_csv(
        seed_dir / "per_condition.csv",
        per_condition_rows,
        [
            "test_set",
            "condition_id",
            "method",
            "seed",
            "best_J",
            "pso_best_J",
            "feas_rate",
            "viol_rate",
            "gap",
            "div",
            "lat",
            "success",
        ],
    )
    write_json(seed_dir / "summary.json", {"method": method, "seed": seed, "candidate_count": candidate_count, "summary": summaries}, force=True)
    return summaries


def aggregate_seed_summaries(run_dir: Path, model_key: str, method: str, seeds: list[int], force: bool = True) -> Path:
    rows: list[dict] = []
    summaries_by_seed = []
    for seed in seeds:
        path = run_dir.resolve() / model_key / "results" / f"seed_{seed}" / "summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        summaries_by_seed.append(payload)
    test_sets = list(summaries_by_seed[0]["summary"].keys())
    for test_set in test_sets:
        row = {"test_set": test_set, "method": method}
        for key in TABLE2_KEYS:
            row[key] = finite_mean(payload["summary"][test_set][key] for payload in summaries_by_seed)
        rows.append(row)
    out = run_dir.resolve() / model_key / "results" / "summary.csv"
    write_csv(out, rows, ["test_set", "method", *TABLE2_KEYS])
    write_json(out.with_suffix(".json"), {"method": method, "seeds": seeds, "rows": rows}, force=force)
    return out


def deterministic_eval_tau_norm(tau_norm: np.ndarray, scene) -> tuple[float, float, float]:
    J = scene.J_chunked(scene.denormalize(np.asarray(tau_norm, dtype=float).reshape(1, -1)))
    ok = bool(np.isfinite(J[0]))
    return (float(J[0]) if ok else float("inf"), float(ok), 0.0)


def generative_eval_tau_norm(tau_norm: np.ndarray, scene) -> tuple[float, float, float]:
    from pgcd_paper import metrics as MM

    tau_norm = np.asarray(tau_norm, dtype=float)
    J = scene.J_chunked(scene.denormalize(tau_norm))
    finite = np.isfinite(J)
    best = float(J[finite].min()) if finite.any() else float("inf")
    div = MM.diversity(tau_norm[finite] if int(finite.sum()) > 1 else tau_norm, ZERO, ONE)
    return best, float(finite.mean()), div
