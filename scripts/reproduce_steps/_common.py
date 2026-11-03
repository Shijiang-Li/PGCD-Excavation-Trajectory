from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = ROOT / "paper_reproduction_results"
DEFAULT_CHECKPOINT_DIR = ROOT / "artifacts" / "checkpoints" / "table2_n8"


def ensure_project_root_on_path() -> None:
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def setup_environment(results_dir: Path) -> Path:
    ensure_project_root_on_path()
    results_dir = results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PGCD_RESULTS_DIR"] = str(results_dir)
    pass  # platform root is resolved by pgcd_paper.config
    return results_dir


def parse_seed_list(raw: str) -> list[int]:
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise argparse.ArgumentTypeError("seed list cannot be empty")
    return values


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def result_cache_path(results_dir: Path, n_idx: int, pop: int, iters: int) -> Path:
    return results_dir / f"cond_data_loo_{n_idx}_{pop}x{iters}.npz"


def load_or_build_loo_data(results_dir: Path, n_idx: int, pop: int, iters: int, *, cache: bool = True):
    setup_environment(results_dir)
    from pgcd_paper import cond_data as CD

    return CD.build_loo(n_idx=n_idx, pop=pop, iters=iters, cache=cache)


def table2_condition_split(
    data,
    *,
    ood_material: str,
    n_fam: int,
    n_ood: int,
    subset_policy: str,
):
    from pgcd_paper import cond_data as CD
    from pgcd_paper.evaluation.selection import select_condition_subset

    split = CD.split_iood(data, ood_material=ood_material)
    meta_codes = data["cond_meta"][:, 0].astype(int)
    subset_count_fam = None if subset_policy == "full" else n_fam
    subset_count_ood = None if subset_policy == "full" else n_ood
    policy = "stratified" if subset_policy == "full" else subset_policy
    fam = list(select_condition_subset(split["fam_test"], meta_codes[split["fam_test"]], subset_count_fam, policy, seed=0))
    ood = list(select_condition_subset(split["ood_test"], meta_codes[split["ood_test"]], subset_count_ood, policy, seed=0))
    return split, fam, ood


def finite_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    ok = np.isfinite(arr)
    return float(arr[ok].mean()) if ok.any() else float("nan")


def run_module(module: str, args: list[str], results_dir: Path) -> None:
    setup_environment(results_dir)
    cmd = [sys.executable, "-m", module, *args]
    env = os.environ.copy()
    print("+ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, env=env)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
