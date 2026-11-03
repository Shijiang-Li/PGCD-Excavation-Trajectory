from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _common import DEFAULT_RESULTS_DIR, load_or_build_loo_data, result_cache_path, setup_environment, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 01: generate or load the PSO condition dataset.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--n-idx", type=int, default=100, help="conditions per material; paper default is 100")
    parser.add_argument("--pop", type=int, default=40, help="PSO population; paper default is 40")
    parser.add_argument("--iters", type=int, default=15, help="PSO iterations; paper default is 15")
    parser.add_argument("--force", action="store_true", help="delete the matching cache before regenerating")
    args = parser.parse_args()
    results_dir = setup_environment(args.results_dir)
    cache = result_cache_path(results_dir, args.n_idx, args.pop, args.iters)
    if args.force and cache.exists():
        cache.unlink()
        print(f"deleted cache: {cache}")

    data = load_or_build_loo_data(results_dir, args.n_idx, args.pop, args.iters, cache=True)
    finite = np.isfinite(data["samp_J"])
    manifest = {
        "cache": str(cache),
        "conditions": int(len(data["conds"])),
        "samples": int(len(data["samp_J"])),
        "finite_samples": int(finite.sum()),
        "n_idx": int(args.n_idx),
        "pso_pop": int(args.pop),
        "pso_iters": int(args.iters),
    }
    write_json(results_dir / "step01_dataset_manifest.json", manifest)
    print(f"dataset_cache: {cache}")
    print(f"conditions: {manifest['conditions']}")
    print(f"samples: {manifest['samples']}")
    print(f"finite_samples: {manifest['finite_samples']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
