from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _plot_common import ACCENT, MM, NEUTRAL_GRAY, ROOT, add_io_args, resolve_fig_dir, save_figure, set_paper_style, style_axis


if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_CASES = [
    ("sand", 1),
    ("soil", 1),
    ("stone", 1),
    ("small_stone", 1),
    ("sand", 2),
    ("stone", 2),
]


def _parse_cases(raw: str) -> list[tuple[str, int]]:
    cases: list[tuple[str, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        material, index = item.split(":")
        cases.append((material, int(index)))
    if not cases:
        raise argparse.ArgumentTypeError("at least one case is required")
    return cases


def _history_cache_path(results_dir: Path, pop: int, iters: int) -> Path:
    return results_dir / f"pso_search_history_{pop}x{iters}.npz"


def _run_one(material: str, file_index: int, *, pop: int, iters: int, seed: int) -> np.ndarray:
    from pgcd_paper import config as C
    from pgcd_paper import scene as SC
    from Trajectory_planning.Cartesian_Space.Cubic_Spline.PSO_multi import PSO

    scene = SC.Scene(material, file_index, end_angle=C.END_ANGLE)
    np.random.seed(seed)
    pso = PSO(func=scene.objective_callable(), dim=SC.TAU_DIM, pop=pop, max_iter=iters, lb=scene.lb, ub=scene.ub)
    _, _, history = pso.run()
    history = np.asarray(history, dtype=float)
    return np.fmin.accumulate(history)


def _write_history_csv(path: Path, labels: list[str], curves: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["iteration", *labels, "mean", "min", "max"])
        iterations = np.arange(1, curves.shape[1] + 1)
        for i, iteration in enumerate(iterations):
            row = curves[:, i]
            writer.writerow([int(iteration), *[f"{v:.8f}" for v in row], f"{row.mean():.8f}", f"{row.min():.8f}", f"{row.max():.8f}"])


def load_or_generate(results_dir: Path, cases: list[tuple[str, int]], *, pop: int, iters: int, force: bool):
    cache = _history_cache_path(results_dir, pop, iters)
    csv_path = cache.with_suffix(".csv")
    if cache.exists() and not force:
        payload = np.load(cache, allow_pickle=True)
        labels = [str(x) for x in payload["labels"]]
        curves = np.asarray(payload["curves"], dtype=float)
        return labels, curves, cache, csv_path

    old_cwd = Path.cwd()
    labels: list[str] = []
    curves: list[np.ndarray] = []
    try:
        for idx, (material, file_index) in enumerate(cases):
            try:
                curve = _run_one(material, file_index, pop=pop, iters=iters, seed=100 + idx)
            except Exception as exc:
                print(f"[plot_fig_pgcd_pso] skip {material}#{file_index}: {type(exc).__name__}: {exc}", flush=True)
                continue
            labels.append(f"{material}#{file_index}")
            curves.append(curve)
            print(f"[plot_fig_pgcd_pso] {labels[-1]}: {curve[0]:.4f} -> {curve[-1]:.4f}", flush=True)
    finally:
        os.chdir(old_cwd)

    if not curves:
        raise SystemExit("no PSO histories were generated; check source data/model paths")

    length = min(len(curve) for curve in curves)
    curve_array = np.asarray([curve[:length] for curve in curves], dtype=float)
    results_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache, labels=np.asarray(labels), curves=curve_array, pop=int(pop), iters=int(iters))
    _write_history_csv(csv_path, labels, curve_array)
    return labels, curve_array, cache, csv_path


def draw(labels: list[str], curves: np.ndarray):
    set_paper_style()
    iterations = np.arange(1, curves.shape[1] + 1)
    fig, ax = plt.subplots(figsize=(89 * MM, 58 * MM))
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.19, top=0.95)
    for curve in curves:
        ax.plot(iterations, curve, color=NEUTRAL_GRAY, lw=0.7, alpha=0.75, zorder=1)
    ax.fill_between(
        iterations,
        curves.min(axis=0),
        curves.max(axis=0),
        color="#DCE6F2",
        alpha=0.75,
        linewidth=0,
        label="Range",
        zorder=0,
    )
    ax.plot(iterations, curves.mean(axis=0), color=ACCENT, lw=1.8, label=f"Mean ($n={len(labels)}$ conditions)", zorder=2)
    ax.set_xlabel("PSO iteration")
    ax.set_ylabel("Best objective, $J$")
    ax.legend(loc="upper right")
    ax.set_xlim(1, int(iterations[-1]))
    style_axis(ax, grid=True)
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw the paper-style PSO search-record figure.")
    add_io_args(parser, default_stem="fig_pgcd_pso")
    parser.add_argument("--pop", type=int, default=40)
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--cases", type=_parse_cases, default=DEFAULT_CASES, help="comma list like sand:1,soil:1,stone:1")
    parser.add_argument("--force", action="store_true", help="Regenerate PSO histories even if cache exists.")
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    fig_dir = resolve_fig_dir(results_dir, args.fig_dir)
    labels, curves, cache, csv_path = load_or_generate(results_dir, args.cases, pop=args.pop, iters=args.iters, force=args.force)
    outputs = save_figure(draw(labels, curves), fig_dir, args.stem)
    print("[plot_fig_pgcd_pso] " + ", ".join(str(path) for path in outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
