from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# Figs. 4, 5, 6, 7 (eta and K sweeps as panels a, b) and 8.
SCRIPTS = [
    "plot_fig_pgcd_pso.py",
    "plot_fig_split_main.py",
    "plot_fig_split_selection.py",
    "plot_fig_split_eta_steps.py",
    "plot_fig_split_ablation.py",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw Figs. 4-8.")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "paper_reproduction_results")
    parser.add_argument("--fig-dir", type=Path, default=None)
    parser.add_argument("--clean", action="store_true", help="Remove the target figure directory before regenerating.")
    args = parser.parse_args()
    fig_dir = (args.fig_dir or args.results_dir / "figures").resolve()
    if args.clean and fig_dir.exists():
        project_root = ROOT.resolve()
        if project_root not in fig_dir.parents:
            raise SystemExit(f"refusing to clean outside project: {fig_dir}")
        shutil.rmtree(fig_dir)
    for script in SCRIPTS:
        cmd = [sys.executable, str(HERE / script), "--results-dir", str(args.results_dir)]
        cmd += ["--fig-dir", str(fig_dir)]
        print("+ " + " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, cwd=ROOT)
        if proc.returncode != 0:
            return proc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
