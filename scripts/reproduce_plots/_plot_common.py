from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = ROOT / "paper_reproduction_results"
# Default output folder.
DEFAULT_FIG_SUBDIR = "figures"
MM = 1 / 25.4

# ---------------------------------------------------------------------------
# Unified palette, taken from ``fig_split_main``.  Every panel in this folder
# reads as one system: a deep blue for our method / the primary series, a
# terracotta for the comparison / secondary series, plus quiet neutrals.
# ---------------------------------------------------------------------------
OURS = "#0F4D92"        # deep blue  -> PG-Diffusion (ours) / primary metric
COMPARE = "#D1845E"     # terracotta -> baseline / secondary metric
FAMILIAR = "#6F91B4"    # medium blue  -> "familiar" test set
UNFAMILIAR = "#D1845E"  # terracotta   -> "unfamiliar" test set
THIRD = "#6F91B4"       # medium blue  -> third metric where needed
OURS_EDGE = "#173D63"   # navy outline used to flag the "ours" element
NEUTRAL_GRAY = "#B8B8B8"

# Retained for backwards compatibility with the shared helpers below.
NEUTRAL = "#5A5A5A"
LIGHT_NEUTRAL = "#D9D9D9"
ACCENT = OURS
BASELINE = COMPARE
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "nature_blue", ["#F7F9FC", "#DCE6F2", "#9DBAD8", "#3775BA", "#0F4D92"]
)

METHOD_DISPLAY_SHORT = {
    "PSO": "PSO",
    "FNN": "FNN",
    "PINN": "PINN",
    "PC-STOMP": "PC-STOMP",
    "CVAE": "CVAE",
    "CondDiff": "CD",
    "PGDiff": "PGCD",
}

METHOD_ORDER = ["PSO", "FNN", "PINN", "PC-STOMP", "CVAE", "CondDiff", "PGDiff"]


def set_paper_style(*, serif: bool = False) -> None:
    # Nature figures use a compact sans-serif system.
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "lines.linewidth": 1.35,
            "lines.markersize": 3.5,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def style_axis(ax: plt.Axes, *, grid: bool = False) -> None:
    """Apply quiet publication styling without decorative chart furniture."""
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(direction="out")
    if grid:
        ax.grid(axis="y", color=LIGHT_NEUTRAL, linewidth=0.45, alpha=0.65)
        ax.set_axisbelow(True)


def add_panel_label(ax: plt.Axes, label: str, *, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color="#111111",
        clip_on=False,
    )


def add_io_args(parser: argparse.ArgumentParser, *, default_stem: str) -> None:
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--fig-dir", type=Path, default=None)
    parser.add_argument("--stem", default=default_stem)


def resolve_fig_dir(results_dir: Path, fig_dir: Path | None) -> Path:
    return (fig_dir or results_dir / DEFAULT_FIG_SUBDIR).resolve()


def save_figure(fig: plt.Figure, fig_dir: Path, stem: str) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    # SVG is primary (editable text), PDF is the manuscript vector version,
    # TIFF is the 600-dpi submission raster, and PNG is a lightweight preview.
    for ext in ("svg", "pdf", "tiff", "png"):
        out = fig_dir / f"{stem}.{ext}"
        dpi = 600 if ext == "tiff" else 300
        fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        outputs.append(out)
    plt.close(fig)
    manifest = fig_dir / "manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text(encoding="utf-8"))
    else:
        old = {"files": []}
    names = set(old.get("files", []))
    names.update(path.name for path in outputs)
    manifest.write_text(json.dumps({"files": sorted(names)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return outputs


def load_model_analysis(results_dir: Path, name: str):
    """Load an analysis of the Table 2 checkpoints written by scripts/reproduce_models/<name>.py."""
    path = results_dir / f"{name}.npz"
    if not path.exists():
        raise SystemExit(f"missing input: {path}\nRun first:\n  python scripts\\reproduce_models\\{name}.py")
    return np.load(path, allow_pickle=True)
