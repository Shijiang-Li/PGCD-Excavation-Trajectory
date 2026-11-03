from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from _plot_common import (
    COMPARE,
    MM,
    OURS,
    add_io_args,
    add_panel_label,
    load_model_analysis,
    resolve_fig_dir,
    save_figure,
    set_paper_style,
    style_axis,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw the paper-style selection-budget figure.")
    add_io_args(parser, default_stem="fig_split_selection")
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    z = load_model_analysis(results_dir, "selection_budget")
    set_paper_style()
    ns = z["NS"]

    # Strong, unified two-series identity: terracotta dashed circles for the
    # baseline, deep-blue solid squares for our method.  Differentiating by
    # colour *and* line style keeps the two curves legible even where they
    # cross or overlap.
    base_kw = dict(color=COMPARE, linestyle="--", marker="o", markersize=4.0,
                   markerfacecolor="white", markeredgecolor=COMPARE, markeredgewidth=1.1,
                   linewidth=1.6, label="CD")
    ours_kw = dict(color=OURS, linestyle="-", marker="s", markersize=4.2,
                   markerfacecolor=OURS, markeredgecolor="white", markeredgewidth=0.6,
                   linewidth=2.0, label="PGCD")

    fig, axes = plt.subplots(2, 2, figsize=(150 * MM, 105 * MM), sharex="col")
    fig.subplots_adjust(hspace=0.30, wspace=0.32, left=0.11, right=0.98, bottom=0.15, top=0.88)
    for row, (tag, label) in enumerate([("fam", "In-distribution"), ("ood", "Unseen-material")]):
        ax0, ax1 = axes[row, 0], axes[row, 1]
        ax0.plot(ns, z[f"sel_{tag}_CD_succ"] * 100, **base_kw)
        ax0.plot(ns, z[f"sel_{tag}_PG_succ"] * 100, **ours_kw)
        ax0.set_xscale("log", base=2)
        ax0.set_ylabel(f"{label}\nsuccess (%)", linespacing=1.6)
        ax0.axvline(1, color="#A8A8A8", ls=":", lw=0.8, zorder=0)
        ax1.plot(ns, z[f"sel_{tag}_CD_gap"] * 100, **base_kw)
        ax1.plot(ns, z[f"sel_{tag}_PG_gap"] * 100, **ours_kw)
        ax1.set_xscale("log", base=2)
        ax1.set_ylabel("Gap to PSO (%)")
        style_axis(ax0, grid=False)
        style_axis(ax1, grid=False)
        if row == 1:
            ax0.set_xlabel("Candidate budget, $N$")
            ax1.set_xlabel("Candidate budget, $N$")

    axes[0, 0].set_title("Plan success", loc="left", fontweight="bold", pad=5)
    axes[0, 1].set_title("Best-of-$N$ optimality gap", loc="left", fontweight="bold", pad=5)
    for label, ax in zip("ABCD", axes.flat):
        add_panel_label(ax, label)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.55, 0.985), ncol=2, handlelength=2.4)
    outputs = save_figure(fig, resolve_fig_dir(results_dir, args.fig_dir), args.stem)
    print("[plot_fig_split_selection] " + ", ".join(str(path) for path in outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
