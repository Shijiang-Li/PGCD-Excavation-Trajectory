from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

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


def draw_eta(ax_left, z) -> None:
    eta = z["eta"]
    feas = z["sweep_feas"] * 100
    div = z["sweep_div"]

    # Feasibility and diversity live on very different scales, so the two former
    # panels are merged onto one axes with twin y-scales: feasibility (deep
    # blue, left) versus candidate diversity (terracotta, right).  The shaded
    # band marks the selected operating point.
    ax_right = ax_left.twinx()

    ax_left.axvspan(0.08, 0.12, color="#DCE6F2", alpha=0.7, linewidth=0, zorder=0)
    ax_left.axvline(0.12, color="#5A5A5A", ls=":", lw=0.9, zorder=1)

    l1, = ax_left.plot(eta, feas, "-o", color=OURS, markersize=4.2, markerfacecolor=OURS,
                       markeredgecolor="white", markeredgewidth=0.6, linewidth=2.0,
                       label="Feasibility", zorder=3)
    l2, = ax_right.plot(eta, div, "--s", color=COMPARE, markersize=4.0, markerfacecolor="white",
                        markeredgecolor=COMPARE, markeredgewidth=1.1, linewidth=1.7,
                        label="Diversity", zorder=3)

    ax_left.set_xlabel(r"Physics-guidance weight, $\eta$")
    ax_left.set_ylabel("Feasibility (%)", color=OURS)
    ax_right.set_ylabel("Candidate diversity", color=COMPARE)
    ax_left.tick_params(axis="y", colors=OURS)
    ax_right.tick_params(axis="y", colors=COMPARE)
    ax_left.spines["left"].set_color(OURS)
    ax_right.spines["right"].set_color(COMPARE)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["left"].set_visible(False)

    style_axis(ax_left)
    ax_left.text(0.125, 0.45, "selected", transform=ax_left.get_xaxis_transform(),
                 ha="left", va="bottom", rotation=90, fontsize=6, color="#4D4D4D")
    ax_left.legend(handles=[l1, l2], loc="upper right", handlelength=2.2)


def draw_guide_steps(ax_f, z) -> None:
    k = np.asarray(z["guide_steps"], dtype=float)
    feas = np.asarray(z["guide_step_feas"], dtype=float) * 100.0
    div = np.asarray(z["guide_step_div"], dtype=float)

    # Feasibility (~75-91 %) and diversity (~0.08-0.14) live on different
    # scales, so they share one panel through twin y-axes: feasibility (blue,
    # left) versus candidate diversity (terracotta, right).
    ax_d = ax_f.twinx()

    # Mark the selected operating point exactly as in panel A: a light
    # shaded band, a dotted vertical line and a rotated "selected" label.
    ax_f.axvspan(34, 40, color="#DCE6F2", alpha=0.7, linewidth=0, zorder=0)
    ax_f.axvline(40, color="#5A5A5A", ls=":", lw=0.9, zorder=1)

    lf, = ax_f.plot(k, feas, "-o", color=OURS, markersize=4.2, markerfacecolor=OURS,
                    markeredgecolor="white", markeredgewidth=0.6, linewidth=2.0,
                    label="Feasibility (%)", zorder=4)
    ld, = ax_d.plot(k, div, "--s", color=COMPARE, markersize=4.0, markerfacecolor="white",
                    markeredgecolor=COMPARE, markeredgewidth=1.1, linewidth=1.7,
                    label="Diversity", zorder=3)

    ax_f.text(42, 0.45, "selected", transform=ax_f.get_xaxis_transform(),
              ha="left", va="bottom", rotation=90, fontsize=6, color="#4D4D4D")

    ax_f.set_xticks(k)
    ax_f.set_xlabel("Guided reverse steps, $K$")
    ax_f.set_ylabel("Feasibility (%)", color=OURS)
    ax_d.set_ylabel("Candidate diversity", color=COMPARE)
    ax_f.tick_params(axis="y", colors=OURS)
    ax_d.tick_params(axis="y", colors=COMPARE)
    ax_f.spines["left"].set_color(OURS)
    ax_d.spines["right"].set_color(COMPARE)
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["left"].set_visible(False)

    style_axis(ax_f)
    ax_f.legend(handles=[lf, ld], loc="upper right", bbox_to_anchor=(0.98, 0.95), handlelength=2.4)


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw the eta sweep (a) and guided-step K sweep (b) side by side.")
    add_io_args(parser, default_stem="fig_split_eta_steps")
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    z = load_model_analysis(results_dir, "guidance_analysis")
    set_paper_style()

    # Two 75 x 60 mm single-panel layouts next to each other.
    fig, axes = plt.subplots(1, 2, figsize=(150 * MM, 60 * MM))
    fig.subplots_adjust(left=0.06, right=0.935, bottom=0.17, top=0.90, wspace=0.42)
    draw_eta(axes[0], z)
    draw_guide_steps(axes[1], z)
    for label, ax in zip("AB", axes):
        add_panel_label(ax, label, x=-0.14, y=1.04)

    outputs = save_figure(fig, resolve_fig_dir(results_dir, args.fig_dir), args.stem)
    print("[plot_fig_split_eta_steps] " + ", ".join(str(path) for path in outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
