from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from _plot_common import (
    COMPARE,
    FAMILIAR,
    MM,
    NEUTRAL_GRAY,
    OURS_EDGE,
    add_io_args,
    add_panel_label,
    load_model_analysis,
    resolve_fig_dir,
    save_figure,
    set_paper_style,
    style_axis,
)


def _add_yaxis_break(ax) -> None:
    """Draw the standard // break symbol on the left spine near the base,
    signalling that the bars sit on a truncated (non-zero) y-axis."""
    d = 0.010
    kwargs = dict(transform=ax.transAxes, color="#333333", lw=0.7, clip_on=False, zorder=5)
    for y in (0.03, 0.055):
        ax.plot([-d, d], [y - d, y + d], **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw the paper-style guidance ablation figure (bar charts).")
    add_io_args(parser, default_stem="fig_split_ablation")
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    z = load_model_analysis(results_dir, "guidance_analysis")
    set_paper_style()
    names = [str(x) for x in z["mode_names"]]
    display = {"none": "no\nguidance", "feasibility": "feasibility\nonly", "feas+perf(full)": "feasibility\n+performance"}
    displays = [display.get(n, n) for n in names]
    x = np.arange(len(names))

    # A bar chart makes the discrete guidance ablation a direct height
    # comparison.  Colours match fig_split_main exactly: neutral grey (no
    # guidance), terracotta (feasibility only) and the main-figure blue (full,
    # ours) with the same navy outline used to flag our configuration there.
    colors = [NEUTRAL_GRAY, COMPARE, FAMILIAR]
    colors = colors[: len(names)]
    ours_idx = names.index("feas+perf(full)") if "feas+perf(full)" in names else len(names) - 1

    fig, axes = plt.subplots(1, 2, figsize=(140 * MM, 62 * MM))
    fig.subplots_adjust(wspace=0.36, left=0.10, right=0.98, bottom=0.24, top=0.90)

    metric_specs = [
        (np.asarray(z["mode_feas"], dtype=float) * 100, "Feasibility (%)", "Feasibility", "{:.1f}", False),
        (np.asarray(z["mode_best"], dtype=float), "Best objective, $J$", "Objective", "{:.3f}", True),
    ]
    for label, ax, (values, ylabel, title, value_fmt, zoom) in zip("AB", axes, metric_specs):
        bars = ax.bar(x, values, width=0.62, color=colors, linewidth=0, zorder=2)
        bars[ours_idx].set_edgecolor(OURS_EDGE)
        bars[ours_idx].set_linewidth(0.9)
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        if zoom:
            # Truncated (zoomed) y-axis so the small objective differences read
            # clearly; a break mark on the left spine flags the non-zero base.
            span = vmax - vmin
            ax.set_ylim(vmin - 0.35 * span, vmax + 0.40 * span)
            _add_yaxis_break(ax)
        else:
            ax.set_ylim(0, vmax * 1.16)
        for xi, value in zip(x, values):
            ax.annotate(value_fmt.format(value), (xi, value), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.2)
        ax.set_xticks(x)
        ax.set_xticklabels(displays, fontsize=6)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold", pad=5)
        # ax.grid(axis="y", color="#E8E8E8", linewidth=0.45, zorder=0)
        ax.set_axisbelow(True)
        add_panel_label(ax, label)
        style_axis(ax, grid=False)

    outputs = save_figure(fig, resolve_fig_dir(results_dir, args.fig_dir), args.stem)
    print("[plot_fig_split_ablation] " + ", ".join(str(path) for path in outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
