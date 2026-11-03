from __future__ import annotations

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np

from _plot_common import (
    ACCENT,
    FAMILIAR,
    METHOD_DISPLAY_SHORT,
    METHOD_ORDER,
    MM,
    OURS_EDGE,
    UNFAMILIAR,
    add_io_args,
    add_panel_label,
    resolve_fig_dir,
    save_figure,
    set_paper_style,
    style_axis,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw the paper-style fixed-split main result figure.")
    add_io_args(parser, default_stem="fig_split_main")
    parser.add_argument("--single-draw", action="store_true",
                        help="plot the single evaluation in split_results.npz instead of the 10-draw mean +/- std "
                             "of CVAE / CD / PGCD from table2_resampling.csv")
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    path = results_dir / "split_results.npz"
    if not path.exists():
        raise SystemExit(f"missing input: {path}")

    set_paper_style(serif=False)
    data = np.load(path, allow_pickle=True)
    methods = [str(m) for m in data["methods"]]
    keys = [str(k) for k in data["keys"]]
    table = data["table"]
    key_index = {k: i for i, k in enumerate(keys)}
    method_index = [methods.index(m) for m in METHOD_ORDER if m in methods]
    methods_ordered = [methods[i] for i in method_index]
    fam = table[0, method_index, :].copy()
    ood = table[1, method_index, :].copy()
    err = np.zeros((2,) + fam.shape)

    # Stochastic methods: mean +/- std over sampling seeds; planning time stays from the original run.
    resampling = results_dir / "table2_resampling.csv"
    if not args.single_draw and resampling.exists():
        with resampling.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["method"] not in methods_ordered or row["metric"] not in key_index or row["metric"] == "lat":
                    continue
                t = 0 if row["test_set"] == "familiar" else 1
                i, k = methods_ordered.index(row["method"]), key_index[row["metric"]]
                (fam if t == 0 else ood)[i, k] = float(row["mean"])
                err[t, i, k] = float(row["std"])

    x = np.arange(len(methods_ordered))
    width = 0.34
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(183 * MM, 105 * MM),
        gridspec_kw={"height_ratios": [1.12, 1.0]},
    )
    fig.subplots_adjust(hspace=0.52, wspace=0.30, left=0.085, right=0.99, bottom=0.19, top=0.86)

    ours_index = methods_ordered.index("PGDiff") if "PGDiff" in methods_ordered else None

    def grouped_bars(
        ax,
        key: str,
        title: str,
        ylabel: str,
        *,
        scale: float = 100.0,
        log: bool = False,
        floor: float | None = None,
        ylim: tuple[float, float] | None = None,
    ) -> None:
        y_fam = fam[:, key_index[key]] * scale
        y_ood = ood[:, key_index[key]] * scale
        if floor is not None:
            y_fam = np.maximum(y_fam, floor)
            y_ood = np.maximum(y_ood, floor)
        if log and floor is not None:
            bars_fam = ax.bar(x - width / 2, y_fam - floor, width, bottom=floor, color=FAMILIAR, label="In-distribution", linewidth=0, zorder=2)
            bars_ood = ax.bar(x + width / 2, y_ood - floor, width, bottom=floor, color=UNFAMILIAR, label="Unseen material", linewidth=0, zorder=2)
        else:
            bars_fam = ax.bar(x - width / 2, y_fam, width, color=FAMILIAR, label="In-distribution", linewidth=0, zorder=2)
            bars_ood = ax.bar(x + width / 2, y_ood, width, color=UNFAMILIAR, label="Unseen material", linewidth=0, zorder=2)
            for offset, y, e in ((-width / 2, y_fam, err[0, :, key_index[key]] * scale),
                                 (width / 2, y_ood, err[1, :, key_index[key]] * scale)):
                m = e > 0
                ax.errorbar(x[m] + offset, y[m], yerr=e[m], fmt="none", ecolor="#333333",
                            elinewidth=0.6, capsize=1.6, capthick=0.6, zorder=3)
        if ours_index is not None:
            for bar in (bars_fam[ours_index], bars_ood[ours_index]):
                bar.set_edgecolor(OURS_EDGE)
                bar.set_linewidth(0.9)

        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_DISPLAY_SHORT[m] for m in methods_ordered], fontsize=5.5)
        if ours_index is not None:
            ax.get_xticklabels()[ours_index].set_fontweight("bold")
            ax.get_xticklabels()[ours_index].set_color(ACCENT)
        ax.set_title(title, loc="left", pad=6, fontweight="bold")
        ax.set_ylabel(ylabel)
        if log:
            ax.set_yscale("log")
        if ylim is not None:
            ax.set_ylim(*ylim)
        # ax.grid(axis="y", color="#E8E8E8", linewidth=0.45, zorder=0)
        ax.set_axisbelow(True)
        style_axis(ax)

    grouped_bars(axes[0, 0], "succ", "Plan success", "Success (%)", ylim=(0, 105))
    grouped_bars(axes[0, 1], "viol_rate", "Constraint violation", "Violation (%)", ylim=(0, 82))
    grouped_bars(axes[1, 0], "gap", "Optimality gap to PSO", "Gap (%)", ylim=(0, 9.4))
    grouped_bars(axes[1, 1], "lat", "Planning time", "Time (s; log scale)", scale=1.0, log=True, floor=2e-4, ylim=(2e-4, 6.0))
    axes[1, 1].tick_params(axis="y", which="minor", left=False)

    for label, ax in zip("ABCD", axes.flat):
        add_panel_label(ax, label, x=-0.12, y=1.05)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.54, 0.985), ncol=2, handlelength=1.8)
    outputs = save_figure(fig, resolve_fig_dir(results_dir, args.fig_dir), args.stem)
    print("[plot_fig_split_main] " + ", ".join(str(path) for path in outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
