# ruff: noqa: PGH004
# ruff: noqa
# type: ignore
"""Generated to create plots from data in Google Sheets."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

out_dir = Path(__file__).resolve().parent

# ---- Data (means + 95% CI half-widths, computed in Sheets) ----

model_labels = ["Claude Haiku 4.5", "Claude Sonnet 4.6", "Claude Opus 4.6"]
model_data = {
    "Precision": [0.8457214286, 0.8166857143, 0.8317285714],
    "Recall": [0.8957714286, 0.9532928571, 0.9433785714],
    "F1": [0.8289714286, 0.8429142857, 0.8450214286],
}
model_ci = {
    "Precision": [0.01549527558, 0.01259810204, 0.009637821991],
    "Recall": [0.02022108954, 0.01195587791, 0.01104550017],
    "F1": [0.01717813589, 0.007413395248, 0.004791243305],
}

temp_labels = ["0", "0.1"]
temp_data = {
    "Precision": [0.8320857143, 0.8306714286],
    "Recall": [0.930847619, 0.9307809524],
    "F1": [0.8395238095, 0.8384142857],
}
temp_ci = {
    "Precision": [0.01212362008, 0.01061200088],
    "Recall": [0.01532501932, 0.01765412408],
    "F1": [0.008574346582, 0.009955254355],
}

toolset_labels = ["FD", "FD+FDV", "FD+SPARQL", "FD+FDV+SPARQL", "FDV", "FDV+SPARQL", "SPARQL"]
toolset_data = {
    "Precision": [0.8406, 0.8201333333, 0.84235, 0.8186, 0.8477166667, 0.84335, 0.8069],
    "Recall": [0.9170166667, 0.94915, 0.9163, 0.9576333333, 0.9287333333, 0.9331166667, 0.91375],
    "F1": [
        0.8378333333,
        0.8392333333,
        0.8403666667,
        0.8444166667,
        0.8489666667,
        0.8464666667,
        0.8155,
    ],
}
toolset_ci = {
    "Precision": [
        0.03159090642,
        0.006160702666,
        0.02927566198,
        0.03087196368,
        0.008934739154,
        0.01648365767,
        0.01837825194,
    ],
    "Recall": [
        0.02012437981,
        0.03511021776,
        0.01357078946,
        0.02404482792,
        0.0198793878,
        0.01918974684,
        0.07831723102,
    ],
    "F1": [
        0.009318058756,
        0.01724496644,
        0.01009918469,
        0.01337417787,
        0.006269152023,
        0.01274684923,
        0.04275792836,
    ],
}

metric_colors = {"F1": "#2a78d6", "Precision": "#eb6834", "Recall": "#1baf7a"}


def dot_plot(
    labels, data, ci_data, title, filename, ylim=None, figsize=(8, 5), bbox_to_anchor=(0.5, -0.15)
):
    metrics = list(data.keys())
    n_groups = len(labels)
    n_metrics = len(metrics)
    x = np.arange(n_groups)
    spread = 0.18
    offsets = np.linspace(-spread, spread, n_metrics) if n_metrics > 1 else [0]

    fig, ax = plt.subplots(figsize=figsize)

    for i, metric in enumerate(metrics):
        values = data[metric]
        errs = ci_data[metric]
        xs = x + offsets[i]
        ax.errorbar(
            xs,
            values,
            yerr=errs,
            fmt="o",
            label=metric,
            color=metric_colors[metric],
            markeredgecolor="black",
            markeredgewidth=0.6,
            markersize=8,
            capsize=4,
            elinewidth=1.2,
            zorder=3,
        )
        for xv, val, err in zip(xs, values, errs):
            ax.text(xv, val + err + 0.004, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels, rotation=15 if n_groups > 4 else 0, ha="right" if n_groups > 4 else "center"
    )
    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.legend(
        title="Metric", frameon=False, loc="upper center", bbox_to_anchor=bbox_to_anchor, ncol=3
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    if ylim:
        ax.set_ylim(ylim)

    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


dot_plot(
    model_labels,
    model_data,
    model_ci,
    "Mean score by model (error bars: 95% CI across toolset x temperature configs)",
    out_dir / "main_effects_model.png",
    ylim=(0.75, 1.0),
    bbox_to_anchor=(0.5, -0.075),
)

dot_plot(
    temp_labels,
    temp_data,
    temp_ci,
    "Mean score by temperature\n(error bars: 95% CI across model x toolset configs)",
    out_dir / "main_effects_temperature.png",
    ylim=(0.75, 1.0),
    figsize=(5, 5),
    bbox_to_anchor=(0.5, -0.075),
)

dot_plot(
    toolset_labels,
    toolset_data,
    toolset_ci,
    "Mean score by toolset (error bars: 95% CI across model x temperature configs)",
    out_dir / "main_effects_toolset.png",
    ylim=(0.75, 1.0),
    figsize=(10, 5.5),
    bbox_to_anchor=(0.5, -0.15),
)

print(f"Saved 3 charts to {out_dir}")
