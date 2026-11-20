# ruff: noqa: PGH004
# ruff: noqa
# type: ignore
"""Generated to create plots from data in Google Sheets."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# ---- Data: mean score per model x toolset combo (averaged over temperature) ----

models = ["Haiku 4.5", "Sonnet 4.6", "Opus 4.6"]
toolsets = ["FD", "FD+FDV", "FD+SPARQL", "FD+FDV+SPARQL", "FDV", "FDV+SPARQL", "SPARQL"]
out_path = Path(__file__).resolve().parent / "model_toolset_heatmap.png"

data = {
    "Precision": np.array(
        [
            [0.8680, 0.8240, 0.8666, 0.8559, 0.8565, 0.8560, 0.7932],
            [0.8036, 0.8142, 0.8079, 0.7959, 0.8470, 0.8496, 0.7987],
            [0.8503, 0.8222, 0.8526, 0.8040, 0.8398, 0.8246, 0.8288],
        ]
    ),
    "Recall": np.array(
        [
            [0.8960, 0.9081, 0.9023, 0.9308, 0.9052, 0.9100, 0.8183],
            [0.9379, 0.9810, 0.9307, 0.9818, 0.9354, 0.9459, 0.9605],
            [0.9173, 0.9585, 0.9160, 0.9603, 0.9457, 0.9435, 0.9625],
        ]
    ),
    "F1 score": np.array(
        [
            [0.8355, 0.8198, 0.8385, 0.8594, 0.8431, 0.8411, 0.7657],
            [0.8309, 0.8549, 0.8321, 0.8419, 0.8514, 0.8608, 0.8285],
            [0.8472, 0.8431, 0.8506, 0.8320, 0.8524, 0.8376, 0.8524],
        ]
    ),
}

fig = plt.figure(figsize=(11, 7.2))
gs = fig.add_gridspec(2, 4, hspace=0.65, wspace=0.55, top=0.90, bottom=0.08, left=0.07, right=0.95)
# Top row: Precision and Recall side by side. Bottom row: F1 centered beneath them.
axes = [
    fig.add_subplot(gs[0, 0:2]),
    fig.add_subplot(gs[0, 2:4]),
    fig.add_subplot(gs[1, 1:3]),
]

for ax, (metric, grid) in zip(axes, data.items()):
    sns.heatmap(
        grid,
        ax=ax,
        cmap="Blues",
        vmin=0.5,
        vmax=1,
        annot=True,
        fmt=".3f",
        annot_kws={"fontsize": 8},
        xticklabels=toolsets,
        yticklabels=models,
        linewidths=1.5,
        linecolor="white",
        square=False,
        cbar_kws={"fraction": 0.046, "pad": 0.04},
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_title(metric, fontsize=12, fontweight="bold")
    ax.collections[0].colorbar.ax.tick_params(labelsize=7)

fig.suptitle(
    "Mean score by model and toolset (averaged over temperature)",
    fontsize=13,
    fontweight="bold",
    y=0.99,
)
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved heatmap to {out_path}")
