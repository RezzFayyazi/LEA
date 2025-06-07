# -*- coding: utf-8 -*-

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


MODEL_FILES = {
    "Gemma-3-27B": "./results/LEA_results_gemma3.json",
    "Mistral-24B": "./results/LEA_results_mistral.json",
    "DeepSeek-8B": "./results/LEA_results_deepseek.json",
    "Llama3.2-3B": "./results/LEA_results_llama.json",
}


LABEL_MAP = {
    "xy=True,x?y=True" : "Found. Knowledge",
    "xy=True,x?y=False": "RAG",
    "xy=False,x?y=False": "Question",
}
METRIC_ORDER = ["xy=False,x?y=False", "xy=True,x?y=False", "xy=True,x?y=True"]  # bottom→top


PALETTE = {
    "xy=False,x?y=False": "#7570b3",
    "xy=True,x?y=False":  "#1b9e77", 
    "xy=True,x?y=True" :  "#f5811b",  
}


plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "axes.edgecolor": "#444444",
    "axes.linewidth": .8,
    "grid.linestyle": ":",
    "grid.alpha": 0.4,
})


def load_block_stats(json_path: str):

    with open(json_path, "r") as f:
        raw = json.load(f)

    df = pd.DataFrame({k: v["percentages"] for k, v in raw.items()}).T
    df.index = df.index.astype(int).sort_values()

    df = df.fillna(0.0)                
    block_idx = df.index // 10

    mean_df = df.groupby(block_idx).mean()
    std_df  = df.groupby(block_idx).std(ddof=0)  
    return mean_df, std_df


mean_frames, std_frames = {}, {}
for model, path in MODEL_FILES.items():
    mean_df, std_df = load_block_stats(path)
    mean_frames[model] = mean_df
    #std_frames[model]  = std_df

wide_mean = (
    pd.concat(mean_frames, axis=1)
      .sort_index(axis=1, level=[0, 1])
)
#wide_std  = (
#    pd.concat(std_frames, axis=1)
#      .sort_index(axis=1, level=[0, 1])
#)


rows, cols = 4, 1
fig, axes = plt.subplots(rows, cols, figsize=(8, 18), sharey=True)
axes = axes.flatten()

bar_width = .7
x = np.arange(wide_mean.index.max() + 1)          # 0,1,2,…   (one bar per “year”)

for ax_i, model in enumerate(MODEL_FILES):
    ax = axes[ax_i]
    bottom = np.zeros_like(x, dtype=float)

    for metric in METRIC_ORDER:
        vals = wide_mean[(model, metric)].reindex(x, fill_value=0.0).values
        #errs = wide_std [(model, metric)].reindex(x, fill_value=0.0).values
        errs = np.nan_to_num(errs, nan=0.0)       # safety


        bars = ax.bar(
            x, vals, bar_width, bottom=bottom,
            color="none" if metric == "xy=False,x?y=False" else PALETTE[metric],
            edgecolor=PALETTE[metric],
            hatch="//" if metric == "xy=False,x?y=False" else None,
            linewidth=.8,
            zorder=1,
        )


        for j, bar in enumerate(bars):
            if vals[j] > 0:
                y = bottom[j] + vals[j] / 2
                txt_col = "black" if metric == "xy=False,x?y=False" else "white"
                ax.text(
                    bar.get_x() + bar.get_width()/2, y,
                    f"{vals[j]:.0f}%", ha="center", va="center",
                    fontsize=10, color=txt_col, weight="bold", zorder=3
                )


        nz = vals > 0
        if nz.any():
            ax.errorbar(
                x[nz], bottom[nz] + vals[nz] / 2, yerr=errs[nz],
                fmt="none", ecolor="k", capsize=3, lw=1, zorder=2
            )

        bottom += vals


    ax.set_title(model)
    ax.set_xticks(x)
    ax.set_xticklabels([str(yr) for yr in range(2025, 2025 - len(x), -1)])
    ax.invert_xaxis()                        # years go from 2016 to 2025
    ax.set_xlabel("Years (10 CVEs per year)")
    ax.grid(axis="y", zorder=0)
    if ax_i % cols == 0:
        ax.set_ylabel("LEA (Mean %)")


for j in range(len(MODEL_FILES), len(axes)):
    fig.delaxes(axes[j])


handles = [
    Patch(facecolor=PALETTE["xy=True,x?y=True"],  label="Found. Knowledge"),
    Patch(facecolor=PALETTE["xy=True,x?y=False"], label="RAG"),
    Patch(facecolor="none", edgecolor=PALETTE["xy=False,x?y=False"],
          hatch="//", label="Question"),
]
fig.legend(handles=handles, ncol=3, loc="upper center",
           bbox_to_anchor=(0.5, 0.98), frameon=True)

fig.tight_layout(rect=(0, 0, 1, 0.96))
plt.show()
