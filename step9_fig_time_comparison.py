"""
Step 9: Fig. 7 -- Response Time Comparison (log scale) Across 20 Datasets
==============================================================================
Produces fig7_time_comparison.png from results_compiled.csv (step7).
DTW is measured on a per-dataset balanced subsample (see dtw_n_per_side
column), not the full test set -- flagged in the legend, not hidden.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("results_compiled.csv")
df = df.sort_values("n_train", ascending=True).reset_index(drop=True)
df["ed_time_s_plot"] = df["ed_time_s"].clip(lower=0.001)

fig, ax = plt.subplots(figsize=(8, 10))
y = np.arange(len(df))
bar_h = 0.25
ax.barh(y + bar_h, df["attn_time_s"], height=bar_h, label="Attention", color="#4B0082")
ax.barh(y, df["ed_time_s_plot"], height=bar_h, label="Euclidean", color="#A9A9A9")
ax.barh(y - bar_h, df["dtw_time_s"], height=bar_h, label="DTW (budgeted subsample)", color="#D2691E")
ax.set_xscale("log")
ax.set_yticks(y)
ax.set_yticklabels(df["dataset"], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("time (s, log scale)")
ax.set_title("Response Time of Attention, Euclidean, and DTW\nacross 20 datasets (sorted by training set size)")
ax.legend(loc="lower right", fontsize=9)
ax.grid(axis="x", alpha=0.3, which="both")
plt.tight_layout()
plt.savefig("fig7_time_comparison.png", dpi=150)
print("Saved fig7_time_comparison.png")
