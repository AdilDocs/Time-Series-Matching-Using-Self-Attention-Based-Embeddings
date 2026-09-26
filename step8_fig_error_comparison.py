"""
Step 8: Fig. 6 -- Error Comparison Across 20 Datasets
==========================================================
Produces fig6_error_comparison.png from results_compiled.csv (step7).
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("results_compiled.csv")
df = df.sort_values("n_train", ascending=True).reset_index(drop=True)
df["attn_err"] = 1 - df["attn_acc"]
df["ed_err"] = 1 - df["ed_acc"]
df["dtw_err"] = 1 - df["dtw_acc"]

fig, ax = plt.subplots(figsize=(8, 10))
y = np.arange(len(df))
bar_h = 0.25
ax.barh(y + bar_h, df["attn_err"], height=bar_h, label="Attention", color="#4B0082")
ax.barh(y, df["ed_err"], height=bar_h, label="Euclidean", color="#A9A9A9")
ax.barh(y - bar_h, df["dtw_err"], height=bar_h, label="DTW", color="#D2691E")
ax.set_yticks(y)
ax.set_yticklabels(df["dataset"], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("error (1 - accuracy)")
ax.set_title("Error of Attention, Euclidean, and DTW\nacross 20 datasets (sorted by training set size)")
ax.legend(loc="lower right", fontsize=9)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("fig6_error_comparison.png", dpi=150)
print("Saved fig6_error_comparison.png")
