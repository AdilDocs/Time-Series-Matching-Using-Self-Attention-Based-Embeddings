"""
Step 12: Fig. 9b -- d_model Trade-off Figure (Table IV / Fig. 9b in the paper)
===================================================================================
Produces fig9b_dmodel_tradeoff.png. Values below are hardcoded from the
paper's actual run (tradeoff_results.csv, step11) -- if you rerun step11
yourself, update these to match your own results (training is
stochastic; exact numbers will differ run to run).

Wafer uses BALANCED accuracy, not raw accuracy: Wafer is severely
class-imbalanced (89.2% majority baseline), and raw accuracy was found
to mask a genuine training failure at d_model=64 (raw acc=92.65% looked
reasonable; balanced acc was only 71.16%, with minority-class recall
collapsing to 43.76%). Computers (125/125 balanced) and StarLightCurves
have no comparable imbalance and use raw accuracy.
"""
import matplotlib.pyplot as plt

D_MODEL_VALUES = [8, 16, 32, 64]

computers_acc = [0.6600, 0.6280, 0.5960, 0.6360]
computers_time = [0.583, 0.615, 0.704, 0.844]

starlight_acc = [0.9025, 0.9197, 0.9017, 0.9100]
starlight_time = [11.021, 11.752, 12.407, 14.414]

wafer_bal_acc = [0.9257, 0.9770, 0.9578, 0.7116]
wafer_time = [8.308, 8.738, 9.371, 11.447]

fig, axes = plt.subplots(3, 1, figsize=(6, 11))
datasets = ["Computers", "StarLightCurves", "Wafer"]
accs = [computers_acc, starlight_acc, wafer_bal_acc]
times = [computers_time, starlight_time, wafer_time]
acc_labels = ["error (1 - accuracy)", "error (1 - accuracy)", "error (1 - balanced accuracy)"]

for ax, name, acc, time_s, acc_label in zip(axes, datasets, accs, times, acc_labels):
    error = [1 - a for a in acc]
    ax2 = ax.twinx()
    l1, = ax.plot(D_MODEL_VALUES, error, marker='^', color='tab:blue', label='error')
    l2, = ax2.plot(D_MODEL_VALUES, time_s, marker='s', linestyle='--', color='tab:green', label='time')
    ax.set_xlabel(r"$d_{model}$")
    ax.set_ylabel(acc_label, color='tab:blue')
    ax2.set_ylabel("time (s)", color='tab:green')
    ax.set_title(name)
    ax.set_xticks(D_MODEL_VALUES)
    ax.legend(handles=[l1, l2], loc='best')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("fig9b_dmodel_tradeoff.png", dpi=150)
print("Saved fig9b_dmodel_tradeoff.png")
