"""
Step 10: Fig. 9 -- Accuracy Sensitivity to k
=================================================
Produces fig9_accuracy_vs_k.png for 6 datasets spanning a range of
training set sizes and class counts. Descriptive only -- NOT the
procedure used to select k for Table I (that uses select_k_via_validation
in step6.py, which never consults the test set).
"""
import numpy as np
import matplotlib.pyplot as plt
from step6 import (get_dataset_paths, load_ucr_nan_safe, adaptive_target_len,
                     paa_downsample_dataset, load_checkpoint, knn_from_similarity,
                     D_MODEL, NUM_HEADS, NUM_LAYERS)
from step4 import encode_all

DATASETS = ["Coffee", "Beef", "StarLightCurves", "Trace", "Gun_Point", "Wafer"]
K_VALUES = [1, 3, 5, 7, 9, 15, 25, 35, 50]

fig, ax = plt.subplots(figsize=(8, 6))
colors = plt.cm.tab10(np.linspace(0, 1, len(DATASETS)))

for name, color in zip(DATASETS, colors):
    train_path, test_path = get_dataset_paths(name)
    train_seq_native, train_labels = load_ucr_nan_safe(train_path)
    test_seq_native, test_labels = load_ucr_nan_safe(test_path)
    target_len = adaptive_target_len(train_seq_native, test_seq_native)
    train_seq_full = paa_downsample_dataset(train_seq_native, target_len)
    test_seq_full = paa_downsample_dataset(test_seq_native, target_len)

    theta, shapes, _ = load_checkpoint(name)
    if theta is None:
        print(f"  {name}: no checkpoint, skipping")
        continue

    train_emb = encode_all(list(train_seq_full), theta, shapes, D_MODEL, NUM_HEADS, NUM_LAYERS)
    test_emb = encode_all(list(test_seq_full), theta, shapes, D_MODEL, NUM_HEADS, NUM_LAYERS)
    S = test_emb @ train_emb.T

    accs, valid_ks = [], []
    for k in K_VALUES:
        if k > len(train_seq_full):
            continue
        acc, _ = knn_from_similarity(S, train_labels, test_labels, k)
        accs.append(acc)
        valid_ks.append(k)
    print(f"{name}: {list(zip(valid_ks, [round(a,3) for a in accs]))}")
    ax.plot(valid_ks, accs, marker='o', label=name, color=color)

ax.set_xlabel("k (neighborhood size)")
ax.set_ylabel("test accuracy")
ax.set_title("Accuracy Sensitivity to k\n(descriptive only -- not the k-selection procedure used for reported results)")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig9_accuracy_vs_k.png", dpi=150)
print("Saved fig9_accuracy_vs_k.png")
