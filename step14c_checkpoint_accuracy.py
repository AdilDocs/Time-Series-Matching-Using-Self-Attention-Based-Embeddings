"""
Step 14c: Table V -- k and Test Accuracy at Each Embedding-Evolution Checkpoint
=====================================================================================
For every StarLightCurves checkpoint saved by step13.py, selects k with the
same validation-only procedure used for Table I (step6.select_k_via_validation,
complete training set, test set never consulted) and reports the resulting
test accuracy on the complete test set.
Output: checkpoint_accuracy.csv
"""
import csv
import numpy as np

from step6 import (get_dataset_paths, load_ucr_nan_safe, adaptive_target_len,
                     paa_downsample_dataset, select_k_via_validation, knn_from_similarity,
                     D_MODEL, NUM_HEADS, NUM_LAYERS)
from step4 import encode_all
from step13_train_with_snapshots import CKPT_DIR, CHECKPOINT_ITERS

NAME = "StarLightCurves"
OUT_CSV = "checkpoint_accuracy.csv"

if __name__ == "__main__":
    tr_p, te_p = get_dataset_paths(NAME)
    tr_raw, ytr = load_ucr_nan_safe(tr_p)
    te_raw, yte = load_ucr_nan_safe(te_p)
    T = adaptive_target_len(tr_raw, te_raw)
    Xtr = paa_downsample_dataset(tr_raw, T)
    Xte = paa_downsample_dataset(te_raw, T)

    rows = []
    for it in CHECKPOINT_ITERS:
        ck = np.load(f"{CKPT_DIR}/{NAME}_iter{it}.npz", allow_pickle=True)
        theta, shapes = ck["theta"], ck["shapes"].item()
        k, _ = select_k_via_validation(theta, shapes, list(Xtr), ytr)
        S = encode_all(list(Xte), theta, shapes, D_MODEL, NUM_HEADS, NUM_LAYERS) @ \
            encode_all(list(Xtr), theta, shapes, D_MODEL, NUM_HEADS, NUM_LAYERS).T
        acc, _ = knn_from_similarity(S, ytr, yte, k)
        row = dict(iteration=it, k_selected=k, test_acc=round(float(acc), 4))
        print(row, flush=True)
        rows.append(row)
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerows(rows)
