"""
Step 17: d_model sweep on Gun_Point
========================================
Same protocol as step11 (retrain from scratch at d_model in {8,16,32,64},
N_h=4, d_ff=32, L=2, validation-selected k), applied to Gun_Point, to
check the capacity-related accuracy drop discussed alongside Table IV.
Output: tradeoff_gunpoint.csv, checkpoints in checkpoints_tradeoff/.
"""
import csv
import time
import numpy as np

from step6 import (get_dataset_paths, load_ucr_nan_safe, adaptive_target_len,
                     paa_downsample_dataset, stratified_subsample_list,
                     select_k_via_validation, knn_from_similarity,
                     NUM_HEADS, NUM_LAYERS, TRAIN_SUBSAMPLE_PER_CLASS)
from step4 import encode_all
from step11_dmodel_tradeoff_sweep import train_at_dmodel, D_MODEL_VALUES

NAME = "Gun_Point"
OUT_CSV = "tradeoff_gunpoint.csv"

if __name__ == "__main__":
    tr_p, te_p = get_dataset_paths(NAME)
    tr_raw, ytr = load_ucr_nan_safe(tr_p)
    te_raw, yte = load_ucr_nan_safe(te_p)
    T = adaptive_target_len(tr_raw, te_raw)
    Xtr = paa_downsample_dataset(tr_raw, T)
    Xte = paa_downsample_dataset(te_raw, T)
    sub, ysub = stratified_subsample_list(list(Xtr), ytr, TRAIN_SUBSAMPLE_PER_CLASS, seed=0)

    rows = []
    for d in D_MODEL_VALUES:
        print(f"=== {NAME}, d_model={d} ===", flush=True)
        theta, shapes, final_loss = train_at_dmodel(NAME, d, sub, ysub)
        k, _ = select_k_via_validation(theta, shapes, list(Xtr), ytr,
                                       d_model=d, num_heads=NUM_HEADS, num_layers=NUM_LAYERS)
        t0 = time.time()
        S = encode_all(list(Xte), theta, shapes, d, NUM_HEADS, NUM_LAYERS) @ \
            encode_all(list(Xtr), theta, shapes, d, NUM_HEADS, NUM_LAYERS).T
        t = time.time() - t0
        acc, _ = knn_from_similarity(S, ytr, yte, k)
        row = dict(dataset=NAME, d_model=d, final_loss=round(float(final_loss), 6), k_used=k,
                   attn_acc=round(acc, 4), attn_time_s=round(t, 3))
        print(row, flush=True)
        rows.append(row)
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerows(rows)
