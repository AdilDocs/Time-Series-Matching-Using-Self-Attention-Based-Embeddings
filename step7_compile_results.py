"""
Step 7: Compile Full Results Table
=======================================
Produces results_compiled.csv, the source of Table I in the paper
("Classification Accuracy Across 20 UCR Datasets") and the source data
for Figs. 6 and 7. Resumable -- skips datasets already in the CSV.

Requires trained checkpoints for all datasets in DATASETS (see step6.py).
"""
import numpy as np
import time
import os
import csv
from step6 import (get_dataset_paths, load_ucr_nan_safe, adaptive_target_len,
                     paa_downsample_dataset, stratified_subsample_list,
                     load_checkpoint, select_k_via_validation, knn_from_similarity,
                     D_MODEL, NUM_HEADS, NUM_LAYERS, DATASETS)
from step4 import encode_all
from step1 import evaluate_1nn_euclidean, evaluate_1nn_dtw

OUT_CSV = "results_compiled.csv"
DTW_N_PER_CLASS = 50
FIELDNAMES = ["dataset", "n_train", "n_test", "n_classes", "k_used",
               "attn_acc", "attn_time_s", "ed_acc", "ed_time_s",
               "dtw_acc", "dtw_time_s", "dtw_n_per_side"]


def load_completed():
    if not os.path.exists(OUT_CSV):
        return {}
    with open(OUT_CSV) as f:
        return {row["dataset"]: row for row in csv.DictReader(f)}


def save_row(row):
    file_exists = os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    completed = load_completed()
    for name in DATASETS:
        if name in completed:
            print(f"{name}: already compiled, skipping")
            continue

        print(f"\n=== {name} ===")
        train_path, test_path = get_dataset_paths(name)
        train_seq_native, train_labels = load_ucr_nan_safe(train_path)
        test_seq_native, test_labels = load_ucr_nan_safe(test_path)
        target_len = adaptive_target_len(train_seq_native, test_seq_native)
        train_seq_full = paa_downsample_dataset(train_seq_native, target_len)
        test_seq_full = paa_downsample_dataset(test_seq_native, target_len)

        theta, shapes, _ = load_checkpoint(name)
        if theta is None:
            print(f"  NO CHECKPOINT -- run step6.py first")
            continue

        best_k, _ = select_k_via_validation(theta, shapes, list(train_seq_full), train_labels)

        t0 = time.time()
        train_emb = encode_all(list(train_seq_full), theta, shapes, D_MODEL, NUM_HEADS, NUM_LAYERS)
        test_emb = encode_all(list(test_seq_full), theta, shapes, D_MODEL, NUM_HEADS, NUM_LAYERS)
        S = test_emb @ train_emb.T
        attn_time = time.time() - t0
        attn_acc, _ = knn_from_similarity(S, train_labels, test_labels, best_k)
        print(f"  Attention (k={best_k}): acc={attn_acc:.4f}, time={attn_time:.2f}s")

        t0 = time.time()
        ed_acc, _ = evaluate_1nn_euclidean(train_seq_full, train_labels, test_seq_full, test_labels)
        ed_time = time.time() - t0
        print(f"  Euclidean: acc={ed_acc:.4f}, time={ed_time:.2f}s")

        n_per_class = DTW_N_PER_CLASS
        dtw_train, dtw_train_labels = stratified_subsample_list(list(train_seq_full), train_labels, n_per_class, seed=0)
        dtw_test, dtw_test_labels = stratified_subsample_list(list(test_seq_full), test_labels, n_per_class, seed=1)
        t0 = time.time()
        dtw_acc, _ = evaluate_1nn_dtw(np.array(dtw_train), dtw_train_labels, np.array(dtw_test), dtw_test_labels)
        dtw_time = time.time() - t0
        print(f"  DTW ({len(dtw_train)}x{len(dtw_test)}): acc={dtw_acc:.4f}, time={dtw_time:.2f}s")

        save_row({
            "dataset": name, "n_train": len(train_seq_full), "n_test": len(test_seq_full),
            "n_classes": len(np.unique(train_labels)), "k_used": best_k,
            "attn_acc": round(attn_acc, 4), "attn_time_s": round(attn_time, 2),
            "ed_acc": round(ed_acc, 4), "ed_time_s": round(ed_time, 2),
            "dtw_acc": round(dtw_acc, 4), "dtw_time_s": round(dtw_time, 2),
            "dtw_n_per_side": len(dtw_train),
        })
        print("  Saved.")
