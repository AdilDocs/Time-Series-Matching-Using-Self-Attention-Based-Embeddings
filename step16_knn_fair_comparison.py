"""
Step 16: Table III -- Effect of k-Selection (Fair Comparison)
==================================================================
For every dataset, reports:
  Att-1NN   : proposed encoder, k = 1
  Att-kNN   : proposed encoder, k selected on a held-out training split
              (identical to Table I / step7)
  ED-1NN    : Euclidean 1-NN (identical to Table I / step7)
  ED-kNN    : Euclidean k-NN, k selected by the SAME validation protocol
  DTW-1NN   : DTW 1-NN on the step7 class-balanced subsample
  DTW-kNN   : DTW k-NN on the same subsample, k selected by the same
              protocol applied within the DTW training subsample

Validation protocol (shared by all k-NN variants, from step6):
  random 20% of the training pool (seed 0) is classified against the
  remaining 80%; k in {1,3,5,7,9,15,25} with the highest accuracy is kept
  (ties -> smallest k). The test set is never consulted.

DTW is the same squared-cost, unconstrained DTW as step1.dtw_distance,
compiled with numba for speed (numerically identical results).
Output: knn_fair_comparison.csv
"""
import os
import csv
import numpy as np
from numba import njit

from step6 import (DATASETS, get_dataset_paths, load_ucr_nan_safe, adaptive_target_len,
                     paa_downsample_dataset, stratified_subsample_list, load_checkpoint,
                     knn_from_similarity, select_k_via_validation,
                     D_MODEL, NUM_HEADS, NUM_LAYERS)
from step1 import train_val_split
from step4 import encode_all

K_CANDIDATES = (1, 3, 5, 7, 9, 15, 25)
DTW_N_PER_CLASS = 50
OUT_CSV = "knn_fair_comparison.csv"
FIELDS = ["dataset", "att_1nn", "att_knn", "k_att", "ed_1nn", "ed_knn", "k_ed",
          "dtw_1nn", "dtw_knn", "k_dtw", "dtw_n_train", "dtw_n_test"]


@njit(cache=True)
def _dtw(a, b):
    n, m = a.shape[0], b.shape[0]
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = (a[i - 1] - b[j - 1]) ** 2
            best = D[i - 1, j]
            if D[i, j - 1] < best:
                best = D[i, j - 1]
            if D[i - 1, j - 1] < best:
                best = D[i - 1, j - 1]
            D[i, j] = c + best
    return np.sqrt(D[n, m])


@njit(cache=True)
def dtw_matrix(A, B):
    out = np.empty((A.shape[0], B.shape[0]))
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            out[i, j] = _dtw(A[i], B[j])
    return out


def ed_matrix(A, B):
    sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return np.sqrt(np.maximum(sq, 0))


def select_k_from_distance(dist_fn, pool, pool_labels, seed=0):
    """Same protocol as step6.select_k_via_validation, but for a distance."""
    tr_idx, val_idx = train_val_split(len(pool), val_frac=0.2, seed=seed)
    Dv = dist_fn(pool[val_idx], pool[tr_idx])
    best_k, best_acc = K_CANDIDATES[0], -1
    for k in K_CANDIDATES:
        if k > len(tr_idx):
            continue
        acc, _ = knn_from_similarity(-Dv, pool_labels[tr_idx], pool_labels[val_idx], k)
        if acc > best_acc:
            best_acc, best_k = acc, k
    return best_k


if __name__ == "__main__":
    rows = []
    for name in DATASETS:
        tr_p, te_p = get_dataset_paths(name)
        tr_raw, ytr = load_ucr_nan_safe(tr_p)
        te_raw, yte = load_ucr_nan_safe(te_p)
        T = adaptive_target_len(tr_raw, te_raw)
        Xtr = paa_downsample_dataset(tr_raw, T)
        Xte = paa_downsample_dataset(te_raw, T)
        ytr, yte = np.asarray(ytr), np.asarray(yte)

        theta, shapes, _ = load_checkpoint(name)
        k_att, _ = select_k_via_validation(theta, shapes, list(Xtr), ytr)
        S = encode_all(list(Xte), theta, shapes, D_MODEL, NUM_HEADS, NUM_LAYERS) @ \
            encode_all(list(Xtr), theta, shapes, D_MODEL, NUM_HEADS, NUM_LAYERS).T
        att_1, _ = knn_from_similarity(S, ytr, yte, 1)
        att_k, _ = knn_from_similarity(S, ytr, yte, k_att)

        k_ed = select_k_from_distance(ed_matrix, Xtr, ytr)
        De = ed_matrix(Xte, Xtr)
        ed_1, _ = knn_from_similarity(-De, ytr, yte, 1)
        ed_k, _ = knn_from_similarity(-De, ytr, yte, k_ed)

        dtr, dytr = stratified_subsample_list(list(Xtr), ytr, DTW_N_PER_CLASS, seed=0)
        dte, dyte = stratified_subsample_list(list(Xte), yte, DTW_N_PER_CLASS, seed=1)
        dtr, dte = np.array(dtr), np.array(dte)
        k_dtw = select_k_from_distance(dtw_matrix, dtr, dytr)
        Dd = dtw_matrix(dte, dtr)
        dtw_1, _ = knn_from_similarity(-Dd, dytr, dyte, 1)
        dtw_k, _ = knn_from_similarity(-Dd, dytr, dyte, k_dtw)

        row = dict(dataset=name, att_1nn=round(att_1, 4), att_knn=round(att_k, 4), k_att=k_att,
                   ed_1nn=round(ed_1, 4), ed_knn=round(ed_k, 4), k_ed=k_ed,
                   dtw_1nn=round(dtw_1, 4), dtw_knn=round(dtw_k, 4), k_dtw=k_dtw,
                   dtw_n_train=len(dtr), dtw_n_test=len(dte))
        print(row, flush=True)
        rows.append(row)
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
    print("Saved", OUT_CSV)
