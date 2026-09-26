"""
Step 11: d_model Trade-off Sweep
=====================================
Retrains the encoder at d_model in {8,16,32,64} for Computers,
StarLightCurves, and Wafer -- the genuine speed/accuracy trade-off
underlying Table IV / Fig. 9b. Substantial compute (full retraining at
each setting). Resumable via tradeoff_results.csv and checkpoints_tradeoff/.
"""
import numpy as np
import time
import os
import csv

from step6 import (get_dataset_paths, load_ucr_nan_safe, adaptive_target_len,
                     paa_downsample_dataset, stratified_subsample_list,
                     robust_init, select_k_via_validation, knn_from_similarity,
                     NUM_HEADS, D_FF, NUM_LAYERS, MARGIN, LR, MINIBATCH_SIZE,
                     TRIPLETS_PER_ANCHOR, TRAIN_SUBSAMPLE_PER_CLASS)
from step4 import train, encode_all, build_triplets

D_MODEL_VALUES = [8, 16, 32, 64]
DATASETS = ["Computers", "StarLightCurves", "Wafer"]

AUTO_ROUND_ITERS = 25
AUTO_MAX_ROUNDS = 8
AUTO_PLATEAU_DELTA = 0.002
AUTO_ABSOLUTE_CONVERGED = 0.01

CKPT_DIR = "checkpoints_tradeoff"
OUT_CSV = "tradeoff_results.csv"
FIELDNAMES = ["dataset", "d_model", "final_loss", "k_used",
               "attn_acc", "attn_time_s", "n_train", "n_test"]


def ckpt_path(dataset_name, d_model):
    os.makedirs(CKPT_DIR, exist_ok=True)
    return os.path.join(CKPT_DIR, f"{dataset_name}_dmodel{d_model}.npz")


def load_completed():
    if not os.path.exists(OUT_CSV):
        return set()
    with open(OUT_CSV) as f:
        return {(row["dataset"], int(row["d_model"])) for row in csv.DictReader(f)}


def save_row(row):
    file_exists = os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def train_at_dmodel(dataset_name, d_model, train_seq, train_labels, verbose=True):
    path = ckpt_path(dataset_name, d_model)
    theta, shapes, optimizer_state = None, None, None
    if os.path.exists(path):
        ckpt = np.load(path, allow_pickle=True)
        theta, shapes = ckpt["theta"], ckpt["shapes"].item()
        optimizer_state = ckpt["optimizer_state"].item() if "optimizer_state" in ckpt else None
        if verbose:
            print(f"  Resuming d_model={d_model} from checkpoint")

    triplets = build_triplets(train_labels.tolist(), triplets_per_anchor=TRIPLETS_PER_ANCHOR, seed=0)

    if theta is None:
        theta, shapes, scale_used, sim = robust_init(
            list(train_seq), d_model, NUM_HEADS, D_FF, NUM_LAYERS, verbose=verbose
        )

    prev_loss = None
    for round_idx in range(AUTO_MAX_ROUNDS):
        theta, shapes, loss_hist, optimizer_state = train(
            list(train_seq), triplets, d_model=d_model, num_heads=NUM_HEADS,
            d_ff=D_FF, num_layers=NUM_LAYERS, num_iters=AUTO_ROUND_ITERS, lr=LR,
            margin=MARGIN, seed=0, minibatch_size=MINIBATCH_SIZE,
            theta_init=theta, shapes=shapes, optimizer_state=optimizer_state,
            verbose_every=None,
        )
        final_loss = loss_hist[-1]
        starting_loss = loss_hist[0]

        if starting_loss < AUTO_ABSOLUTE_CONVERGED and final_loss >= starting_loss:
            prev_loss = starting_loss
            break

        np.savez(path, theta=theta, shapes=shapes, optimizer_state=optimizer_state)
        if verbose:
            print(f"  d_model={d_model} round {round_idx+1}: loss {loss_hist[0]:.6f} -> {final_loss:.6f}")

        if final_loss < AUTO_ABSOLUTE_CONVERGED:
            prev_loss = final_loss
            break
        if prev_loss is not None and abs(prev_loss - final_loss) < AUTO_PLATEAU_DELTA:
            prev_loss = final_loss
            break
        prev_loss = final_loss

    return theta, shapes, prev_loss


if __name__ == "__main__":
    completed = load_completed()
    for name in DATASETS:
        train_path, test_path = get_dataset_paths(name)
        train_seq_native, train_labels = load_ucr_nan_safe(train_path)
        test_seq_native, test_labels = load_ucr_nan_safe(test_path)
        target_len = adaptive_target_len(train_seq_native, test_seq_native)
        train_seq_full = paa_downsample_dataset(train_seq_native, target_len)
        test_seq_full = paa_downsample_dataset(test_seq_native, target_len)
        train_seq_sub, train_labels_sub = stratified_subsample_list(
            list(train_seq_full), train_labels, TRAIN_SUBSAMPLE_PER_CLASS, seed=0
        )

        for d_model in D_MODEL_VALUES:
            if (name, d_model) in completed:
                print(f"{name}, d_model={d_model}: already done, skipping")
                continue

            print(f"\n=== {name}, d_model={d_model} ===")
            theta, shapes, final_loss = train_at_dmodel(name, d_model, train_seq_sub, train_labels_sub)

            best_k, _ = select_k_via_validation(theta, shapes, list(train_seq_full), train_labels,
                                                   d_model=d_model, num_heads=NUM_HEADS, num_layers=NUM_LAYERS)

            t0 = time.time()
            train_emb = encode_all(list(train_seq_full), theta, shapes, d_model, NUM_HEADS, NUM_LAYERS)
            test_emb = encode_all(list(test_seq_full), theta, shapes, d_model, NUM_HEADS, NUM_LAYERS)
            S = test_emb @ train_emb.T
            attn_time = time.time() - t0
            attn_acc, _ = knn_from_similarity(S, train_labels, test_labels, best_k)

            print(f"  Result: acc={attn_acc:.4f}, time={attn_time:.2f}s, final_loss={final_loss}")

            save_row({
                "dataset": name, "d_model": d_model,
                "final_loss": round(final_loss, 6) if final_loss else None,
                "k_used": best_k, "attn_acc": round(attn_acc, 4), "attn_time_s": round(attn_time, 3),
                "n_train": len(train_seq_full), "n_test": len(test_seq_full),
            })
            print("  Saved.")
