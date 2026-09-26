"""
Step 5: Train and Evaluate on a Single Dataset (StarLightCurves default)
=============================================================================
Simpler, single-dataset predecessor to step6.py's multi-dataset driver.
Checkpoint path is anchored to the script's own directory (not the
working directory an IDE happens to launch from) to avoid checkpoints
appearing to "disappear" between runs.
"""

import numpy as np
import time
import os
import sys

from step1 import load_ucr, paa_downsample, stratified_subsample, evaluate_1nn_euclidean, evaluate_1nn_dtw
from step4 import train, encode_all, build_triplets

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(SCRIPT_DIR, "attention_checkpoint.npz")
TRAIN_PATH = os.path.join(SCRIPT_DIR, "StarLightCurves_TRAIN")
TEST_PATH = os.path.join(SCRIPT_DIR, "StarLightCurves_TEST")

TARGET_LEN = 64
D_MODEL = 16
NUM_HEADS = 4
D_FF = 32
NUM_LAYERS = 2
MARGIN = 0.3
LR = 0.015
MINIBATCH_SIZE = 16
TRIPLETS_PER_ANCHOR = 2
TRAIN_SUBSAMPLE_PER_CLASS = 50

AUTO_ROUND_ITERS = 25
AUTO_MAX_ROUNDS = 12
AUTO_PLATEAU_DELTA = 0.002
AUTO_ABSOLUTE_CONVERGED = 0.01


def knn_from_similarity(S, train_labels, test_labels, k):
    train_labels_arr = np.array(train_labels)
    preds = []
    for i in range(S.shape[0]):
        top_k = np.argsort(-S[i])[:k]
        votes = train_labels_arr[top_k]
        cls, cnt = np.unique(votes, return_counts=True)
        preds.append(cls[np.argmax(cnt)])
    preds = np.array(preds)
    acc = np.mean(preds == np.array(test_labels))
    return acc, preds


def load_checkpoint():
    if not os.path.exists(CKPT_PATH):
        return None, None, None
    ckpt = np.load(CKPT_PATH, allow_pickle=True)
    theta = ckpt["theta"]
    shapes = ckpt["shapes"].item()
    optimizer_state = ckpt["optimizer_state"].item() if "optimizer_state" in ckpt else None
    return theta, shapes, optimizer_state


def save_checkpoint(theta, shapes, optimizer_state):
    np.savez(CKPT_PATH, theta=theta, shapes=shapes, optimizer_state=optimizer_state)


def load_training_data():
    train_seq_raw, train_labels_full = load_ucr(TRAIN_PATH)
    train_seq_sub, train_labels = stratified_subsample(
        train_seq_raw, train_labels_full, TRAIN_SUBSAMPLE_PER_CLASS, seed=0
    )
    train_seq = np.array([paa_downsample(x, TARGET_LEN) for x in train_seq_sub])
    triplets = build_triplets(train_labels.tolist(), triplets_per_anchor=TRIPLETS_PER_ANCHOR, seed=0)
    return train_seq, train_labels, triplets


def run_compare(theta, shapes, dtw_n_per_class=50):
    train_seq_raw, train_labels = load_ucr(TRAIN_PATH)
    test_seq_raw, test_labels = load_ucr(TEST_PATH)
    train_seq = np.array([paa_downsample(x, TARGET_LEN) for x in train_seq_raw])
    test_seq = np.array([paa_downsample(x, TARGET_LEN) for x in test_seq_raw])

    t0 = time.time()
    train_emb = encode_all(list(train_seq), theta, shapes, D_MODEL, NUM_HEADS, NUM_LAYERS)
    test_emb = encode_all(list(test_seq), theta, shapes, D_MODEL, NUM_HEADS, NUM_LAYERS)
    S = test_emb @ train_emb.T
    t_attn = time.time() - t0

    best_k, best_acc = None, -1
    for k in [1, 3, 5, 7, 9, 15, 25]:
        acc, _ = knn_from_similarity(S, train_labels, test_labels, k)
        if acc > best_acc:
            best_acc, best_k = acc, k
    print(f"Attention (best k={best_k}): acc={best_acc:.4f}, time={t_attn:.1f}s")

    t0 = time.time()
    acc_ed, _ = evaluate_1nn_euclidean(train_seq, train_labels, test_seq, test_labels)
    t_ed = time.time() - t0
    print(f"Euclidean: acc={acc_ed:.4f}, time={t_ed:.2f}s")

    train_sub, train_labels_sub = stratified_subsample(train_seq_raw, train_labels, dtw_n_per_class, seed=0)
    test_sub, test_labels_sub = stratified_subsample(test_seq_raw, test_labels, dtw_n_per_class, seed=1)
    train_sub_ds = np.array([paa_downsample(x, TARGET_LEN) for x in train_sub])
    test_sub_ds = np.array([paa_downsample(x, TARGET_LEN) for x in test_sub])
    t0 = time.time()
    acc_dtw, _ = evaluate_1nn_dtw(train_sub_ds, train_labels_sub, test_sub_ds, test_labels_sub)
    t_dtw = time.time() - t0
    print(f"DTW ({dtw_n_per_class}/class): acc={acc_dtw:.4f}, time={t_dtw:.1f}s")

    print(f"\n{'Method':<15}{'Accuracy':>12}{'Time':>10}")
    print(f"{'Attention':<15}{best_acc:>12.2%}{t_attn:>9.1f}s")
    print(f"{'Euclidean':<15}{acc_ed:>12.2%}{t_ed:>9.2f}s")
    print(f"{'DTW':<15}{acc_dtw:>12.2%}{t_dtw:>9.1f}s")


if __name__ == "__main__":
    print(f"StarLightCurves: Self-Attention Encoder (checkpoint: {CKPT_PATH})")
    print("1) Train / continue training")
    print("2) Compare (3-row summary table)")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        raw = input("Number of iterations [default 20]: ").strip()
        n_iters = int(raw) if raw else 20
        train_seq, train_labels, triplets = load_training_data()
        theta_init, shapes, optimizer_state = load_checkpoint()
        theta, shapes, loss_hist, opt_state = train(
            list(train_seq), triplets, d_model=D_MODEL, num_heads=NUM_HEADS,
            d_ff=D_FF, num_layers=NUM_LAYERS, num_iters=n_iters, lr=LR,
            margin=MARGIN, seed=0, minibatch_size=MINIBATCH_SIZE,
            theta_init=theta_init, shapes=shapes, optimizer_state=optimizer_state,
            verbose_every=5,
        )
        print(f"Loss: {loss_hist[0]:.6f} -> {loss_hist[-1]:.6f}")
        save_checkpoint(theta, shapes, opt_state)
    elif choice == "2":
        theta, shapes, _ = load_checkpoint()
        if theta is None:
            print("No checkpoint found -- choose option 1 first.")
            sys.exit(1)
        run_compare(theta, shapes)
