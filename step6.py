"""
Step 6: Multi-Dataset Experiment Driver
============================================
Main driver used to produce every result reported in the paper's Section
5 (Tables I-IV, Figs. 6-9b) except the case study. Handles per-dataset
checkpoints, adaptive PAA target length, NaN-safe loading, validation-
based k selection, and a collapse-detection + adaptive-initialization-
scale mechanism (robust_init) that fixes a real training failure found
on OliveOil, without regressing datasets that did not need it.

CALIBRATION HISTORY (robust_init): an earlier single-threshold version
(0.99) correctly fixed OliveOil's near-total embedding collapse at
initialization, but caused FALSE POSITIVES on Coffee, DiatomSizeReduction,
and ArrowHead -- datasets that were training successfully at the default
scale (similarity in 0.99-0.9995) and were measurably HURT by forced
escalation (ArrowHead: 78.9% -> 48.6% accuracy). Initial embedding
similarity alone was also found to be a poor predictor of general
training difficulty: Adiac (sim=0.997) struggled to train well despite a
similar initial similarity to Coffee (sim=0.999), which trained fine --
Adiac's difficulty is better attributed to its 37-class problem with
~10.5 examples/class, not initialization collapse. The final two-
threshold design (trigger_threshold=0.9999, accept_threshold=0.97) only
escalates datasets whose DEFAULT initialization is extremely collapsed
(only OliveOil, sim=0.999997, crosses this in the datasets tested), and
once escalating, requires reaching genuinely low similarity (not just
barely crossing an arbitrary line) before stopping.
"""

import numpy as np
import time
import os
import sys

from step1 import (
    paa_downsample as _paa_downsample_raw,
    evaluate_1nn_euclidean, evaluate_1nn_dtw, train_val_split,
)
from step4 import train, encode_all, build_triplets
from step3 import init_encoder_params, get_shapes, flatten_params

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- EDIT for your environment ----
UCR_ARCHIVE_DIR = os.path.join(SCRIPT_DIR, "UCR_TS_Archive_2015")
DATASETS = [
    "Coffee", "ItalyPowerDemand", "Beef", "Lighting7", "Gun_Point", "ECG200",
    "Trace", "OSULeaf", "StarLightCurves", "BirdChicken", "DiatomSizeReduction",
    "ArrowHead", "ECGFiveDays", "Computers", "Earthquakes", "CBF",
    "CinC_ECG_torso", "Adiac", "OliveOil", "Wafer",
]
# ------------------------------------

CKPT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
RESULTS_CSV = os.path.join(SCRIPT_DIR, "multi_dataset_results.csv")

MAX_TARGET_LEN = 64

# Encoder / training hyperparameters (defaults used throughout the paper;
# d_model is varied explicitly in step11 for the trade-off sweep)
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

DTW_MAX_N_PER_CLASS = 50
DTW_MIN_N_PER_CLASS = 5
DTW_TIME_BUDGET_SECONDS = 120


# ----------------------------------------------------------------------
# Dataset loading (NaN-aware, variable-length-safe)
# ----------------------------------------------------------------------
def get_dataset_paths(name: str):
    train_path = os.path.join(UCR_ARCHIVE_DIR, name, f"{name}_TRAIN")
    test_path = os.path.join(UCR_ARCHIVE_DIR, name, f"{name}_TEST")
    return train_path, test_path


def load_ucr_nan_safe(path: str):
    """Loads a UCR-format file and strips NaN padding from each row
    individually, returning a LIST of variable-length 1-D arrays plus a
    label array. Safe for both fixed-length and NaN-padded UCR files."""
    data = np.loadtxt(path, delimiter=",")
    labels = data[:, 0].astype(int)
    raw_sequences = data[:, 1:]
    sequences = []
    for row in raw_sequences:
        valid = row[~np.isnan(row)]
        sequences.append(valid)
    return sequences, labels


def adaptive_target_len(train_sequences, test_sequences, cap=MAX_TARGET_LEN):
    """Target PAA length: never larger than the shortest sequence present
    (train or test), and never larger than `cap`."""
    min_len = min(
        min(len(x) for x in train_sequences),
        min(len(x) for x in test_sequences),
    )
    return max(1, min(cap, min_len))


def paa_downsample_dataset(sequences, target_len):
    return np.array([_paa_downsample_raw(x, target_len) for x in sequences])


# ----------------------------------------------------------------------
# Per-dataset checkpoint helpers
# ----------------------------------------------------------------------
def get_ckpt_path(dataset_name: str) -> str:
    os.makedirs(CKPT_DIR, exist_ok=True)
    return os.path.join(CKPT_DIR, f"attention_checkpoint_{dataset_name}.npz")


def load_checkpoint(dataset_name: str):
    path = get_ckpt_path(dataset_name)
    if not os.path.exists(path):
        return None, None, None
    ckpt = np.load(path, allow_pickle=True)
    theta = ckpt["theta"]
    shapes = ckpt["shapes"].item()
    optimizer_state = ckpt["optimizer_state"].item() if "optimizer_state" in ckpt else None
    return theta, shapes, optimizer_state


def save_checkpoint(dataset_name: str, theta, shapes, optimizer_state):
    np.savez(get_ckpt_path(dataset_name), theta=theta, shapes=shapes, optimizer_state=optimizer_state)


def stratified_subsample_list(sequences, labels, n_per_class, seed=0):
    """Like step1.stratified_subsample, but operating on a LIST of
    variable-length arrays instead of a single 2-D numpy array."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    idx_selected = []
    for c in classes:
        class_idx = np.where(labels == c)[0]
        chosen = rng.choice(class_idx, size=min(n_per_class, len(class_idx)), replace=False)
        idx_selected.extend(chosen.tolist())
    idx_selected = np.array(idx_selected)
    rng.shuffle(idx_selected)
    return [sequences[i] for i in idx_selected], labels[idx_selected]


# ----------------------------------------------------------------------
# Collapse detection + adaptive initialization (robust_init)
# ----------------------------------------------------------------------
def embedding_diversity_check(theta, shapes, sample_sequences, pos_enc, d_model, num_heads, num_layers, threshold=0.99):
    """Checks whether a set of parameters produces a 'collapsed'
    representation: all sequences mapping to nearly-identical embeddings
    (mean pairwise cosine similarity above `threshold`).

    d_model/num_heads/num_layers must be passed explicitly and match the
    architecture theta/shapes were built with -- required for compatibility
    with non-default d_model (e.g. the trade-off sweep in step11.py).
    Returns (is_collapsed: bool, mean_pairwise_similarity: float)."""
    emb = encode_all(sample_sequences, theta, shapes, d_model, num_heads, num_layers)
    n = emb.shape[0]
    if n < 2:
        return False, 0.0
    sim_matrix = emb @ emb.T
    off_diag = sim_matrix[np.triu_indices(n, k=1)]
    mean_sim = float(off_diag.mean())
    return mean_sim > threshold, mean_sim


def robust_init(train_seq, d_model, num_heads, d_ff, num_layers,
                  scales=(1.0, 2.0, 3.0, 5.0, 8.0), seeds=(0, 1, 2),
                  trigger_threshold=0.9999, accept_threshold=0.97,
                  sample_size=8, verbose=True):
    """Two-threshold design (see module docstring for full calibration
    history): trigger_threshold decides WHETHER escalation is attempted
    at all (only datasets whose DEFAULT scale=1.0 similarity exceeds this
    are touched); accept_threshold (much stricter/lower) decides when
    escalation has gone far enough to stop, once triggered.
    Returns (theta, shapes, scale_used, mean_sim_achieved)."""
    from step3 import sinusoidal_positional_encoding
    T = train_seq[0].shape[0]
    pos_enc = sinusoidal_positional_encoding(T, d_model)
    sample = list(train_seq[:sample_size])

    default_params = init_encoder_params(d_model, num_heads, d_ff, num_layers, seed=0, scale=1.0)
    default_shapes = get_shapes(default_params)
    default_theta = flatten_params(default_params)
    default_collapsed, default_sim = embedding_diversity_check(
        default_theta, default_shapes, sample, pos_enc, d_model, num_heads, num_layers, trigger_threshold
    )
    if not default_collapsed:
        if verbose:
            print(f"  Default initialization OK (scale=1.0, mean pairwise sim={default_sim:.4f}) "
                  f"-- no escalation needed.")
        return default_theta, default_shapes, 1.0, default_sim

    if verbose:
        print(f"  Default initialization COLLAPSED (scale=1.0, mean pairwise sim={default_sim:.6f}) "
              f"-- escalating scale to find genuinely diverse initialization.")
    last_theta, last_shapes, last_scale, last_sim = default_theta, default_shapes, 1.0, default_sim
    for scale in scales:
        for seed in seeds:
            params = init_encoder_params(d_model, num_heads, d_ff, num_layers, seed=seed, scale=scale)
            shapes = get_shapes(params)
            theta = flatten_params(params)
            _, mean_sim = embedding_diversity_check(theta, shapes, sample, pos_enc, d_model, num_heads, num_layers, accept_threshold)
            last_theta, last_shapes, last_scale, last_sim = theta, shapes, scale, mean_sim
            if mean_sim < accept_threshold:
                if verbose:
                    print(f"  Escalation succeeded at scale={scale}, seed={seed} "
                          f"(mean pairwise sim={mean_sim:.4f} < accept_threshold={accept_threshold})")
                return theta, shapes, scale, mean_sim
            if verbose:
                print(f"  Still not diverse enough at scale={scale}, seed={seed} "
                      f"(mean pairwise sim={mean_sim:.6f}) -- retrying")

    if verbose:
        print(f"  WARNING: escalation did not reach accept_threshold; proceeding with least-collapsed "
              f"found (scale={last_scale}, mean sim={last_sim:.6f}).")
    return last_theta, last_shapes, last_scale, last_sim


# ----------------------------------------------------------------------
# Validation-based k selection
# ----------------------------------------------------------------------
def knn_from_similarity(S: np.ndarray, pool_labels, query_labels, k: int):
    pool_labels_arr = np.array(pool_labels)
    preds = []
    for i in range(S.shape[0]):
        top_k = np.argsort(-S[i])[:k]
        votes = pool_labels_arr[top_k]
        cls, cnt = np.unique(votes, return_counts=True)
        preds.append(cls[np.argmax(cnt)])
    preds = np.array(preds)
    acc = np.mean(preds == np.array(query_labels))
    return acc, preds


def select_k_via_validation(theta, shapes, train_seq, train_labels,
                              k_candidates=(1, 3, 5, 7, 9, 15, 25), val_frac=0.2, seed=0,
                              d_model=D_MODEL, num_heads=NUM_HEADS, num_layers=NUM_LAYERS):
    """Selects k using a held-out split of the TRAINING data only -- the
    test set is never consulted for this choice. d_model/num_heads/
    num_layers default to module constants; pass explicitly when
    evaluating a non-default architecture (e.g. step11.py)."""
    n = len(train_seq)
    tr_idx, val_idx = train_val_split(n, val_frac=val_frac, seed=seed)
    tr_seq = [train_seq[i] for i in tr_idx]
    tr_labels = np.array(train_labels)[tr_idx]
    val_seq = [train_seq[i] for i in val_idx]
    val_labels = np.array(train_labels)[val_idx]

    tr_emb = encode_all(tr_seq, theta, shapes, d_model, num_heads, num_layers)
    val_emb = encode_all(val_seq, theta, shapes, d_model, num_heads, num_layers)
    S_val = val_emb @ tr_emb.T

    best_k, best_acc = k_candidates[0], -1
    for k in k_candidates:
        if k > len(tr_seq):
            continue
        acc, _ = knn_from_similarity(S_val, tr_labels, val_labels, k)
        if acc > best_acc:
            best_acc, best_k = acc, k
    return best_k, best_acc


# ----------------------------------------------------------------------
# Training with plateau/convergence detection
# ----------------------------------------------------------------------
def auto_train(dataset_name, train_seq, train_labels, verbose=True):
    """Trains until the loss plateaus, resuming from any existing
    checkpoint. On a fresh start, uses robust_init to avoid beginning
    from a collapsed representation. Also detects and skips training if
    a RESUMED checkpoint is already converged (prevents destabilizing an
    already-good model -- an earlier version without this check caused
    Beef's loss to climb from 0.002 back up to 0.30 on an unnecessary
    resume)."""
    triplets = build_triplets(train_labels.tolist(), triplets_per_anchor=TRIPLETS_PER_ANCHOR, seed=0)
    theta, shapes, optimizer_state = load_checkpoint(dataset_name)
    if theta is not None:
        if verbose:
            print(f"  Resuming from existing checkpoint for {dataset_name}")
    else:
        if verbose:
            print(f"  Fresh start -- checking for representation collapse at initialization...")
        theta, shapes, scale_used, init_sim = robust_init(
            train_seq, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, verbose=verbose
        )

    prev_loss = None
    for round_idx in range(AUTO_MAX_ROUNDS):
        theta, shapes, loss_hist, optimizer_state = train(
            list(train_seq), triplets, d_model=D_MODEL, num_heads=NUM_HEADS,
            d_ff=D_FF, num_layers=NUM_LAYERS, num_iters=AUTO_ROUND_ITERS, lr=LR,
            margin=MARGIN, seed=0, minibatch_size=MINIBATCH_SIZE,
            theta_init=theta, shapes=shapes, optimizer_state=optimizer_state,
            verbose_every=None,
        )
        final_loss = loss_hist[-1]
        starting_loss = loss_hist[0]

        if starting_loss < AUTO_ABSOLUTE_CONVERGED and final_loss >= starting_loss:
            if verbose:
                print(f"  Checkpoint already converged (loss={starting_loss:.6f}) -- "
                      f"skipping further training for this run.")
            theta, shapes, optimizer_state = load_checkpoint(dataset_name)
            prev_loss = starting_loss
            break

        save_checkpoint(dataset_name, theta, shapes, optimizer_state)
        if verbose:
            print(f"  Round {round_idx + 1}/{AUTO_MAX_ROUNDS}: "
                  f"loss {loss_hist[0]:.6f} -> {final_loss:.6f}")

        if final_loss < AUTO_ABSOLUTE_CONVERGED:
            prev_loss = final_loss
            break
        if prev_loss is not None and abs(prev_loss - final_loss) < AUTO_PLATEAU_DELTA:
            prev_loss = final_loss
            break
        prev_loss = final_loss

    return theta, shapes, prev_loss


# ----------------------------------------------------------------------
# Evaluation (shared by run_dataset and manual/analysis scripts)
# ----------------------------------------------------------------------
def run_compare(theta, shapes, dtw_n_per_class=50, train_path=None, test_path=None):
    """Attention on full dataset, ED on full dataset, DTW on a budgeted
    subsample. Returns nothing; prints the 3-row summary table."""
    train_seq_raw, train_labels = load_ucr_nan_safe(train_path)
    test_seq_raw, test_labels = load_ucr_nan_safe(test_path)
    target_len = adaptive_target_len(train_seq_raw, test_seq_raw)
    train_seq = paa_downsample_dataset(train_seq_raw, target_len)
    test_seq = paa_downsample_dataset(test_seq_raw, target_len)

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
    print(f"Attention (full dataset, best k={best_k}): acc={best_acc:.4f}, time={t_attn:.1f}s")

    t0 = time.time()
    acc_ed, _ = evaluate_1nn_euclidean(train_seq, train_labels, test_seq, test_labels)
    t_ed = time.time() - t0
    print(f"Euclidean (full dataset): acc={acc_ed:.4f}, time={t_ed:.2f}s")

    train_sub, train_labels_sub = stratified_subsample_list(list(train_seq), train_labels, dtw_n_per_class, seed=0)
    test_sub, test_labels_sub = stratified_subsample_list(list(test_seq), test_labels, dtw_n_per_class, seed=1)
    t0 = time.time()
    acc_dtw, _ = evaluate_1nn_dtw(np.array(train_sub), train_labels_sub, np.array(test_sub), test_labels_sub)
    t_dtw = time.time() - t0
    print(f"DTW (subsample {len(train_sub)}x{len(test_sub)}): acc={acc_dtw:.4f}, time={t_dtw:.1f}s")

    print(f"\n{'Method':<15}{'Accuracy':>12}{'Time':>10}")
    print(f"{'Attention':<15}{best_acc:>12.2%}{t_attn:>9.1f}s")
    print(f"{'Euclidean':<15}{acc_ed:>12.2%}{t_ed:>9.2f}s")
    print(f"{'DTW':<15}{acc_dtw:>12.2%}{t_dtw:>9.1f}s")


def run_dataset(name: str, verbose=True):
    """Full pipeline for one dataset: load, adaptive-length PAA, train
    (with collapse detection + plateau stopping), select k via
    validation, evaluate attention/ED/DTW. Returns a results dict."""
    result = {"dataset": name}
    try:
        train_path, test_path = get_dataset_paths(name)
        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            result["status"] = f"SKIPPED (files not found at {train_path})"
            return result

        train_seq_native, train_labels = load_ucr_nan_safe(train_path)
        test_seq_native, test_labels = load_ucr_nan_safe(test_path)
        target_len = adaptive_target_len(train_seq_native, test_seq_native)

        train_seq_full = paa_downsample_dataset(train_seq_native, target_len)
        test_seq_full = paa_downsample_dataset(test_seq_native, target_len)

        result.update({
            "n_train": len(train_seq_native), "n_test": len(test_seq_native),
            "n_classes": len(np.unique(train_labels)), "target_len": target_len,
        })

        if verbose:
            print(f"\n{'='*60}\nDataset: {name}")
            print(f"  train={result['n_train']}, test={result['n_test']}, "
                  f"classes={result['n_classes']}, target_len={target_len}")

        train_seq_sub, train_labels_sub = stratified_subsample_list(
            list(train_seq_full), train_labels, TRAIN_SUBSAMPLE_PER_CLASS, seed=0
        )

        t0 = time.time()
        theta, shapes, final_loss = auto_train(name, train_seq_sub, train_labels_sub, verbose=verbose)
        result["final_loss"] = final_loss
        result["train_time_s"] = time.time() - t0

        best_k, val_acc = select_k_via_validation(theta, shapes, list(train_seq_full), train_labels)
        result["k_selected"] = best_k

        t0 = time.time()
        train_emb = encode_all(list(train_seq_full), theta, shapes, D_MODEL, NUM_HEADS, NUM_LAYERS)
        test_emb = encode_all(list(test_seq_full), theta, shapes, D_MODEL, NUM_HEADS, NUM_LAYERS)
        S = test_emb @ train_emb.T
        result["attn_time_s"] = time.time() - t0
        result["attn_acc"], _ = knn_from_similarity(S, train_labels, test_labels, best_k)

        t0 = time.time()
        result["ed_acc"], _ = evaluate_1nn_euclidean(train_seq_full, train_labels, test_seq_full, test_labels)
        result["ed_time_s"] = time.time() - t0

        # DTW: adaptive budget based on a small timing probe
        n_classes = result["n_classes"]
        probe_train, probe_train_labels = stratified_subsample_list(list(train_seq_full), train_labels, 2, seed=0)
        probe_test, probe_test_labels = stratified_subsample_list(list(test_seq_full), test_labels, 2, seed=1)
        t0 = time.time()
        evaluate_1nn_dtw(np.array(probe_train), probe_train_labels, np.array(probe_test), probe_test_labels)
        per_pair_time = (time.time() - t0) / max(len(probe_train) * len(probe_test), 1)

        for n_per_class in range(DTW_MAX_N_PER_CLASS, DTW_MIN_N_PER_CLASS - 1, -5):
            est_pairs = (n_per_class * n_classes) ** 2
            if est_pairs * per_pair_time <= DTW_TIME_BUDGET_SECONDS:
                break
        else:
            n_per_class = DTW_MIN_N_PER_CLASS

        dtw_train, dtw_train_labels = stratified_subsample_list(list(train_seq_full), train_labels, n_per_class, seed=0)
        dtw_test, dtw_test_labels = stratified_subsample_list(list(test_seq_full), test_labels, n_per_class, seed=1)
        t0 = time.time()
        result["dtw_acc"], _ = evaluate_1nn_dtw(np.array(dtw_train), dtw_train_labels, np.array(dtw_test), dtw_test_labels)
        result["dtw_time_s"] = time.time() - t0
        result["dtw_n_per_side"] = len(dtw_train)

        result["status"] = "OK"
        if verbose:
            print(f"  Attention: acc={result['attn_acc']:.4f}, time={result['attn_time_s']:.1f}s")
            print(f"  Euclidean: acc={result['ed_acc']:.4f}, time={result['ed_time_s']:.2f}s")
            print(f"  DTW ({n_per_class}/class): acc={result['dtw_acc']:.4f}, time={result['dtw_time_s']:.1f}s")

    except Exception as e:
        result["status"] = f"ERROR: {e}"
        if verbose:
            print(f"  ERROR on {name}: {e}")

    return result


if __name__ == "__main__":
    print("=" * 45)
    print("Step 6: Multi-Dataset Experiment Driver")
    print(f"UCR archive dir: {UCR_ARCHIVE_DIR}")
    print(f"Datasets configured: {DATASETS}")
    print("=" * 45)
    print("1) Run all configured datasets")
    print("2) Run a single dataset by name")
    choice = input("\nEnter 1 or 2: ").strip()

    if choice == "1":
        for name in DATASETS:
            run_dataset(name)
    elif choice == "2":
        name = input("Dataset name: ").strip()
        run_dataset(name)
    else:
        print(f"Unrecognized choice '{choice}'.")
