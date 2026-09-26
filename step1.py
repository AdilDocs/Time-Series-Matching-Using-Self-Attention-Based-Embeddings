"""
Step 1: Data Loading, Preprocessing, and Baseline Utilities
================================================================
Loads UCR-format datasets, preprocesses sequences (PAA downsampling),
and provides the classical baseline methods (Euclidean 1-NN, DTW 1-NN)
compared against the proposed self-attention encoder throughout the paper.
"""

import numpy as np


def load_ucr(path: str):
    """Loads a UCR-format file (comma-delimited; first column = integer
    class label, remaining columns = sequence values). Returns
    (sequences, labels) as numpy arrays. Assumes fixed-length rows with
    no NaN padding -- use step6.load_ucr_nan_safe for NaN-padded or
    variable-length archives."""
    data = np.loadtxt(path, delimiter=",")
    labels = data[:, 0].astype(int)
    sequences = data[:, 1:]
    return sequences, labels


def paa_downsample(x: np.ndarray, target_len: int) -> np.ndarray:
    """Piecewise Aggregate Approximation: averages consecutive segments to
    reduce sequence length while preserving coarse shape."""
    n = len(x)
    indices = np.linspace(0, n, target_len + 1).astype(int)
    return np.array([x[indices[i]:indices[i + 1]].mean() for i in range(target_len)])


def stratified_subsample(sequences: np.ndarray, labels: np.ndarray, n_per_class: int, seed: int = 0):
    """Balanced random subsample: up to n_per_class examples per class."""
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    idx_selected = []
    for c in classes:
        class_idx = np.where(labels == c)[0]
        chosen = rng.choice(class_idx, size=min(n_per_class, len(class_idx)), replace=False)
        idx_selected.extend(chosen.tolist())
    idx_selected = np.array(idx_selected)
    rng.shuffle(idx_selected)
    return sequences[idx_selected], labels[idx_selected]


def train_val_split(n: int, val_frac: float = 0.2, seed: int = 0):
    """Returns (train_idx, val_idx) index arrays for a held-out validation split."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    return idx[n_val:], idx[:n_val]


def evaluate_1nn_euclidean(train_seq, train_labels, test_seq, test_labels):
    """Euclidean distance 1-nearest-neighbor classification."""
    preds = []
    for x in test_seq:
        dists = np.linalg.norm(train_seq - x[None, :], axis=1)
        preds.append(train_labels[np.argmin(dists)])
    preds = np.array(preds)
    acc = np.mean(preds == test_labels)
    return acc, preds


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Standard Dynamic Time Warping distance (O(n*m) dynamic programming,
    squared-Euclidean local cost)."""
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = (ai - b[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return np.sqrt(D[n, m])


def evaluate_1nn_dtw(train_seq, train_labels, test_seq, test_labels):
    """DTW distance 1-nearest-neighbor classification. O(n_test * n_train *
    L^2) -- only tractable at modest scale; used on budgeted subsamples
    throughout the paper's experiments (see step6.py)."""
    preds = []
    for x in test_seq:
        dists = np.array([dtw_distance(x, y) for y in train_seq])
        preds.append(train_labels[np.argmin(dists)])
    preds = np.array(preds)
    acc = np.mean(preds == test_labels)
    return acc, preds


if __name__ == "__main__":
    print("step1.py: data loading, PAA downsampling, Euclidean/DTW baselines.")
    print("Run against a UCR dataset directory to sanity-check, e.g.:")
    print("  train_seq, train_labels = load_ucr('StarLightCurves_TRAIN')")
