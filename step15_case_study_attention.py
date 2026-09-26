"""
Step 15: Case Study -- Attention Weight Visualization (Gun_Point)
========================================================================
Reproduces the paper's Case Study subsection: finds a test example the
proposed encoder classifies correctly but Euclidean distance classifies
incorrectly, extracts the trained encoder's attention weights for that
example and its retrieved nearest neighbor, and visualizes attention
intensity overlaid on both sequences.

Requires an existing trained checkpoint for Gun_Point (step6.py).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from step6 import (get_dataset_paths, load_ucr_nan_safe, adaptive_target_len,
                     paa_downsample_dataset, load_checkpoint,
                     D_MODEL, NUM_HEADS, NUM_LAYERS)
from step4 import encode_all
from step3 import (encode_sequence_with_attention, build_param_vars,
                     sinusoidal_positional_encoding, unflatten_params)

DATASET_NAME = "Gun_Point"


def find_illustrative_example(train_seq_full, train_labels, test_seq_full, test_labels, theta, shapes):
    train_emb = encode_all(list(train_seq_full), theta, shapes, D_MODEL, NUM_HEADS, NUM_LAYERS)
    test_emb = encode_all(list(test_seq_full), theta, shapes, D_MODEL, NUM_HEADS, NUM_LAYERS)
    S = test_emb @ train_emb.T

    attn_preds = np.array([train_labels[np.argmax(S[i])] for i in range(S.shape[0])])

    ed_preds = []
    for x in test_seq_full:
        dists = np.linalg.norm(train_seq_full - x[None, :], axis=1)
        ed_preds.append(train_labels[np.argmin(dists)])
    ed_preds = np.array(ed_preds)

    candidates = np.where((attn_preds == test_labels) & (ed_preds != test_labels))[0]
    return candidates, train_emb, test_emb, ed_preds


def extract_attention_for_example(query_seq, param_vars, pos_enc):
    _, attn_weights_by_layer = encode_sequence_with_attention(
        query_seq, param_vars, pos_enc, NUM_HEADS, NUM_LAYERS
    )
    final_layer = attn_weights_by_layer[-1]
    avg_attn = np.mean(final_layer, axis=0)
    return avg_attn.sum(axis=0)


def plot_attention_overlay(seq, attn, title, ax):
    t = np.arange(len(seq))
    points = np.array([t, seq]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(attn.min(), attn.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(attn[:-1])
    lc.set_linewidth(2.5)
    line = ax.add_collection(lc)
    ax.scatter(t, seq, c=attn, cmap='viridis', norm=norm, s=15, zorder=3)
    ax.set_xlim(t.min(), t.max())
    ax.set_ylim(seq.min() - 0.5, seq.max() + 0.5)
    ax.set_title(title, fontsize=10)
    return line


if __name__ == "__main__":
    train_path, test_path = get_dataset_paths(DATASET_NAME)
    train_seq_native, train_labels = load_ucr_nan_safe(train_path)
    test_seq_native, test_labels = load_ucr_nan_safe(test_path)
    target_len = adaptive_target_len(train_seq_native, test_seq_native)
    train_seq_full = paa_downsample_dataset(train_seq_native, target_len)
    test_seq_full = paa_downsample_dataset(test_seq_native, target_len)

    theta, shapes, _ = load_checkpoint(DATASET_NAME)
    if theta is None:
        print(f"No checkpoint found for {DATASET_NAME}. Run step6.py first.")
        exit(1)

    candidates, train_emb, test_emb, ed_preds = find_illustrative_example(
        train_seq_full, train_labels, test_seq_full, test_labels, theta, shapes
    )
    print(f"Found {len(candidates)} examples where attention is correct and ED is wrong: {candidates}")

    query_idx = candidates[0]  # paper uses test index 5, the first candidate
    query_seq = test_seq_full[query_idx]
    query_label = test_labels[query_idx]

    sims = test_emb[query_idx] @ train_emb.T
    nn_idx = np.argmax(sims)
    nn_seq = train_seq_full[nn_idx]
    nn_label = train_labels[nn_idx]

    print(f"Query idx={query_idx}, true label={query_label}")
    print(f"Attention's nearest neighbor: idx={nn_idx}, label={nn_label}, similarity={sims[nn_idx]:.4f}")
    print(f"ED's (incorrect) nearest neighbor label: {ed_preds[query_idx]}")

    params = unflatten_params(theta, shapes)
    param_vars = build_param_vars(params)
    pos_enc = sinusoidal_positional_encoding(target_len, D_MODEL)

    attn_received_q = extract_attention_for_example(query_seq, param_vars, pos_enc)
    attn_received_nn = extract_attention_for_example(nn_seq, param_vars, pos_enc)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6.5))
    line1 = plot_attention_overlay(query_seq, attn_received_q,
                                      f"Query (test example, true class {query_label})", axes[0])
    line2 = plot_attention_overlay(nn_seq, attn_received_nn,
                                      f"Matched neighbor (train example, class {nn_label}, "
                                      f"similarity={sims[nn_idx]:.4f})", axes[1])
    for ax, line in zip(axes, [line1, line2]):
        cbar = fig.colorbar(line, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("attention received", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"case_study_{DATASET_NAME.lower()}_attention.png", dpi=150, bbox_inches="tight")
    print(f"Saved case_study_{DATASET_NAME.lower()}_attention.png")
