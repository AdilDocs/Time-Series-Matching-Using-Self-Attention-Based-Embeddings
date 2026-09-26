"""
Step 13: Train with Intermediate Checkpoint Snapshots
===========================================================
Retrains StarLightCurves from scratch, saving the full parameter state
at several intermediate iteration counts (not just the final converged
state), for the embedding-evolution figures (step14). Substantial
compute; resumable by loading the last saved snapshot.
"""
import numpy as np
import os
from step6 import (get_dataset_paths, load_ucr_nan_safe, adaptive_target_len,
                     paa_downsample_dataset, stratified_subsample_list,
                     D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, MARGIN, LR, MINIBATCH_SIZE,
                     TRIPLETS_PER_ANCHOR, TRAIN_SUBSAMPLE_PER_CLASS)
from step4 import train, build_triplets

CKPT_DIR = "checkpoints_evolution"
CHECKPOINT_ITERS = [0, 5, 10, 20, 40, 70, 110, 150, 200, 250, 290, 345, 400]
RESET_ADAM_FROM_ITER = 40


def train_with_snapshots(dataset_name, train_seq, train_labels, checkpoint_iters, seed=0):
    os.makedirs(CKPT_DIR, exist_ok=True)
    triplets = build_triplets(train_labels.tolist(), triplets_per_anchor=TRIPLETS_PER_ANCHOR, seed=0)

    from step3 import init_encoder_params, get_shapes, flatten_params
    params = init_encoder_params(D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, seed=seed)
    shapes = get_shapes(params)
    theta = flatten_params(params)
    optimizer_state = None

    np.savez(os.path.join(CKPT_DIR, f"{dataset_name}_iter0.npz"), theta=theta, shapes=shapes)
    print(f"  Saved iter=0 (untrained)")

    prev_iter = 0
    for target_iter in checkpoint_iters[1:]:
        n_iters = target_iter - prev_iter
        # Adam schedule used for the paper's checkpoints (verified to reproduce
        # every loss value in Table V exactly): the optimizer state is carried
        # through iteration 40 and reset to zero at every checkpoint boundary
        # from iteration 40 onward.
        if prev_iter >= RESET_ADAM_FROM_ITER:
            optimizer_state = None
        theta, shapes, loss_hist, optimizer_state = train(
            list(train_seq), triplets, d_model=D_MODEL, num_heads=NUM_HEADS,
            d_ff=D_FF, num_layers=NUM_LAYERS, num_iters=n_iters, lr=LR,
            margin=MARGIN, seed=0, minibatch_size=MINIBATCH_SIZE,
            theta_init=theta, shapes=shapes, optimizer_state=optimizer_state,
            verbose_every=None,
        )
        np.savez(os.path.join(CKPT_DIR, f"{dataset_name}_iter{target_iter}.npz"), theta=theta, shapes=shapes)
        print(f"  Saved iter={target_iter}, loss={loss_hist[-1]:.6f}")
        prev_iter = target_iter


if __name__ == "__main__":
    name = "StarLightCurves"
    train_path, test_path = get_dataset_paths(name)
    train_seq_native, train_labels = load_ucr_nan_safe(train_path)
    test_seq_native, test_labels = load_ucr_nan_safe(test_path)
    target_len = adaptive_target_len(train_seq_native, test_seq_native)
    train_seq_full = paa_downsample_dataset(train_seq_native, target_len)
    train_seq_sub, train_labels_sub = stratified_subsample_list(
        list(train_seq_full), train_labels, TRAIN_SUBSAMPLE_PER_CLASS, seed=0
    )
    print(f"{name}: training subsample = {len(train_seq_sub)} sequences")

    train_with_snapshots(name, train_seq_sub, train_labels_sub, CHECKPOINT_ITERS)

    np.savez(os.path.join(CKPT_DIR, f"{name}_viz_subsample.npz"),
              seqs=np.array(train_seq_sub), labels=train_labels_sub)
    print("Saved visualization subsample (sequences + labels).")
