"""
Step 4: Training Loop (Triplet-Margin Loss + Adam)
=======================================================
Trains the self-attention encoder via exact backpropagation (Section 3.3
of the paper) using the autodiff engine from step2.py and the encoder
from step3.py.
"""

import numpy as np
from step2 import Var
from step3 import (
    init_encoder_params, get_shapes, flatten_params, unflatten_params,
    build_param_vars, encode_sequence, embedding_similarity,
    sinusoidal_positional_encoding,
)


class AdamOptimizer:
    def __init__(self, num_params, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = np.zeros(num_params)
        self.v = np.zeros(num_params)
        self.t = 0

    def step(self, theta, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return theta - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def flatten_grad(param_vars: dict) -> np.ndarray:
    return np.concatenate([param_vars[k].grad.reshape(-1) for k in sorted(param_vars.keys())])


def build_triplets(labels, triplets_per_anchor: int = 2, seed: int = 0):
    """Random triplets: for each anchor, `triplets_per_anchor` (positive,
    negative) pairs drawn from same-class / other-class examples."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    triplets = []
    for i in range(n):
        same_class = [k for k in range(n) if labels[k] == labels[i] and k != i]
        other_class = [k for k in range(n) if labels[k] != labels[i]]
        if not same_class or not other_class:
            continue
        for _ in range(triplets_per_anchor):
            j = int(rng.choice(same_class))
            k = int(rng.choice(other_class))
            triplets.append((i, j, k))
    return triplets


def batched_triplet_loss_and_grad(theta_flat, shapes, sequences, triplets, margin,
                                     pos_enc, num_heads, num_layers, minibatch_size=16):
    """Mini-batched triplet margin loss with exact analytic gradients."""
    total_loss_value = 0.0
    grad_accum = np.zeros_like(theta_flat)
    n_triplets = len(triplets)

    for start in range(0, n_triplets, minibatch_size):
        batch = triplets[start:start + minibatch_size]
        params = unflatten_params(theta_flat, shapes)
        pv = build_param_vars(params)

        needed_idx = sorted(set(i for t in batch for i in t))
        embeddings = {}
        for idx in needed_idx:
            embeddings[idx] = encode_sequence(sequences[idx], pv, pos_enc, num_heads, num_layers)

        batch_total = Var(np.array(0.0))
        for i, j, k in batch:
            s_ij = embedding_similarity(embeddings[i], embeddings[j])
            s_ik = embedding_similarity(embeddings[i], embeddings[k])
            hinge = (Var(np.array(margin)) - s_ij + s_ik).maximum_scalar(0.0)
            batch_total = batch_total + hinge

        batch_mean = batch_total * (1.0 / n_triplets)
        batch_mean.backward()

        grad_accum += flatten_grad(pv)
        total_loss_value += float(batch_mean.value)

    return total_loss_value, grad_accum


def train(sequences, triplets, d_model=16, num_heads=4, d_ff=32, num_layers=2,
           num_iters=100, lr=0.015, margin=0.3, seed=0, minibatch_size=16,
           theta_init=None, shapes=None, optimizer_state=None, verbose_every=10):
    """Trains the encoder for num_iters Adam steps. Returns
    (theta, shapes, loss_history, optimizer_state) -- pass theta_init,
    shapes, and optimizer_state back in to resume training later
    (optimizer_state persistence matters: resetting Adam's momentum on
    every resume causes oscillation instead of smooth convergence)."""
    T = sequences[0].shape[0]
    pos_enc = sinusoidal_positional_encoding(T, d_model)

    if theta_init is None:
        params = init_encoder_params(d_model, num_heads, d_ff, num_layers, seed=seed)
        shapes = get_shapes(params)
        theta = flatten_params(params)
    else:
        theta = theta_init

    optimizer = AdamOptimizer(len(theta), lr=lr)
    if optimizer_state is not None:
        optimizer.m = optimizer_state["m"].copy()
        optimizer.v = optimizer_state["v"].copy()
        optimizer.t = int(optimizer_state["t"])

    loss_history = []
    for it in range(num_iters):
        loss_val, grad = batched_triplet_loss_and_grad(
            theta, shapes, sequences, triplets, margin, pos_enc,
            num_heads, num_layers, minibatch_size=minibatch_size,
        )
        theta = optimizer.step(theta, grad)
        loss_history.append(loss_val)
        if verbose_every and (it % verbose_every == 0 or it == num_iters - 1):
            print(f"  iter {it}: loss={loss_val:.6f}, grad_norm={np.linalg.norm(grad):.6f}", flush=True)

    final_optimizer_state = {"m": optimizer.m, "v": optimizer.v, "t": optimizer.t}
    return theta, shapes, loss_history, final_optimizer_state


def encode_all(sequences, theta_flat, shapes, d_model, num_heads, num_layers):
    """Encodes a list of sequences into a (N, d_model) embedding matrix.
    Forward-only -- the fast path used at evaluation time: encode once,
    then compare via dot products."""
    T = sequences[0].shape[0]
    pos_enc = sinusoidal_positional_encoding(T, d_model)
    params = unflatten_params(theta_flat, shapes)
    pv = build_param_vars(params)
    embeddings = []
    for x in sequences:
        e = encode_sequence(x, pv, pos_enc, num_heads, num_layers)
        embeddings.append(e.value)
    return np.array(embeddings)


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    def make_sine(length, noise=0.1):
        t = np.linspace(0, 2 * np.pi, length)
        return np.sin(t) + noise * rng.standard_normal(length)

    def make_randomwalk(length, noise=1.0):
        return np.cumsum(noise * rng.standard_normal(length))

    T = 32
    class_A = [make_sine(T) for _ in range(8)]
    class_B = [make_randomwalk(T) for _ in range(8)]
    sequences = class_A + class_B
    labels = [0] * 8 + [1] * 8

    triplets = build_triplets(labels, triplets_per_anchor=3, seed=1)
    print(f"Training on {len(sequences)} synthetic sequences, {len(triplets)} triplets")

    theta, shapes, loss_hist, _ = train(
        sequences, triplets, d_model=16, num_heads=4, d_ff=32, num_layers=2,
        num_iters=60, lr=0.02, margin=0.3, seed=0, minibatch_size=12, verbose_every=10,
    )
    print(f"\nLoss: {loss_hist[0]:.6f} -> {loss_hist[-1]:.6f}")

    embeddings = encode_all(sequences, theta, shapes, 16, 4, 2)
    same_sims, diff_sims = [], []
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            sim = float(np.dot(embeddings[i], embeddings[j]))
            (same_sims if labels[i] == labels[j] else diff_sims).append(sim)
    print(f"Same-class mean similarity: {np.mean(same_sims):.4f}")
    print(f"Diff-class mean similarity: {np.mean(diff_sims):.4f}")
