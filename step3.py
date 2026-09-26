"""
Step 3: Self-Attention Sequence Encoder
============================================
Maps a variable-length time-series sequence to a fixed-size embedding
via a self-attention encoder (Sections 3.1-3.2 of the paper), built on
the autodiff engine from step2.py. Also provides attention-weight
extraction variants (used only for post-hoc, inference-time
visualization -- e.g. the Case Study in the paper -- never during
training).
"""

import numpy as np
from step2 import Var, concat_last_axis


def sinusoidal_positional_encoding(T: int, d_model: int) -> np.ndarray:
    pe = np.zeros((T, d_model))
    position = np.arange(T)[:, None]
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe


def init_encoder_params(d_model: int, num_heads: int, d_ff: int, num_layers: int,
                          seed: int = 0, scale: float = 1.0):
    """Returns a dict of numpy parameter arrays (Xavier-ish init, scaled
    by `scale`). Default scale=1.0 reproduces standard behavior; a larger
    scale is used as a fallback for datasets where the default scale
    produces a collapsed initial representation (see step6.robust_init)."""
    rng = np.random.default_rng(seed)
    assert d_model % num_heads == 0
    d_k = d_model // num_heads

    def xavier(shape):
        fan_in = shape[0]
        return scale * rng.standard_normal(shape) / np.sqrt(fan_in)

    params = {
        "W_in": xavier((1, d_model)),
        "b_in": np.zeros(d_model),
    }
    for l in range(num_layers):
        for h in range(num_heads):
            params[f"Wq_{l}_{h}"] = xavier((d_model, d_k))
            params[f"Wk_{l}_{h}"] = xavier((d_model, d_k))
            params[f"Wv_{l}_{h}"] = xavier((d_model, d_k))
        params[f"Wo_{l}"] = xavier((d_model, d_model))
        params[f"W1_{l}"] = xavier((d_model, d_ff))
        params[f"b1_{l}"] = np.zeros(d_ff)
        params[f"W2_{l}"] = xavier((d_ff, d_model))
        params[f"b2_{l}"] = np.zeros(d_model)
    return params


def flatten_params(params: dict) -> np.ndarray:
    return np.concatenate([params[k].reshape(-1) for k in sorted(params.keys())])


def unflatten_params(theta_flat: np.ndarray, shapes: dict) -> dict:
    params = {}
    idx = 0
    for k in sorted(shapes.keys()):
        shape = shapes[k]
        size = int(np.prod(shape))
        params[k] = theta_flat[idx:idx + size].reshape(shape)
        idx += size
    return params


def get_shapes(params: dict) -> dict:
    return {k: v.shape for k, v in params.items()}


def build_param_vars(params: dict) -> dict:
    return {k: Var(v) for k, v in params.items()}


def multi_head_attention(X: Var, param_vars: dict, layer_idx: int, num_heads: int) -> Var:
    """X: (T, d_model) Var. Returns (T, d_model) Var."""
    head_outputs = []
    for h in range(num_heads):
        Wq = param_vars[f"Wq_{layer_idx}_{h}"]
        Wk = param_vars[f"Wk_{layer_idx}_{h}"]
        Wv = param_vars[f"Wv_{layer_idx}_{h}"]
        Q = X.matmul(Wq)
        K = X.matmul(Wk)
        V = X.matmul(Wv)
        d_k = Wq.shape[1]
        scores = Q.matmul(K.transpose_last2()) * (1.0 / np.sqrt(d_k))
        attn = scores.softmax(axis=-1)
        head_out = attn.matmul(V)
        head_outputs.append(head_out)
    concatenated = concat_last_axis(head_outputs)
    Wo = param_vars[f"Wo_{layer_idx}"]
    return concatenated.matmul(Wo)


def multi_head_attention_with_weights(X: Var, param_vars: dict, layer_idx: int, num_heads: int):
    """Identical computation to multi_head_attention, but additionally
    returns the raw (T, T) attention weight matrix for each head as plain
    numpy arrays. Used only for post-hoc inference-time visualization of
    a trained model -- never called during training."""
    head_outputs = []
    attn_weights = []
    for h in range(num_heads):
        Wq = param_vars[f"Wq_{layer_idx}_{h}"]
        Wk = param_vars[f"Wk_{layer_idx}_{h}"]
        Wv = param_vars[f"Wv_{layer_idx}_{h}"]
        Q = X.matmul(Wq)
        K = X.matmul(Wk)
        V = X.matmul(Wv)
        d_k = Wq.shape[1]
        scores = Q.matmul(K.transpose_last2()) * (1.0 / np.sqrt(d_k))
        attn = scores.softmax(axis=-1)
        attn_weights.append(attn.value.copy())
        head_out = attn.matmul(V)
        head_outputs.append(head_out)
    concatenated = concat_last_axis(head_outputs)
    Wo = param_vars[f"Wo_{layer_idx}"]
    return concatenated.matmul(Wo), attn_weights


def feedforward(X: Var, param_vars: dict, layer_idx: int) -> Var:
    W1 = param_vars[f"W1_{layer_idx}"]
    b1 = param_vars[f"b1_{layer_idx}"]
    W2 = param_vars[f"W2_{layer_idx}"]
    b2 = param_vars[f"b2_{layer_idx}"]
    h = (X.matmul(W1) + b1).relu()
    return h.matmul(W2) + b2


def encode_sequence(x: np.ndarray, param_vars: dict, pos_encoding: np.ndarray,
                      num_heads: int, num_layers: int) -> Var:
    """Full forward pass: x (T,) raw sequence -> (d_model,) embedding Var."""
    T = x.shape[0]
    x_std = (x - x.mean()) / (x.std() + 1e-8)
    x_col = Var(x_std.reshape(T, 1))
    W_in = param_vars["W_in"]
    b_in = param_vars["b_in"]
    X = x_col.matmul(W_in) + b_in
    X = X + Var(pos_encoding)

    for l in range(num_layers):
        attn_out = multi_head_attention(X, param_vars, l, num_heads)
        X = (X + attn_out).layernorm(axis=-1)
        ff_out = feedforward(X, param_vars, l)
        X = (X + ff_out).layernorm(axis=-1)

    embedding = Var(np.ones((1, T)) / T).matmul(X)
    embedding = embedding.reshape(-1)
    norm = (embedding * embedding).sum() ** 0.5
    embedding = embedding * (norm ** -1.0)
    return embedding


def encode_sequence_with_attention(x: np.ndarray, param_vars: dict, pos_encoding: np.ndarray,
                                      num_heads: int, num_layers: int):
    """Identical computation to encode_sequence, but additionally returns
    the attention weight matrices from every layer and head. Used for
    post-hoc visualization (the paper's Case Study, Section 5.3
    equivalent) -- never called during training.
    Returns (embedding: Var, attn_weights_by_layer: list of num_layers
    lists, each containing num_heads (T,T) numpy arrays)."""
    T = x.shape[0]
    x_std = (x - x.mean()) / (x.std() + 1e-8)
    x_col = Var(x_std.reshape(T, 1))
    W_in = param_vars["W_in"]
    b_in = param_vars["b_in"]
    X = x_col.matmul(W_in) + b_in
    X = X + Var(pos_encoding)

    attn_weights_by_layer = []
    for l in range(num_layers):
        attn_out, attn_weights = multi_head_attention_with_weights(X, param_vars, l, num_heads)
        attn_weights_by_layer.append(attn_weights)
        X = (X + attn_out).layernorm(axis=-1)
        ff_out = feedforward(X, param_vars, l)
        X = (X + ff_out).layernorm(axis=-1)

    embedding = Var(np.ones((1, T)) / T).matmul(X)
    embedding = embedding.reshape(-1)
    norm = (embedding * embedding).sum() ** 0.5
    embedding = embedding * (norm ** -1.0)
    return embedding, attn_weights_by_layer


def embedding_similarity(emb_a: Var, emb_b: Var) -> Var:
    """Cosine similarity between two L2-normalized embeddings (dot product)."""
    return (emb_a * emb_b).sum()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T = 16
    d_model, num_heads, d_ff, num_layers = 8, 2, 16, 1

    params = init_encoder_params(d_model, num_heads, d_ff, num_layers, seed=0)
    shapes = get_shapes(params)
    theta_flat = flatten_params(params)
    pos_enc = sinusoidal_positional_encoding(T, d_model)

    x1 = rng.standard_normal(T)
    x2 = rng.standard_normal(T)

    def similarity_of_theta(theta):
        p = unflatten_params(theta, shapes)
        pv = build_param_vars(p)
        e1 = encode_sequence(x1, pv, pos_enc, num_heads, num_layers)
        e2 = encode_sequence(x2, pv, pos_enc, num_heads, num_layers)
        return embedding_similarity(e1, e2)

    print("Testing forward pass and gradient correctness...")
    pv = build_param_vars(params)
    e1 = encode_sequence(x1, pv, pos_enc, num_heads, num_layers)
    e2 = encode_sequence(x2, pv, pos_enc, num_heads, num_layers)
    print(f"Embedding norms: {float((e1.value**2).sum())**0.5:.4f}, "
          f"{float((e2.value**2).sum())**0.5:.4f} (expect ~1.0 each)")
    sim = embedding_similarity(e1, e2)
    print(f"Cosine similarity: {float(sim.value):.4f}")

    sim.backward()
    key = "Wq_0_0"
    idx_start = sum(int(np.prod(shapes[k])) for k in sorted(shapes.keys()) if k < key)
    analytic = pv[key].grad.flatten()[0]
    eps = 1e-5
    tp = theta_flat.copy(); tp[idx_start] += eps
    tm = theta_flat.copy(); tm[idx_start] -= eps
    numeric = (float(similarity_of_theta(tp).value) - float(similarity_of_theta(tm).value)) / (2 * eps)
    print(f"\nGradient check on {key}[0]: analytic={analytic:.6f}, numeric={numeric:.6f}, "
          f"diff={abs(analytic-numeric):.8f}")
    print(f"\nTotal parameters: {len(theta_flat)}")
