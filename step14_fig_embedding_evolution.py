"""
Step 14: Embedding Evolution Figure (PCA + t-SNE)
=======================================================
Produces the combined PCA/t-SNE embedding-evolution figures from the
checkpoints saved by step13.py. Two output modes are provided: the full
0-400 range (sparse checkpoints), and a zoomed 150-400 range (dense
checkpoints) used to pinpoint the non-monotonic separation pattern
discussed in the paper.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from step6 import D_MODEL, NUM_HEADS, NUM_LAYERS
from step4 import encode_all

CKPT_DIR = "checkpoints_evolution"
DATASET_NAME = "StarLightCurves"
TSNE_PERPLEXITY = 30

FULL_RANGE_ITERS = [0, 20, 70, 110, 150, 250, 400]
ZOOMED_RANGE_ITERS = [150, 200, 250, 290, 345, 400]


def build_figure(name, checkpoint_iters, perplexity, out_path):
    data = np.load(f"{CKPT_DIR}/{name}_viz_subsample.npz", allow_pickle=True)
    seqs, labels = list(data["seqs"]), data["labels"]

    n_checkpoints = len(checkpoint_iters)
    fig, axes = plt.subplots(2, n_checkpoints, figsize=(3.0 * n_checkpoints, 6.4))

    classes = np.unique(labels)
    cmap = plt.cm.tab10
    colors = {c: cmap(i) for i, c in enumerate(classes)}

    for col, it in enumerate(checkpoint_iters):
        ckpt = np.load(f"{CKPT_DIR}/{name}_iter{it}.npz", allow_pickle=True)
        theta, shapes = ckpt["theta"], ckpt["shapes"].item()
        emb = encode_all(seqs, theta, shapes, D_MODEL, NUM_HEADS, NUM_LAYERS)

        pca_2d = PCA(n_components=2, random_state=0).fit_transform(emb)
        tsne_2d = TSNE(n_components=2, perplexity=perplexity, random_state=0,
                         init="pca").fit_transform(emb)

        for row, proj in enumerate([pca_2d, tsne_2d]):
            ax = axes[row, col]
            for c in classes:
                mask = labels == c
                ax.scatter(proj[mask, 0], proj[mask, 1], s=16, color=colors[c], alpha=0.8)
            if row == 0:
                ax.set_title(f"iter={it}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel("PCA" if row == 0 else "t-SNE", fontsize=11)
        print(f"  iter={it} done")

    fig.suptitle(f"Embedding Space Evolution -- {name} (PCA top, t-SNE bottom, perplexity={perplexity})", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    build_figure(DATASET_NAME, FULL_RANGE_ITERS, TSNE_PERPLEXITY,
                  f"embedding_evolution_{DATASET_NAME}_combined_full.png")
    build_figure(DATASET_NAME, ZOOMED_RANGE_ITERS, TSNE_PERPLEXITY,
                  f"embedding_evolution_{DATASET_NAME}_zoomed.png")
