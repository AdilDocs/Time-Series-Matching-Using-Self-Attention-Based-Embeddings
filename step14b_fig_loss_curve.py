"""
Step 14b: Loss Curve Companion Figure
==========================================
Produces embedding_evolution_StarLightCurves_loss_full.png -- the loss
trajectory accompanying the embedding-evolution figures (step14.py).
Loss values below are the triplet margin loss evaluated at each saved
checkpoint's parameters (step13 checkpoints), i.e. the same quantity at
every iteration, as reported in Table V.
"""
import matplotlib.pyplot as plt

iters = [0, 5, 10, 20, 40, 70, 110, 150, 200, 250, 290, 345, 400]
losses = [0.242420, 0.132652, 0.101919, 0.067147, 0.046124, 0.026441, 0.015611,
          0.008827, 0.002162, 0.005043, 0.003020, 0.001567, 0.000357]

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(iters, losses, marker='o', color='tab:blue')
for it in [150, 200, 250, 290, 345, 400]:
    ax.axvline(it, color='gray', alpha=0.15, linestyle='--')
ax.set_xlabel("training iteration")
ax.set_ylabel("triplet margin loss")
ax.set_title("Training Loss -- StarLightCurves (full range)")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("embedding_evolution_StarLightCurves_loss_full.png", dpi=150)
print("Saved embedding_evolution_StarLightCurves_loss_full.png")
