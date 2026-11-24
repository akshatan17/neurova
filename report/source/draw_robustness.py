"""Grouped bar chart: share of test predictions that change under each perturbation, per model."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
m = json.load(open(os.path.join(ROOT, "results", "mri", "metrics.json")))
NAMES = {"neurova_cnn": "NeurovaCNN (scratch)", "efficientnet_b0_tf": "EfficientNet-B0",
         "efficientnet_b0_tf_robust": "EfficientNet-B0 + fingerprint scrub"}
COLS = {"neurova_cnn": "#b8b0a6", "efficientnet_b0_tf": "#a8690f", "efficientnet_b0_tf_robust": "#4f7a59"}
PERT = [("noise_0.05", "Faint noise\n(σ = 0.05)"), ("noise_0.1", "Noise\n(σ = 0.1)"),
        ("blur_7", "Blur\n(7 px)"), ("brightness_0.3", "Brightness\n(+0.3)")]
keys = [k for k in NAMES if k in m["results"] and "robustness" in m["results"][k]]

fig, ax = plt.subplots(figsize=(8.2, 3.9))
w = 0.8 / max(len(keys), 1)
for i, k in enumerate(keys):
    vals = [m["results"][k]["robustness"][p] for p, _ in PERT]
    xs = np.arange(len(PERT)) + (i - (len(keys) - 1) / 2) * w
    bars = ax.bar(xs, vals, w * 0.92, color=COLS[k], label=NAMES[k])
    for x, v in zip(xs, vals):
        ax.text(x, v + 1.2, f"{v:.0f}%", ha="center", va="bottom", fontsize=8.5, color="#2f2620")
ax.set_xticks(range(len(PERT))); ax.set_xticklabels([l for _, l in PERT], fontsize=9.5)
ax.set_ylabel("test predictions that change (%)", fontsize=9.5)
ax.set_ylim(0, max(100, ax.get_ylim()[1]))
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.25)
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.set_title("How often a perturbation changes the prediction (lower is better)", fontsize=11, loc="left")
out = os.path.join(ROOT, "results", "mri", "robustness.png")
fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print("wrote", out)
