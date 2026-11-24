"""System architecture diagram for the report and slides (matplotlib, explicit layout)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, MUTED, SAGE, WASH, LINE = "#2f2620", "#6f655c", "#4f7a59", "#e3ece1", "#c9d3c6"
OUT = os.path.dirname(os.path.abspath(__file__))

W, H = 172, 78
fig, ax = plt.subplots(figsize=(17.2, 7.8))
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def box(x, y, w, h, title, lines=(), fill=WASH, edge=SAGE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.4", fc=fill, ec=edge, lw=1.4))
    ax.text(x + w / 2, y + h - 2.4, title, ha="center", va="top", fontsize=11.5, color=INK, fontweight="bold")
    for i, l in enumerate(lines):
        ax.text(x + w / 2, y + h - 6.6 - i * 3.3, l, ha="center", va="top", fontsize=9.4, color=MUTED)


def arrow(x1, y1, x2, y2, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15, color=MUTED, lw=1.4,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}"))


def tag(x, y, text):
    ax.text(x, y, text, ha="center", va="center", fontsize=9.2, color=INK,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=LINE, lw=0.8))


ax.text(0.5, H - 0.5, "Neurova: system architecture", fontsize=17, fontweight="bold", color=INK, va="top")
ax.text(0.5, H - 5.2, "How one upload becomes a prediction on screen", fontsize=11, color=MUTED, va="top")

bw, gap, y, h = 21.5, 8.0, 30, 21
xs = [0.5 + i * (bw + gap) for i in range(6)]
stages = [
    ("User", ["patient, clinician", "or administrator", "in a web browser"], "white", LINE),
    ("Frontend", ["HTML, CSS, JavaScript", "sign-in, results,", "sharing, review"], WASH, SAGE),
    ("Backend API", ["FastAPI, Python", "sign-in tokens, roles,", "consent, validation"], WASH, SAGE),
    ("Preprocessing", ["MRI: brain crop,", "resize 192 px, normalise", "EEG: Fpz-Cz, 100 Hz"], WASH, SAGE),
    ("Deep learning", ["EfficientNet-B0 (MRI)", "CNN + BiGRU (sleep)", "ONNX Runtime, CPU"], "#d4e4d1", SAGE),
    ("Result", ["calibrated probabilities,", "Grad-CAM, uncertainty,", "hypnogram, summary"], WASH, SAGE),
]
for x, (t, l, f, e) in zip(xs, stages):
    box(x, y, bw, h, t, l, f, e)
for i in range(5):
    arrow(xs[i] + bw + 0.4, y + h / 2, xs[i + 1] - 0.4, y + h / 2)
for i, t in ((1, "HTTPS"), (2, "JSON +\ntoken"), (4, "tensor")):
    ax.text(xs[i] - gap / 2, y + h / 2 - 2.2, t, ha="center", va="top", fontsize=8.8, color=MUTED, linespacing=1.1)

# Return path: result back to the frontend dashboard, drawn above the row
top = y + h + 1.0
arrow(xs[5] + bw / 2, top, xs[1] + bw / 2, top, rad=0.16)
tag((xs[5] + xs[1] + bw) / 2, y + h + 8.4, "JSON result shown to the patient and the clinician")

# Storage row
sy, sh = 2, 15
store = [
    (xs[1] + 6, "Database", ["SQLite or MySQL: users,", "access, studies, audit log"]),
    (xs[3] - 6, "Training, done once", ["PyTorch on public datasets,", "then exported to ONNX"]),
    (xs[4] + 12, "Model files", ["mri.onnx (16 MB)", "sleep.onnx (0.8 MB)"]),
]
sw = 28
for x, t, l in store:
    box(x, sy, sw, sh, t, l, "white", LINE)
arrow(xs[2] + bw / 2, y - 0.4, store[0][0] + sw / 2, sy + sh + 0.4)
arrow(store[1][0] + sw + 0.4, sy + sh / 2, store[2][0] - 0.4, sy + sh / 2, ls="--")
ax.text((store[1][0] + sw + store[2][0]) / 2, sy + sh / 2 + 1.2, "export", ha="center", va="bottom", fontsize=9.2, color=MUTED)
arrow(store[2][0] + sw / 2, sy + sh + 0.4, xs[4] + bw / 2, y - 0.4, ls="--")
ax.text(store[2][0] + sw / 2 + 2.5, (sy + sh + y) / 2, "loaded at start", ha="left", va="center", fontsize=9.2, color=MUTED)

plt.savefig(os.path.join(OUT, "architecture.png"), dpi=170, bbox_inches="tight", facecolor="white")
print("wrote architecture.png")
