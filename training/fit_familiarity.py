"""
Teach the app to notice scans that are unlike anything the MRI model was trained on.

    python training/fit_familiarity.py        (after export_onnx.py; needs the MRI dataset)

The classifier has only four possible answers, so a scan of something else (an eye-socket
tumour, a different kind of MRI, a picture that is not an MRI at all) is still forced into
one of them. This script measures where the training scans sit in the network's feature
space, so the app can tell when a new scan sits far away from all of them.

Method (Mahalanobis distance, Lee et al. 2018):
  1. Run every training scan through the network and keep its 1280 pooled features.
  2. Reduce them to a few dimensions with PCA.
  3. Store the centre of each class and how the features spread around those centres.
  4. A new scan's distance is how far it is from the nearest class centre, measured in
     units of that spread. The threshold is the distance that 95% of validation scans stay under.

Writes  models/mri_familiar.npz        what the app needs
        results/mri/familiarity.json   how well the check works
        results/mri/familiarity.png    the same as a figure
"""
import argparse
import json
import os
import sys

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from app import mri                                          # noqa: E402
from app.preprocess import prepare_mri                       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(ROOT, "datasets", "brain_tumor_mri"))
ap.add_argument("--unfamiliar", default=None, help="optional folder of pictures the model was not trained for")
args = ap.parse_args()


def features(img_bgr):
    """The 1280 numbers the network boils a scan down to, and its predicted class."""
    x, _ = prepare_mri(img_bgr)
    scores, maps = mri.run_network(x)
    return mri.pooled_features(maps[0]), int(scores[0].argmax())


def load_split():
    """Use the split of the latest training run, or the one shipped with the results."""
    fresh = os.path.join(ROOT, "training", "outputs", "mri_split.json")
    if os.path.exists(fresh):
        return json.load(open(fresh)), ""
    return json.load(open(os.path.join(ROOT, "results", "mri", "split.json"))), args.data


split, base = load_split()
sets = {}
for name in ("train", "val", "test"):
    F, Y, P, images = [], [], [], []
    for path, label in split[name]:
        img = cv2.imread(os.path.join(base, path))
        f, p = features(img)
        F.append(f), Y.append(label), P.append(p)
        if name == "test":
            images.append(img)
    sets[name] = (np.array(F), np.array(Y), np.array(P))
    print(name, len(F), "scans", flush=True)
Ftrain, Ytrain, _ = sets["train"]
Fval, Yval, Pval = sets["val"]
Ftest, Ytest, Ptest = sets["test"]


def fit(k):
    centre = Ftrain.mean(axis=0)
    axes = np.linalg.svd(Ftrain - centre, full_matrices=False)[2][:k].T          # PCA: the k main directions
    Z = (Ftrain - centre) @ axes
    class_centres = np.array([Z[Ytrain == c].mean(axis=0) for c in range(len(mri.CLASSES))])
    spread = sum((Z[Ytrain == c] - class_centres[c]).T @ (Z[Ytrain == c] - class_centres[c]) for c in range(len(mri.CLASSES))) / len(Z)
    return {"centre": centre, "axes": axes, "class_centres": class_centres, "inverse_spread": np.linalg.inv(spread)}


def distance(model, F):
    Z = (np.atleast_2d(F) - model["centre"]) @ model["axes"]
    d = [np.einsum("ij,jk,ik->i", Z - c, model["inverse_spread"], Z - c) for c in model["class_centres"]]
    return np.sqrt(np.min(d, axis=0))


# Choose the number of PCA dimensions on VALIDATION data only: the one whose 95% threshold
# flags the most validation mistakes. The test set and the unfamiliar pictures are not used here.
best = None
for k in (8, 16, 32, 64):
    model = fit(k)
    d = distance(model, Fval)
    threshold = float(np.percentile(d, 95))
    caught = int((d[Pval != Yval] > threshold).sum())
    print(f"k={k}: threshold {threshold:.2f}, flags {caught} of {(Pval != Yval).sum()} validation mistakes")
    if best is None or caught > best[0]:
        best = (caught, k, model, threshold)
_, k, model, threshold = best

# ---------------------------------------------------------------- how well does it work? (test set, used once)
d_test = distance(model, Ftest)
wrong = Ptest != Ytest
upside_down = distance(model, np.array([features(cv2.rotate(i, cv2.ROTATE_180))[0] for i in images]))
inverted = distance(model, np.array([features(255 - i)[0] for i in images]))
report = {
    "pca_dimensions": k, "threshold": round(threshold, 3),
    "ordinary_test_scans_flagged_percent": round(float((d_test[~wrong] > threshold).mean() * 100), 1),
    "test_mistakes": int(wrong.sum()), "test_mistakes_flagged": int((d_test[wrong] > threshold).sum()),
    "upside_down_scans_flagged_percent": round(float((upside_down > threshold).mean() * 100), 1),
    "inverted_brightness_scans_flagged_percent": round(float((inverted > threshold).mean() * 100), 1),
}
other = np.array([])
if args.unfamiliar:
    files = sorted(os.listdir(args.unfamiliar))
    other = distance(model, np.array([features(cv2.imread(os.path.join(args.unfamiliar, f)))[0] for f in files]))
    report["other_pictures"] = len(files)
    report["other_pictures_flagged"] = int((other > threshold).sum())
    report["other_pictures_note"] = "one orbital tumour MRI found online and four charts that are not MRI scans; not redistributed"
print(json.dumps(report, indent=2))

np.savez(os.path.join(ROOT, "models", "mri_familiar.npz"), threshold=np.float32(threshold),
         **{name: value.astype(np.float32) for name, value in model.items()})
json.dump(report, open(os.path.join(ROOT, "results", "mri", "familiarity.json"), "w"), indent=2)

fig, ax = plt.subplots(figsize=(8.2, 3.6))
bins = np.linspace(0, max(d_test.max(), upside_down.max(), inverted.max()) * 1.02, 60)
ax.hist(d_test[~wrong], bins, color="#4f7a59", alpha=0.9, label="test scans, answered correctly")
ax.hist(upside_down, bins, histtype="step", color="#3f679e", lw=1.6, label="the same scans upside down")
ax.hist(inverted, bins, histtype="step", color="#a8690f", lw=1.6, label="the same scans with brightness inverted")
ax.plot(d_test[wrong], np.full(wrong.sum(), 4), "v", color="#b42318", ms=8, label="test scans, answered wrongly")
ax.axvline(threshold, color="#2f2620", ls="--", lw=1.2)
ax.text(threshold - 0.12, ax.get_ylim()[1] * 0.62, "flagged beyond\nthis line", fontsize=9, ha="right", va="center")
ax.set_xlabel("distance from the training scans (higher = less familiar)", fontsize=9.5)
ax.set_ylabel("number of scans", fontsize=9.5)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8.5)
ax.set_title("Scans far from the training data are flagged as unfamiliar", fontsize=11, loc="left")
fig.savefig(os.path.join(ROOT, "results", "mri", "familiarity.png"), dpi=160, bbox_inches="tight", facecolor="white")
print("wrote models/mri_familiar.npz, results/mri/familiarity.json and familiarity.png")
