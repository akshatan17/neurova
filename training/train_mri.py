"""
Train and evaluate the brain MRI tumour classifiers. Runs on a CPU in about two hours.

    python training/get_data.py          (once, downloads the datasets)
    python training/train_mri.py
    python training/export_onnx.py       (puts the new model into the app)

Method: remove duplicate images with a perceptual hash, make a fresh stratified
70/15/15 split, crop to the brain exactly as the app does, augment, train with
AdamW + cosine decay + label smoothing, stop early on validation macro-F1, fit
the calibration temperature on validation, and evaluate on the test set once.

Three models are trained and compared: a small CNN from scratch (baseline),
EfficientNet-B0 fine-tuned from ImageNet, and the same EfficientNet trained with
"fingerprint scrub" augmentation. Writes checkpoints, figures and metrics to
training/outputs/.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.preprocess import crop_brain_region              # noqa: E402  (the same preprocessing the app runs)
from networks import build_mri_model, cam_layer as _cam_layer   # noqa: E402

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
LABELS = ["Glioma", "Meningioma", "No tumour", "Pituitary tumour"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SEED = 42


def seed_all(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ data
def gather(root):
    recs = []
    for split in ("Training", "Testing"):
        for c in CLASSES:
            d = os.path.join(root, split, c)
            for f in sorted(os.listdir(d)):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    recs.append({"path": os.path.join(d, f), "label": c, "orig_split": split})
    return recs


def dedup(recs):
    import imagehash
    from PIL import Image
    for r in recs:
        with Image.open(r["path"]) as im:
            r["hash"] = str(imagehash.phash(im.convert("L"), hash_size=12))
    groups = defaultdict(list)
    for r in recs:
        groups[r["hash"]].append(r)
    dup = [g for g in groups.values() if len(g) > 1]
    report = {
        "images_total": len(recs),
        "unique_images": len(groups),
        "duplicate_groups": len(dup),
        "images_in_duplicate_groups": sum(len(g) for g in dup),
        "groups_spanning_official_train_and_test": sum(1 for g in dup if len({r["orig_split"] for r in g}) > 1),
        "groups_with_conflicting_labels_dropped": sum(1 for g in dup if len({r["label"] for r in g}) > 1),
    }
    keep = [g[0] for g in groups.values() if len({r["label"] for r in g}) == 1]
    return keep, report


def fingerprint_scrub(img, rng):
    """Degrade fine texture so the network can't key on a source collection's resolution,
    compression or noise signature: random down-up resampling, JPEG re-encode, blur, noise."""
    h, w = img.shape[:2]
    if rng.random() < 0.5:
        f = rng.uniform(0.45, 0.9)
        small = cv2.resize(img, (max(16, int(w * f)), max(16, int(h * f))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.5:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 91))])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if rng.random() < 0.3:
        k = int(rng.choice([3, 5, 7]))
        img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.6:
        sigma = rng.uniform(0, 0.045) * 255
        img = np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)
    return img


class MRIDataset(Dataset):
    def __init__(self, items, cache, augment, robust=False):
        self.items, self.cache, self.augment, self.robust = items, cache, augment, robust
        self.rng = np.random.default_rng()
        if augment:
            import torchvision.transforms.v2 as T
            self.aug = T.Compose([T.RandomHorizontalFlip(),
                                  T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                                  T.ColorJitter(brightness=0.15, contrast=0.15)])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        img = self.cache[p]
        if self.augment and self.robust:
            img = fingerprint_scrub(img, self.rng)
        x = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
        if self.augment:
            x = self.aug(x)
        x = x.float() / 255.0
        x = (x - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None]
        return x, y


# ------------------------------------------------------------------ models
def make_model(arch, pretrained_path=None):
    m = build_mri_model(arch, 4)
    if arch == "efficientnet_b0_tf" and pretrained_path:
        sd = torch.load(pretrained_path, map_location="cpu")
        sd = {k: v for k, v in sd.items() if not k.startswith("_fc.")}      # new 4-class head
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert set(missing) == {"_fc.weight", "_fc.bias"} and not unexpected, (missing, unexpected)
        log(f"loaded ImageNet weights into EfficientNet-B0 ({len(sd)} tensors)")
    return m


# ------------------------------------------------------------------ training
def run_epoch(model, loader, opt=None):
    train = opt is not None
    model.train(train)
    L, Y, P = [], [], []
    for x, y in loader:
        with torch.set_grad_enabled(train):
            out = model(x)
            loss = F.cross_entropy(out, y, label_smoothing=0.1)
        if train:
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        L.append(loss.item()); Y.append(y); P.append(out.argmax(1))
    from sklearn.metrics import accuracy_score, f1_score
    y, p = torch.cat(Y).numpy(), torch.cat(P).numpy()
    return float(np.mean(L)), accuracy_score(y, p), f1_score(y, p, average="macro")


@torch.no_grad()
def logits_of(model, loader):
    model.eval()
    L, Y = [], []
    for x, y in loader:
        L.append(model(x)); Y.append(y)
    return torch.cat(L), torch.cat(Y)


def fit_temperature(logits, y):
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(min(max(log_t.exp().item(), 0.05), 10.0))


def ece(probs, y, bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    e = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def train_one(arch, dl_tr, dl_va, epochs, lr, patience, pretrained, hist_path):
    seed_all()
    model = make_model(arch, pretrained)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, best_state, wait, hist = -1, None, 0, []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        te = time.time()
        tl, ta, tf = run_epoch(model, dl_tr, opt)
        vl, va, vf = run_epoch(model, dl_va)
        sched.step()
        hist.append({"epoch": ep, "train_loss": tl, "train_acc": ta, "val_loss": vl, "val_acc": va, "val_f1": vf})
        mark = ""
        if vf > best:
            best, wait, mark = vf, 0, " *"
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        log(f"{arch} ep {ep:2d}/{epochs}  train loss {tl:.3f} acc {ta:.3f} | val loss {vl:.3f} acc {va:.3f} "
            f"macro-F1 {vf:.3f}{mark}  ({(time.time() - te) / 60:.1f} min)")
        json.dump(hist, open(hist_path, "w"))
        if wait >= patience:
            log("early stop"); break
    model.load_state_dict(best_state)
    return model, hist, time.time() - t0


@torch.no_grad()
def robustness_eval(model, loader, T):
    """Share of test predictions that change under each perturbation (normalised input space)."""
    model.eval()
    g = torch.Generator().manual_seed(0)
    tests = {"noise_0.05": lambda x: x + torch.randn(x.shape, generator=g) * 0.05,
             "noise_0.1": lambda x: x + torch.randn(x.shape, generator=g) * 0.1,
             "blur_7": None, "brightness_0.3": lambda x: x + 0.3}
    flips = {k: 0 for k in tests}; n = 0
    for x, _ in loader:
        base = (model(x) / T).argmax(1)
        for k, fn in tests.items():
            if fn is None:
                arr = x.numpy().transpose(0, 2, 3, 1)
                xt = torch.from_numpy(np.stack([cv2.GaussianBlur(a, (7, 7), 0) for a in arr]).transpose(0, 3, 1, 2))
            else:
                xt = fn(x)
            flips[k] += int(((model(xt) / T).argmax(1) != base).sum())
        n += len(x)
    return {k: round(v / n * 100, 1) for k, v in flips.items()}


def cpu_latency_ms(model, size, n=30):
    model.eval()
    x = torch.randn(1, 3, size, size)
    with torch.no_grad():
        for _ in range(3):
            model(x)
        ts = []
        for _ in range(n):
            t = time.perf_counter(); model(x); ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))


def gradcam(model, arch, x, cls):
    acts, grads = {}, {}
    layer = _cam_layer(model, arch)
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    model.eval(); model.zero_grad()
    out = model(x.clone().requires_grad_(True)); out[0, cls].backward()
    h1.remove(); h2.remove()
    cam = F.relu((grads["g"].mean(dim=(2, 3), keepdim=True) * acts["a"]).sum(1))[0].detach().numpy()
    return cam / (cam.max() + 1e-8)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "datasets", "brain_tumor_mri"))
    ap.add_argument("--pretrained", default=os.path.join(ROOT, "datasets", "pretrained", "efficientnet-b0-355c32eb.pth"))
    ap.add_argument("--out", default=os.path.join(ROOT, "training", "outputs"))
    ap.add_argument("--img", type=int, default=192)
    ap.add_argument("--effnet-epochs", type=int, default=8)
    ap.add_argument("--cnn-epochs", type=int, default=20)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N images per class")
    ap.add_argument("--cnn-lr", type=float, default=5e-4)
    ap.add_argument("--cnn-patience", type=int, default=8)
    ap.add_argument("--only", default="", help="retrain just this model key and merge with an earlier run's results")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    R = os.path.join(a.out, "results", "mri")
    os.makedirs(R, exist_ok=True)
    os.makedirs(os.path.join(a.out, "weights"), exist_ok=True)

    recs = gather(a.data)
    if a.limit:
        per = defaultdict(int); kept = []
        for r in recs:
            if per[(r["label"], r["orig_split"])] < a.limit:
                kept.append(r); per[(r["label"], r["orig_split"])] += 1
        recs = kept
    log(len(recs), "images", dict(Counter((r["orig_split"], r["label"]) for r in recs)))
    uniq, dedup_report = dedup(recs)
    log("dedup:", dedup_report)

    from sklearn.model_selection import train_test_split
    P = [r["path"] for r in uniq]; Y = [CLASSES.index(r["label"]) for r in uniq]
    trp, rp, try_, ry = train_test_split(P, Y, test_size=0.30, stratify=Y, random_state=SEED)
    vap, tep, vay, tey = train_test_split(rp, ry, test_size=0.50, stratify=ry, random_state=SEED)
    splits = {n: {CLASSES[k]: v for k, v in sorted(Counter(y).items())} for n, y in (("train", try_), ("val", vay), ("test", tey))}
    log("splits:", splits)
    json.dump({"train": list(zip(trp, try_)), "val": list(zip(vap, vay)), "test": list(zip(tep, tey))},
              open(os.path.join(a.out, "mri_split.json"), "w"))

    t0 = time.time()
    cache = {}
    for p in trp + vap + tep:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        cache[p] = cv2.resize(crop_brain_region(img), (a.img, a.img), interpolation=cv2.INTER_AREA)
    log(f"preprocessed {len(cache)} images in {time.time() - t0:.0f}s")

    tr, va, te = list(zip(trp, try_)), list(zip(vap, vay)), list(zip(tep, tey))
    dl_tr = DataLoader(MRIDataset(tr, cache, True), batch_size=32, shuffle=True, drop_last=True)
    dl_va = DataLoader(MRIDataset(va, cache, False), batch_size=64)
    dl_te = DataLoader(MRIDataset(te, cache, False), batch_size=64)

    # Figures about the data
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for i, (n, y) in enumerate((("train", try_), ("val", vay), ("test", tey))):
        c = Counter(y); ax.bar(np.arange(4) + (i - 1) * 0.26, [c[k] for k in range(4)], 0.26, label=n)
    ax.set_xticks(range(4)); ax.set_xticklabels(LABELS); ax.set_ylabel("images")
    ax.set_title("Class balance after de-duplication"); ax.legend(frameon=False)
    save(fig, os.path.join(R, "class_balance.png"))
    fig, axes = plt.subplots(2, 6, figsize=(12, 4.4))
    for j in range(6):
        for row, aug in ((0, False), (1, True)):
            x, y = MRIDataset(tr[:6], cache, aug)[j]
            axes[row, j].imshow((x.permute(1, 2, 0).numpy() * STD + MEAN).clip(0, 1)); axes[row, j].axis("off")
            axes[row, j].set_title("augmented" if row else LABELS[y], fontsize=9)
    save(fig, os.path.join(R, "augmentation_examples.png"))

    # key, architecture, epochs, lr, patience, pretrained weights, fingerprint-scrub augmentation
    plan = [("neurova_cnn", "neurova_cnn", a.cnn_epochs, a.cnn_lr, a.cnn_patience, None, False),
            ("efficientnet_b0_tf", "efficientnet_b0_tf", a.effnet_epochs, 3e-4, 3, a.pretrained, False),
            ("efficientnet_b0_tf_robust", "efficientnet_b0_tf", a.effnet_epochs, 3e-4, 3, a.pretrained, True)]
    ARCH_OF = {k: arc for k, arc, *_ in plan}
    trained, results = {}, {}
    if a.only:
        # Keep the other architecture exactly as trained before: reload its checkpoint and recompute its outputs.
        prev = torch.load(os.path.join(a.out, "weights", "neurova_mri_v2.pt"), map_location="cpu", weights_only=False)
        for arch, rr in prev["comparison"].items():
            if arch == a.only:
                continue
            ck_path = os.path.join(a.out, "weights", f"mri_{arch}.pt")
            m = build_mri_model(ARCH_OF[arch], 4); m.load_state_dict(torch.load(ck_path, map_location="cpu")); m.eval()
            hist = json.load(open(os.path.join(R, f"history_{arch}.json")))
            if "robustness" not in rr:
                rr["robustness"] = robustness_eval(m, dl_te, rr["temperature"])
            tl, ty = logits_of(m, dl_te)
            T = rr["temperature"]
            trained[arch] = {"model": m, "history": hist, "T": T, "y": ty.numpy(), "probs": F.softmax(tl / T, 1).numpy(),
                             "pred": F.softmax(tl / T, 1).numpy().argmax(1), "raw": F.softmax(tl, 1).numpy()}
            results[arch] = rr
            log(f"reused {arch}: test acc {rr['test_accuracy']:.4f}")
        plan = [p for p in plan if p[0] == a.only]
    for key, arch, E, lr, pat, pre, robust in plan:
        log(f"=== training {key}")
        loader = DataLoader(MRIDataset(tr, cache, True, robust), batch_size=32, shuffle=True, drop_last=True) if robust else dl_tr
        model, hist, secs = train_one(arch, loader, dl_va, E, lr, pat, pre, os.path.join(R, f"history_{key}.json"))
        vl, vy = logits_of(model, dl_va)
        T = fit_temperature(vl, vy)
        tl, ty = logits_of(model, dl_te)
        y = ty.numpy()
        p_raw, p_cal = F.softmax(tl, 1).numpy(), F.softmax(tl / T, 1).numpy()
        pred = p_cal.argmax(1)
        from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
        pr, rc, f1, sp = precision_recall_fscore_support(y, pred, labels=range(4), zero_division=0)
        results[key] = {
            "arch": arch, "fingerprint_scrub": robust,
            "test_accuracy": accuracy_score(y, pred), "test_macro_f1": f1_score(y, pred, average="macro"),
            "test_macro_auc": roc_auc_score(y, p_cal, multi_class="ovr", average="macro"),
            "ece_before": ece(p_raw, y), "ece_after": ece(p_cal, y), "temperature": T,
            "best_val_macro_f1": max(h["val_f1"] for h in hist),
            "per_class": {CLASSES[i]: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]), "support": int(sp[i])} for i in range(4)},
            "params_m": sum(q.numel() for q in model.parameters()) / 1e6,
            "cpu_latency_ms": cpu_latency_ms(model, a.img), "train_minutes": secs / 60, "epochs_run": len(hist),
            "pretrained": pre is not None,
            "robustness": robustness_eval(model, dl_te, T),
        }
        trained[key] = {"model": model, "history": hist, "T": T, "y": y, "probs": p_cal, "pred": pred, "raw": p_raw}
        log(f"{key} TEST acc {results[key]['test_accuracy']:.4f} macro-F1 {results[key]['test_macro_f1']:.4f} "
            f"AUC {results[key]['test_macro_auc']:.4f} ECE {results[key]['ece_before']:.3f}->{results[key]['ece_after']:.3f} T={T:.3f} "
            f"robustness {results[key]['robustness']}")
        json.dump(results, open(os.path.join(R, "metrics_partial.json"), "w"), indent=2, default=float)
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, os.path.join(a.out, "weights", f"mri_{key}.pt"))

    best_val = max(r["best_val_macro_f1"] for r in results.values())
    robust_ok = [k for k, r in results.items() if r.get("fingerprint_scrub") and r["best_val_macro_f1"] >= best_val - 0.01]
    # Rule fixed before the robust model was trained: prefer it if its VALIDATION macro-F1 is within
    # one point of the best, because it is far less sensitive to scanner noise. Test scores play no part.
    chosen = robust_ok[0] if robust_ok else max(results, key=lambda k: (round(results[k]["best_val_macro_f1"], 3), -results[k]["cpu_latency_ms"]))
    D = trained[chosen]
    log("deploying", chosen)

    # ---- figures
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for arch, d in trained.items():
        e = [h["epoch"] for h in d["history"]]
        axes[0].plot(e, [h["train_loss"] for h in d["history"]], "--", label=f"{arch} train")
        axes[0].plot(e, [h["val_loss"] for h in d["history"]], label=f"{arch} val")
        axes[1].plot(e, [h["val_acc"] for h in d["history"]], marker="o", label=arch)
        axes[2].plot(e, [h["val_f1"] for h in d["history"]], marker="o", label=arch)
    for ax, t in zip(axes, ["Loss", "Validation accuracy", "Validation macro-F1"]):
        ax.set_title(t); ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, frameon=False); axes[2].legend(fontsize=8, frameon=False)
    save(fig, os.path.join(R, "training_curves.png"))

    from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
    fig, axes = plt.subplots(1, len(trained), figsize=(5.2 * len(trained), 4.6))
    for ax, (arch, d) in zip(np.atleast_1d(axes), trained.items()):
        cm = confusion_matrix(d["y"], d["pred"], labels=range(4)); cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
        ax.imshow(cmn, cmap="Greens", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.0%}", ha="center", va="center", fontsize=8, color="white" if cmn[i, j] > 0.6 else "black")
        ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels(LABELS, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(LABELS, fontsize=8); ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{arch}{'  (deployed)' if arch == chosen else ''}\nacc {results[arch]['test_accuracy']:.3f}, macro-F1 {results[arch]['test_macro_f1']:.3f}", fontsize=10)
    save(fig, os.path.join(R, "confusion_matrices.png"))

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for i, c in enumerate(LABELS):
        yb = (D["y"] == i).astype(int); fpr, tpr, _ = roc_curve(yb, D["probs"][:, i])
        ax.plot(fpr, tpr, label=f"{c} (AUC {roc_auc_score(yb, D['probs'][:, i]):.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1); ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title(f"ROC, one-vs-rest ({chosen})"); ax.legend(fontsize=8, frameon=False)
    save(fig, os.path.join(R, "roc_curves.png"))

    def rel(ax, probs, y, title):
        conf, pred = probs.max(1), probs.argmax(1); xs, accs, ns = [], [], []
        for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 3:
                xs.append(conf[m].mean()); accs.append((pred[m] == y[m]).mean()); ns.append(m.sum())
        ax.plot([0, 1], [0, 1], "k:", lw=1); ax.plot(xs, accs, "o-"); ax.set_title(title, fontsize=10)
        ax.set_xlabel("confidence"); ax.set_ylabel("accuracy"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    rel(axes[0], D["raw"], D["y"], f"Before (ECE {results[chosen]['ece_before']:.3f})")
    rel(axes[1], D["probs"], D["y"], f"After T={D['T']:.2f} (ECE {results[chosen]['ece_after']:.3f})")
    save(fig, os.path.join(R, "calibration.png"))

    def cases(idx, title, name):
        if not idx:
            return
        n = len(idx)
        fig, axes = plt.subplots(2, n, figsize=(2.3 * n, 5)); axes = np.array(axes).reshape(2, n)
        for j, i in enumerate(idx):
            img = cache[tep[i]]; x, _ = MRIDataset([te[i]], cache, False)[0]
            cam = cv2.resize(gradcam(D["model"], ARCH_OF[chosen], x[None], int(D["pred"][i])), (a.img, a.img))
            over = cv2.addWeighted(img, 0.6, cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET), 0.4, 0)
            axes[0, j].imshow(img[:, :, ::-1]); axes[1, j].imshow(over[:, :, ::-1])
            axes[0, j].set_title(f"true: {LABELS[D['y'][i]]}\npred: {LABELS[D['pred'][i]]} {D['probs'][i].max():.0%}", fontsize=8)
            axes[0, j].axis("off"); axes[1, j].axis("off")
        fig.suptitle(title); save(fig, os.path.join(R, name))
    rng = np.random.default_rng(SEED)
    ok = [i for i in range(len(tey)) if D["pred"][i] == D["y"][i]]
    bad = sorted([i for i in range(len(tey)) if D["pred"][i] != D["y"][i]], key=lambda i: -D["probs"][i].max())
    cases(sorted(rng.choice(ok, min(8, len(ok)), replace=False).tolist()), "Correct predictions (top) with Grad-CAM (bottom)", "examples_correct.png")
    cases(bad[:8], "Most confident mistakes", "examples_incorrect.png")
    mistakes = [{"file": os.path.basename(tep[i]), "true": CLASSES[D["y"][i]], "pred": CLASSES[D["pred"][i]],
                 "confidence": round(float(D["probs"][i].max()), 3)} for i in bad]

    # ---- checkpoint + demo
    m = D["model"]
    ckpt = {"format": "neurova-mri-v2", "arch": ARCH_OF[chosen], "model_key": chosen, "classes": CLASSES, "class_labels": LABELS,
            "img_size": a.img, "mean": MEAN.tolist(), "std": STD.tolist(), "preprocess": "crop_brain_region+resize",
            "temperature": D["T"], "state_dict": {k: v.cpu() for k, v in m.state_dict().items()},
            "metrics": results[chosen], "comparison": results,
            "dataset": {"name": "Brain Tumor MRI Dataset (Nickparvar 2021; IEEE DataPort doi:10.21227/1jny-g144)",
                        "license": "CC BY 4.0", "dedup": dedup_report, "split_counts": splits,
                        "split": "stratified 70/15/15 after perceptual-hash de-duplication, seed 42"},
            "trained_at": datetime.now(timezone.utc).isoformat(), "test_mode": bool(a.limit),
            "training_hardware": "CPU (2 threads)"}
    torch.save(ckpt, os.path.join(a.out, "weights", "neurova_mri_v2.pt"))
    json.dump({"chosen": chosen, "results": results, "dedup": dedup_report, "splits": splits,
               "history": {k: v["history"] for k, v in trained.items()}, "test_mistakes": mistakes},
              open(os.path.join(R, "metrics.json"), "w"), indent=2, default=float)
    demo = os.path.join(a.out, "demo", "mri"); os.makedirs(demo, exist_ok=True)
    for c in range(4):
        picks = [p for p, y in zip(tep, tey) if y == c][:2]
        for k, p in enumerate(picks):
            shutil.copy(p, os.path.join(demo, f"{CLASSES[c]}_{k + 1}{os.path.splitext(p)[1].lower()}"))
    log("done; deployed", chosen, json.dumps({k: round(v, 4) for k, v in results[chosen].items() if isinstance(v, float)}))


if __name__ == "__main__":
    main()
