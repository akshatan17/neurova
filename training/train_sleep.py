"""
Train and evaluate the EEG sleep-staging model (CNN epoch encoder + BiGRU).

    python training/get_data.py --sleep  (once, downloads the recordings, about 1 GB)
    python training/train_sleep.py       (about 5 minutes on a CPU)
    python training/export_onnx.py       (puts the new model into the app)

Data: Sleep-EDF Expanded, sleep-cassette subjects 0-9, both nights (20 PSG +
hypnogram pairs). Single channel Fpz-Cz at 100 Hz. Stages 3 and 4 merge into
N3; wake is trimmed to 30 minutes either side of sleep. Subjects (not nights,
not epochs) are split into train / validation / test so the test score reflects
a person the model has never seen.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone

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
from app.preprocess import robust_scale                   # noqa: E402  (the same preprocessing the app runs)
from networks import SleepCNNGRU, SleepCNNTransformer, predict_recording   # noqa: E402

STAGES = ["W", "N1", "N2", "N3", "REM"]
ANNOT = {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2, "Sleep stage 3": 3,
         "Sleep stage 4": 3, "Sleep stage R": 4}
FS, EP = 100, 30
SPE = FS * EP
SEQ = 20
SEED = 42


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_night(psg, hyp):
    import mne
    raw = mne.io.read_raw_edf(psg, preload=True, verbose="error")
    raw.pick(["EEG Fpz-Cz"])
    sig = robust_scale(raw.get_data()[0] * 1e6).astype(np.float32)
    ann = mne.read_annotations(hyp)
    spans = [(o, d, ANNOT[s]) for o, d, s in zip(ann.onset, ann.duration, ann.description) if s in ANNOT]
    sleep = [(o, d) for o, d, l in spans if l != 0]
    t0 = max(0.0, min(o for o, _ in sleep) - 1800)
    t1 = max(o + d for o, d in sleep) + 1800
    X, Y, T = [], [], []
    for o, d, lab in spans:
        for k in range(int(d // EP)):
            t = o + k * EP
            if t < t0 or t + EP > t1:
                continue
            a = int(round(t * FS))
            seg = sig[a:a + SPE]
            if len(seg) == SPE:
                X.append(seg); Y.append(lab); T.append(t)
    order = np.argsort(T)
    return np.asarray(X, np.float32)[order], np.asarray(Y, int)[order], np.asarray(T)[order], sig, spans


class SeqDS(Dataset):
    def __init__(self, nights, augment):
        self.nights, self.augment, self.items = nights, augment, []
        for ni, n in enumerate(nights):
            for a in range(0, len(n["X"]) - SEQ + 1, SEQ // 2):
                self.items.append((ni, a))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ni, a = self.items[i]
        x = self.nights[ni]["X"][a:a + SEQ].copy()
        y = self.nights[ni]["Y"][a:a + SEQ]
        if self.augment:
            x = x * np.random.uniform(0.9, 1.1) + np.random.normal(0, 0.02, x.shape).astype(np.float32)
            x = np.roll(x, np.random.randint(-50, 50), axis=1)
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y)


def evaluate(model, nights):
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
    ys, ps = [], []
    for n in nights:
        ys.append(n["Y"]); ps.append(predict_recording(model, n["X"], SEQ).argmax(1))
    y, p = np.concatenate(ys), np.concatenate(ps)
    return y, p, accuracy_score(y, p), f1_score(y, p, average="macro", labels=range(5), zero_division=0), cohen_kappa_score(y, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "datasets", "sleep_edf"))
    ap.add_argument("--out", default=os.path.join(ROOT, "training", "outputs"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--arch", choices=["gru", "transformer"], default="gru",
                    help="gru = the deployed model; transformer = the comparison (writes to results/sleep_transformer)")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="random starting weights; the people in train/validation/test stay the same whatever the seed")
    a = ap.parse_args()
    comparison = a.arch != "gru" or a.seed != SEED            # anything but the deployed run is a comparison run
    if a.threads:
        torch.set_num_threads(a.threads)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    R = os.path.join(a.out, "results", "sleep" if not comparison else f"sleep_{a.arch}" + (f"_seed{a.seed}" if a.seed != SEED else ""))
    os.makedirs(R, exist_ok=True); os.makedirs(os.path.join(a.out, "weights"), exist_ok=True)

    psgs = sorted(glob.glob(os.path.join(a.data, "*-PSG.edf")))
    nights = []
    for p in psgs:
        stem = os.path.basename(p)[:6]                         # SC4ssN
        hyp = glob.glob(os.path.join(a.data, stem + "*-Hypnogram.edf"))[0]
        X, Y, T, sig, spans = load_night(p, hyp)
        nights.append({"subject": int(stem[3:5]), "night": int(stem[5]), "file": os.path.basename(p),
                       "X": X, "Y": Y, "T": T, "sig": sig, "spans": spans})
        log(f"{os.path.basename(p)}: {len(X)} epochs {dict(Counter(STAGES[v] for v in Y))}")
    tot = Counter(np.concatenate([n["Y"] for n in nights]).tolist())
    log("total", {STAGES[k]: v for k, v in sorted(tot.items())})

    subs = sorted({n["subject"] for n in nights})
    rng = np.random.default_rng(SEED); rng.shuffle(subs)
    test_s, val_s, train_s = sorted(subs[:2]), sorted(subs[2:3]), sorted(subs[3:])
    pick = lambda S: [n for n in nights if n["subject"] in S]
    tr, va, te = pick(train_s), pick(val_s), pick(test_s)
    log(f"subjects train {train_s} ({len(tr)} nights), val {val_s}, test {test_s}")

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(STAGES, [tot.get(i, 0) for i in range(5)], color="#4f7a59")
    ax.set_ylabel("30 s epochs"); ax.set_title("Sleep stage distribution, 20 nights")
    save(fig, os.path.join(R, "stage_distribution.png"))

    ds = SeqDS(tr, True)
    dl = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)
    ytr = np.concatenate([n["Y"] for n in tr])
    freq = np.bincount(ytr, minlength=5).astype(float) + 1
    w = torch.tensor((freq.sum() / freq) ** 0.5, dtype=torch.float32); w = w / w.mean()
    log("class weights", dict(zip(STAGES, w.numpy().round(2))), "|", len(ds), "training sequences")

    model = SleepCNNTransformer() if a.arch == "transformer" else SleepCNNGRU()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    best, best_state, wait, hist = -1, None, 0, []
    t_start = time.time()
    for ep in range(1, a.epochs + 1):
        te0 = time.time(); model.train(); L = []
        for x, y in dl:
            out = model(x)
            loss = F.cross_entropy(out.reshape(-1, 5), y.reshape(-1), weight=w)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
            L.append(loss.item())
        sched.step()
        _, _, vacc, vf1, vk = evaluate(model, va)
        hist.append({"epoch": ep, "train_loss": float(np.mean(L)), "val_acc": vacc, "val_f1": vf1, "val_kappa": vk})
        mark = ""
        if vf1 > best:
            best, wait, mark = vf1, 0, " *"
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        log(f"ep {ep:2d}/{a.epochs} loss {np.mean(L):.3f} | val acc {vacc:.3f} macro-F1 {vf1:.3f} kappa {vk:.3f}{mark} ({(time.time() - te0) / 60:.1f} min)")
        json.dump(hist, open(os.path.join(R, "history.json"), "w"))
        if wait >= a.patience:
            log("early stop"); break
    model.load_state_dict(best_state); model.eval()
    train_min = (time.time() - t_start) / 60

    from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
    y, p, acc, mf1, kap = evaluate(model, te)
    pr, rc, f1, sp = precision_recall_fscore_support(y, p, labels=range(5), zero_division=0)
    cm = confusion_matrix(y, p, labels=range(5))
    t = time.perf_counter(); predict_recording(model, te[0]["X"], SEQ); el = time.perf_counter() - t
    results = {"test_accuracy": acc, "test_macro_f1": mf1, "test_kappa": kap,
               "per_class": {STAGES[i]: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]), "support": int(sp[i])} for i in range(5)},
               "confusion_matrix": cm.tolist(), "params_m": sum(q.numel() for q in model.parameters()) / 1e6,
               "train_minutes": train_min, "cpu_seconds_per_night": el * 960 / len(te[0]["X"]),
               "subjects": {"train": train_s, "val": val_s, "test": test_s},
               "per_night": [{"file": n["file"], **dict(zip(("accuracy", "macro_f1", "kappa"), evaluate(model, [n])[2:]))} for n in te]}
    results["arch"], results["seed"] = a.arch, a.seed
    results["best_val_macro_f1"] = best
    log("TEST", f"acc {acc:.4f} macro-F1 {mf1:.4f} kappa {kap:.4f}")
    if comparison:                              # a comparison run: keep its numbers and curves, nothing else
        json.dump({"results": results, "history": hist}, open(os.path.join(R, "metrics.json"), "w"), indent=2, default=float)
        log("comparison run done")
        return
    print(classification_report(y, p, labels=range(5), target_names=STAGES, digits=3, zero_division=0))

    # figures
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
    ax[0].plot([h["train_loss"] for h in hist], marker="o"); ax[0].set_title("Training loss"); ax[0].set_xlabel("epoch")
    ax[1].plot([h["val_f1"] for h in hist], marker="o", label="macro-F1"); ax[1].plot([h["val_kappa"] for h in hist], marker="o", label="Cohen's kappa")
    ax[1].set_title("Validation (held-out subject)"); ax[1].legend(frameon=False); ax[1].set_xlabel("epoch")
    save(fig, os.path.join(R, "training_curves.png"))
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    ax.imshow(cmn, cmap="Greens", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.0%}", ha="center", va="center", fontsize=8, color="white" if cmn[i, j] > 0.6 else "black")
    ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.set_xticklabels(STAGES); ax.set_yticklabels(STAGES)
    ax.set_xlabel("predicted"); ax.set_ylabel("expert")
    ax.set_title(f"Sleep staging, unseen subjects {test_s}\nacc {acc:.3f}, macro-F1 {mf1:.3f}, κ {kap:.3f}", fontsize=10)
    save(fig, os.path.join(R, "confusion_matrix.png"))
    n0 = te[0]; P0 = predict_recording(model, n0["X"], SEQ).argmax(1)
    order = [0, 4, 1, 2, 3]; pos = {c: k for k, c in enumerate(order)}
    fig, ax = plt.subplots(2, 1, figsize=(12, 4.4), sharex=True)
    hrs = (n0["T"] - n0["T"][0]) / 3600
    ax[0].step(hrs, [pos[v] for v in n0["Y"]], where="post", color="#2f2620", lw=1); ax[0].set_title(f"Expert scoring ({n0['file'][:8]})", loc="left")
    ax[1].step(hrs, [pos[v] for v in P0], where="post", color="#4f7a59", lw=1); ax[1].set_title(f"Model ({(P0 == n0['Y']).mean():.0%} agreement)", loc="left")
    for x in ax:
        x.set_yticks(range(5)); x.set_yticklabels([STAGES[c] for c in order]); x.invert_yaxis()
    ax[1].set_xlabel("hours")
    save(fig, os.path.join(R, "hypnogram_test_subject.png"))

    # checkpoint
    torch.save({"format": "neurova-sleep-v1", "arch": "SleepCNNGRU", "classes": STAGES, "sfreq": FS, "epoch_seconds": EP,
                "seq_len": SEQ, "channel_hint": "Fpz-Cz", "scaling": "robust_per_recording(median, IQR)",
                "state_dict": model.state_dict(), "metrics": results,
                "dataset": {"name": "Sleep-EDF Expanded, sleep-cassette (PhysioNet)", "license": "ODC-By 1.0",
                            "nights": len(nights), "subjects": len(subs), "citation": "Kemp et al., IEEE-BME 47(9):1185-1194, 2000"},
                "trained_at": datetime.now(timezone.utc).isoformat(), "test_mode": False, "training_hardware": "CPU (2 threads)"},
               os.path.join(a.out, "weights", "neurova_sleep_v1.pt"))
    json.dump({"results": results, "history": hist, "nights": [{k: n[k] for k in ("subject", "night", "file")} | {"epochs": len(n["X"])} for n in nights]},
              open(os.path.join(R, "metrics.json"), "w"), indent=2, default=float)

    # demo: first 2 h of sleep from the first test night, plus the expert's scoring
    import pyedflib
    sig, spans = n0["sig"], n0["spans"]
    first = min(o for o, d, l in spans if l != 0)
    t0 = max(0.0, first - 900); tlen = int(min(7200, len(sig) / FS - t0) // EP * EP)
    seg = sig[int(t0 * FS):int(t0 * FS) + tlen * FS]
    demo = os.path.join(a.out, "demo", "eeg"); os.makedirs(demo, exist_ok=True)
    path = os.path.join(demo, "sleep_edf_test_subject.edf")
    wr = pyedflib.EdfWriter(path, 1, file_type=pyedflib.FILETYPE_EDFPLUS)
    lim = float(np.ceil(np.abs(seg).max() + 1))
    wr.setSignalHeaders([{"label": "EEG Fpz-Cz", "dimension": "a.u.", "sample_frequency": FS, "physical_max": lim,
                          "physical_min": -lim, "digital_max": 32767, "digital_min": -32768, "transducer": "", "prefilter": ""}])
    wr.writeSamples([seg.astype(np.float64)]); wr.close()
    truth = [{"t": round(o + k * EP - t0, 1), "stage": STAGES[l]} for o, d, l in spans for k in range(int(d // EP))
             if t0 <= o + k * EP and o + (k + 1) * EP <= t0 + tlen]
    json.dump({"source": f"Sleep-EDF Expanded {n0['file'][:8]} (held-out subject {n0['subject']})", "license": "ODC-By 1.0",
               "start_offset_s": t0, "epochs": sorted(truth, key=lambda e: e["t"])},
              open(os.path.join(demo, "sleep_edf_test_subject.hypnogram.json"), "w"))
    log(f"demo EDF {tlen / 3600:.1f} h, {len(truth)} scored epochs; done")


if __name__ == "__main__":
    main()
