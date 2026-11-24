"""
Compare the GRU sleep model (deployed) with a Transformer version, fairly.

    python training/compare_sleep.py          (about 30 minutes on a CPU)

Both models share everything except the part that reads the 20-epoch sequence:
the same CNN epoch encoder, data, people in each split, class weights and
training settings. Each is trained three times with different random starting
weights (seeds 42, 1 and 2), because the test set is only two people and a
single run can be lucky or unlucky by a couple of points. Runs that already
exist in training/outputs are reused.

Writes results/sleep/comparison.json, which the model page and the report read.
"""
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.join(ROOT, "training", "outputs")
DATA = sys.argv[sys.argv.index("--data") + 1] if "--data" in sys.argv else os.path.join(ROOT, "datasets", "sleep_edf")
SEEDS = [42, 1, 2]
NAMES = {"gru": "CNN + bidirectional GRU", "transformer": "CNN + Transformer encoder"}


def result_file(arch: str, seed: int) -> str:
    folder = "sleep" if (arch, seed) == ("gru", 42) else f"sleep_{arch}" + ("" if seed == 42 else f"_seed{seed}")
    return os.path.join(OUT, "results", folder, "metrics.json")


runs = {arch: [] for arch in NAMES}
for arch in NAMES:
    for seed in SEEDS:
        path = result_file(arch, seed)
        if not os.path.exists(path):
            print(f"training {arch}, seed {seed}", flush=True)
            subprocess.run([sys.executable, os.path.join(ROOT, "training", "train_sleep.py"), "--arch", arch,
                            "--seed", str(seed), "--data", DATA, "--out", OUT], check=True)
        m = json.load(open(path))
        r = m["results"]
        runs[arch].append({"seed": seed, "test_accuracy": r["test_accuracy"], "test_macro_f1": r["test_macro_f1"],
                           "test_kappa": r["test_kappa"], "best_val_macro_f1": max(h["val_f1"] for h in m["history"]),
                           "n1_f1": r["per_class"]["N1"]["f1"], "wake_f1": r["per_class"]["W"]["f1"],
                           "params_m": r["params_m"], "train_minutes": r["train_minutes"]})

summary = []
for arch, rows in runs.items():
    row = {"key": arch, "name": NAMES[arch], "deployed": arch == "gru", "runs": rows, "params_m": rows[0]["params_m"]}
    for k in ("test_accuracy", "test_macro_f1", "test_kappa", "best_val_macro_f1", "n1_f1", "wake_f1", "train_minutes"):
        values = np.array([r[k] for r in rows])
        row[k] = {"mean": float(values.mean()), "min": float(values.min()), "max": float(values.max())}
    summary.append(row)

json.dump({"seeds": SEEDS, "models": summary}, open(os.path.join(ROOT, "results", "sleep", "comparison.json"), "w"), indent=2)
for row in summary:
    a, k = row["test_accuracy"], row["test_kappa"]
    print(f"{row['name']:28s} accuracy {a['mean']:.3f} ({a['min']:.3f}-{a['max']:.3f})  kappa {k['mean']:.3f} ({k['min']:.3f}-{k['max']:.3f})")
print("wrote results/sleep/comparison.json")
