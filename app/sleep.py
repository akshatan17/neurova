"""
Sleep staging from one EEG channel.

An overnight recording is cut into 30-second pieces called epochs. The model
labels each epoch as Wake, N1, N2, N3 (deep sleep) or REM, which gives the
night's hypnogram. The network is a 1-D CNN followed by a bidirectional GRU,
trained in PyTorch (see training/train_sleep.py) and exported to ONNX.
"""
import json
import os

import numpy as np
import onnxruntime as ort

from .edf import read_channel, read_header
from .preprocess import robust_scale

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
INFO = json.load(open(os.path.join(MODELS, "info.json")))["sleep"]
STAGES = INFO["classes"]                                  # W, N1, N2, N3, REM
RATE, EPOCH_S, SEQ = INFO["sample_rate"], INFO["epoch_seconds"], INFO["sequence_length"]
STAGE_NAMES = {"W": "Wake", "N1": "N1 (light sleep)", "N2": "N2", "N3": "N3 (deep sleep)", "REM": "REM"}

_session = ort.InferenceSession(os.path.join(MODELS, "sleep.onnx"), providers=["CPUExecutionProvider"])


def pick_channel(labels: list[str]) -> tuple[int, str | None]:
    """The model was trained on Fpz-Cz. Use it if present, otherwise the nearest frontal channel."""
    lower = [name.lower() for name in labels]
    for wanted in ("fpz", "fp1", "fp2", "fz", "f3", "f4"):
        for i, name in enumerate(lower):
            if wanted in name and "eog" not in name and "emg" not in name:
                return i, None if wanted == "fpz" else f"{labels[i]} was used in place of Fpz-Cz, so accuracy may be lower."
    return 0, f"No frontal EEG channel was found, so '{labels[0]}' was used. Accuracy may be lower."


def resample(signal: np.ndarray, rate: float, new_rate: float) -> np.ndarray:
    """Change the sampling rate by trimming or padding the signal's frequency spectrum."""
    if abs(rate - new_rate) < 1e-6:
        return signal
    n_new = int(round(len(signal) * new_rate / rate))
    spectrum = np.fft.rfft(signal)
    return np.fft.irfft(spectrum[: n_new // 2 + 1], n_new) * (n_new / len(signal))


def predict_epochs(epochs: np.ndarray) -> np.ndarray:
    """epochs: N x 3000. Returns N x 5 stage probabilities.

    The GRU reads 20 epochs at a time. Windows overlap by half, and each epoch's
    probabilities are averaged over the windows that contain it, so no epoch is
    judged only from the edge of a window.
    """
    n = len(epochs)
    length = min(SEQ, n)
    starts = list(range(0, n - length + 1, max(length // 2, 1)))
    if starts[-1] != n - length:
        starts.append(n - length)
    windows = np.stack([epochs[s: s + length] for s in starts]).astype(np.float32)
    scores = _session.run(None, {"epochs": windows})[0]                      # windows x length x 5
    e = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probs = e / e.sum(axis=-1, keepdims=True)
    total, count = np.zeros((n, len(STAGES)), np.float32), np.zeros(n, np.float32)
    for s, p in zip(starts, probs):
        total[s: s + length] += p
        count[s: s + length] += 1
    return total / count[:, None]


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Agreement between two scorers, corrected for the agreement expected by chance."""
    a, b = np.array(a), np.array(b)
    observed = float((a == b).mean())
    by_chance = sum(float((a == s).mean() * (b == s).mean()) for s in STAGES)
    return (observed - by_chance) / (1 - by_chance) if by_chance < 1 else 1.0


def analyze(edf_bytes: bytes, expert: dict | None = None) -> dict:
    """Stage a whole recording. `expert` is the technician's scoring, shipped with the demo night."""
    header = read_header(edf_bytes)
    eeg = [i for i, name in enumerate(header["labels"]) if "annotation" not in name.lower()]
    if not eeg:
        raise ValueError("The EDF file has no signal channels.")
    which, channel_note = pick_channel([header["labels"][i] for i in eeg])
    channel = eeg[which]
    signal, rate = read_channel(edf_bytes, header, channel)
    signal = resample(signal, rate, RATE)

    per_epoch = int(RATE * EPOCH_S)                                           # 3,000 samples
    n = len(signal) // per_epoch
    if n < 10:
        raise ValueError(f"Sleep staging needs at least 5 minutes of EEG. This recording has {len(signal) / RATE / 60:.1f}.")
    epochs = robust_scale(signal[: n * per_epoch]).reshape(n, per_epoch)
    probs = predict_epochs(epochs)
    stages = [STAGES[i] for i in probs.argmax(axis=1)]
    confidence = probs.max(axis=1)

    minutes = {s: round(stages.count(s) * EPOCH_S / 60, 1) for s in STAGES}
    recorded = n * EPOCH_S / 60
    asleep = round(sum(m for s, m in minutes.items() if s != "W"), 1)
    sleeping = [i for i, s in enumerate(stages) if s != "W"]
    night = {"recorded_minutes": round(recorded, 1), "asleep_minutes": asleep,
             "efficiency_percent": round(asleep / recorded * 100, 1),
             "fell_asleep_after_minutes": round(sleeping[0] * EPOCH_S / 60, 1) if sleeping else None}

    agreement, expert_stages = None, None
    if expert:
        scored = {int(round(e["t"] / EPOCH_S)): e["stage"] for e in expert["epochs"]}
        both = [i for i in range(n) if i in scored]
        if both:
            expert_stages = [scored.get(i) for i in range(n)]
            mine, theirs = [stages[i] for i in both], [scored[i] for i in both]
            agreement = {"epochs": len(both), "percent": round(float(np.mean([m == t for m, t in zip(mine, theirs)])) * 100, 1),
                         "kappa": round(cohen_kappa(mine, theirs), 3), "source": expert.get("source")}

    summary = f"Over {recorded / 60:.1f} hours the model scored {asleep / 60:.1f} hours as sleep (efficiency {night['efficiency_percent']}%)."
    if sleeping:
        most = max((s for s in STAGES if s != "W"), key=lambda s: minutes[s])
        summary += (f" Sleep began after {night['fell_asleep_after_minutes']:.0f} minutes. Most of it was {STAGE_NAMES[most]}, "
                    f"with {minutes['N3']:.0f} minutes of deep sleep and {minutes['REM']:.0f} minutes of REM.")
    if agreement:
        summary += f" The model agreed with the expert's scoring of this night on {agreement['percent']}% of epochs (kappa {agreement['kappa']})."

    return {
        "kind": "sleep",
        "headline": f"{asleep / 60:.1f} h asleep, {night['efficiency_percent']:.0f}% efficiency",
        "channel": header["labels"][channel], "channel_note": channel_note,
        "epochs": n, "stages": stages, "expert_stages": expert_stages,
        "minutes": minutes, "night": night, "agreement": agreement,
        "average_confidence": round(float(confidence.mean()) * 100, 1),
        "unsure_epochs": int((confidence < 0.5).sum()),
        "summary": summary, "needs_review": True,
        "caveat": "Trained on 20 nights from healthy adults. Clinical sleep scoring uses several EEG, eye and muscle channels and a trained technician.",
    }
