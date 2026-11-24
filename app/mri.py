"""
Brain MRI analysis: which of four classes a scan belongs to, and how far to trust it.

The network is EfficientNet-B0, trained in PyTorch (see training/train_mri.py) and
exported to ONNX, so the app runs it with ONNX Runtime and plain NumPy.

For every scan the app reports four things:
  1. the predicted class and a calibrated confidence (temperature scaling)
  2. a Grad-CAM heatmap of where the network looked
  3. an uncertainty level, from Monte Carlo dropout
  4. a stability check: does the answer survive noise, blur and brightness changes?
  5. a familiarity check: is this scan like the scans the network was trained on?
"""
import base64
import hashlib
import json
import os

import cv2
import numpy as np
import onnxruntime as ort

from .preprocess import prepare_mri

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
INFO = json.load(open(os.path.join(MODELS, "info.json")))["mri"]
CLASSES, LABELS, TEMPERATURE = INFO["classes"], INFO["labels"], INFO["temperature"]

_session = ort.InferenceSession(os.path.join(MODELS, "mri.onnx"), providers=["CPUExecutionProvider"])
_head = np.load(os.path.join(MODELS, "mri_head.npz"))
HEAD_W, HEAD_B = _head["weight"], _head["bias"]        # final layer: 4 classes x 1280 features

_familiar = np.load(os.path.join(MODELS, "mri_familiar.npz"))       # made by training/fit_familiarity.py

DROPOUT = 0.3            # the dropout rate the network was trained with
MC_PASSES = 20


# ------------------------------------------------------------------ small maths helpers
def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def run_network(x):
    """x: N x 3 x 192 x 192. Returns class scores (N x 4) and feature maps (N x 1280 x 6 x 6)."""
    return _session.run(None, {"image": x.astype(np.float32)})


def to_png(img_bgr) -> str:
    return base64.b64encode(cv2.imencode(".png", img_bgr)[1]).decode()


def pooled_features(maps):
    """Boil the 1280 x 6 x 6 feature maps down to 1280 numbers, as the network itself does before its last layer."""
    return (maps * sigmoid(maps)).mean(axis=(1, 2))


# ------------------------------------------------------------------ 2. Grad-CAM
def grad_cam(maps, cls):
    """Where did the evidence for class `cls` come from?

    Grad-CAM weights each feature map by how much the class score changes when
    that map changes (its gradient), then adds the weighted maps together.

    After the feature maps the network only does: swish activation -> average
    over the 6 x 6 grid -> one linear layer. That is simple enough to
    differentiate by hand, so no deep learning framework is needed here:

        d score / d map[k, i, j] = HEAD_W[cls, k] * swish'(map[k, i, j]) / 36
    """
    s = sigmoid(maps)
    swish_slope = s * (1 + maps * (1 - s))                               # derivative of x * sigmoid(x)
    gradient = HEAD_W[cls][:, None, None] * swish_slope / maps[0].size   # 1280 x 6 x 6
    importance = gradient.mean(axis=(1, 2))                              # one weight per map
    cam = np.maximum((importance[:, None, None] * maps).sum(axis=0), 0)  # keep positive evidence only
    return cam / (cam.max() + 1e-8)                                      # 6 x 6, scaled to 0..1


def draw_heatmap(shown_bgr, cam):
    """Stretch the 6 x 6 map over the image and blend it in as colour."""
    big = cv2.resize(cam.astype(np.float32), shown_bgr.shape[1::-1], interpolation=cv2.INTER_CUBIC).clip(0, 1)
    colour = cv2.applyColorMap((big * 255).astype(np.uint8), cv2.COLORMAP_JET)
    alpha = (big[..., None] * 0.6).astype(np.float32)                    # stronger colour where attention is higher
    return (shown_bgr * (1 - alpha) + colour * alpha).astype(np.uint8), big


def describe_position(big):
    """Say roughly where in the picture the attention sits (a single slice has no anatomical labels)."""
    ys, xs = np.nonzero(big > 0.5)
    if len(xs) == 0:
        return "centre"
    x, y = xs.mean() / big.shape[1], ys.mean() / big.shape[0]
    vertical = "upper" if y < 0.38 else "lower" if y > 0.62 else ""
    horizontal = "left" if x < 0.38 else "right" if x > 0.62 else ""
    return " ".join(p for p in (vertical, horizontal) if p) or "centre"


# ------------------------------------------------------------------ 3. uncertainty
def mc_dropout(maps, seed):
    """Run the final layer 20 times with dropout left on and see how much the answers differ.

    Dropout randomly switches off 30% of the 1280 features. If the prediction
    depends on a few fragile features, the 20 answers disagree, and that
    disagreement is a measure of how unsure the model is about this scan.
    """
    pooled = pooled_features(maps)                                       # 1280 features
    rng = np.random.default_rng(seed)
    keep = rng.random((MC_PASSES, pooled.size)) > DROPOUT
    scores = (pooled * keep / (1 - DROPOUT)) @ HEAD_W.T + HEAD_B          # 20 x 4
    mean = softmax(scores / TEMPERATURE).mean(axis=0)
    entropy = float(-(mean * np.log(mean + 1e-12)).sum() / np.log(len(mean)))   # 0 = certain, 1 = no idea
    level = "Low" if entropy < 0.25 else "Moderate" if entropy < 0.6 else "High"
    note = {"Low": "Twenty repeated passes with dropout agree closely.",
            "Moderate": "Repeated passes vary somewhat, so read the confidence as approximate.",
            "High": "Repeated passes disagree. This scan is close to a decision boundary."}[level]
    return {"level": level, "score": round(entropy, 3), "note": note}


# ------------------------------------------------------------------ 4. stability
def stability_check(x, predicted, seed):
    """Disturb the image slightly in seven ways and see whether the answer changes."""
    rng = np.random.default_rng(seed)
    picture = x[0].transpose(1, 2, 0)
    tests = [(f"Noise, strength {s}", x + rng.standard_normal(x.shape).astype(np.float32) * s) for s in (0.05, 0.1, 0.2)]
    tests += [(f"Brightness {d:+.1f}", x + d) for d in (-0.3, 0.3)]
    tests += [(f"Blur, {k} pixels", cv2.GaussianBlur(picture, (k, k), 0).transpose(2, 0, 1)[None]) for k in (3, 7)]
    scores, _ = run_network(np.concatenate([t for _, t in tests]))
    probs = softmax(scores / TEMPERATURE)
    rows = [{"test": name, "prediction": LABELS[int(p.argmax())], "confidence": round(float(p.max()) * 100, 1),
             "changed": int(p.argmax()) != predicted} for (name, _), p in zip(tests, probs)]
    changed = [r["test"] for r in rows if r["changed"]]
    verdict = ("The prediction held under all seven disturbances." if not changed else
               f"The prediction changed under {len(changed)} of {len(rows)} disturbances. Treat it with extra caution.")
    return {"held": not changed, "verdict": verdict, "tests": rows}


# ------------------------------------------------------------------ 5. familiarity
def familiarity_check(maps):
    """Is this scan like the scans the network learned from?

    The network can only answer with one of its four classes, even for a scan of
    something it never saw. So we measure how far this scan's features are from
    the nearest class of training scans (Mahalanobis distance: distance counted
    in units of how much the training scans themselves vary). 95% of ordinary
    scans stay under the threshold; a scan above it is flagged as unfamiliar.
    """
    f = _familiar
    z = (pooled_features(maps) - f["centre"]) @ f["axes"]                # reduce 1280 features to a few
    gaps = z - f["class_centres"]                                        # offset from each class centre
    distance = float(np.sqrt(np.einsum("ij,jk,ik->i", gaps, f["inverse_spread"], gaps).min()))
    familiar = distance <= float(f["threshold"])
    note = ("This scan resembles the scans the model was trained on." if familiar else
            "This scan is unlike the scans the model was trained on. The model only knows glioma, meningioma, "
            "pituitary tumour and no tumour in the brain, so its answer here may mean little.")
    return {"familiar": bool(familiar), "distance": round(distance, 2), "threshold": round(float(f["threshold"]), 2), "note": note}


# ------------------------------------------------------------------ the whole analysis
PHRASE = {"glioma": "a pattern most consistent with a glioma", "meningioma": "a pattern most consistent with a meningioma",
          "pituitary": "a pattern most consistent with a pituitary tumour", "notumor": "no pattern suggestive of a tumour"}


def analyze(image_bytes: bytes) -> dict:
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR) if image_bytes else None
    if img is None:
        raise ValueError("That file could not be read as an image. Upload a PNG or JPG brain MRI slice.")
    seed = int(hashlib.sha256(image_bytes).hexdigest()[:8], 16)       # same scan -> same random numbers -> same result

    x, shown = prepare_mri(img)
    scores, maps = run_network(x)
    probs = softmax(scores[0] / TEMPERATURE)     # 1. temperature scaling makes the confidence honest
    order = np.argsort(-probs)
    top, second = int(order[0]), int(order[1])
    is_tumour = CLASSES[top] != "notumor"
    close_call = probs[top] - probs[second] < 0.3

    cam = grad_cam(maps[0], top)
    overlay, big = draw_heatmap(shown, cam)
    uncertainty = mc_dropout(maps[0], seed)
    stability = stability_check(x, top, seed)
    familiarity = familiarity_check(maps[0])
    position = describe_position(big) if is_tumour else None

    percent = "over 99%" if probs[top] >= 0.995 else f"{probs[top] * 100:.0f}%"
    summary = f"The scan shows {PHRASE[CLASSES[top]]}, with a confidence of {percent}."
    if position:
        summary += f" The model's attention is {'in the centre' if position == 'centre' else 'in the ' + position + ' part'} of the image."
    if close_call:
        summary += f" The second most likely class is {LABELS[second].lower()} at {probs[second] * 100:.0f}%, so this one is close."
    if not familiarity["familiar"]:
        summary += " This scan looks unlike the scans the model was trained on, so treat the answer with caution."
    needs_review = is_tumour or close_call or uncertainty["level"] != "Low" or not stability["held"] or not familiarity["familiar"]

    return {
        "kind": "mri",
        "prediction": CLASSES[top], "label": LABELS[top], "is_tumour": is_tumour,
        "confidence": round(float(probs[top]) * 100, 1),
        "confidence_text": percent, "headline": f"{LABELS[top]}, {percent} confidence",
        "probabilities": [{"label": LABELS[i], "percent": round(float(probs[i]) * 100, 1)} for i in order],
        "uncertainty": uncertainty, "stability": stability, "familiarity": familiarity, "position": position,
        "summary": summary, "needs_review": bool(needs_review),
        "image": to_png(shown), "heatmap": to_png(overlay),
    }
