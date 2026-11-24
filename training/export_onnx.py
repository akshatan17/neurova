"""
Convert the trained PyTorch checkpoints into the small files the app runs.

    python training/export_onnx.py
    python training/fit_familiarity.py     (afterwards, so the unfamiliar-scan check matches the new model)

Reads   training/outputs/weights/neurova_mri_v2.pt and neurova_sleep_v1.pt
Writes  models/mri.onnx, models/mri_head.npz, models/sleep.onnx, models/info.json
Copies  the figures and metrics of a new training run into results/, and its example files into samples/

Why ONNX: the app then needs only ONNX Runtime (a few tens of MB) instead of the
whole of PyTorch (about 1 GB), and predictions are faster on a CPU.
"""
import json
import os
import shutil
import sys

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from networks import SleepCNNGRU, build_mri_model   # noqa: E402

WEIGHTS = os.path.join(ROOT, "training", "outputs", "weights")
OUT = os.path.join(ROOT, "models")


class MRIForExport(nn.Module):
    """EfficientNet that also returns its last feature maps.

    The app needs those maps to draw the Grad-CAM heatmap, so the exported model
    gives back two things: the class scores and the 1280 x 6 x 6 feature maps.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        n = self.net
        x = n._swish(n._bn0(n._conv_stem(x)))
        for block in n._blocks:
            x = block(x)
        maps = n._bn1(n._conv_head(x))                       # feature maps, before the activation
        pooled = n._avg_pooling(n._swish(maps)).flatten(1)   # one number per map
        return n._fc(pooled), maps                           # class scores, feature maps


def export_mri():
    ck = torch.load(os.path.join(WEIGHTS, "neurova_mri_v2.pt"), map_location="cpu", weights_only=False)
    net = build_mri_model(ck["arch"], len(ck["classes"]))
    net.load_state_dict(ck["state_dict"])
    net.eval()
    net.set_swish(memory_efficient=False)        # the plain version of the activation can be exported
    size = ck["img_size"]
    x = torch.randn(1, 3, size, size)
    path = os.path.join(OUT, "mri.onnx")
    with torch.no_grad():
        want = net(x).numpy()                    # PyTorch's answer, to compare against below
    torch.onnx.export(MRIForExport(net), x, path, input_names=["image"], output_names=["scores", "feature_maps"],
                      dynamic_axes={"image": {0: "batch"}, "scores": {0: "batch"}, "feature_maps": {0: "batch"}},
                      opset_version=17, dynamo=False)
    # The final layer's weights, saved separately: the app uses them for Grad-CAM and MC dropout.
    np.savez(os.path.join(OUT, "mri_head.npz"), weight=net._fc.weight.detach().numpy(), bias=net._fc.bias.detach().numpy())
    got = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"image": x.numpy()})[0]
    print(f"mri.onnx    {os.path.getsize(path) / 1e6:5.1f} MB   largest difference from PyTorch: {np.abs(want - got).max():.2e}")
    return ck


def export_sleep():
    ck = torch.load(os.path.join(WEIGHTS, "neurova_sleep_v1.pt"), map_location="cpu", weights_only=False)
    net = SleepCNNGRU(len(ck["classes"]))
    net.load_state_dict(ck["state_dict"])
    net.eval()
    samples = int(ck["sfreq"] * ck["epoch_seconds"])
    x = torch.randn(1, ck["seq_len"], samples)
    path = os.path.join(OUT, "sleep.onnx")
    torch.onnx.export(net, x, path, input_names=["epochs"], output_names=["scores"],
                      dynamic_axes={"epochs": {0: "batch", 1: "length"}, "scores": {0: "batch", 1: "length"}},
                      opset_version=17, dynamo=False)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    worst = 0.0
    for length in (ck["seq_len"], 12):           # also check a shorter sequence than it was traced with
        x = torch.randn(2, length, samples)
        with torch.no_grad():
            want = net(x).numpy()
        worst = max(worst, float(np.abs(want - sess.run(None, {"epochs": x.numpy()})[0]).max()))
    print(f"sleep.onnx  {os.path.getsize(path) / 1e6:5.1f} MB   largest difference from PyTorch: {worst:.2e}")
    return ck


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    mri, sleep = export_mri(), export_sleep()
    info = {
        "mri": {"architecture": "EfficientNet-B0", "classes": mri["classes"], "labels": mri["class_labels"],
                "input_size": mri["img_size"], "temperature": round(float(mri["temperature"]), 4),
                "trained_at": mri["trained_at"], "dataset": mri["dataset"]},
        "sleep": {"architecture": "1-D CNN + bidirectional GRU", "classes": sleep["classes"],
                  "sample_rate": sleep["sfreq"], "epoch_seconds": sleep["epoch_seconds"], "sequence_length": sleep["seq_len"],
                  "channel": sleep["channel_hint"], "trained_at": sleep["trained_at"], "dataset": sleep["dataset"]},
    }
    json.dump(info, open(os.path.join(OUT, "info.json"), "w"), indent=2)
    print("wrote models/info.json")

    # If the training scripts have just been run, put their figures and example files where the app looks.
    produced = os.path.join(ROOT, "training", "outputs")
    for kind in ("mri", "sleep"):
        src = os.path.join(produced, "results", kind)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(ROOT, "results", kind), dirs_exist_ok=True)
            print(f"copied results/{kind}")
    if os.path.isdir(os.path.join(produced, "demo", "mri")):
        shutil.copytree(os.path.join(produced, "demo", "mri"), os.path.join(ROOT, "samples", "mri"), dirs_exist_ok=True)
    night = os.path.join(produced, "demo", "eeg", "sleep_edf_test_subject")
    if os.path.exists(night + ".edf"):
        shutil.copy(night + ".edf", os.path.join(ROOT, "samples", "sleep", "night.edf"))
        shutil.copy(night + ".hypnogram.json", os.path.join(ROOT, "samples", "sleep", "night.expert.json"))
