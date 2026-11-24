"""
The neural networks, written in PyTorch. Used for training and for the export to ONNX.

  build_mri_model("efficientnet_b0")  EfficientNet-B0, the deployed MRI classifier
  build_mri_model("neurova_cnn")      a small CNN trained from scratch, the baseline
  SleepCNNGRU                         1-D CNN + bidirectional GRU for sleep staging (deployed)
  SleepCNNTransformer                 the same, with a Transformer encoder instead of the GRU (comparison)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ MRI
class ConvBlock(nn.Module):
    """Convolution -> batch norm -> ReLU -> halve the image size."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch),
                                   nn.ReLU(inplace=True), nn.MaxPool2d(2))

    def forward(self, x):
        return self.block(x)


class NeurovaCNN(nn.Module):
    """Baseline: four conv blocks, global average pooling, two dense layers. About 0.1 M parameters."""

    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(ConvBlock(3, 16), ConvBlock(16, 32), ConvBlock(32, 64), ConvBlock(64, 128))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(128, 64), nn.ReLU(inplace=True),
                                        nn.Dropout(0.3), nn.Linear(64, num_classes))

    def forward(self, x):
        return self.classifier(self.gap(self.features(x)))


def build_mri_model(arch: str, num_classes: int = 4) -> nn.Module:
    if arch == "neurova_cnn":
        return NeurovaCNN(num_classes)
    if arch in ("efficientnet_b0", "efficientnet_b0_tf"):
        # The original Google EfficientNet-B0 (lukemelas port, Apache 2.0), so the official
        # ImageNet weights load without any conversion.
        from efficientnet_pytorch import EfficientNet
        return EfficientNet.from_name("efficientnet-b0", num_classes=num_classes, dropout_rate=0.3)
    raise ValueError(f"Unknown architecture: {arch}")


def cam_layer(model: nn.Module, arch: str) -> nn.Module:
    """The last layer that still has a spatial map; Grad-CAM reads it."""
    return model.features[-1] if arch == "neurova_cnn" else model._bn1


# ------------------------------------------------------------------ sleep
class EpochEncoder(nn.Module):
    """Turns one 30-second epoch (3,000 samples) into 128 features.

    The first convolution is 50 samples wide (half a second), so its filters can
    learn to respond to EEG rhythms such as delta waves and sleep spindles.
    """

    def __init__(self, d=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 50, stride=6, bias=False), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(8), nn.Dropout(0.3),
            nn.Conv1d(32, 64, 8, padding=4, bias=False), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 8, padding=4, bias=False), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, d, 8, padding=4, bias=False), nn.BatchNorm1d(d), nn.ReLU(), nn.MaxPool1d(4),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3))

    def forward(self, x):
        return self.net(x[:, None, :])


class SleepCNNGRU(nn.Module):
    """Encode each epoch with the CNN, then let a bidirectional GRU read 20 epochs in a row.

    A sleep stage depends on what came before and after it, which is why a human
    scorer looks at the neighbouring epochs. The GRU gives the model that context.
    """

    def __init__(self, n_classes=5, d=128, hidden=64):
        super().__init__()
        self.encoder = EpochEncoder(d)
        self.gru = nn.GRU(d, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * hidden, n_classes))

    def forward(self, x):                       # x: batch x epochs x samples
        B, L, S = x.shape
        z = self.encoder(x.reshape(B * L, S)).reshape(B, L, -1)
        h, _ = self.gru(z)
        return self.head(h)                     # batch x epochs x 5 stage scores


class SleepCNNTransformer(nn.Module):
    """The same epoch encoder, but a Transformer reads the 20 epochs instead of a GRU.

    A GRU reads the epochs one after another and carries a memory forward. A
    Transformer looks at all 20 at once: self-attention lets every epoch weigh
    every other epoch directly. Attention has no built-in sense of order, so a
    learned position embedding tells it which epoch is which.
    """

    def __init__(self, n_classes=5, d=128, heads=4, layers=2, max_len=20):
        super().__init__()
        self.encoder = EpochEncoder(d)
        self.position = nn.Parameter(torch.randn(1, max_len, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.2, activation="gelu", batch_first=True, norm_first=True)
        self.attention = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.3), nn.Linear(d, n_classes))

    def forward(self, x):                       # x: batch x epochs x samples
        B, L, S = x.shape
        z = self.encoder(x.reshape(B * L, S)).reshape(B, L, -1)
        z = self.attention(z + self.position[:, :L])
        return self.head(z)                     # batch x epochs x 5 stage scores


@torch.no_grad()
def predict_recording(model, epochs: np.ndarray, seq: int) -> np.ndarray:
    """Stage a whole night: slide a window of `seq` epochs along it, overlapping by half,
    and average each epoch's probabilities over the windows that contain it.
    (app/sleep.py does the same with the exported model.)"""
    model.eval()
    n = len(epochs)
    total, count = np.zeros((n, 5), np.float32), np.zeros(n, np.float32)
    length = min(seq, n)
    starts = list(range(0, max(n - length, 0) + 1, max(length // 2, 1)))
    if starts[-1] != n - length:
        starts.append(n - length)
    for s in starts:
        x = torch.from_numpy(epochs[s:s + length][None].astype(np.float32))
        total[s:s + length] += F.softmax(model(x)[0], -1).numpy()
        count[s:s + length] += 1
    return total / count[:, None]
