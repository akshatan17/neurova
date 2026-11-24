"""
Download the public datasets used for training. Only needed if you want to retrain;
the app already contains the trained models.

    python training/get_data.py            MRI images and ImageNet weights (about 180 MB)
    python training/get_data.py --sleep    also the 20 Sleep-EDF nights from PhysioNet (about 1 GB)

Everything goes into datasets/. Files that are already there are skipped.
"""
import os
import shutil
import sys
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "datasets")

# Brain Tumor MRI Dataset (M. Nickparvar, 2021, CC BY 4.0). A public GitHub copy with the same
# per-class counts as the Kaggle original, so no Kaggle account is needed.
MRI_ZIP = "https://github.com/jamshaid120/Brain-Tumor-MRI-Dataset/archive/HEAD.zip"
# Official EfficientNet-B0 ImageNet weights (EfficientNet-PyTorch project, Apache 2.0).
WEIGHTS = "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth"
# Sleep-EDF Expanded (Kemp et al. 2000, PhysioNet, ODC-By 1.0): subjects 0-9, both nights.
SLEEP = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"
NIGHTS = ["SC4001E0-PSG", "SC4001EC-Hypnogram", "SC4002E0-PSG", "SC4002EC-Hypnogram", "SC4011E0-PSG", "SC4011EH-Hypnogram",
          "SC4012E0-PSG", "SC4012EC-Hypnogram", "SC4021E0-PSG", "SC4021EH-Hypnogram", "SC4022E0-PSG", "SC4022EJ-Hypnogram",
          "SC4031E0-PSG", "SC4031EC-Hypnogram", "SC4032E0-PSG", "SC4032EP-Hypnogram", "SC4041E0-PSG", "SC4041EC-Hypnogram",
          "SC4042E0-PSG", "SC4042EC-Hypnogram", "SC4051E0-PSG", "SC4051EC-Hypnogram", "SC4052E0-PSG", "SC4052EC-Hypnogram",
          "SC4061E0-PSG", "SC4061EC-Hypnogram", "SC4062E0-PSG", "SC4062EC-Hypnogram", "SC4071E0-PSG", "SC4071EC-Hypnogram",
          "SC4072E0-PSG", "SC4072EH-Hypnogram", "SC4081E0-PSG", "SC4081EC-Hypnogram", "SC4082E0-PSG", "SC4082EP-Hypnogram",
          "SC4091E0-PSG", "SC4091EC-Hypnogram", "SC4092E0-PSG", "SC4092EC-Hypnogram"]


def download(url: str, path: str):
    if os.path.exists(path):
        return
    print("downloading", os.path.basename(path), flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "neurova-student-project"})
    with urllib.request.urlopen(request) as reply, open(path + ".part", "wb") as out:
        shutil.copyfileobj(reply, out)
    os.replace(path + ".part", path)


def get_mri():
    target = os.path.join(DATA, "brain_tumor_mri")
    if os.path.isdir(os.path.join(target, "Training")):
        print("MRI dataset already there")
        return
    archive, unpacked = os.path.join(DATA, "mri.zip"), os.path.join(DATA, "_mri")
    download(MRI_ZIP, archive)
    print("unpacking", flush=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(unpacked)
    os.makedirs(target, exist_ok=True)
    for folder, subfolders, _ in os.walk(unpacked):          # find the Training and Testing folders wherever they are
        for name in ("Training", "Testing"):
            if name in subfolders and not os.path.exists(os.path.join(target, name)):
                shutil.move(os.path.join(folder, name), os.path.join(target, name))
    shutil.rmtree(unpacked)
    os.remove(archive)
    count = sum(len(files) for _, _, files in os.walk(target))
    print(f"MRI dataset ready: {count} images (expected 7023)")


def get_weights():
    os.makedirs(os.path.join(DATA, "pretrained"), exist_ok=True)
    download(WEIGHTS, os.path.join(DATA, "pretrained", "efficientnet-b0-355c32eb.pth"))


def get_sleep():
    os.makedirs(os.path.join(DATA, "sleep_edf"), exist_ok=True)
    for i, name in enumerate(NIGHTS, 1):
        print(f"[{i}/{len(NIGHTS)}]", end=" ")
        download(SLEEP + name + ".edf", os.path.join(DATA, "sleep_edf", name + ".edf"))
        print(name, "ok")


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    get_mri()
    get_weights()
    if "--sleep" in sys.argv:
        get_sleep()
    print("done")
