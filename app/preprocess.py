"""
Preprocessing shared by training and the app.

Both the training scripts and the running app import these functions, so a scan
is always prepared in exactly the same way it was during training.
"""
import cv2
import numpy as np

MRI_SIZE = 192                                   # the network's input is 192 x 192 pixels
MRI_MEAN = np.array([0.485, 0.456, 0.406], np.float32)   # ImageNet statistics, because the
MRI_STD = np.array([0.229, 0.224, 0.225], np.float32)    # network was pretrained on ImageNet


def crop_brain_region(img_bgr: np.ndarray) -> np.ndarray:
    """Cut away the black border around the head so every brain fills the frame."""
    gray = cv2.GaussianBlur(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, mask = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(cv2.erode(mask, None, iterations=2), None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < 0.02 * img_bgr.shape[0] * img_bgr.shape[1]:
        return img_bgr
    x, y, w, h = cv2.boundingRect(biggest)
    pad = 5
    crop = img_bgr[max(0, y - pad): y + h + pad, max(0, x - pad): x + w + pad]
    return crop if crop.size else img_bgr


def prepare_mri(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (network input of shape 1x3x192x192, the cropped image for display)."""
    shown = cv2.resize(crop_brain_region(img_bgr), (MRI_SIZE, MRI_SIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(shown, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = ((rgb - MRI_MEAN) / MRI_STD).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(x, np.float32), shown


def robust_scale(signal: np.ndarray) -> np.ndarray:
    """Centre an EEG recording on its median and divide by its interquartile range.

    Unlike mean and standard deviation, these are not thrown off by the large
    spikes that movement causes in an overnight recording.
    """
    median = np.median(signal)
    iqr = np.subtract(*np.percentile(signal, [75, 25]))
    return (signal - median) / (iqr if iqr > 1e-6 else 1.0)
