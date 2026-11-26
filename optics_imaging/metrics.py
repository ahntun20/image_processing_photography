"""Image-quality metrics used to compare algorithms objectively."""
from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(reference: np.ndarray, test: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB (higher is better). Images in [0, 1]."""
    return float(peak_signal_noise_ratio(reference, np.clip(test, 0, 1), data_range=1.0))


def ssim(reference: np.ndarray, test: np.ndarray) -> float:
    """Structural similarity index, 0..1 (higher is better)."""
    kw = {"channel_axis": -1} if reference.ndim == 3 else {}
    return float(structural_similarity(reference, np.clip(test, 0, 1), data_range=1.0, **kw))


def angular_error_deg(est_rgb, true_rgb) -> float:
    """Angle between estimated and true illuminant colour vectors.

    This is the standard error measure in colour-constancy research;
    under ~3 degrees is generally hard to notice.
    """
    a = np.asarray(est_rgb, float)
    b = np.asarray(true_rgb, float)
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
