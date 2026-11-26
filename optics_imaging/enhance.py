"""Tone / contrast tools: histogram matching and region-targeted enhancement."""
from __future__ import annotations

import numpy as np


def histogram_match(source, reference):
    """Remap each channel of `source` so its CDF matches `reference` (a simple 'look transfer')."""
    out = np.empty_like(source)
    for c in range(source.shape[-1]):
        s = source[..., c].ravel()
        r = reference[..., c].ravel()
        s_vals, s_idx, s_counts = np.unique(s, return_inverse=True, return_counts=True)
        r_vals, r_counts = np.unique(r, return_counts=True)
        s_cdf = np.cumsum(s_counts) / s.size
        r_cdf = np.cumsum(r_counts) / r.size
        out[..., c] = np.interp(s_cdf, r_cdf, r_vals)[s_idx].reshape(source.shape[:-1])
    return out


def target_and_enhance(image, threshold=0.75, gain=1.5):
    """Find bright 'target' regions by luminance threshold and stretch their contrast.

    Returns (enhanced image, binary mask).
    """
    lum = image @ np.array([0.299, 0.587, 0.114])
    mask = lum > threshold
    out = image.copy()
    if mask.any():
        region = out[mask]
        mean = region.mean(axis=0)
        out[mask] = np.clip(mean + gain * (region - mean), 0, 1)
    return out, mask
