"""Colour constancy / white balance.

Model: observed = reflectance * illuminant (per RGB channel, von Kries).
Each method estimates the illuminant, then divides it out.
"""
from __future__ import annotations

import numpy as np


def apply_illuminant(image, illum_rgb):
    """Simulate shooting under a coloured light (e.g. warm tungsten)."""
    illum = np.asarray(illum_rgb, float)
    return np.clip(image * illum / illum.max(), 0, 1)


def _correct(image, illum):
    illum = np.asarray(illum, float)
    gain = illum.mean() / illum
    return np.clip(image * gain, 0, 1)


def estimate_gray_world(image):
    """Assumes the scene averages to neutral grey."""
    return image.reshape(-1, 3).mean(axis=0)


def estimate_max_rgb(image):
    """Assumes the brightest value in each channel is a white surface."""
    return image.reshape(-1, 3).max(axis=0)


def estimate_white_patch(image, top_percent=1.0):
    """Robust White-Patch Retinex: mean colour of the brightest top_percent of pixels."""
    flat = image.reshape(-1, 3)
    lum = flat.sum(axis=1)
    k = max(1, int(len(flat) * top_percent / 100))
    idx = np.argpartition(lum, -k)[-k:]
    return flat[idx].mean(axis=0)


ESTIMATORS = {
    "gray_world": estimate_gray_world,
    "max_rgb": estimate_max_rgb,
    "white_patch": estimate_white_patch,
}


def white_balance(image, method="gray_world"):
    """Return (corrected image, estimated illuminant)."""
    illum = ESTIMATORS[method](image)
    return _correct(image, illum), illum
