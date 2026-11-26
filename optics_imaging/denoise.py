"""Noise models and denoising filters."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from skimage import util
from skimage.restoration import denoise_bilateral, denoise_nl_means, estimate_sigma


def add_noise(image, mode="gaussian", seed=0, **kw):
    """mode: 'gaussian' (read noise), 'poisson' (shot noise), 's&p' (dead/hot pixels)."""
    return util.random_noise(image, mode=mode, rng=seed, **kw)


def gaussian(image, sigma=1.0):
    """Linear low-pass filter: removes noise but also blurs edges."""
    s = (sigma, sigma, 0) if image.ndim == 3 else sigma
    return gaussian_filter(image, sigma=s)


def median(image, size=3):
    """Non-linear rank filter: best for impulse (salt & pepper / hot pixel) noise."""
    s = (size, size, 1) if image.ndim == 3 else size
    return median_filter(image, size=s)


def bilateral(image, sigma_color=0.08, sigma_spatial=2):
    """Edge-preserving: averages only neighbours that are close in both space and intensity."""
    return denoise_bilateral(image, sigma_color=sigma_color, sigma_spatial=sigma_spatial,
                             channel_axis=-1 if image.ndim == 3 else None)


def nl_means(image):
    """Non-local means: averages similar patches anywhere in the image."""
    ca = -1 if image.ndim == 3 else None
    sigma = float(np.mean(estimate_sigma(image, channel_axis=ca)))
    return denoise_nl_means(image, h=0.8 * sigma, sigma=sigma, patch_size=5,
                            patch_distance=6, fast_mode=True, channel_axis=ca)


METHODS = {
    "gaussian": gaussian,
    "median": median,
    "bilateral": bilateral,
    "nl_means": nl_means,
}
