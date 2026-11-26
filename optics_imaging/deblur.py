"""Image formation (blur + noise) and deconvolution algorithms.

Forward model:  g = h (*) f + n
  f = sharp scene, h = PSF, n = sensor noise, g = recorded image.
Deconvolution estimates f from g when h is known (non-blind).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve
from skimage import restoration


def _per_channel(fn, img, *args, **kw):
    if img.ndim == 2:
        return fn(img, *args, **kw)
    return np.stack([fn(img[..., c], *args, **kw) for c in range(img.shape[-1])], axis=-1)


def _padded(fn, img, psf, *args, **kw):
    """Run an FFT-based deconvolution on a reflect-padded copy, then crop.

    FFT methods assume the image wraps around; padding stops the left edge
    'ringing' into the right edge.
    """
    p = 2 * max(psf.shape)
    def one(ch):
        out = fn(np.pad(ch, p, mode="reflect"), psf, *args, **kw)
        return out[p:-p, p:-p]
    return _per_channel(one, img)


def blur(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """Convolve with the PSF (reflect-padded so borders don't darken)."""
    def one(ch):
        p = max(psf.shape)
        padded = np.pad(ch, p, mode="reflect")
        return fftconvolve(padded, psf, mode="same")[p:-p, p:-p]
    return np.clip(_per_channel(one, image), 0, 1)


def add_gaussian_noise(image, sigma=0.01, seed=0):
    """Read noise / thermal noise model."""
    rng = np.random.default_rng(seed)
    return np.clip(image + sigma * rng.standard_normal(image.shape), 0, 1)


def add_poisson_noise(image, photons=1000, seed=0):
    """Photon shot noise: each pixel collects Poisson(photons * intensity)."""
    rng = np.random.default_rng(seed)
    return np.clip(rng.poisson(image * photons) / photons, 0, 1)


def psf2otf(psf: np.ndarray, shape) -> np.ndarray:
    """FFT of the PSF, zero-padded and *centred at the origin*.

    The original deblurred_3.py padded the PSF into the top-left corner without
    shifting it, which translates the result by half the kernel width.
    """
    pad = np.zeros(shape)
    h, w = psf.shape
    pad[:h, :w] = psf
    pad = np.roll(pad, (-(h // 2), -(w // 2)), axis=(0, 1))
    return np.fft.fft2(pad)


def tikhonov(image, psf, alpha=0.01):
    """Regularised inverse filter: F = H* G / (|H|^2 + alpha)."""
    def one(ch, psf):
        H = psf2otf(psf, ch.shape)
        G = np.fft.fft2(ch)
        return np.real(np.fft.ifft2(np.conj(H) * G / (np.abs(H) ** 2 + alpha)))
    return np.clip(_padded(one, image, psf), 0, 1)


def wiener(image, psf, balance=0.01):
    """Wiener filter (balance ~ noise-to-signal power ratio)."""
    return np.clip(_padded(restoration.wiener, image, psf, balance), 0, 1)


def unsupervised_wiener(image, psf):
    """Wiener filter whose regularisation is estimated from the data (Bayesian, Gibbs sampling)."""
    def one(ch, psf):
        out, _ = restoration.unsupervised_wiener(ch, psf, rng=0)
        return out
    return np.clip(_padded(one, image, psf), 0, 1)


def richardson_lucy(image, psf, num_iter=30):
    """Richardson-Lucy: maximum-likelihood deconvolution for Poisson (photon) noise.

    Used in astronomy and fluorescence microscopy (and famously on Hubble images).
    """
    return np.clip(_padded(restoration.richardson_lucy, image, psf, num_iter=num_iter), 0, 1)


# Default regularisation chosen by a parameter sweep at noise sigma = 0.01
# (see README "Tuning"). Stronger noise -> raise alpha/balance, lower num_iter.
METHODS = {
    "tikhonov": lambda g, h: tikhonov(g, h, alpha=0.02),
    "wiener": lambda g, h: wiener(g, h, balance=0.05),
    "unsupervised_wiener": unsupervised_wiener,
    "richardson_lucy": lambda g, h: richardson_lucy(g, h, num_iter=20),
}
