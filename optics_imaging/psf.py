"""Point spread function (PSF) models and MTF analysis.

A PSF describes how an optical system images a single point of light.
Every blur in this project is modelled as: blurred = sharp (*) PSF + noise.
"""
from __future__ import annotations

import numpy as np
from scipy.special import j1


def _odd(n: int) -> int:
    n = int(np.ceil(n))
    return n if n % 2 == 1 else n + 1


def _grid(size: int):
    size = _odd(size)
    r = (size - 1) / 2
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x, y


def normalize(psf: np.ndarray) -> np.ndarray:
    """Scale a PSF so it sums to 1 (conserves energy / image brightness)."""
    psf = np.asarray(psf, dtype=float)
    return psf / psf.sum()


def gaussian_psf(sigma: float = 1.0, size: int | None = None) -> np.ndarray:
    """Gaussian PSF: a common approximation for aberrations + sensor blur."""
    size = size or _odd(6 * sigma + 1)
    x, y = _grid(size)
    return normalize(np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)))


def box_psf(size: int = 5) -> np.ndarray:
    """Uniform square PSF (the kernel the original course scripts used)."""
    return normalize(np.ones((size, size)))


def defocus_psf(radius: float = 3.0, size: int | None = None) -> np.ndarray:
    """Geometric-optics defocus: a uniform disk (the 'circle of confusion')."""
    size = size or _odd(2 * radius + 3)
    x, y = _grid(size)
    # 4x supersampling gives anti-aliased disk edges
    ss = 4
    offs = (np.arange(ss) + 0.5) / ss - 0.5
    acc = np.zeros_like(x, dtype=float)
    for dy in offs:
        for dx in offs:
            acc += ((x + dx) ** 2 + (y + dy) ** 2) <= radius ** 2
    return normalize(acc)


def motion_psf(length: float = 9.0, angle_deg: float = 0.0, size: int | None = None) -> np.ndarray:
    """Linear motion blur (camera shake / moving subject during exposure)."""
    size = size or _odd(length + 2)
    x, y = _grid(size)
    t = np.deg2rad(angle_deg)
    along = x * np.cos(t) + y * np.sin(t)
    across = -x * np.sin(t) + y * np.cos(t)
    psf = (np.abs(along) <= length / 2) * np.clip(1 - np.abs(across), 0, 1)
    return normalize(psf)


def airy_psf(wavelength_um: float = 0.55, f_number: float = 8.0,
             pixel_pitch_um: float = 1.5, size: int | None = None) -> np.ndarray:
    """Diffraction-limited PSF of a circular aperture (Airy pattern), sampled on a sensor.

    I(r) = [2 J1(v) / v]^2,  v = pi * r / (lambda * N)
    First dark ring at r = 1.22 * lambda * N.
    """
    first_zero_px = 1.22 * wavelength_um * f_number / pixel_pitch_um
    size = size or _odd(max(9, 8 * first_zero_px))
    x, y = _grid(size)
    ss = 5  # supersample each pixel to integrate over its area
    offs = (np.arange(ss) + 0.5) / ss - 0.5
    acc = np.zeros_like(x, dtype=float)
    for dy in offs:
        for dx in offs:
            r_um = np.hypot(x + dx, y + dy) * pixel_pitch_um
            v = np.pi * r_um / (wavelength_um * f_number)
            with np.errstate(invalid="ignore", divide="ignore"):
                val = np.where(v == 0, 1.0, (2 * j1(v) / v) ** 2)
            acc += val
    return normalize(acc)


def airy_radius_um(wavelength_um: float, f_number: float) -> float:
    """Radius of the Airy disk (first zero) in micrometres."""
    return 1.22 * wavelength_um * f_number


def diffraction_mtf(freq_cyc_per_mm: np.ndarray, wavelength_um: float, f_number: float) -> np.ndarray:
    """Analytic MTF of an aberration-free circular aperture (incoherent light)."""
    cutoff = 1000.0 / (wavelength_um * f_number)  # cycles/mm
    s = np.clip(np.asarray(freq_cyc_per_mm) / cutoff, 0, 1)
    return (2 / np.pi) * (np.arccos(s) - s * np.sqrt(1 - s ** 2))


def mtf_from_psf(psf: np.ndarray, pixel_pitch_um: float = 1.0, n_fft: int = 256):
    """Radially averaged MTF computed as |FFT(PSF)|.

    Returns (frequency in cycles/mm, MTF). Nyquist = 1000 / (2 * pitch) cycles/mm.
    """
    pad = np.zeros((n_fft, n_fft))
    h, w = psf.shape
    pad[:h, :w] = psf
    pad = np.roll(pad, (-(h // 2), -(w // 2)), axis=(0, 1))
    otf = np.fft.fftshift(np.abs(np.fft.fft2(pad)))
    otf /= otf.max()
    c = n_fft // 2
    y, x = np.indices(otf.shape)
    r = np.hypot(x - c, y - c).astype(int)
    radial = np.bincount(r.ravel(), otf.ravel()) / np.bincount(r.ravel())
    radial = radial[: c + 1]
    freq_cyc_per_px = np.arange(c + 1) / n_fft
    return freq_cyc_per_px * 1000.0 / pixel_pitch_um, radial


def mtf50(freq: np.ndarray, mtf: np.ndarray) -> float:
    """Frequency where MTF first falls to 0.5 (a standard sharpness spec)."""
    below = np.nonzero(mtf < 0.5)[0]
    if len(below) == 0:
        return float("nan")
    i = below[0]
    f0, f1, m0, m1 = freq[i - 1], freq[i], mtf[i - 1], mtf[i]
    return float(f0 + (0.5 - m0) * (f1 - f0) / (m1 - m0))


PSF_BUILDERS = {
    "gaussian": lambda a: gaussian_psf(sigma=a.sigma),
    "box": lambda a: box_psf(size=a.size),
    "defocus": lambda a: defocus_psf(radius=a.radius),
    "motion": lambda a: motion_psf(length=a.length, angle_deg=a.angle),
    "airy": lambda a: airy_psf(a.wavelength, a.fnum, a.pitch),
}
