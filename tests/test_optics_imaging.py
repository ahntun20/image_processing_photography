"""Physics and correctness checks. Run with:  python -m pytest -q"""
import numpy as np
import pytest

from optics_imaging import color, deblur, enhance, psf as P
from optics_imaging.metrics import angular_error_deg, psnr


@pytest.mark.parametrize("h", [P.gaussian_psf(1.5), P.defocus_psf(3), P.motion_psf(9, 30),
                               P.airy_psf(0.55, 8, 1.5), P.box_psf(5)])
def test_psf_energy_conserved_and_centred(h):
    assert h.sum() == pytest.approx(1.0)
    assert h.shape[0] % 2 == 1 and h.shape[1] % 2 == 1
    yy, xx = np.indices(h.shape)
    c = (h.shape[0] - 1) / 2
    assert (h * yy).sum() == pytest.approx(c, abs=1e-6)
    assert (h * xx).sum() == pytest.approx(c, abs=1e-6)


def test_airy_first_zero_radius():
    assert P.airy_radius_um(0.55, 8) == pytest.approx(5.368, rel=1e-3)


def test_sampled_mtf_matches_diffraction_theory_when_well_sampled():
    pitch = 0.5  # finely sampled -> no aliasing
    h = P.airy_psf(0.55, 2.8, pitch, size=201)
    f, m = P.mtf_from_psf(h, pitch, n_fft=512)
    for fq in (100, 200, 333):
        theory = P.diffraction_mtf(fq, 0.55, 2.8) * np.sinc(fq * pitch / 1000)
        assert np.interp(fq, f, m) == pytest.approx(theory, abs=0.01)


def test_tikhonov_has_no_spatial_shift():
    """Regression test for the uncentred-PSF bug in legacy/deblurred_3.py."""
    img = np.zeros((64, 64))
    img[32, 32] = 1.0
    h = P.gaussian_psf(1.5)
    out = deblur.tikhonov(deblur.blur(img, h), h, alpha=1e-3)
    assert np.unravel_index(out.argmax(), out.shape) == (32, 32)


def test_deconvolution_improves_psnr():
    rng = np.random.default_rng(1)
    sharp = np.clip(rng.random((96, 96)) > 0.7, 0, 1).astype(float)
    h = P.defocus_psf(2)
    g = deblur.add_gaussian_noise(deblur.blur(sharp, h), 0.005)
    for name, fn in deblur.METHODS.items():
        if name == "unsupervised_wiener":
            continue
        assert psnr(sharp, fn(g, h)) > psnr(sharp, g), name


def test_white_patch_recovers_illuminant_when_white_present():
    scene = np.full((10, 10, 3), 0.3)
    scene[0, 0] = 1.0  # a white surface
    ill = (1.0, 0.7, 0.4)
    _, est = color.white_balance(color.apply_illuminant(scene, ill), "white_patch")
    assert angular_error_deg(est, ill) < 0.5


def test_histogram_match_copies_distribution():
    rng = np.random.default_rng(0)
    src, ref = rng.random((50, 50, 3)), rng.random((50, 50, 3)) ** 3
    out = enhance.histogram_match(src, ref)
    assert np.allclose(np.sort(out[..., 0].ravel()), np.sort(ref[..., 0].ravel()), atol=0.02)
