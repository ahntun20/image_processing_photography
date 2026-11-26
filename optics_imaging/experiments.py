"""Reproducible experiments. Each writes figures + a metrics table to the output folder."""
from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import data

from . import color, deblur, denoise, enhance, psf as P
from .io_utils import load_image, save_grid, save_image
from .metrics import angular_error_deg, psnr, ssim

DATA = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------- sample data
def make_sample_data(folder=DATA):
    """Write test images (from scikit-image's bundled set) + a synthetic colour chart."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    save_image(folder / "sharp.png", data.astronaut() / 255.0)
    save_image(folder / "reference_style.png", data.coffee() / 255.0)
    save_image(folder / "cat.png", data.chelsea() / 255.0)
    save_image(folder / "color_chart.png", srgb_encode(color_chart_linear()))
    return folder


# Approximate sRGB values of the 24-patch Macbeth/X-Rite ColorChecker
_CHECKER = [
    (115, 82, 68), (194, 150, 130), (98, 122, 157), (87, 108, 67), (133, 128, 177), (103, 189, 170),
    (214, 126, 44), (80, 91, 166), (193, 90, 99), (94, 60, 108), (157, 188, 64), (224, 163, 46),
    (56, 61, 150), (70, 148, 73), (175, 54, 60), (231, 199, 31), (187, 86, 149), (8, 133, 161),
    (243, 243, 242), (200, 200, 200), (160, 160, 160), (122, 122, 121), (85, 85, 85), (52, 52, 52),
]


def srgb_decode(x):
    x = np.asarray(x, float)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def srgb_encode(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def color_chart_linear(patch=60, gap=8):
    """Linear-RGB image of a 6x4 colour chart on a mid-grey background."""
    h = 4 * patch + 5 * gap
    w = 6 * patch + 7 * gap
    img = np.full((h, w, 3), srgb_decode(0.35))
    for i, rgb in enumerate(_CHECKER):
        r, c = divmod(i, 6)
        y, x = gap + r * (patch + gap), gap + c * (patch + gap)
        img[y:y + patch, x:x + patch] = srgb_decode(np.array(rgb) / 255.0)
    return img


def _sharp():
    p = DATA / "sharp.png"
    if not p.exists():
        make_sample_data()
    return load_image(p)


def _write_table(path, rows):
    path = Path(path)
    with open(path.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    keys = list(rows[0])
    md = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in rows:
        md.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r.values()) + " |")
    path.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


# ------------------------------------------------------------ 1. PSF and MTF
def exp_psf_mtf(out: Path, wavelength=0.55, pitch=1.5, f_numbers=(2.8, 8, 16)):
    """Diffraction-limited lens: Airy PSF vs f-number and the resulting MTF."""
    nyq = 1000 / (2 * pitch)
    fig, axes = plt.subplots(1, len(f_numbers) + 1, figsize=(4.2 * (len(f_numbers) + 1), 4))
    rows = []
    for ax, N in zip(axes, f_numbers):
        h = P.airy_psf(wavelength, N, pitch, size=31)
        ax.imshow(h ** 0.5, cmap="inferno")
        d_px = 2 * P.airy_radius_um(wavelength, N) / pitch
        ax.set_title(f"Airy PSF  f/{N}\nAiry disk = {d_px:.1f} px", fontsize=10)
        ax.axis("off")
        f, m = P.mtf_from_psf(h, pitch)
        line, = axes[-1].plot(f, m, label=f"f/{N} sampled")
        theory = P.diffraction_mtf(f, wavelength, N) * np.sinc(f * pitch / 1000)  # optics x pixel aperture
        axes[-1].plot(f, theory, "--", color=line.get_color(), alpha=0.6, label=f"f/{N} theory")
        rows.append({
            "f_number": N,
            "airy_diameter_um": 2 * P.airy_radius_um(wavelength, N),
            "airy_diameter_px": d_px,
            "cutoff_cyc_per_mm": 1000 / (wavelength * N),
            "mtf50_cyc_per_mm": P.mtf50(f, m),
            "mtf_at_nyquist_sampled": float(np.interp(nyq, f, m)),
            "mtf_at_nyquist_theory": float(P.diffraction_mtf(nyq, wavelength, N) * np.sinc(0.5)),
        })
    ax = axes[-1]
    ax.axvline(nyq, color="k", lw=1, ls=":")
    ax.text(nyq, 0.5, "Nyquist ", fontsize=9, ha="right")
    ax.set(xlabel="spatial frequency (cycles/mm)", ylabel="MTF", ylim=(0, 1.02), xlim=(0, nyq * 1.1),
           title=f"MTF  (λ={wavelength} µm, pitch={pitch} µm)")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "01_psf_mtf.png", dpi=110)
    plt.close(fig)

    # Simulate the photo through each aperture
    sharp = _sharp()
    crop = sharp[60:316, 140:396]
    sims = [crop] + [deblur.blur(crop, P.airy_psf(wavelength, N, pitch)) for N in f_numbers]
    save_grid(out / "01_diffraction_sim.png", sims,
              ["Ideal (no diffraction)"] + [f"Diffraction at f/{N}" for N in f_numbers])
    _write_table(out / "01_psf_mtf", rows)


def exp_psf_gallery(out: Path, pitch=1.5):
    """Common blur PSFs and their MTFs side by side."""
    psfs = {
        "Gaussian σ=1.5": P.gaussian_psf(1.5, size=21),
        "Defocus r=4": P.defocus_psf(4, size=21),
        "Motion 11px @30°": P.motion_psf(11, 30, size=21),
        "Airy f/16": P.airy_psf(0.55, 16, pitch, size=21),
    }
    fig, axes = plt.subplots(1, len(psfs) + 1, figsize=(4 * (len(psfs) + 1), 4))
    for ax, (name, h) in zip(axes, psfs.items()):
        ax.imshow(h, cmap="magma")
        ax.set_title(name, fontsize=10)
        ax.axis("off")
        f, m = P.mtf_from_psf(h, pitch)
        axes[-1].plot(f, m, label=name)
    axes[-1].set(xlabel="cycles/mm", ylabel="MTF (radial avg)", ylim=(0, 1.02), xlim=(0, 1000 / (2 * pitch)),
                 title="MTF comparison")
    axes[-1].legend(fontsize=8)
    axes[-1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "02_psf_gallery.png", dpi=110)
    plt.close(fig)


# --------------------------------------------------------- 2. Deconvolution
def exp_deblur(out: Path, noise_sigma=0.01):
    sharp = _sharp()
    cases = {"defocus_r3": P.defocus_psf(3), "motion_11px": P.motion_psf(11, 20)}
    rows = []
    for case, h in cases.items():
        g = deblur.add_gaussian_noise(deblur.blur(sharp, h), noise_sigma)
        rows.append({"blur": case, "method": "(blurred input)", "psnr_db": psnr(sharp, g),
                     "ssim": ssim(sharp, g), "seconds": 0.0})
        imgs, titles = [sharp, g], ["Ground truth", f"Blurred + noise\n{psnr(sharp, g):.2f} dB"]
        for name, fn in deblur.METHODS.items():
            t = time.perf_counter()
            r = fn(g, h)
            dt = time.perf_counter() - t
            rows.append({"blur": case, "method": name, "psnr_db": psnr(sharp, r), "ssim": ssim(sharp, r),
                         "seconds": dt})
            imgs.append(r)
            titles.append(f"{name}\n{psnr(sharp, r):.2f} dB / SSIM {ssim(sharp, r):.3f}")
        crop = (slice(40, 296), slice(150, 406))
        save_grid(out / f"03_deblur_{case}.png", [im[crop] for im in imgs], titles, ncols=3)
    _write_table(out / "03_deblur", rows)


def exp_deblur_noise_sweep(out: Path):
    """How each method degrades as sensor noise rises — the core trade-off of deconvolution."""
    sharp = _sharp()[100:356, 100:356]
    h = P.defocus_psf(3)
    sigmas = [0.0025, 0.005, 0.01, 0.02, 0.04]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    res = {"blurred": []} | {k: [] for k in deblur.METHODS}
    for s in sigmas:
        g = deblur.add_gaussian_noise(deblur.blur(sharp, h), s)
        res["blurred"].append(psnr(sharp, g))
        for k, fn in deblur.METHODS.items():
            res[k].append(psnr(sharp, fn(g, h)))
    for k, v in res.items():
        ax.plot(sigmas, v, marker="o", label=k, ls="--" if k == "blurred" else "-")
    ax.set(xscale="log", xlabel="noise σ (fraction of full scale)", ylabel="PSNR (dB)",
           title="Deconvolution vs noise level (defocus r=3)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "04_deblur_noise_sweep.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- 3. Denoise
def exp_denoise(out: Path):
    sharp = _sharp()
    rows = []
    crop = (slice(40, 296), slice(150, 406))
    for noise, kw in [("gaussian", {"var": 0.01}), ("s&p", {"amount": 0.05})]:
        g = denoise.add_noise(sharp, noise, **kw)
        rows.append({"noise": noise, "method": "(noisy input)", "psnr_db": psnr(sharp, g), "ssim": ssim(sharp, g)})
        imgs, titles = [g], [f"Noisy ({noise})\n{psnr(sharp, g):.2f} dB"]
        for name, fn in denoise.METHODS.items():
            r = fn(g)
            rows.append({"noise": noise, "method": name, "psnr_db": psnr(sharp, r), "ssim": ssim(sharp, r)})
            imgs.append(r)
            titles.append(f"{name}\n{psnr(sharp, r):.2f} dB / SSIM {ssim(sharp, r):.3f}")
        tag = "gaussian" if noise == "gaussian" else "saltpepper"
        save_grid(out / f"05_denoise_{tag}.png", [im[crop] for im in imgs], titles, ncols=5)
    _write_table(out / "05_denoise", rows)


# ---------------------------------------------------------- 4. White balance
ILLUMINANTS = {  # linear RGB of the light source (relative)
    "tungsten_2850K": (1.0, 0.70, 0.38),
    "shade_7500K": (0.80, 0.90, 1.0),
    "green_fluorescent": (0.85, 1.0, 0.75),
}


def exp_white_balance(out: Path):
    chart = color_chart_linear()
    photo = srgb_decode(load_image(DATA / "cat.png" if (DATA / "cat.png").exists() else make_sample_data() / "cat.png"))
    rows = []
    for ill_name, ill in ILLUMINANTS.items():
        cast = color.apply_illuminant(chart, ill)
        for m in color.ESTIMATORS:
            _, est = color.white_balance(cast, m)
            rows.append({"illuminant": ill_name, "method": m, "angular_error_deg": angular_error_deg(est, ill)})
    # visuals: chart + photo under tungsten
    ill = ILLUMINANTS["tungsten_2850K"]
    for tag, scene in [("chart", chart), ("photo", photo)]:
        cast = color.apply_illuminant(scene, ill)
        imgs, titles = [scene, cast], ["Original", "Under tungsten light"]
        for m in color.ESTIMATORS:
            r, est = color.white_balance(cast, m)
            imgs.append(r)
            titles.append(f"{m}\nerror {angular_error_deg(est, ill):.1f}°" if tag == "chart" else m)
        save_grid(out / f"06_white_balance_{tag}.png", [srgb_encode(i) for i in imgs], titles, ncols=5)
    _write_table(out / "06_white_balance", rows)


# --------------------------------------------------- 5. Tone / enhancement
def exp_enhance(out: Path):
    content = _sharp()
    style = load_image(DATA / "reference_style.png")
    matched = enhance.histogram_match(content, style)
    save_grid(out / "07_histogram_match.png", [content, style, matched],
              ["Content", "Reference look", "Histogram matched"])
    enh, mask = enhance.target_and_enhance(content, threshold=0.75, gain=1.6)
    save_grid(out / "08_targeted_enhance.png", [content, mask.astype(float), enh],
              ["Original", "Target mask (luminance > 0.75)", "Contrast-enhanced target"])


ALL = [exp_psf_mtf, exp_psf_gallery, exp_deblur, exp_deblur_noise_sweep, exp_denoise, exp_white_balance, exp_enhance]


def run_all(out="results", quick=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    make_sample_data()
    exps = [e for e in ALL if not (quick and e is exp_deblur_noise_sweep)]
    for e in exps:
        print(f"\n== {e.__name__}: {e.__doc__.strip().splitlines()[0] if e.__doc__ else ''}")
        t = time.perf_counter()
        e(out)
        print(f"   done in {time.perf_counter() - t:.1f}s")
    print(f"\nAll results written to {out.resolve()}")
