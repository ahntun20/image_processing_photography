"""Command-line interface.

    python -m optics_imaging demo                      # run every experiment -> results/
    python -m optics_imaging deblur  -i photo.jpg --psf gaussian --sigma 1.5 --method richardson_lucy
    python -m optics_imaging denoise -i photo.jpg --method nl_means
    python -m optics_imaging wb      -i photo.jpg --method white_patch
    python -m optics_imaging match   -i photo.jpg -r look.jpg
    python -m optics_imaging psf     --psf airy --fnum 11 --pitch 1.2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import color, deblur, denoise, enhance, experiments
from . import psf as P
from .io_utils import load_image, save_grid, save_image


def _add_psf_args(p):
    g = p.add_argument_group("PSF (blur model)")
    g.add_argument("--psf", choices=list(P.PSF_BUILDERS), default="gaussian")
    g.add_argument("--sigma", type=float, default=1.5, help="gaussian sigma (px)")
    g.add_argument("--size", type=int, default=5, help="box size (px)")
    g.add_argument("--radius", type=float, default=3, help="defocus radius (px)")
    g.add_argument("--length", type=float, default=9, help="motion length (px)")
    g.add_argument("--angle", type=float, default=0, help="motion angle (deg)")
    g.add_argument("--wavelength", type=float, default=0.55, help="airy wavelength (um)")
    g.add_argument("--fnum", type=float, default=8, help="airy f-number")
    g.add_argument("--pitch", type=float, default=1.5, help="pixel pitch (um)")


def _out(args, suffix):
    return Path(args.output) if args.output else Path("results") / f"{Path(args.input).stem}_{suffix}.png"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="optics_imaging", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run all experiments and write figures + metrics")
    d.add_argument("--out", default="results")
    d.add_argument("--quick", action="store_true", help="skip the slow noise sweep")

    sub.add_parser("sample-data", help="write test images into data/")

    b = sub.add_parser("deblur", help="deconvolve your own image with a chosen PSF")
    b.add_argument("-i", "--input", required=True)
    b.add_argument("-o", "--output")
    b.add_argument("--method", choices=list(deblur.METHODS), default="richardson_lucy")
    b.add_argument("--simulate", action="store_true",
                   help="treat input as sharp: blur + add noise first, then restore")
    _add_psf_args(b)

    n = sub.add_parser("denoise")
    n.add_argument("-i", "--input", required=True)
    n.add_argument("-o", "--output")
    n.add_argument("--method", choices=list(denoise.METHODS), default="nl_means")

    w = sub.add_parser("wb", help="white balance / colour constancy")
    w.add_argument("-i", "--input", required=True)
    w.add_argument("-o", "--output")
    w.add_argument("--method", choices=list(color.ESTIMATORS), default="white_patch")

    m = sub.add_parser("match", help="histogram-match input to a reference image")
    m.add_argument("-i", "--input", required=True)
    m.add_argument("-r", "--reference", required=True)
    m.add_argument("-o", "--output")

    p = sub.add_parser("psf", help="plot a PSF and print its MTF50")
    p.add_argument("-o", "--output", default="results/psf.png")
    _add_psf_args(p)

    a = ap.parse_args(argv)

    if a.cmd == "demo":
        experiments.run_all(a.out, quick=a.quick)
    elif a.cmd == "sample-data":
        print("wrote", experiments.make_sample_data())
    elif a.cmd == "deblur":
        img = load_image(a.input)
        h = P.PSF_BUILDERS[a.psf](a)
        src = deblur.add_gaussian_noise(deblur.blur(img, h), 0.01) if a.simulate else img
        res = deblur.METHODS[a.method](src, h)
        save_image(_out(a, a.method), res)
        save_grid(_out(a, a.method + "_compare"), [src, res], ["Input", f"Restored ({a.method})"])
        print("saved", _out(a, a.method))
    elif a.cmd == "denoise":
        img = load_image(a.input)
        res = denoise.METHODS[a.method](img)
        save_image(_out(a, a.method), res)
        print("saved", _out(a, a.method))
    elif a.cmd == "wb":
        img = experiments.srgb_decode(load_image(a.input))
        res, est = color.white_balance(img, a.method)
        save_image(_out(a, "wb_" + a.method), experiments.srgb_encode(res))
        print("estimated illuminant (linear RGB):", np.round(est / est.max(), 3))
        print("saved", _out(a, "wb_" + a.method))
    elif a.cmd == "match":
        res = enhance.histogram_match(load_image(a.input), load_image(a.reference))
        save_image(_out(a, "matched"), res)
        print("saved", _out(a, "matched"))
    elif a.cmd == "psf":
        h = P.PSF_BUILDERS[a.psf](a)
        f, mtf = P.mtf_from_psf(h, a.pitch)
        save_grid(a.output, [h / h.max()], [f"{a.psf} PSF ({h.shape[0]}x{h.shape[1]})"], cmap="magma")
        print(f"MTF50 = {P.mtf50(f, mtf):.1f} cycles/mm  (Nyquist = {1000 / (2 * a.pitch):.1f})")
        print("saved", a.output)


if __name__ == "__main__":
    main()
