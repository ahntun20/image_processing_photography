"""Loading, saving and plotting helpers (headless-safe)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures to disk; works on servers without a display
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, transform


def load_image(path, max_side: int | None = 768) -> np.ndarray:
    """Load an image as float RGB in [0, 1], optionally downsized for speed."""
    img = io.imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = img[..., :3]
    img = img.astype(float) / (65535.0 if img.dtype == np.uint16 else 255.0)
    if max_side and max(img.shape[:2]) > max_side:
        s = max_side / max(img.shape[:2])
        img = transform.rescale(img, s, channel_axis=-1, anti_aliasing=True)
    return np.clip(img, 0, 1)


def save_image(path, img):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    io.imsave(str(path), (np.clip(img, 0, 1) * 255).round().astype(np.uint8), check_contrast=False)


def save_grid(path, images, titles, ncols=None, cmap=None, size=4):
    """Save a row/grid of images with titles to one PNG."""
    n = len(images)
    ncols = ncols or n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, im, t in zip(axes.ravel(), images, titles):
        ax.imshow(np.clip(im, 0, 1), cmap=cmap if im.ndim == 2 else None)
        ax.set_title(t, fontsize=10)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
