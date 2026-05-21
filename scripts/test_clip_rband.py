#!/usr/bin/env python3
"""Standalone sanity check for r-band sigma-clipping (V0.5).

Pick one object from an images directory, run the sigma-clip detection on its
rSDSS cutout, write a clipped FITS next to the original, and produce a
3-panel diagnostic PNG:

    [ raw rSDSS cutout ]  [ detection mask ]  [ clipped cutout (NaN outside) ]

This is the fastest way to eyeball whether the threshold / dilation give a
sensible galaxy footprint before running ``palfitology make-cutouts`` over
the whole catalog.

Usage
-----
    python scripts/test_clip_rband.py \
        --image-dir ~/PALFITology/images/<object-id> \
        --sigma 3.0 \
        --dilate 0 \
        --out test_clip_rband.png

If --image-dir is omitted the script auto-picks the first object found under
``./images/``.

Run from the repository root (so 'palfitology' is importable in editable mode):
    pip install -e .
    python scripts/test_clip_rband.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402

from palfitology.cutouts import derive_clipped_dir, write_clipped_fits  # noqa: E402
from palfitology.detect import (  # noqa: E402
    DEFAULT_CLIP_DILATE,
    DEFAULT_DETECT_SIGMA,
    make_clipped_cutout,
)
from palfitology.images import locate_band_fits  # noqa: E402

logger = logging.getLogger("test_clip_rband")


def _auto_pick_object(images_root: Path) -> Path:
    """Return the first object directory that contains an rSDSS cutout."""
    if not images_root.is_dir():
        raise SystemExit(f"images_root does not exist: {images_root}")
    for sub in sorted(images_root.iterdir()):
        if not sub.is_dir():
            continue
        if locate_band_fits(sub, "rSDSS") is not None:
            return sub
    raise SystemExit(f"No subdirectory with an rSDSS cutout under {images_root}")


def _percentile_norm(arr: np.ndarray, lo: float = 1.0, hi: float = 99.0):
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return -1.0, 1.0
    return np.percentile(finite, lo), np.percentile(finite, hi)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--image-dir", type=Path, default=None,
                        help="One object's directory (parent of fits_images_*/).")
    parser.add_argument("--images-root", type=Path, default=Path("images"),
                        help="Used only when --image-dir is omitted (default: ./images).")
    parser.add_argument("--band", type=str, default="rSDSS",
                        help="Detection band (default: rSDSS).")
    parser.add_argument("--sigma", type=float, default=DEFAULT_DETECT_SIGMA,
                        help=f"Sigma threshold (default: {DEFAULT_DETECT_SIGMA}).")
    parser.add_argument("--dilate", type=int, default=DEFAULT_CLIP_DILATE,
                        help=f"Mask dilation in pixels (default: {DEFAULT_CLIP_DILATE}).")
    parser.add_argument("--write-fits", action="store_true",
                        help="Also write the clipped FITS to the sibling folder.")
    parser.add_argument("--out", type=Path, default=Path("test_clip_rband.png"),
                        help="Where to save the diagnostic PNG.")
    args = parser.parse_args(argv)

    image_dir = args.image_dir or _auto_pick_object(args.images_root)
    logger.info(f"Using object directory: {image_dir}")

    band_fits = locate_band_fits(image_dir, args.band)
    if band_fits is None:
        logger.error(f"No {args.band}_cutout.fits in {image_dir}")
        return 1
    logger.info(f"Loading {band_fits}")

    with fits.open(band_fits) as hdul:
        data = hdul[0].data.astype(float)

    clipped, mask, det = make_clipped_cutout(
        data,
        sigma_threshold=args.sigma,
        dilate=args.dilate,
    )

    logger.info(
        f"detection: status={det.status} npix={det.npix} "
        f"centre=({det.x0:.1f},{det.y0:.1f}) pa={det.pa_deg:.1f} eps={det.eps:.3f} "
        f"bg={det.background:.3g} rms={det.background_rms:.3g}"
    )

    if args.write_fits and det.status == "ok":
        out_dir = derive_clipped_dir(band_fits.parent)
        out_fits = out_dir / f"{args.band}_cutout.fits"
        write_clipped_fits(
            clipped=clipped,
            source_fits=band_fits,
            out_path=out_fits,
            detection=det,
            dilate=args.dilate,
        )
        logger.info(f"Wrote clipped FITS -> {out_fits}")

    # ------------------------------------------------------------------
    # Diagnostic PNG
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    vmin, vmax = _percentile_norm(data, 1.0, 99.0)

    axes[0].imshow(data, origin="lower", vmin=vmin, vmax=vmax, cmap="gray")
    axes[0].set_title(f"Raw {args.band}\n{image_dir.name}")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    axes[1].imshow(data, origin="lower", vmin=vmin, vmax=vmax, cmap="gray")
    axes[1].contour(mask.astype(int), levels=[0.5], colors=["lime"], linewidths=1.5)
    if det.status == "ok":
        axes[1].plot(det.x0, det.y0, "r+", ms=12, mew=2)
    axes[1].set_title(
        f"Detection mask (sigma={args.sigma}, dilate={args.dilate})\n"
        f"status={det.status} npix={det.npix}"
    )
    axes[1].set_xticks([]); axes[1].set_yticks([])

    # Clipped panel uses the same stretch and a colourmap that shows NaN clearly.
    cmap = plt.cm.gray.copy()
    cmap.set_bad("crimson")  # NaN pixels stand out in red
    axes[2].imshow(clipped, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
    axes[2].set_title("Clipped cutout (NaN = red)")
    axes[2].set_xticks([]); axes[2].set_yticks([])

    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    plt.close(fig)
    logger.info(f"Diagnostic PNG -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
