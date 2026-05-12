"""FITS cutout location and loading helpers.

The standard J-PLUS cutout layout is::

    images/<id>/
        fits_images_<ra>_<dec>/
            <band>_cutout.fits
            ...
        psfs_<ra>_<dec>/
            psf_<ra>_<dec>_<band>.fits
            ...
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from astropy.io import fits

logger = logging.getLogger(__name__)


def locate_band_fits(image_dir: Path, band: str) -> Optional[Path]:
    """Return the path to the FITS cutout for one band, or None if missing."""
    return next(image_dir.glob(f"fits_images_*/{band}_cutout.fits"), None)


def locate_band_psf(image_dir: Path, band: str) -> Optional[Path]:
    """Return the path to the PSF FITS for one band, or None if missing."""
    return next(image_dir.glob(f"psfs_*/psf_*_{band}.fits"), None)


def open_fits_array(path: Path) -> Optional[np.ndarray]:
    """Open a FITS file and return its primary HDU data as a float array."""
    try:
        with fits.open(path) as hdul:
            return hdul[0].data.astype(float)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to open {path}: {e}")
        return None


def locate_band_with_fallback(
    image_dir: Path,
    band: str,
    fallback_band: Optional[str] = None,
) -> Tuple[Optional[Path], Optional[str]]:
    """Try to find ``band``, else fall back to ``fallback_band``.

    Returns (path, band_used) -- band_used reflects which band actually had a
    cutout. Returns (None, None) if neither was found.
    """
    primary = locate_band_fits(image_dir, band)
    if primary:
        return primary, band
    if fallback_band:
        fb = locate_band_fits(image_dir, fallback_band)
        if fb:
            logger.warning(
                f"Falling back from {band} to {fallback_band} for {image_dir.name}"
            )
            return fb, fallback_band
    return None, None
