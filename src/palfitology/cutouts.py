"""Clipped-cutout I/O — V0.5.

This module handles **on-disk persistence** of sigma-clipped cutouts produced
by :func:`palfitology.detect.make_clipped_cutout`.  The layout mirrors the
existing J-PLUS images directory:

::

    images/<id>/
        fits_images_<ra>_<dec>/<band>_cutout.fits        (original — untouched)
        psfs_<ra>_<dec>/psf_<ra>_<dec>_<band>.fits       (original — untouched)
        clipped_cutouts_<ra>_<dec>/<band>_cutout.fits    (new, written here)

The sibling-folder choice means:

1. The originals are never modified.  A user can always delete the
   ``clipped_cutouts_*`` folder and re-run from scratch.
2. The pipeline can transparently prefer the clipped version when present,
   falling back to the original when absent (see :func:`locate_clipped_or_original`).
3. The ``<ra>_<dec>`` suffix is inherited from the original folder so each
   object is self-contained.

The clipped FITS files preserve the original header (so WCS information,
band metadata, exposure time, etc. are kept) and add a few HISTORY records
documenting which sigma threshold produced the clip.

Functions
---------
- :func:`derive_clipped_dir` — translate ``fits_images_<ra>_<dec>`` ->
  ``clipped_cutouts_<ra>_<dec>``.
- :func:`write_clipped_fits` — write a 2D array as a FITS file copying the
  source header and appending HISTORY entries.
- :func:`locate_clipped_band_fits` — find ``clipped_cutouts_*/<band>_cutout.fits``.
- :func:`locate_clipped_or_original` — prefer the clipped version, fall back
  to the original (used by the fitter when ``--use-clipped-cutouts`` is set).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from astropy.io import fits

from .detect import DEFAULT_CLIP_DILATE, DEFAULT_DETECT_SIGMA, DetectionResult

logger = logging.getLogger(__name__)

__all__ = [
    "derive_clipped_dir",
    "write_clipped_fits",
    "locate_clipped_band_fits",
    "locate_clipped_or_original",
    "make_cutouts_for_catalog",
    "ClipReport",
]


# Sibling-folder prefix.  We keep the trailing ``_<ra>_<dec>`` part of the
# original directory name so each object's clipped cutouts live next to
# their originals.
_FITS_PREFIX = "fits_images_"
_CLIPPED_PREFIX = "clipped_cutouts_"


def derive_clipped_dir(fits_dir: Path) -> Path:
    """Return the sibling ``clipped_cutouts_<ra>_<dec>`` folder for ``fits_dir``.

    ``fits_dir`` is expected to be a path whose final component begins with
    ``fits_images_``.  If it doesn't (e.g. someone re-organised the layout),
    we fall back to a generic ``clipped_cutouts`` folder inside the parent.
    """
    name = fits_dir.name
    if name.startswith(_FITS_PREFIX):
        suffix = name[len(_FITS_PREFIX):]
        return fits_dir.parent / f"{_CLIPPED_PREFIX}{suffix}"
    logger.debug(
        f"{fits_dir} doesn't match the '{_FITS_PREFIX}*' convention -- "
        f"using a plain 'clipped_cutouts' sibling"
    )
    return fits_dir.parent / "clipped_cutouts"


def write_clipped_fits(
    *,
    clipped: np.ndarray,
    source_fits: Path,
    out_path: Path,
    detection: DetectionResult,
    dilate: int = DEFAULT_CLIP_DILATE,
) -> Path:
    """Write a clipped cutout to disk, copying the source FITS header.

    Parameters
    ----------
    clipped : 2D float array
        The masked cutout (NaN outside the detected source).
    source_fits : Path
        The original cutout, whose primary HDU header we copy so WCS,
        exposure, band tags, etc. are preserved.
    out_path : Path
        Where to write the new FITS.  Parent directory will be created.
    detection : DetectionResult
        Used to fill HISTORY records documenting the threshold used.
    dilate : int
        Mask-dilation radius (also recorded in HISTORY for reproducibility).

    Returns
    -------
    out_path
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy header from the source — fail gracefully if the original is gone
    # by writing with a minimal header instead.
    header: Optional[fits.Header]
    try:
        with fits.open(source_fits) as hdul:
            header = hdul[0].header.copy()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            f"Could not read source header from {source_fits}: {e} -- "
            f"writing clipped FITS with a fresh header"
        )
        header = None

    # Add HISTORY records.  These survive in the file forever and let a future
    # reader know exactly which detection params produced this clipped image.
    if header is None:
        header = fits.Header()
    header["HISTORY"] = "palfitology: sigma-clipped cutout"
    header["HISTORY"] = (
        f"  sigma_threshold = {detection.sigma_threshold:.3f}"
    )
    header["HISTORY"] = (
        f"  background      = {detection.background:.6g}"
    )
    header["HISTORY"] = (
        f"  background_rms  = {detection.background_rms:.6g}"
    )
    header["HISTORY"] = (
        f"  detect_npix     = {detection.npix}"
    )
    header["HISTORY"] = (
        f"  dilate_pixels   = {int(dilate)}"
    )
    header["HISTORY"] = (
        f"  source_centre   = ({detection.x0:.2f}, {detection.y0:.2f})"
    )

    # Use .astype(np.float32) so the clipped file is roughly the same size as
    # the input (J-PLUS cutouts are 32-bit floats).  np.float32 preserves NaN.
    hdu = fits.PrimaryHDU(data=clipped.astype(np.float32), header=header)
    hdu.writeto(out_path, overwrite=True)
    logger.debug(
        f"Wrote clipped cutout to {out_path} (npix_kept={int(np.isfinite(clipped).sum())}, "
        f"sigma={detection.sigma_threshold:.2f})"
    )
    return out_path


def locate_clipped_band_fits(image_dir: Path, band: str) -> Optional[Path]:
    """Return the path to the clipped cutout for one band, or None.

    Looks for ``image_dir / clipped_cutouts_*/<band>_cutout.fits``.
    """
    return next(image_dir.glob(f"{_CLIPPED_PREFIX}*/{band}_cutout.fits"), None)


def locate_clipped_or_original(
    image_dir: Path,
    band: str,
) -> Tuple[Optional[Path], str]:
    """Prefer the clipped cutout if present, otherwise return the original.

    Returns
    -------
    (path, source)
        ``source`` is either ``'clipped'`` or ``'original'`` so callers can
        record where the data came from.  Returns ``(None, 'missing')`` when
        neither exists.
    """
    clipped = locate_clipped_band_fits(image_dir, band)
    if clipped is not None:
        return clipped, "clipped"

    # Late import to avoid a circular import at module-load time.
    from .images import locate_band_fits

    original = locate_band_fits(image_dir, band)
    if original is not None:
        return original, "original"
    return None, "missing"


# ---------------------------------------------------------------------------
# High-level driver for `palfitology make-cutouts`
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402  -- bottom-grouped helpers
from typing import Iterable, List  # noqa: E402

import pandas as pd  # noqa: E402

from .detect import make_clipped_cutout  # noqa: E402
from .images import locate_band_fits  # noqa: E402


@dataclass
class ClipReport:
    """Per-object summary written to make-cutouts' CSV log."""
    id: str
    band: str
    status: str  # 'ok', 'no_detection', 'missing_input', 'failed'
    sigma_threshold: float
    npix_kept: int
    detect_x0: float
    detect_y0: float
    out_path: str


def make_cutouts_for_catalog(
    *,
    images_root: Path,
    catalog: pd.DataFrame,
    detect_band: str,
    apply_bands: Iterable[str],
    sigma_threshold: float = DEFAULT_DETECT_SIGMA,
    dilate: int = DEFAULT_CLIP_DILATE,
    overwrite: bool = True,
) -> List[ClipReport]:
    """Generate clipped FITS cutouts for every object in ``catalog``.

    The detection mask is **always** built from ``detect_band`` (typically
    rSDSS).  That same mask is then applied to each band listed in
    ``apply_bands`` so the same pixels survive across every wavelength —
    which is exactly what the cross-band PA fit needs.

    Parameters
    ----------
    images_root : Path
        Root directory containing one subfolder per object.
    catalog : DataFrame
        Must have an ``'id'`` column.  Each id corresponds to a subfolder
        under ``images_root``.
    detect_band : str
        Band used to *build* the mask (e.g. 'rSDSS').
    apply_bands : iterable of str
        Bands the mask is *applied* to.  Pass ``[detect_band]`` for the
        first sanity check; pass the full 12-band list to make a complete
        clipped dataset.
    sigma_threshold : float
        Sigma threshold for the detection (default 3.0).
    dilate : int
        Mask-dilation radius before applying (default 0).
    overwrite : bool
        If False, skip objects whose clipped FITS already exists.  Default
        True so re-running with a different threshold replaces stale data.

    Returns
    -------
    List of :class:`ClipReport`, one entry per (object, band) processed.
    """
    apply_bands = list(apply_bands)
    reports: List[ClipReport] = []

    for _, row in catalog.iterrows():
        objectid = str(row["id"])
        image_dir = images_root / objectid
        if not image_dir.is_dir():
            logger.debug(f"[{objectid}] object directory missing -- skipping")
            for band in apply_bands:
                reports.append(ClipReport(
                    id=objectid, band=band, status="missing_input",
                    sigma_threshold=sigma_threshold, npix_kept=0,
                    detect_x0=float("nan"), detect_y0=float("nan"),
                    out_path="",
                ))
            continue

        # --------------------------------------------------------------
        # Build the mask once from the detect band.
        # --------------------------------------------------------------
        detect_path = locate_band_fits(image_dir, detect_band)
        if detect_path is None:
            logger.warning(
                f"[{objectid}] detect-band '{detect_band}' cutout missing -- "
                f"cannot build clipped cutouts for this object"
            )
            for band in apply_bands:
                reports.append(ClipReport(
                    id=objectid, band=band, status="missing_input",
                    sigma_threshold=sigma_threshold, npix_kept=0,
                    detect_x0=float("nan"), detect_y0=float("nan"),
                    out_path="",
                ))
            continue

        try:
            with fits.open(detect_path) as hdul:
                detect_data = hdul[0].data.astype(float)
        except Exception as e:  # noqa: BLE001
            logger.error(f"[{objectid}] failed to read {detect_path}: {e}")
            for band in apply_bands:
                reports.append(ClipReport(
                    id=objectid, band=band, status="failed",
                    sigma_threshold=sigma_threshold, npix_kept=0,
                    detect_x0=float("nan"), detect_y0=float("nan"),
                    out_path="",
                ))
            continue

        _, mask, det = make_clipped_cutout(
            detect_data,
            sigma_threshold=sigma_threshold,
            dilate=dilate,
        )

        if det.status != "ok":
            logger.info(
                f"[{objectid}] detection returned '{det.status}' "
                f"(sigma={sigma_threshold}) -- skipping clipped cutout write"
            )
            for band in apply_bands:
                reports.append(ClipReport(
                    id=objectid, band=band, status=det.status,
                    sigma_threshold=sigma_threshold, npix_kept=0,
                    detect_x0=det.x0, detect_y0=det.y0,
                    out_path="",
                ))
            continue

        clipped_dir = derive_clipped_dir(detect_path.parent)

        # --------------------------------------------------------------
        # Apply the mask to each requested band.  Bands missing on disk
        # are logged but don't fail the whole object.
        # --------------------------------------------------------------
        for band in apply_bands:
            band_path = locate_band_fits(image_dir, band)
            if band_path is None:
                logger.debug(
                    f"[{objectid}/{band}] original cutout missing -- "
                    f"cannot apply mask for this band"
                )
                reports.append(ClipReport(
                    id=objectid, band=band, status="missing_input",
                    sigma_threshold=sigma_threshold, npix_kept=0,
                    detect_x0=det.x0, detect_y0=det.y0,
                    out_path="",
                ))
                continue

            out_path = clipped_dir / f"{band}_cutout.fits"
            if out_path.exists() and not overwrite:
                logger.debug(f"[{objectid}/{band}] exists, skipping (overwrite=False)")
                reports.append(ClipReport(
                    id=objectid, band=band, status="ok",
                    sigma_threshold=sigma_threshold,
                    npix_kept=int(mask.sum()),
                    detect_x0=det.x0, detect_y0=det.y0,
                    out_path=str(out_path),
                ))
                continue

            try:
                with fits.open(band_path) as hdul:
                    band_data = hdul[0].data.astype(float)
            except Exception as e:  # noqa: BLE001
                logger.error(f"[{objectid}/{band}] failed to read {band_path}: {e}")
                reports.append(ClipReport(
                    id=objectid, band=band, status="failed",
                    sigma_threshold=sigma_threshold, npix_kept=0,
                    detect_x0=det.x0, detect_y0=det.y0,
                    out_path="",
                ))
                continue

            if band_data.shape != mask.shape:
                logger.warning(
                    f"[{objectid}/{band}] cutout shape {band_data.shape} != "
                    f"detect-band shape {mask.shape} -- skipping"
                )
                reports.append(ClipReport(
                    id=objectid, band=band, status="failed",
                    sigma_threshold=sigma_threshold, npix_kept=0,
                    detect_x0=det.x0, detect_y0=det.y0,
                    out_path="",
                ))
                continue

            band_clipped = np.where(mask, band_data, float("nan"))
            write_clipped_fits(
                clipped=band_clipped,
                source_fits=band_path,
                out_path=out_path,
                detection=det,
                dilate=dilate,
            )
            reports.append(ClipReport(
                id=objectid, band=band, status="ok",
                sigma_threshold=sigma_threshold,
                npix_kept=int(mask.sum()),
                detect_x0=det.x0, detect_y0=det.y0,
                out_path=str(out_path),
            ))

    return reports
