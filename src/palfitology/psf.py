"""PSF loading, FWHM estimation, and Wiener deconvolution.

V0.2 introduces a conservative PSF-aware preprocessing step that runs *before*
the isophotal fit. The motivating observation: for J-PLUS narrowband cutouts,
when the seeing-disk FWHM is a non-negligible fraction of the galaxy size, the
isophote position angle gets biased toward zero ellipticity (the PSF
circularizes the source). Deconvolving the cutout sharpens the source enough
that the existing `fit_pa_with_fallbacks` returns a better PA prior for GALFIT
to refine downstream.

The approach is intentionally conservative:

1. Load the per-band PSF cutout from ``psfs_*/psf_*_<band>.fits``.
2. Fit a circular 2D Gaussian to estimate the PSF FWHM in pixels.
3. Compute the ratio ``psf_fwhm / r_eff_pixels``. If this ratio is below the
   gate threshold (default 0.2), the PSF effect is negligible -- skip
   deconvolution and fit the raw cutout.
4. Otherwise run a Wiener deconvolution with a noise term tied to the
   background RMS of the image. The deconvolved image is then handed to the
   normal fit pipeline.
5. The pipeline caller is expected to retry on the raw cutout if the
   deconvolved fit fails; this module only provides the preprocessing.

GALFIT (the next stage of the project) does proper convolved-model fitting,
so palfitology's job is just to give it good *priors*. Wiener deconvolution
is the minimum effort that meaningfully helps without reinventing GALFIT.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "PSFInfo",
    "DEFAULT_PSF_GATE",
    "estimate_psf_fwhm",
    "wiener_deconvolve",
    "preprocess_for_fit",
]


# Sigma -> FWHM for a Gaussian: FWHM = 2 * sqrt(2 * ln 2) * sigma.
_FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))

# Conservative default: if PSF FWHM is less than 20% of the galaxy effective
# radius, the PSF barely smears the source -- skip deconvolution and avoid the
# noise amplification it would introduce.
DEFAULT_PSF_GATE = 0.2


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

PsfMode = Literal["raw", "deconv", "missing_psf", "off"]


@dataclass(frozen=True)
class PSFInfo:
    """What `preprocess_for_fit` returns to the pipeline.

    Attributes
    ----------
    image_for_fit:
        The 2D array the pipeline should pass to ``fit_pa_with_fallbacks``.
        Either the deconvolved image or the original raw cutout.
    mode:
        How the image was prepared. One of:
        - ``"raw"``   -- no deconvolution (gate passed, or psf-mode=off).
        - ``"deconv"`` -- Wiener-deconvolved.
        - ``"missing_psf"`` -- requested but no PSF file found; falls back to raw.
        - ``"off"`` -- caller passed mode='off'.
    psf_fwhm_pixels:
        Measured FWHM of the PSF in pixels, or NaN if no PSF was loaded.
    """
    image_for_fit: np.ndarray
    mode: PsfMode
    psf_fwhm_pixels: float


# ---------------------------------------------------------------------------
# PSF FWHM estimation
# ---------------------------------------------------------------------------

def estimate_psf_fwhm(psf: np.ndarray) -> float:
    """Estimate FWHM (in pixels) of a 2D PSF by fitting a circular Gaussian.

    Uses second moments rather than a nonlinear fit -- robust, fast, and
    accurate to within ~1% on a clean Gaussian PSF. Returns NaN for degenerate
    inputs (all-zero, negative-only, non-finite, or vanishing second moment).
    """
    if psf is None or psf.size == 0:
        return float("nan")

    arr = np.asarray(psf, dtype=float)
    if not np.all(np.isfinite(arr)):
        # Replace NaNs with zero rather than abort; PSF files occasionally have
        # NaN borders from the cutout extraction.
        arr = np.where(np.isfinite(arr), arr, 0.0)

    # Normalise to a non-negative weight grid centred on its peak.
    arr = arr - np.min(arr)
    total = arr.sum()
    if not np.isfinite(total) or total <= 0.0:
        return float("nan")

    ny, nx = arr.shape
    ys, xs = np.mgrid[0:ny, 0:nx]
    weights = arr / total

    x0 = float((weights * xs).sum())
    y0 = float((weights * ys).sum())
    var_x = float((weights * (xs - x0) ** 2).sum())
    var_y = float((weights * (ys - y0) ** 2).sum())
    var = 0.5 * (var_x + var_y)  # circularised
    if var <= 0.0:
        return float("nan")

    sigma = np.sqrt(var)
    return float(_FWHM_PER_SIGMA * sigma)


# ---------------------------------------------------------------------------
# Wiener deconvolution
# ---------------------------------------------------------------------------

def _background_rms(image: np.ndarray) -> float:
    """Estimate background RMS via the MAD of the lower half of pixel values.

    Robust to bright sources in the centre of the cutout. Returns 0.0 if the
    image is degenerate (all-zero or non-finite).
    """
    arr = image[np.isfinite(image)]
    if arr.size == 0:
        return 0.0
    # Lower-half MAD: take pixels below the median, mirror about the median,
    # and use 1.4826 * MAD as the sigma estimate. This avoids contamination
    # from the galaxy.
    med = float(np.median(arr))
    lower = arr[arr < med]
    if lower.size == 0:
        return 0.0
    mad = float(np.median(med - lower))
    return 1.4826 * mad


def wiener_deconvolve(
    image: np.ndarray,
    psf: np.ndarray,
    noise_rms: Optional[float] = None,
    regularization_floor: float = 1e-6,
) -> np.ndarray:
    """Wiener-deconvolve ``image`` by ``psf`` in the Fourier domain.

    The Wiener filter is::

        F(x) = ifft(  conj(H) / (|H|^2 + K)  *  fft(image_padded) )

    where ``H`` is the FFT of the PSF (centred and zero-padded to the image
    shape), and ``K`` is a regularization parameter. We set::

        K = max(regularization_floor, (noise_rms / image_dynamic_range)^2)

    so that K grows with the relative noise level. This gives a stable result
    on both deep broadband cutouts (low noise -> small K -> sharper recovery)
    and shallow narrowband cutouts (higher noise -> larger K -> heavier
    regularization).

    The PSF is normalised to unit sum before transformation; the deconvolved
    result is real-valued.

    Parameters
    ----------
    image, psf
        2D arrays. Need not be the same shape; ``psf`` is zero-padded to
        ``image.shape`` and shifted so its centroid sits at the origin (FFT
        convention).
    noise_rms
        Background RMS. If None, estimated from the image.
    regularization_floor
        Lower bound on ``K`` to guarantee numerical stability.
    """
    img = np.asarray(image, dtype=float)
    psf_arr = np.asarray(psf, dtype=float)

    if img.ndim != 2 or psf_arr.ndim != 2:
        raise ValueError("wiener_deconvolve requires 2D inputs")

    if not np.isfinite(psf_arr).any() or psf_arr.sum() <= 0:
        # Nothing usable to deconvolve with; return the input unchanged so the
        # caller can fall back to raw.
        return img

    psf_norm = psf_arr / psf_arr.sum()

    # Pad PSF to image shape and centre it at (0, 0) for the FFT, which is what
    # numpy's FFT expects: a kernel whose origin is the top-left, with the rest
    # wrapping around.
    padded = np.zeros_like(img)
    py, px = psf_norm.shape
    iy, ix = img.shape
    if py > iy or px > ix:
        # PSF larger than the image -- crop to the central piece.
        sy = (py - iy) // 2
        sx = (px - ix) // 2
        psf_norm = psf_norm[sy:sy + iy, sx:sx + ix]
        py, px = psf_norm.shape
    oy = (iy - py) // 2
    ox = (ix - px) // 2
    padded[oy:oy + py, ox:ox + px] = psf_norm
    # Roll so the PSF centre is at (0, 0).
    padded = np.roll(padded, shift=(-(oy + py // 2), -(ox + px // 2)), axis=(0, 1))

    if noise_rms is None:
        noise_rms = _background_rms(img)

    dynamic_range = float(np.nanmax(img) - np.nanmin(img))
    if dynamic_range <= 0.0 or not np.isfinite(dynamic_range):
        return img
    rel_noise = max(0.0, float(noise_rms)) / dynamic_range
    k = max(regularization_floor, rel_noise ** 2)

    H = np.fft.fft2(padded)
    G = np.fft.fft2(img)
    filt = np.conj(H) / (np.abs(H) ** 2 + k)
    out = np.fft.ifft2(filt * G).real

    if not np.all(np.isfinite(out)):
        logger.debug("wiener_deconvolve produced non-finite pixels -- returning input")
        return img
    return out


# ---------------------------------------------------------------------------
# Pipeline-facing entry point
# ---------------------------------------------------------------------------

def _load_psf_array(psf_path: Optional[Path]) -> Optional[np.ndarray]:
    """Open a PSF FITS file. Returns None if the file is missing or unreadable.

    J-PLUS' PSFEx-derived PSF cutouts store the array in extension HDU[1]
    with HDU[0] holding only header metadata. Other surveys (and the
    synthetic PSFs used in our unit tests) put the image in HDU[0]. We try
    HDU[0] first and fall back to HDU[1] if [0] is empty or not 2D.

    Defers the astropy import so that callers using ``psf-mode=off`` never
    touch astropy via this module.
    """
    if psf_path is None or not psf_path.is_file():
        return None
    try:
        from astropy.io import fits  # local import: see docstring
        with fits.open(psf_path) as hdul:
            data = None
            for hdu in hdul:
                arr = hdu.data
                if arr is None:
                    continue
                arr = np.asarray(arr)
                if arr.ndim == 2 and arr.size > 0:
                    data = arr
                    break
        if data is None:
            return None
        return np.asarray(data, dtype=float)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to read PSF at {psf_path}: {e}")
        return None


def preprocess_for_fit(
    image: np.ndarray,
    psf_path: Optional[Path],
    r_eff_pixels: float,
    mode: Literal["auto", "on", "off"] = "auto",
    gate: float = DEFAULT_PSF_GATE,
) -> PSFInfo:
    """Decide whether to deconvolve, and return the image the pipeline should fit.

    Parameters
    ----------
    image:
        Raw cutout, 2D float array.
    psf_path:
        Path to the per-band PSF FITS file, or None if not available.
    r_eff_pixels:
        Effective radius of the galaxy in pixels (from the catalog).
    mode:
        ``'off'`` -- never deconvolve. Returns ``mode='off'``.
        ``'on'``  -- deconvolve whenever a PSF is available, regardless of gate.
        ``'auto'`` -- deconvolve only when ``psf_fwhm / r_eff_pixels >= gate``.
    gate:
        Threshold for the auto mode. Default 0.2 (skip when PSF is ~5x smaller
        than the galaxy).

    Returns
    -------
    PSFInfo with the image to fit, the mode it ended up in, and the measured
    PSF FWHM (NaN if no PSF was loaded).
    """
    if mode == "off":
        return PSFInfo(image_for_fit=image, mode="off", psf_fwhm_pixels=float("nan"))

    psf = _load_psf_array(psf_path)
    if psf is None:
        return PSFInfo(image_for_fit=image, mode="missing_psf", psf_fwhm_pixels=float("nan"))

    fwhm = estimate_psf_fwhm(psf)

    if mode == "auto":
        if not np.isfinite(r_eff_pixels) or r_eff_pixels <= 0.0:
            # No usable size prior -- be conservative, skip deconv.
            return PSFInfo(image_for_fit=image, mode="raw", psf_fwhm_pixels=fwhm)
        if not np.isfinite(fwhm) or fwhm <= 0.0:
            return PSFInfo(image_for_fit=image, mode="raw", psf_fwhm_pixels=fwhm)
        ratio = fwhm / r_eff_pixels
        if ratio < gate:
            return PSFInfo(image_for_fit=image, mode="raw", psf_fwhm_pixels=fwhm)

    deconvolved = wiener_deconvolve(image, psf)
    return PSFInfo(image_for_fit=deconvolved, mode="deconv", psf_fwhm_pixels=fwhm)
