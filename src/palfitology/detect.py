"""r-band sigma-clipping detection — V0.4.

This module implements a SExtractor-style source detection pass:

1. Estimate the sky background level and noise using astropy's sigma-clipped
   statistics (robust against the central galaxy contaminating the mean).
2. Threshold the image at ``background + sigma_threshold * rms`` to produce a
   binary mask of "bright" pixels.
3. Label connected components (scipy.ndimage) and select the one whose
   centroid is closest to the cutout centre.  This guards against bright
   stars/neighbours that sit off-centre dominating the mask.
4. Compute first- and second-moments of that component to derive an initial
   estimate of the source centroid, ellipticity, and position angle.

The result is a ``DetectionResult`` dataclass.  The pipeline passes it into
every band's fit so that ``fit_pa_with_fallbacks`` gets moment-derived priors
instead of catalog priors, and ``_fit_once`` seeds its ``EllipseGeometry``
from the real source shape rather than a blind guess.

Typical usage
-------------
::

    from palfitology.detect import detect_source, DEFAULT_DETECT_SIGMA

    result = detect_source(r_band_image, sigma_threshold=DEFAULT_DETECT_SIGMA)
    if result.status == "ok":
        # use result.pa_deg, result.eps, result.x0, result.y0 as fit priors
        ...

Design notes
------------
* We deliberately do NOT mask bad-pixels in this module — the pipeline already
  handles FITS NaN values by opening as float; any NaNs propagate through
  sigma_clipped_stats cleanly (it ignores NaN by default).
* The moment-derived PA and eps are good enough as *initial geometry seeds*.
  The isophote fitter in fit.py refines them; we are not trying to replace it.
* If the detection band itself has no signal (blank field, very low S/N), the
  function returns status='no_detection' and the pipeline falls back to the
  original catalog priors — identical behaviour to v0.3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import binary_dilation, label

logger = logging.getLogger(__name__)

__all__ = [
    "DetectionResult",
    "DEFAULT_DETECT_SIGMA",
    "DEFAULT_DETECT_BAND",
    "DEFAULT_CLIP_DILATE",
    "detect_source",
    "build_detection_mask",
    "make_clipped_cutout",
]

# Default sigma threshold — 3σ above background, matching SExtractor's DETECT_THRESH.
DEFAULT_DETECT_SIGMA: float = 3.0

# The band whose image is used to build the detection mask applied to all bands.
DEFAULT_DETECT_BAND: str = "rSDSS"

# How many pixels to dilate the binary detection mask outward when making the
# clipped cutout.  0 = use the raw connected-component mask.  A small (1-3 px)
# dilation keeps a thin ring of "edge" pixels alive, which prevents the
# downstream isophote fitter from immediately running into a NaN boundary when
# the outer ellipse approaches the source edge.
DEFAULT_CLIP_DILATE: int = 0


DetectStatus = Literal["ok", "no_detection"]


@dataclass(frozen=True)
class DetectionResult:
    """Output of a single ``detect_source`` call.

    Attributes
    ----------
    status : 'ok' or 'no_detection'
        'no_detection' means the image is below threshold everywhere, or the
        labelled region is too small / too far from centre to be trusted.
    x0, y0 : float
        Intensity-weighted centroid of the detected component, in pixel coords
        (0-indexed, matching numpy axis convention x=col, y=row).
    pa_deg : float
        Position angle of the source, derived from second moments, in degrees
        CCW from the +x axis. NaN on no_detection.
    eps : float
        Ellipticity 1 - b/a from second moments, clamped to [0, 0.99].
        NaN on no_detection.
    npix : int
        Number of pixels in the selected connected component. 0 on no_detection.
    background : float
        Sigma-clipped mean of the sky background.
    background_rms : float
        Sigma-clipped standard deviation of the sky background.
    sigma_threshold : float
        The sigma multiple used to build the threshold.
    """
    status: DetectStatus
    x0: float
    y0: float
    pa_deg: float
    eps: float
    npix: int
    background: float
    background_rms: float
    sigma_threshold: float


def _second_moment_shape(
    image: np.ndarray,
    mask: np.ndarray,
    x0: float,
    y0: float,
) -> tuple[float, float]:
    """Return (eps, pa_deg) from intensity-weighted second moments.

    Parameters
    ----------
    image : 2D array
        The pixel values — used as weights so brighter pixels count more.
    mask : 2D bool array
        Only pixels where mask is True contribute.
    x0, y0 : float
        Centroid (already computed), used as the origin for moment offsets.

    Returns
    -------
    (eps, pa_deg)
        eps  in [0, 0.99], pa_deg in (-90, 90] degrees.
        Returns (0.3, 0.0) as a safe neutral default if moments are degenerate.
    """
    ny, nx = image.shape
    ys, xs = np.mgrid[0:ny, 0:nx]

    weights = np.where(mask, np.maximum(image, 0.0), 0.0)
    total = weights.sum()
    if total <= 0.0:
        return 0.3, 0.0

    dx = xs - x0
    dy = ys - y0

    Mxx = float((weights * dx * dx).sum()) / total
    Myy = float((weights * dy * dy).sum()) / total
    Mxy = float((weights * dx * dy).sum()) / total

    # Eigenvalues of the moment matrix give the principal axes.
    trace = Mxx + Myy
    det = Mxx * Myy - Mxy ** 2
    discriminant = max(0.0, (trace / 2) ** 2 - det)
    sqrt_disc = np.sqrt(discriminant)

    lambda1 = trace / 2 + sqrt_disc  # larger eigenvalue -> major axis
    lambda2 = trace / 2 - sqrt_disc  # smaller eigenvalue -> minor axis

    if lambda1 <= 0.0:
        return 0.3, 0.0

    # Ellipticity: 1 - sqrt(lambda2/lambda1) clamped to valid range.
    ratio = max(0.0, lambda2 / lambda1) if lambda1 > 0 else 0.0
    eps = float(np.clip(1.0 - np.sqrt(ratio), 0.0, 0.99))

    # PA: angle of the major-axis eigenvector.
    # Eigenvector for lambda1: solve (Mxx - lambda1)*vx + Mxy*vy = 0.
    if abs(Mxy) < 1e-12:
        pa_deg = 0.0 if Mxx >= Myy else 90.0
    else:
        # vy = 1, vx = -Mxy / (Mxx - lambda1)
        denom = Mxx - lambda1
        if abs(denom) < 1e-12:
            pa_deg = 90.0
        else:
            vx = -Mxy / denom
            vy = 1.0
            pa_deg = float(np.degrees(np.arctan2(vy, vx)))
            # Bring into (-90, 90] — same half-turn ambiguity as photutils.
            pa_deg = float(pa_deg % 180.0)
            if pa_deg > 90.0:
                pa_deg -= 180.0

    return eps, pa_deg


def detect_source(
    image: np.ndarray,
    sigma_threshold: float = DEFAULT_DETECT_SIGMA,
    min_npix: int = 5,
    max_centre_offset_frac: float = 0.4,
) -> DetectionResult:
    """Detect the central source in a cutout using sigma-clipping.

    Parameters
    ----------
    image : 2D float array
        The image cutout (already opened as float, NaNs allowed).
    sigma_threshold : float
        Number of sigma above background to use as the detection threshold.
        Default 3.0 (matches SExtractor's DETECT_THRESH=3.0).
    min_npix : int
        Minimum number of pixels a connected component must have to be
        considered a real detection.  Smaller blobs are noise spikes.
    max_centre_offset_frac : float
        The centroid of the selected component must lie within this fraction
        of the image half-width from the cutout centre.  If nothing passes
        this cut, status='no_detection'.  Default 0.4 (40% of half-width).

    Returns
    -------
    DetectionResult
    """
    arr = np.asarray(image, dtype=float)
    ny, nx = arr.shape

    # ------------------------------------------------------------------
    # 1. Sigma-clipped background stats
    # ------------------------------------------------------------------
    # sigma_clipped_stats ignores NaN by default (mask_value=None).
    # Returns (mean, median, std); we use mean as the background level
    # and std as the noise estimate, consistent with SExtractor.
    try:
        bg_mean, _bg_med, bg_std = sigma_clipped_stats(
            arr, sigma=3.0, maxiters=5
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"sigma_clipped_stats failed: {exc} -- returning no_detection")
        return DetectionResult(
            status="no_detection",
            x0=nx / 2.0, y0=ny / 2.0,
            pa_deg=float("nan"), eps=float("nan"),
            npix=0,
            background=float("nan"), background_rms=float("nan"),
            sigma_threshold=sigma_threshold,
        )

    if not np.isfinite(bg_std) or bg_std <= 0.0:
        logger.debug("Background RMS is zero or non-finite -- no_detection")
        return DetectionResult(
            status="no_detection",
            x0=nx / 2.0, y0=ny / 2.0,
            pa_deg=float("nan"), eps=float("nan"),
            npix=0,
            background=float(bg_mean) if np.isfinite(bg_mean) else float("nan"),
            background_rms=float(bg_std),
            sigma_threshold=sigma_threshold,
        )

    threshold = bg_mean + sigma_threshold * bg_std

    # ------------------------------------------------------------------
    # 2. Build binary detection mask
    # ------------------------------------------------------------------
    detect_mask = np.isfinite(arr) & (arr > threshold)

    if not detect_mask.any():
        logger.debug(
            f"No pixels above {sigma_threshold}σ threshold "
            f"(bg={bg_mean:.3g}, rms={bg_std:.3g}) -- no_detection"
        )
        return DetectionResult(
            status="no_detection",
            x0=nx / 2.0, y0=ny / 2.0,
            pa_deg=float("nan"), eps=float("nan"),
            npix=0,
            background=float(bg_mean), background_rms=float(bg_std),
            sigma_threshold=sigma_threshold,
        )

    # ------------------------------------------------------------------
    # 3. Label connected components; select the one nearest the centre
    # ------------------------------------------------------------------
    labelled, n_labels = label(detect_mask)
    if n_labels == 0:
        return DetectionResult(
            status="no_detection",
            x0=nx / 2.0, y0=ny / 2.0,
            pa_deg=float("nan"), eps=float("nan"),
            npix=0,
            background=float(bg_mean), background_rms=float(bg_std),
            sigma_threshold=sigma_threshold,
        )

    cx_img = (nx - 1) / 2.0  # cutout geometric centre (x = col)
    cy_img = (ny - 1) / 2.0  # cutout geometric centre (y = row)
    half_width = min(nx, ny) / 2.0
    max_offset = max_centre_offset_frac * half_width

    best_label = None
    best_dist = float("inf")

    for lbl in range(1, n_labels + 1):
        comp = labelled == lbl
        npix = int(comp.sum())
        if npix < min_npix:
            continue
        # Centroid of this component (unweighted — we'll refine with flux weights later).
        ys_comp, xs_comp = np.where(comp)
        cy_comp = float(ys_comp.mean())
        cx_comp = float(xs_comp.mean())
        dist = np.hypot(cx_comp - cx_img, cy_comp - cy_img)
        if dist < best_dist:
            best_dist = dist
            best_label = lbl

    if best_label is None or best_dist > max_offset:
        logger.debug(
            f"No component with npix>={min_npix} within {max_centre_offset_frac:.0%} "
            f"of image centre (best dist={best_dist:.1f} px) -- no_detection"
        )
        return DetectionResult(
            status="no_detection",
            x0=cx_img, y0=cy_img,
            pa_deg=float("nan"), eps=float("nan"),
            npix=0,
            background=float(bg_mean), background_rms=float(bg_std),
            sigma_threshold=sigma_threshold,
        )

    best_mask = labelled == best_label
    npix = int(best_mask.sum())

    # ------------------------------------------------------------------
    # 4. Intensity-weighted centroid
    # ------------------------------------------------------------------
    weights = np.where(best_mask, np.maximum(arr - bg_mean, 0.0), 0.0)
    total_w = weights.sum()
    ys_grid, xs_grid = np.mgrid[0:ny, 0:nx]

    if total_w > 0.0:
        x0 = float((weights * xs_grid).sum() / total_w)
        y0 = float((weights * ys_grid).sum() / total_w)
    else:
        # Fall back to geometric centroid of the mask pixels.
        ys_comp, xs_comp = np.where(best_mask)
        x0 = float(xs_comp.mean())
        y0 = float(ys_comp.mean())

    # ------------------------------------------------------------------
    # 5. Second-moment shape (eps, PA)
    # ------------------------------------------------------------------
    eps, pa_deg = _second_moment_shape(arr - bg_mean, best_mask, x0, y0)

    logger.debug(
        f"detect_source: npix={npix} centre=({x0:.1f},{y0:.1f}) "
        f"eps={eps:.3f} pa={pa_deg:.1f}° dist_from_centre={best_dist:.1f}px"
    )

    return DetectionResult(
        status="ok",
        x0=x0,
        y0=y0,
        pa_deg=pa_deg,
        eps=eps,
        npix=npix,
        background=float(bg_mean),
        background_rms=float(bg_std),
        sigma_threshold=sigma_threshold,
    )


# ---------------------------------------------------------------------------
# V0.5: clipped-cutout generation
# ---------------------------------------------------------------------------
#
# The functions below take the same sigma-clipping logic used by
# ``detect_source`` and turn it into a *mask*, then apply that mask to the
# input image to produce a new "clipped" cutout where pixels outside the
# detected galaxy region are set to NaN.  The clipped cutout is written to
# disk as a new FITS file (see ``palfitology.cutouts``) and re-used as input
# to the PA fitter for every band.
#
# Design choice: NaN-fill (rather than zero or background) was chosen because
# ``photutils.isophote`` and ``numpy.nanmean``-style operations ignore NaN
# cleanly, so the downstream fit sees only the source pixels without any
# stray background structure contaminating the moments.


def build_detection_mask(
    image: np.ndarray,
    sigma_threshold: float = DEFAULT_DETECT_SIGMA,
    min_npix: int = 5,
    max_centre_offset_frac: float = 0.4,
    dilate: int = DEFAULT_CLIP_DILATE,
) -> tuple[np.ndarray, DetectionResult]:
    """Return the binary mask of the central source plus a DetectionResult.

    The mask is True for pixels belonging to the central detected component
    (after optional dilation by ``dilate`` pixels) and False elsewhere.  If
    detection fails the returned mask is all-False and the DetectionResult's
    status is ``'no_detection'``.

    This is the same logic as :func:`detect_source` but exposed in mask form
    so it can be applied to the image to build a clipped cutout.
    """
    arr = np.asarray(image, dtype=float)
    ny, nx = arr.shape

    # ------------------------------------------------------------------
    # Reuse detect_source for stats + moment-derived geometry.
    # ------------------------------------------------------------------
    det = detect_source(
        arr,
        sigma_threshold=sigma_threshold,
        min_npix=min_npix,
        max_centre_offset_frac=max_centre_offset_frac,
    )

    if det.status != "ok":
        return np.zeros((ny, nx), dtype=bool), det

    # ------------------------------------------------------------------
    # Rebuild the binary mask.  We repeat the labelling here (cheap) so we
    # don't have to refactor detect_source's internal state into the public
    # return type.
    # ------------------------------------------------------------------
    threshold = det.background + sigma_threshold * det.background_rms
    detect_mask = np.isfinite(arr) & (arr > threshold)
    labelled, n_labels = label(detect_mask)

    if n_labels == 0:
        return np.zeros((ny, nx), dtype=bool), det

    # The "central" component is the one whose centroid lies closest to
    # ``(det.x0, det.y0)`` — that's exactly the one detect_source picked.
    best_label = None
    best_dist = float("inf")
    for lbl in range(1, n_labels + 1):
        comp = labelled == lbl
        if comp.sum() < min_npix:
            continue
        ys_comp, xs_comp = np.where(comp)
        dist = np.hypot(xs_comp.mean() - det.x0, ys_comp.mean() - det.y0)
        if dist < best_dist:
            best_dist = dist
            best_label = lbl

    if best_label is None:
        return np.zeros((ny, nx), dtype=bool), det

    mask = labelled == best_label

    if dilate > 0:
        mask = binary_dilation(mask, iterations=int(dilate))

    return mask, det


def make_clipped_cutout(
    image: np.ndarray,
    sigma_threshold: float = DEFAULT_DETECT_SIGMA,
    min_npix: int = 5,
    max_centre_offset_frac: float = 0.4,
    dilate: int = DEFAULT_CLIP_DILATE,
    fill_value: float = float("nan"),
) -> tuple[np.ndarray, np.ndarray, DetectionResult]:
    """Sigma-clip an image and return a NaN-outside clipped cutout.

    Parameters
    ----------
    image : 2D float array
        The input cutout (typically the rSDSS band).
    sigma_threshold : float
        Threshold above background (in sigma) used to build the detection
        mask.  Default 3.0.
    min_npix, max_centre_offset_frac
        Forwarded to :func:`build_detection_mask` — see that function.
    dilate : int
        Number of pixels to dilate the binary mask outward before applying
        it.  0 = no dilation.  Useful when the downstream isophote fit
        needs a little breathing room around the source edge.
    fill_value : float
        Value to write into pixels *outside* the detected source.  Defaults
        to ``NaN`` so photutils ignores them cleanly.

    Returns
    -------
    clipped : 2D float array
        Same shape as ``image``: pixel values are preserved inside the
        detection mask, replaced by ``fill_value`` outside.
    mask : 2D bool array
        The mask itself (True = source pixel kept, False = clipped).
    det : DetectionResult
        The underlying detection result (status, centroid, moments, etc.).

    Notes
    -----
    * If detection fails (status='no_detection') the returned cutout is
      *entirely* ``fill_value`` and ``mask`` is all-False.  The pipeline
      should treat this as "do not write a clipped FITS for this object"
      so the original cutout remains the source of truth.
    * The input image is **never** modified in place; a fresh ndarray is
      always returned.
    """
    arr = np.asarray(image, dtype=float)
    mask, det = build_detection_mask(
        arr,
        sigma_threshold=sigma_threshold,
        min_npix=min_npix,
        max_centre_offset_frac=max_centre_offset_frac,
        dilate=dilate,
    )

    clipped = np.where(mask, arr, fill_value).astype(float, copy=True)
    return clipped, mask, det
