"""Single-band isophotal fit + fallback ladder.

The public entry point is `fit_pa_with_fallbacks`, which takes a 2D image
array, catalog priors for ellipticity and PA, and returns the best-scoring
`FitCandidate` across many initialisation attempts.

The fallback ladder runs photutils with a sequence of starting geometries:
catalog priors first, then orthogonal/perpendicular variants, then 50
reproducible Monte-Carlo seeds spread across (eps, pa) space. Each pass is
also retried with a Gaussian-smoothed image (sigma=2) if no strong fit was
found at full resolution. As soon as `keep_best_of` strong candidates have
been collected, the best-scoring one (lowest pa_err/sma) is returned.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from photutils.isophote import Ellipse, EllipseGeometry
from scipy.ndimage import gaussian_filter

from .selection import select_isophote

logger = logging.getLogger(__name__)


@dataclass
class FitCandidate:
    """One scored, selected isophote from a single photutils fit."""

    pa_deg: float           # position angle in degrees (CCW from +x)
    sma: float              # semi-major axis in pixels
    ell: float              # ellipticity = 1 - b/a
    x0: float               # fitted center x
    y0: float               # fitted center y
    pa_err: float           # uncertainty on pa, degrees
    score: float            # pa_err / sma; lower = better
    config_tag: str         # which fallback config produced this
    smoothing: float        # Gaussian sigma applied to image before fitting
    weak: bool              # True if no isophote cleared the SMA floor


def _fit_once(
    data: np.ndarray,
    eps: float,
    pa_init_deg: float,
    min_sma_abs: float,
    min_sma_frac: float,
    sma_guess_frac: float = 0.25,
) -> Optional[FitCandidate]:
    """Run one photutils ellipse fit + isophote selection.

    Returns None on hard failure (no isophotes, all rows rejected).

    Parameters
    ----------
    sma_guess_frac : float
        Initial SMA guess as a fraction of image width. Default 0.25 (i.e.
        ``x_shape / 4``) -- cutouts are pre-centred on the target, so a
        galaxy filling ~half the cutout has true SMA close to x_shape/4.
    """
    y_shape, x_shape = data.shape
    x0, y0 = x_shape / 2.0, y_shape / 2.0
    sma_guess = x_shape * sma_guess_frac
    half_width = min(x_shape, y_shape) / 2.0

    geometry = EllipseGeometry(x0, y0, sma_guess, eps, np.deg2rad(pa_init_deg))
    ellipse = Ellipse(data, geometry=geometry)
    isolist = ellipse.fit_image()

    if not isolist:
        return None

    iso_table = isolist.to_table()
    if len(iso_table) < 2:
        return None

    idx, weak = select_isophote(iso_table, half_width, min_sma_abs, min_sma_frac)
    if idx is None:
        return None

    pa_val = float(iso_table["pa"][idx].value)  # photutils stores deg
    sma_val = float(iso_table["sma"][idx])
    ell_val = float(iso_table["ellipticity"][idx])
    x0_val = float(iso_table["x0"][idx])
    y0_val = float(iso_table["y0"][idx])
    pa_err_val = float(iso_table["pa_err"][idx].value)

    score = pa_err_val / sma_val if sma_val > 0 else float("inf")

    return FitCandidate(
        pa_deg=pa_val,
        sma=sma_val,
        ell=ell_val,
        x0=x0_val,
        y0=y0_val,
        pa_err=pa_err_val,
        score=score,
        config_tag="",  # caller fills in
        smoothing=0.0,
        weak=weak,
    )


def _generate_fallback_configs(eps_prior: float, pa_prior: float):
    """Yield (tag, eps_init, pa_init) tuples for the fallback ladder.

    The order matters: scientific priors come first, then orthogonal /
    perpendicular variants, then 50 Monte-Carlo random seeds. Most galaxies
    converge on the first or second config; the MC seeds are insurance for
    pathological cases.
    """
    safe_eps = eps_prior if np.isfinite(eps_prior) else 0.3
    safe_pa = pa_prior if np.isfinite(pa_prior) else 30.0

    yield ("catalog_prior", safe_eps, safe_pa)
    yield ("default_03_30", 0.3, 30.0)
    yield (
        "orthogonal_inv",
        max(0.01, min(0.99, 1.0 - safe_eps)),
        -safe_pa,
    )
    yield ("perp_pa", 0.5, safe_pa + 90.0)
    yield ("e04_pa-45", 0.4, -45.0)
    yield ("e01_pa+45", 0.1, safe_pa + 45.0)
    yield ("e08_pa135", 0.8, 135.0)

    np.random.seed(42)
    random_eps = np.random.uniform(0.05, 0.95, size=50)
    random_pa = np.random.uniform(0.0, 180.0, size=50)
    for i, (e, p) in enumerate(zip(random_eps, random_pa)):
        yield (f"mc_{i}", float(e), float(p))


def fit_pa_with_fallbacks(
    data: np.ndarray,
    eps_prior: float,
    pa_prior: float,
    min_sma_abs: float = 3.0,
    min_sma_frac: float = 0.05,
    keep_best_of: int = 8,
) -> Tuple[Optional[FitCandidate], int]:
    """Fit one image with many initialisations and return the best candidate.

    Parameters
    ----------
    data : np.ndarray
        2D image array (a FITS cutout).
    eps_prior : float
        Catalog ellipticity prior, typically ``B_WORLD / A_WORLD``.
    pa_prior : float
        Catalog position angle prior in degrees.
    min_sma_abs, min_sma_frac : float
        Forwarded to `select_isophote`.
    keep_best_of : int
        Stop collecting strong candidates once this many have succeeded; return
        the best-scoring one. Set higher for more thorough searches.

    Returns
    -------
    (candidate, n_tried) : tuple
        `candidate` is the best-scoring `FitCandidate`, or None if photutils
        refused to fit anything. `n_tried` is the total number of init
        configurations attempted.
    """
    strong: List[FitCandidate] = []
    weak: List[FitCandidate] = []
    n_tried = 0

    for smoothing in (0.0, 2.0):
        work = data if smoothing == 0 else gaussian_filter(data, sigma=smoothing)
        for tag, eps, pa_init in _generate_fallback_configs(eps_prior, pa_prior):
            n_tried += 1
            try:
                cand = _fit_once(work, eps, pa_init, min_sma_abs, min_sma_frac)
            except Exception as e:  # noqa: BLE001 -- photutils raises lots of types
                logger.debug(f"  fit threw ({tag}, smooth={smoothing}): {e}")
                continue
            if cand is None:
                continue
            cand.config_tag = tag
            cand.smoothing = float(smoothing)
            if cand.weak:
                weak.append(cand)
            else:
                strong.append(cand)
                if len(strong) >= keep_best_of:
                    return min(strong, key=lambda c: c.score), n_tried

        # If we got at least one strong fit at this smoothing level, prefer it
        # over running another full ladder with extra Gaussian blur.
        if strong:
            return min(strong, key=lambda c: c.score), n_tried

    if strong:
        return min(strong, key=lambda c: c.score), n_tried
    if weak:
        # Among weak fits, prefer the largest SMA (most galaxy-scale info).
        return max(weak, key=lambda c: c.sma), n_tried
    return None, n_tried
