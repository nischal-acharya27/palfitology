"""Isophote-row selection rule for the photutils fit table.

photutils' `Ellipse.fit_image()` returns a table of isophotes at increasing
SMA. The naive "smallest pa_err" rule that early versions used (V4/V5) was
brittle: for face-on / low-S/N galaxies photutils generates many sub-pixel
near-center isophotes with artificially small pa_err (no flux gradient =
no apparent uncertainty), so the rule can collapse to a degenerate row with
sma ~ 0.5 px.

This module implements the V6 fix:

1. Geometric floor: candidate isophotes must have
   ``sma >= max(min_sma_abs, min_sma_frac * image_half_width)``.
2. Sanity mask: drop non-finite pa_err, the dummy pa_err==0 row, and
   under-sampled isophotes (ndata < 5).
3. Score = ``pa_err / sma``. Lower is better -- penalises both
   high-uncertainty and tiny-SMA fits.
4. Weak fallback: if no isophote clears the floor, return the outermost
   surviving (largest-SMA) row with a `weak=True` flag.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def select_isophote(
    iso_table,
    image_half_width: float,
    min_sma_abs: float = 3.0,
    min_sma_frac: float = 0.05,
) -> Tuple[Optional[int], bool]:
    """Choose the best isophote row from a photutils fit table.

    Parameters
    ----------
    iso_table : astropy Table
        ``isolist.to_table()`` from a `photutils.isophote.Ellipse` fit.
    image_half_width : float
        ``min(shape) / 2`` of the original image, used to derive the SMA floor.
    min_sma_abs : float
        Absolute lower bound on the chosen SMA, in pixels.
    min_sma_frac : float
        Fractional lower bound on the chosen SMA, as a fraction of
        ``image_half_width``.

    Returns
    -------
    (index, weak) : tuple
        ``index`` is the row index in ``iso_table`` (or None if no row was
        usable). ``weak`` is True when no isophote cleared the SMA floor and
        we fell back to the outermost surviving row.
    """
    sma_arr = np.asarray(iso_table["sma"].data, dtype=float)
    pa_err_arr = np.asarray(iso_table["pa_err"].data, dtype=float)
    ndata_arr = (
        np.asarray(iso_table["ndata"].data, dtype=float)
        if "ndata" in iso_table.colnames
        else np.full_like(sma_arr, np.inf)
    )

    sma_floor = max(min_sma_abs, min_sma_frac * image_half_width)

    sane = (
        np.isfinite(pa_err_arr)
        & (pa_err_arr > 0)
        & np.isfinite(sma_arr)
        & (sma_arr > 0)
        & (ndata_arr >= 5)
    )

    primary = sane & (sma_arr >= sma_floor)

    if np.any(primary):
        scores = np.where(primary, pa_err_arr / sma_arr, np.inf)
        return int(np.argmin(scores)), False

    if np.any(sane):
        sane_idx = np.where(sane)[0]
        outermost = sane_idx[np.argmax(sma_arr[sane])]
        return int(outermost), True

    return None, False
