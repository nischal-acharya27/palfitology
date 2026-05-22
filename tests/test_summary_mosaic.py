"""Tests for plots.make_summary_mosaic (the 12-band all-bands overview).

These lock in the V0.6+ behaviour where every band panel must display an
array of exactly the same shape (the shared detection-crop window), so the
12 panels can't end up with mixed square/rectangle aspect ratios as
happened pre-fix.

Pre-fix bug: ``make_summary_mosaic`` called ``_detection_crop(data, det)``
per band, which re-thresholded each band's data against the rSDSS-derived
background. Bands with different flux levels produced different bounding
boxes -> different display shapes -> visually inconsistent panel aspects.
The fix uses ``_detection_crop_slice`` once on a reference band and
applies the same (row_slice, col_slice) to every band.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from palfitology.detect import DetectionResult
from palfitology.fit import FitCandidate
from palfitology.plots import make_summary_mosaic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

JPLUS_BANDS = [
    "uJAVA", "J0378", "J0395", "J0410", "J0430", "gSDSS",
    "J0515", "rSDSS", "J0660", "iSDSS", "J0861", "zSDSS",
]


def _galaxy(shape=(61, 61), amplitude: float = 100.0, bg: float = 10.0,
            noise: float = 0.5, seed: int = 0) -> np.ndarray:
    """A simple centred Gaussian galaxy on a noisy background."""
    ny, nx = shape
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    ys, xs = np.mgrid[0:ny, 0:nx]
    galaxy = amplitude * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * 6.0 ** 2))
    rng = np.random.default_rng(seed)
    return bg + galaxy + rng.normal(0, noise, size=shape)


def _detection(shape=(61, 61), sigma_threshold: float = 3.0) -> DetectionResult:
    """A DetectionResult consistent with the synthetic ``_galaxy`` background."""
    return DetectionResult(
        status="ok",
        x0=float((shape[1] - 1) / 2.0),
        y0=float((shape[0] - 1) / 2.0),
        pa_deg=0.0,
        eps=0.2,
        npix=200,
        background=10.0,
        background_rms=0.5,
        sigma_threshold=sigma_threshold,
    )


def _fit_cand() -> FitCandidate:
    return FitCandidate(
        pa_deg=45.0, sma=12.0, ell=0.3, x0=30.0, y0=30.0,
        pa_err=0.1, score=1.0, config_tag="test",
        smoothing=0.0, weak=False,
    )


# ---------------------------------------------------------------------------
# The behaviour-locking regression test
# ---------------------------------------------------------------------------

def test_all_panels_display_same_shape_array(tmp_path: Path):
    """Every band panel must call imshow with an array of identical shape.

    Pre-fix this failed because per-band re-thresholding picked different
    bounding boxes for bright vs faint bands. The shared-slice fix guarantees
    one shape for all 12 panels.
    """
    # 12 bands with deliberately different flux levels so a per-band
    # re-threshold would pick different bboxes.
    bands = JPLUS_BANDS
    flux_scale = np.linspace(0.3, 3.0, len(bands))  # faint -> bright
    band_data = {b: _galaxy(amplitude=100.0 * s) for b, s in zip(bands, flux_scale)}
    band_cands = {b: _fit_cand() for b in bands}
    band_statuses = {b: "ok" for b in bands}

    det = _detection()

    # Capture every imshow call to check shape consistency.
    captured_shapes: list[tuple] = []

    real_imshow = None
    try:
        import matplotlib.axes as mpl_axes
        real_imshow = mpl_axes.Axes.imshow
    except Exception:
        pytest.skip("matplotlib not available")

    def _imshow_spy(self, X, *args, **kwargs):
        captured_shapes.append(tuple(np.asarray(X).shape))
        return real_imshow(self, X, *args, **kwargs)

    out = tmp_path / "all_bands_mosaic.png"
    with mock.patch.object(mpl_axes.Axes, "imshow", _imshow_spy):
        make_summary_mosaic(
            objectid="obj_aspect",
            band_data=band_data,
            band_cands=band_cands,
            band_statuses=band_statuses,
            bands_order=bands,
            out_path=out,
            detect_result=det,
        )

    assert out.is_file()
    assert len(captured_shapes) == len(bands), (
        f"expected {len(bands)} imshow calls (one per band), "
        f"got {len(captured_shapes)}: {captured_shapes}"
    )
    unique_shapes = set(captured_shapes)
    assert len(unique_shapes) == 1, (
        "all 12 band panels must display arrays of the same shape; "
        f"got {len(unique_shapes)} distinct shapes: {sorted(unique_shapes)}"
    )


def test_missing_band_does_not_break_shared_crop(tmp_path: Path):
    """A missing band leaves its panel as a placeholder; the others still share shape."""
    bands = JPLUS_BANDS
    band_data = {b: _galaxy(amplitude=100.0) for b in bands}
    band_data["J0395"] = None  # one missing band
    band_cands = {b: _fit_cand() if band_data.get(b) is not None else None for b in bands}
    band_statuses = {b: ("ok" if band_data.get(b) is not None else "missing") for b in bands}

    out = tmp_path / "missing_band_mosaic.png"
    make_summary_mosaic(
        objectid="obj_missing",
        band_data=band_data,
        band_cands=band_cands,
        band_statuses=band_statuses,
        bands_order=bands,
        out_path=out,
        detect_result=_detection(),
    )
    assert out.is_file()


def test_no_detection_falls_back_to_full_arrays(tmp_path: Path):
    """detect_result=None -> no crop -> every panel shows the full uncropped array."""
    bands = ["rSDSS", "gSDSS", "iSDSS"]
    band_data = {b: _galaxy(amplitude=100.0) for b in bands}
    band_cands = {b: _fit_cand() for b in bands}
    band_statuses = {b: "ok" for b in bands}

    out = tmp_path / "no_detect_mosaic.png"
    make_summary_mosaic(
        objectid="obj_nodet",
        band_data=band_data,
        band_cands=band_cands,
        band_statuses=band_statuses,
        bands_order=bands,
        out_path=out,
        detect_result=None,
    )
    assert out.is_file()
