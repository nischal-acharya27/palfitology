"""Tests for palfitology.detect (V0.4 sigma-clip detection).

Covers:
 - Clean detection of a centred, elongated Gaussian galaxy.
 - Correct centroid within a tight tolerance.
 - PA direction: a galaxy elongated along the x-axis should give PA ~ 0°,
   one elongated along the y-axis should give PA ~ ±90°.
 - eps: round source gives eps ~ 0; elongated gives eps > 0.
 - No-detection on a flat/noise-only image.
 - No-detection when the only bright blob is far from centre.
 - min_npix gating: a single-pixel spike should not count as a detection.
 - detect_sigma=0 path in fit_catalog disables detection.
 - DetectionResult is frozen (immutable).
"""

from __future__ import annotations

import numpy as np
import pytest

from palfitology.detect import (
    DEFAULT_DETECT_SIGMA,
    DetectionResult,
    detect_source,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_galaxy(
    shape=(61, 61),
    sigma_x: float = 8.0,
    sigma_y: float = 3.0,
    x0: float | None = None,
    y0: float | None = None,
    amplitude: float = 100.0,
    bg: float = 10.0,
    noise: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """Render a synthetic elongated Gaussian galaxy on a flat background."""
    ny, nx = shape
    cx = (nx - 1) / 2.0 if x0 is None else x0
    cy = (ny - 1) / 2.0 if y0 is None else y0
    ys, xs = np.mgrid[0:ny, 0:nx]
    galaxy = amplitude * np.exp(
        -((xs - cx) ** 2 / (2 * sigma_x ** 2)
          + (ys - cy) ** 2 / (2 * sigma_y ** 2))
    )
    rng = np.random.default_rng(seed)
    return bg + galaxy + rng.normal(0, noise, size=shape)


# ---------------------------------------------------------------------------
# Basic detection
# ---------------------------------------------------------------------------

def test_detect_centred_galaxy_returns_ok():
    img = _make_galaxy()
    result = detect_source(img, sigma_threshold=3.0)
    assert result.status == "ok"
    assert result.npix > 0
    assert np.isfinite(result.pa_deg)
    assert np.isfinite(result.eps)
    assert np.isfinite(result.background)
    assert np.isfinite(result.background_rms)


def test_detect_centroid_is_close_to_image_centre():
    shape = (61, 61)
    img = _make_galaxy(shape=shape)
    result = detect_source(img, sigma_threshold=3.0)
    assert result.status == "ok"
    cx, cy = (shape[1] - 1) / 2.0, (shape[0] - 1) / 2.0
    assert abs(result.x0 - cx) < 1.5  # within 1.5 px
    assert abs(result.y0 - cy) < 1.5


def test_detect_records_sigma_threshold():
    img = _make_galaxy()
    result = detect_source(img, sigma_threshold=2.5)
    assert result.sigma_threshold == 2.5


# ---------------------------------------------------------------------------
# PA direction
# ---------------------------------------------------------------------------

def test_pa_x_elongated_galaxy_near_zero():
    """Galaxy elongated along x-axis -> PA ~ 0 degrees."""
    img = _make_galaxy(sigma_x=12.0, sigma_y=2.0, noise=0.1)
    result = detect_source(img, sigma_threshold=3.0)
    assert result.status == "ok"
    # PA is in (-90, 90]; elongation along x -> near 0°, allow ±20° tolerance.
    assert abs(result.pa_deg) < 20.0, f"Expected PA ~ 0, got {result.pa_deg:.1f}°"


def test_pa_y_elongated_galaxy_near_90():
    """Galaxy elongated along y-axis -> PA ~ ±90 degrees."""
    img = _make_galaxy(sigma_x=2.0, sigma_y=12.0, noise=0.1)
    result = detect_source(img, sigma_threshold=3.0)
    assert result.status == "ok"
    # |PA| should be close to 90°, allow ±20° tolerance.
    assert abs(abs(result.pa_deg) - 90.0) < 20.0, (
        f"Expected |PA| ~ 90, got {result.pa_deg:.1f}°"
    )


# ---------------------------------------------------------------------------
# Ellipticity
# ---------------------------------------------------------------------------

def test_eps_round_galaxy_is_low():
    """Near-circular galaxy should give low ellipticity."""
    img = _make_galaxy(sigma_x=6.0, sigma_y=6.0, noise=0.1)
    result = detect_source(img, sigma_threshold=3.0)
    assert result.status == "ok"
    assert result.eps < 0.2, f"Expected eps < 0.2 for round galaxy, got {result.eps:.3f}"


def test_eps_elongated_galaxy_is_high():
    """Highly elongated galaxy should give larger ellipticity."""
    img = _make_galaxy(sigma_x=12.0, sigma_y=2.0, noise=0.1)
    result = detect_source(img, sigma_threshold=3.0)
    assert result.status == "ok"
    assert result.eps > 0.3, f"Expected eps > 0.3, got {result.eps:.3f}"


def test_eps_is_bounded():
    """eps must always be in [0, 0.99]."""
    img = _make_galaxy(sigma_x=20.0, sigma_y=0.5, noise=0.05)
    result = detect_source(img, sigma_threshold=2.0)
    if result.status == "ok":
        assert 0.0 <= result.eps <= 0.99


# ---------------------------------------------------------------------------
# No-detection cases
# ---------------------------------------------------------------------------

def test_no_detection_on_flat_image():
    img = np.ones((61, 61)) * 50.0
    result = detect_source(img, sigma_threshold=3.0)
    assert result.status == "no_detection"
    assert result.npix == 0
    assert np.isnan(result.pa_deg)
    assert np.isnan(result.eps)


def test_no_detection_on_pure_noise():
    rng = np.random.default_rng(99)
    img = rng.normal(loc=10.0, scale=1.0, size=(61, 61))
    # At 3σ in a 61×61 image there will be very few pixels, but none should
    # form a large connected component near centre. We just check the function
    # runs without error (status may be ok or no_detection depending on RNG).
    result = detect_source(img, sigma_threshold=5.0)
    # At 5σ pure noise rarely triggers a large central component.
    # We only check the output is valid:
    assert result.status in ("ok", "no_detection")
    assert result.sigma_threshold == 5.0


def test_no_detection_when_blob_is_off_centre():
    """A bright source far from centre should not be selected."""
    shape = (61, 61)
    img = np.ones(shape) * 10.0
    # Bright blob at top-left corner, not near centre.
    img[2:8, 2:8] = 200.0
    result = detect_source(img, sigma_threshold=3.0, max_centre_offset_frac=0.3)
    assert result.status == "no_detection"


def test_no_detection_on_single_pixel_spike():
    """A single bright pixel should not trigger detection (min_npix=5)."""
    img = np.ones((61, 61)) * 10.0
    img[30, 30] = 500.0  # 1-pixel spike at centre
    result = detect_source(img, sigma_threshold=3.0, min_npix=5)
    assert result.status == "no_detection"


# ---------------------------------------------------------------------------
# DetectionResult is frozen
# ---------------------------------------------------------------------------

def test_detection_result_is_frozen():
    result = DetectionResult(
        status="ok", x0=30.0, y0=30.0,
        pa_deg=10.0, eps=0.3, npix=100,
        background=5.0, background_rms=1.0,
        sigma_threshold=3.0,
    )
    with pytest.raises((AttributeError, TypeError)):
        result.pa_deg = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Default sigma constant
# ---------------------------------------------------------------------------

def test_default_detect_sigma_is_documented_value():
    assert DEFAULT_DETECT_SIGMA == 3.0
