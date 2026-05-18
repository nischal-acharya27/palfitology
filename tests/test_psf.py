"""Tests for palfitology.psf.

Covers:
 - FWHM estimation from a synthetic Gaussian PSF.
 - Wiener deconvolution recovers a sharp source from a blurred one.
 - Mode gating: 'off' bypasses everything, 'auto' skips deconv on small PSFs,
   'on' deconvolves whenever a PSF is present.
 - 'missing_psf' returns when no PSF file exists on disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from palfitology.psf import (
    DEFAULT_PSF_GATE,
    PSFInfo,
    estimate_psf_fwhm,
    preprocess_for_fit,
    wiener_deconvolve,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian_2d(shape: tuple[int, int], sigma: float, x0=None, y0=None) -> np.ndarray:
    """Render a circular 2D Gaussian, peak-normalised."""
    ny, nx = shape
    y0 = (ny - 1) / 2 if y0 is None else y0
    x0 = (nx - 1) / 2 if x0 is None else x0
    ys, xs = np.mgrid[0:ny, 0:nx]
    g = np.exp(-((xs - x0) ** 2 + (ys - y0) ** 2) / (2 * sigma ** 2))
    return g


def _write_fits(path: Path, data: np.ndarray) -> None:
    hdu = fits.PrimaryHDU(data=data.astype(np.float32))
    hdu.writeto(path, overwrite=True)


def _write_fits_in_ext1(path: Path, data: np.ndarray) -> None:
    """Mimic J-PLUS PSFEx output: empty primary HDU, image in extension 1."""
    primary = fits.PrimaryHDU()
    image = fits.ImageHDU(data=data.astype(np.float32))
    fits.HDUList([primary, image]).writeto(path, overwrite=True)


# ---------------------------------------------------------------------------
# FWHM estimation
# ---------------------------------------------------------------------------

def test_estimate_fwhm_circular_gaussian():
    # FWHM = 2*sqrt(2*ln2) * sigma. sigma=3 -> FWHM ~ 7.0645.
    psf = _gaussian_2d((41, 41), sigma=3.0)
    fwhm = estimate_psf_fwhm(psf)
    assert np.isfinite(fwhm)
    assert abs(fwhm - 2.0 * np.sqrt(2.0 * np.log(2.0)) * 3.0) < 0.05


def test_estimate_fwhm_nan_on_degenerate_input():
    assert np.isnan(estimate_psf_fwhm(np.zeros((10, 10))))
    assert np.isnan(estimate_psf_fwhm(np.full((10, 10), np.nan)))


def test_estimate_fwhm_handles_nan_borders():
    # Cutouts sometimes have NaN borders; the helper should treat them as 0.
    psf = _gaussian_2d((41, 41), sigma=3.0)
    psf[:2, :] = np.nan
    psf[-2:, :] = np.nan
    fwhm = estimate_psf_fwhm(psf)
    assert np.isfinite(fwhm)
    assert abs(fwhm - 2.0 * np.sqrt(2.0 * np.log(2.0)) * 3.0) < 0.3


# ---------------------------------------------------------------------------
# Wiener deconvolution
# ---------------------------------------------------------------------------

def test_wiener_sharpens_a_blurred_galaxy():
    """A point-like source convolved with a known PSF should be sharper after deconv."""
    rng = np.random.default_rng(0)
    shape = (61, 61)

    # An elongated 'galaxy' modelled as an anisotropic Gaussian.
    ys, xs = np.mgrid[0:shape[0], 0:shape[1]]
    yc, xc = 30, 30
    galaxy = np.exp(-((xs - xc) ** 2 / (2 * 2.0 ** 2)
                      + (ys - yc) ** 2 / (2 * 8.0 ** 2)))

    psf = _gaussian_2d(shape, sigma=2.5)
    psf /= psf.sum()

    # Convolve in Fourier space.
    F_psf = np.fft.fft2(np.fft.ifftshift(psf))
    blurred = np.fft.ifft2(np.fft.fft2(galaxy) * F_psf).real
    noisy = blurred + rng.normal(0, 0.001, size=shape)

    recovered = wiener_deconvolve(noisy, psf, noise_rms=0.001)

    # Peak should be higher post-deconv (we're undoing smoothing).
    assert recovered.max() > blurred.max()


def test_wiener_returns_input_on_degenerate_psf():
    img = np.random.default_rng(0).normal(size=(20, 20))
    bad = np.zeros_like(img)
    out = wiener_deconvolve(img, bad)
    # Degenerate PSF -> no-op
    np.testing.assert_allclose(out, img)


def test_wiener_returns_input_on_flat_image():
    flat = np.ones((20, 20))
    psf = _gaussian_2d((20, 20), sigma=2.0)
    out = wiener_deconvolve(flat, psf)
    # Zero dynamic range -> no-op
    np.testing.assert_allclose(out, flat)


# ---------------------------------------------------------------------------
# preprocess_for_fit -- mode + gating
# ---------------------------------------------------------------------------

def test_mode_off_short_circuits(tmp_path: Path):
    img = _gaussian_2d((30, 30), sigma=4.0)
    info = preprocess_for_fit(img, psf_path=None, r_eff_pixels=10.0, mode="off")
    assert isinstance(info, PSFInfo)
    assert info.mode == "off"
    assert np.isnan(info.psf_fwhm_pixels)
    np.testing.assert_array_equal(info.image_for_fit, img)


def test_mode_auto_skips_when_psf_is_small(tmp_path: Path):
    # Small PSF (sigma=0.5 -> FWHM ~ 1.18 px) vs big galaxy (R_EFF=20 px).
    # Ratio 1.18/20 = 0.059 < default gate 0.2 -> should be 'raw'.
    psf = _gaussian_2d((11, 11), sigma=0.5)
    psf_path = tmp_path / "psf.fits"
    _write_fits(psf_path, psf)

    img = _gaussian_2d((40, 40), sigma=8.0)
    info = preprocess_for_fit(img, psf_path=psf_path, r_eff_pixels=20.0, mode="auto")
    assert info.mode == "raw"
    np.testing.assert_array_equal(info.image_for_fit, img)
    assert info.psf_fwhm_pixels > 0  # we still measured it


def test_mode_auto_deconvolves_when_psf_is_large(tmp_path: Path):
    # Big PSF (sigma=3 -> FWHM ~ 7.06 px) vs small galaxy (R_EFF=4 px).
    # Ratio 7.06/4 = 1.77 > gate 0.2 -> should deconvolve.
    psf = _gaussian_2d((25, 25), sigma=3.0)
    psf_path = tmp_path / "psf.fits"
    _write_fits(psf_path, psf)

    img = _gaussian_2d((40, 40), sigma=4.0)
    info = preprocess_for_fit(img, psf_path=psf_path, r_eff_pixels=4.0, mode="auto")
    assert info.mode == "deconv"
    assert not np.array_equal(info.image_for_fit, img)


def test_mode_on_always_deconvolves_when_psf_present(tmp_path: Path):
    # Same small PSF that 'auto' would skip -- 'on' should still deconvolve.
    psf = _gaussian_2d((11, 11), sigma=0.5)
    psf_path = tmp_path / "psf.fits"
    _write_fits(psf_path, psf)

    img = _gaussian_2d((40, 40), sigma=8.0)
    info = preprocess_for_fit(img, psf_path=psf_path, r_eff_pixels=20.0, mode="on")
    assert info.mode == "deconv"


def test_missing_psf_file_falls_back_to_raw(tmp_path: Path):
    img = _gaussian_2d((30, 30), sigma=4.0)
    bogus = tmp_path / "does-not-exist.fits"
    info = preprocess_for_fit(img, psf_path=bogus, r_eff_pixels=10.0, mode="auto")
    assert info.mode == "missing_psf"
    np.testing.assert_array_equal(info.image_for_fit, img)


def test_auto_no_r_eff_falls_back_to_raw(tmp_path: Path):
    # Without a usable R_EFF we can't evaluate the gate -> be conservative.
    psf = _gaussian_2d((25, 25), sigma=3.0)
    psf_path = tmp_path / "psf.fits"
    _write_fits(psf_path, psf)

    img = _gaussian_2d((40, 40), sigma=4.0)
    info = preprocess_for_fit(img, psf_path=psf_path, r_eff_pixels=float("nan"), mode="auto")
    assert info.mode == "raw"


def test_gate_threshold_is_honoured(tmp_path: Path):
    """A custom --psf-gate setting should change the auto outcome."""
    psf = _gaussian_2d((25, 25), sigma=3.0)  # FWHM ~ 7.06 px
    psf_path = tmp_path / "psf.fits"
    _write_fits(psf_path, psf)

    img = _gaussian_2d((40, 40), sigma=4.0)
    r_eff = 10.0  # ratio ~ 0.706

    # gate=0.5 -> 0.706 > 0.5 -> deconv
    info_low_gate = preprocess_for_fit(img, psf_path, r_eff, mode="auto", gate=0.5)
    assert info_low_gate.mode == "deconv"

    # gate=0.9 -> 0.706 < 0.9 -> raw
    info_high_gate = preprocess_for_fit(img, psf_path, r_eff, mode="auto", gate=0.9)
    assert info_high_gate.mode == "raw"


def test_default_gate_is_documented_value():
    # Guard the doc: if we change the default, make us update the docstring too.
    assert DEFAULT_PSF_GATE == 0.2


def test_psf_loader_handles_image_in_ext1(tmp_path: Path):
    """J-PLUS PSFEx output stores the PSF in HDU[1], not HDU[0]."""
    psf = _gaussian_2d((25, 25), sigma=3.0)
    psf_path = tmp_path / "psf_ext1.fits"
    _write_fits_in_ext1(psf_path, psf)

    img = _gaussian_2d((40, 40), sigma=4.0)
    info = preprocess_for_fit(img, psf_path=psf_path, r_eff_pixels=4.0, mode="auto")
    # FWHM should have been measured -- proving the loader found the image in [1].
    assert np.isfinite(info.psf_fwhm_pixels)
    assert info.psf_fwhm_pixels > 0
    # And ratio 7.06/4 ~ 1.77 > 0.2 -> deconv.
    assert info.mode == "deconv"
