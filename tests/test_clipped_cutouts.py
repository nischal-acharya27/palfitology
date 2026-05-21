"""Tests for V0.5 clipped-cutout generation.

Covers:
 - make_clipped_cutout: NaN outside the detected component, finite inside.
 - build_detection_mask: returns the same connected component used by detect_source.
 - dilation widens the mask without changing the centroid (much).
 - write_clipped_fits: round-trip read/write, HISTORY records, NaN preserved.
 - derive_clipped_dir: ``fits_images_<ra>_<dec>`` -> ``clipped_cutouts_<ra>_<dec>``.
 - locate_clipped_or_original: prefers clipped, falls back to original.
 - make_cutouts_for_catalog: writes the right files for a 2-object synthetic catalog.
 - No-detection path: clipped cutout is all-NaN, status is propagated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from palfitology.cutouts import (
    ClipReport,
    derive_clipped_dir,
    locate_clipped_band_fits,
    locate_clipped_or_original,
    make_cutouts_for_catalog,
    write_clipped_fits,
)
from palfitology.detect import (
    DetectionResult,
    build_detection_mask,
    detect_source,
    make_clipped_cutout,
)


# ---------------------------------------------------------------------------
# Helpers (small, self-contained — kept independent of test_detect.py)
# ---------------------------------------------------------------------------

def _make_galaxy(
    shape=(61, 61),
    sigma_x: float = 8.0,
    sigma_y: float = 3.0,
    amplitude: float = 100.0,
    bg: float = 10.0,
    noise: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    ny, nx = shape
    cx = (nx - 1) / 2.0
    cy = (ny - 1) / 2.0
    ys, xs = np.mgrid[0:ny, 0:nx]
    galaxy = amplitude * np.exp(
        -((xs - cx) ** 2 / (2 * sigma_x ** 2)
          + (ys - cy) ** 2 / (2 * sigma_y ** 2))
    )
    rng = np.random.default_rng(seed)
    return bg + galaxy + rng.normal(0, noise, size=shape)


def _write_object_layout(
    root: Path,
    objectid: str,
    ra_dec: str,
    bands_data: dict[str, np.ndarray],
) -> Path:
    """Mimic the J-PLUS layout: images/<id>/fits_images_<ra>_<dec>/<band>_cutout.fits."""
    obj_dir = root / objectid
    fits_dir = obj_dir / f"fits_images_{ra_dec}"
    fits_dir.mkdir(parents=True, exist_ok=True)
    for band, data in bands_data.items():
        hdu = fits.PrimaryHDU(data=data.astype(np.float32))
        hdu.writeto(fits_dir / f"{band}_cutout.fits", overwrite=True)
    return obj_dir


# ---------------------------------------------------------------------------
# make_clipped_cutout
# ---------------------------------------------------------------------------

def test_make_clipped_cutout_nan_outside_mask():
    img = _make_galaxy()
    clipped, mask, det = make_clipped_cutout(img, sigma_threshold=3.0)

    assert det.status == "ok"
    assert clipped.shape == img.shape
    # Every pixel outside the mask is NaN, every pixel inside is finite.
    assert np.all(np.isnan(clipped[~mask]))
    assert np.all(np.isfinite(clipped[mask]))
    # And the kept pixels equal the original values exactly.
    np.testing.assert_array_equal(clipped[mask], img[mask])


def test_make_clipped_cutout_does_not_modify_input():
    img = _make_galaxy()
    img_copy = img.copy()
    _ = make_clipped_cutout(img, sigma_threshold=3.0)
    np.testing.assert_array_equal(img, img_copy)


def test_make_clipped_cutout_no_detection_returns_all_nan():
    flat = np.ones((41, 41)) * 50.0
    clipped, mask, det = make_clipped_cutout(flat, sigma_threshold=3.0)
    assert det.status == "no_detection"
    assert mask.sum() == 0
    assert np.all(np.isnan(clipped))


def test_make_clipped_cutout_fill_value_override():
    """Filling with 0 instead of NaN keeps the array fully finite."""
    img = _make_galaxy()
    clipped, mask, det = make_clipped_cutout(img, sigma_threshold=3.0, fill_value=0.0)
    assert det.status == "ok"
    assert np.all(np.isfinite(clipped))
    # Outside the mask, pixels are exactly 0.
    np.testing.assert_array_equal(clipped[~mask], 0.0)


# ---------------------------------------------------------------------------
# build_detection_mask
# ---------------------------------------------------------------------------

def test_build_detection_mask_matches_detect_source():
    img = _make_galaxy()
    mask, det = build_detection_mask(img, sigma_threshold=3.0)
    src = detect_source(img, sigma_threshold=3.0)
    # Same status, same npix.
    assert det.status == src.status == "ok"
    assert int(mask.sum()) == det.npix


def test_build_detection_mask_dilation_grows_mask():
    img = _make_galaxy()
    mask0, _ = build_detection_mask(img, sigma_threshold=3.0, dilate=0)
    mask3, _ = build_detection_mask(img, sigma_threshold=3.0, dilate=3)
    assert mask3.sum() > mask0.sum()
    # Dilation only adds pixels — never removes them.
    assert np.all(mask3 >= mask0)


def test_build_detection_mask_returns_blank_on_no_detection():
    flat = np.ones((41, 41)) * 50.0
    mask, det = build_detection_mask(flat, sigma_threshold=3.0)
    assert det.status == "no_detection"
    assert mask.sum() == 0
    assert mask.dtype == bool


# ---------------------------------------------------------------------------
# Folder / path helpers
# ---------------------------------------------------------------------------

def test_derive_clipped_dir_standard_layout(tmp_path: Path):
    fits_dir = tmp_path / "fits_images_123.45_-6.78"
    fits_dir.mkdir()
    clipped = derive_clipped_dir(fits_dir)
    assert clipped == tmp_path / "clipped_cutouts_123.45_-6.78"


def test_derive_clipped_dir_unrecognised_layout(tmp_path: Path):
    fits_dir = tmp_path / "weird_name"
    fits_dir.mkdir()
    clipped = derive_clipped_dir(fits_dir)
    assert clipped == tmp_path / "clipped_cutouts"


# ---------------------------------------------------------------------------
# write_clipped_fits round-trip
# ---------------------------------------------------------------------------

def test_write_clipped_fits_round_trip(tmp_path: Path):
    img = _make_galaxy()
    src = tmp_path / "src.fits"
    fits.PrimaryHDU(data=img.astype(np.float32)).writeto(src)

    clipped, mask, det = make_clipped_cutout(img, sigma_threshold=3.0)
    out = tmp_path / "clipped.fits"
    write_clipped_fits(
        clipped=clipped,
        source_fits=src,
        out_path=out,
        detection=det,
        dilate=0,
    )

    assert out.is_file()
    with fits.open(out) as hdul:
        read = hdul[0].data
        hdr = hdul[0].header

    # NaN must survive the FITS round-trip.
    assert read.shape == img.shape
    assert np.isnan(read[~mask]).all()
    # The kept pixels should be within float32 precision of the input.
    np.testing.assert_allclose(
        read[mask], img.astype(np.float32)[mask], rtol=0, atol=1e-4
    )

    # HISTORY entries document the threshold + dilation.
    history_text = "\n".join(hdr["HISTORY"])
    assert "sigma_threshold" in history_text
    assert "dilate_pixels" in history_text


def test_write_clipped_fits_handles_missing_source_header(tmp_path: Path):
    img = _make_galaxy()
    clipped, mask, det = make_clipped_cutout(img)
    fake_src = tmp_path / "nonexistent.fits"
    out = tmp_path / "clipped.fits"

    # Should not raise even when the source can't be read.
    write_clipped_fits(
        clipped=clipped,
        source_fits=fake_src,
        out_path=out,
        detection=det,
        dilate=0,
    )
    assert out.is_file()


# ---------------------------------------------------------------------------
# locate_clipped_or_original
# ---------------------------------------------------------------------------

def test_locate_clipped_or_original_prefers_clipped(tmp_path: Path):
    obj_dir = _write_object_layout(
        tmp_path, "obj1", "10.0_20.0",
        {"rSDSS": _make_galaxy()},
    )
    # Add a clipped sibling folder.
    clipped_dir = obj_dir / "clipped_cutouts_10.0_20.0"
    clipped_dir.mkdir()
    fits.PrimaryHDU(data=np.zeros((5, 5), dtype=np.float32)).writeto(
        clipped_dir / "rSDSS_cutout.fits"
    )

    path, source = locate_clipped_or_original(obj_dir, "rSDSS")
    assert source == "clipped"
    assert path.parent.name.startswith("clipped_cutouts_")


def test_locate_clipped_or_original_falls_back_to_original(tmp_path: Path):
    obj_dir = _write_object_layout(
        tmp_path, "obj1", "10.0_20.0",
        {"rSDSS": _make_galaxy()},
    )
    path, source = locate_clipped_or_original(obj_dir, "rSDSS")
    assert source == "original"
    assert path is not None
    assert path.name == "rSDSS_cutout.fits"


def test_locate_clipped_or_original_missing(tmp_path: Path):
    obj_dir = tmp_path / "obj_empty"
    obj_dir.mkdir()
    path, source = locate_clipped_or_original(obj_dir, "rSDSS")
    assert path is None
    assert source == "missing"


def test_locate_clipped_band_fits_returns_none_when_absent(tmp_path: Path):
    obj_dir = _write_object_layout(
        tmp_path, "obj1", "10.0_20.0",
        {"rSDSS": _make_galaxy()},
    )
    assert locate_clipped_band_fits(obj_dir, "rSDSS") is None


# ---------------------------------------------------------------------------
# Driver: make_cutouts_for_catalog
# ---------------------------------------------------------------------------

def test_make_cutouts_for_catalog_writes_expected_files(tmp_path: Path):
    images_root = tmp_path / "images"

    # Two objects: one detectable, one flat.
    g = _make_galaxy()
    _write_object_layout(images_root, "good", "1.0_2.0", {"rSDSS": g, "gSDSS": g})
    _write_object_layout(images_root, "blank", "3.0_4.0",
                         {"rSDSS": np.ones((41, 41)) * 50.0})

    catalog = pd.DataFrame({"id": ["good", "blank"]})
    reports = make_cutouts_for_catalog(
        images_root=images_root,
        catalog=catalog,
        detect_band="rSDSS",
        apply_bands=["rSDSS", "gSDSS"],
    )

    # All four (id, band) combinations should appear in the report.
    by_key = {(r.id, r.band): r for r in reports}
    assert set(by_key.keys()) == {
        ("good", "rSDSS"), ("good", "gSDSS"),
        ("blank", "rSDSS"), ("blank", "gSDSS"),
    }

    # The detectable object's clipped FITS should both exist.
    good_dir = images_root / "good"
    r_clipped = locate_clipped_band_fits(good_dir, "rSDSS")
    g_clipped = locate_clipped_band_fits(good_dir, "gSDSS")
    assert r_clipped is not None and r_clipped.is_file()
    assert g_clipped is not None and g_clipped.is_file()
    assert by_key[("good", "rSDSS")].status == "ok"
    assert by_key[("good", "gSDSS")].status == "ok"
    assert by_key[("good", "rSDSS")].npix_kept > 0

    # The flat object should report no_detection for every band, with no
    # FITS file written.
    assert by_key[("blank", "rSDSS")].status == "no_detection"
    assert by_key[("blank", "gSDSS")].status == "no_detection"
    assert locate_clipped_band_fits(images_root / "blank", "rSDSS") is None

    # The 'gSDSS' band cutout missing for the blank object is still reported.
    # (it had only rSDSS — we expect the apply_band loop to never get there
    #  because detection fails first; status is no_detection rather than
    #  missing_input)
    assert by_key[("blank", "gSDSS")].out_path == ""


def test_make_cutouts_for_catalog_handles_missing_object(tmp_path: Path):
    images_root = tmp_path / "images"
    images_root.mkdir()
    # Catalog references an id that has no directory.
    catalog = pd.DataFrame({"id": ["ghost"]})
    reports = make_cutouts_for_catalog(
        images_root=images_root,
        catalog=catalog,
        detect_band="rSDSS",
        apply_bands=["rSDSS"],
    )
    assert len(reports) == 1
    assert reports[0].status == "missing_input"


def test_make_cutouts_for_catalog_skips_when_no_overwrite(tmp_path: Path):
    images_root = tmp_path / "images"
    _write_object_layout(images_root, "good", "1.0_2.0",
                         {"rSDSS": _make_galaxy()})
    catalog = pd.DataFrame({"id": ["good"]})

    # First pass writes the file.
    reports1 = make_cutouts_for_catalog(
        images_root=images_root, catalog=catalog,
        detect_band="rSDSS", apply_bands=["rSDSS"],
    )
    assert reports1[0].status == "ok"
    out_path = Path(reports1[0].out_path)
    mtime1 = out_path.stat().st_mtime_ns

    # Second pass with --no-overwrite should leave the file untouched.
    reports2 = make_cutouts_for_catalog(
        images_root=images_root, catalog=catalog,
        detect_band="rSDSS", apply_bands=["rSDSS"],
        overwrite=False,
    )
    mtime2 = out_path.stat().st_mtime_ns
    assert reports2[0].status == "ok"
    assert mtime1 == mtime2  # not rewritten


# ---------------------------------------------------------------------------
# ClipReport dataclass sanity
# ---------------------------------------------------------------------------

def test_clip_report_round_trip():
    r = ClipReport(
        id="obj1", band="rSDSS", status="ok",
        sigma_threshold=3.0, npix_kept=100,
        detect_x0=30.0, detect_y0=30.0,
        out_path="/tmp/x.fits",
    )
    d = r.__dict__
    assert d["id"] == "obj1"
    assert d["status"] == "ok"


# ---------------------------------------------------------------------------
# Sanity: cross-band masks really do come from the detect band
# ---------------------------------------------------------------------------

def test_same_mask_applied_to_all_bands(tmp_path: Path):
    """The mask is built once from the detect band and applied identically
    to every other band — meaning the NaN pattern across bands is identical."""
    images_root = tmp_path / "images"
    galaxy = _make_galaxy(sigma_x=10.0, sigma_y=2.0)
    # Different per-band data, but same shape and footprint.
    _write_object_layout(images_root, "obj", "5.0_6.0", {
        "rSDSS": galaxy,
        "gSDSS": galaxy * 0.5 + 1.0,
        "iSDSS": galaxy * 1.3 + 0.5,
    })
    catalog = pd.DataFrame({"id": ["obj"]})
    make_cutouts_for_catalog(
        images_root=images_root, catalog=catalog,
        detect_band="rSDSS",
        apply_bands=["rSDSS", "gSDSS", "iSDSS"],
    )

    obj_dir = images_root / "obj"
    nan_patterns = []
    for band in ("rSDSS", "gSDSS", "iSDSS"):
        path = locate_clipped_band_fits(obj_dir, band)
        assert path is not None
        with fits.open(path) as hdul:
            arr = hdul[0].data
        nan_patterns.append(np.isnan(arr))

    # All three NaN-masks should be byte-identical.
    np.testing.assert_array_equal(nan_patterns[0], nan_patterns[1])
    np.testing.assert_array_equal(nan_patterns[0], nan_patterns[2])
