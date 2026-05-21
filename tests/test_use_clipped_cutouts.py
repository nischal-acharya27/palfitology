"""Tests for V0.6 fit-pa consumption of clipped cutouts.

Covers:
 - With use_clipped_cutouts=True and a clipped sibling present, the worker
   reads from the clipped FITS and the result row records cutout_source='clipped'.
 - With use_clipped_cutouts=True and NO clipped sibling, the worker falls
   back to the original raw cutout and records cutout_source='original'.
 - With use_clipped_cutouts=False (default), the worker always reads from
   the original raw cutout and records cutout_source='original'.
 - With no cutout at all, _missing_row records cutout_source='missing'.
 - fit_catalog accepts and threads the use_clipped_cutouts kwarg.
 - The RESULT_COLUMNS schema includes 'cutout_source'.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from palfitology.cutouts import make_cutouts_for_catalog
from palfitology.pipeline import RESULT_COLUMNS, fit_catalog, process_one_band


# ---------------------------------------------------------------------------
# Helpers (self-contained, mirror the other test files)
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
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    ys, xs = np.mgrid[0:ny, 0:nx]
    galaxy = amplitude * np.exp(
        -((xs - cx) ** 2 / (2 * sigma_x ** 2)
          + (ys - cy) ** 2 / (2 * sigma_y ** 2))
    )
    rng = np.random.default_rng(seed)
    return bg + galaxy + rng.normal(0, noise, size=shape)


def _write_layout(root: Path, objectid: str, ra_dec: str,
                  bands_data: dict[str, np.ndarray]) -> Path:
    obj_dir = root / objectid
    fits_dir = obj_dir / f"fits_images_{ra_dec}"
    fits_dir.mkdir(parents=True, exist_ok=True)
    for band, data in bands_data.items():
        fits.PrimaryHDU(data=data.astype(np.float32)).writeto(
            fits_dir / f"{band}_cutout.fits", overwrite=True
        )
    return obj_dir


def _base_task(images_root: Path, output_dir: Path, objectid: str, band: str,
               *, use_clipped_cutouts: bool = False) -> dict:
    """Build a minimal task dict mirroring what fit_catalog would assemble."""
    return {
        "objectid": objectid,
        "band": band,
        "images_root": str(images_root),
        "output_dir": str(output_dir),
        "eps_prior": 0.3,
        "pa_prior": 30.0,
        "min_sma_abs": 3.0,
        "min_sma_frac": 0.05,
        "keep_best_of": 8,
        "psf_mode": "off",
        "psf_gate": 0.2,
        "r_eff_pixels": float("nan"),
        "detect_result": None,
        "use_clipped_cutouts": use_clipped_cutouts,
    }


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_result_columns_includes_cutout_source():
    """The CSV schema must expose the new column so external consumers can rely on it."""
    assert "cutout_source" in RESULT_COLUMNS


# ---------------------------------------------------------------------------
# process_one_band path
# ---------------------------------------------------------------------------

def test_process_one_band_reads_original_by_default(tmp_path: Path):
    images_root = tmp_path / "images"
    output_dir = tmp_path / "out"
    _write_layout(images_root, "obj", "1.0_2.0", {"rSDSS": _make_galaxy()})

    row = process_one_band(_base_task(images_root, output_dir, "obj", "rSDSS"))

    assert row["cutout_source"] == "original"
    assert "fits_images_" in row["fits_path"]


def test_process_one_band_prefers_clipped_when_flag_set(tmp_path: Path):
    images_root = tmp_path / "images"
    output_dir = tmp_path / "out"
    _write_layout(images_root, "obj", "1.0_2.0", {"rSDSS": _make_galaxy()})
    # Generate the clipped sibling via the public API.
    catalog = pd.DataFrame({"id": ["obj"]})
    make_cutouts_for_catalog(
        images_root=images_root, catalog=catalog,
        detect_band="rSDSS", apply_bands=["rSDSS"],
    )

    row = process_one_band(
        _base_task(images_root, output_dir, "obj", "rSDSS", use_clipped_cutouts=True)
    )

    assert row["cutout_source"] == "clipped"
    assert "clipped_cutouts_" in row["fits_path"]


def test_process_one_band_falls_back_to_original_when_no_clipped(tmp_path: Path):
    images_root = tmp_path / "images"
    output_dir = tmp_path / "out"
    _write_layout(images_root, "obj", "1.0_2.0", {"rSDSS": _make_galaxy()})
    # Note: no make_cutouts_for_catalog call -> no clipped sibling exists.

    row = process_one_band(
        _base_task(images_root, output_dir, "obj", "rSDSS", use_clipped_cutouts=True)
    )

    assert row["cutout_source"] == "original"
    assert "fits_images_" in row["fits_path"]


def test_process_one_band_missing_cutout_records_missing_source(tmp_path: Path):
    images_root = tmp_path / "images"
    output_dir = tmp_path / "out"
    # Create the object directory but with NO cutout files.
    (images_root / "obj" / "fits_images_1.0_2.0").mkdir(parents=True, exist_ok=True)

    row = process_one_band(
        _base_task(images_root, output_dir, "obj", "rSDSS", use_clipped_cutouts=True)
    )

    assert row["cutout_source"] == "missing"
    assert row["status"] == "missing"


# ---------------------------------------------------------------------------
# fit_catalog driver path
# ---------------------------------------------------------------------------

def test_fit_catalog_threads_use_clipped_cutouts_flag(tmp_path: Path):
    images_root = tmp_path / "images"
    output_dir = tmp_path / "out"
    g = _make_galaxy()
    _write_layout(images_root, "obj_clip", "1.0_2.0", {"rSDSS": g})
    _write_layout(images_root, "obj_no_clip", "3.0_4.0", {"rSDSS": g})

    # Only the first object gets clipped cutouts.
    make_cutouts_for_catalog(
        images_root=images_root,
        catalog=pd.DataFrame({"id": ["obj_clip"]}),
        detect_band="rSDSS", apply_bands=["rSDSS"],
    )

    catalog = pd.DataFrame({
        "id": ["obj_clip", "obj_no_clip"],
        "A_WORLD": [1.0, 1.0],
        "B_WORLD": [0.5, 0.5],
        "pa_jplus": [30.0, 30.0],
    })

    df = fit_catalog(
        images_root=images_root,
        output_dir=output_dir,
        catalog=catalog,
        bands=["rSDSS"],
        workers=1,
        make_summary=False,
        psf_mode="off",
        detect_sigma=0.0,  # disable detection so the test is deterministic
        use_clipped_cutouts=True,
    )

    by_id = {r["id"]: r for _, r in df.iterrows()}
    assert by_id["obj_clip"]["cutout_source"] == "clipped"
    assert by_id["obj_no_clip"]["cutout_source"] == "original"


def test_fit_catalog_default_uses_original_cutouts(tmp_path: Path):
    """Without the flag, even when a clipped sibling exists, fit-pa reads from
    the original.  This guarantees backward-compatible behaviour for v0.5 users."""
    images_root = tmp_path / "images"
    output_dir = tmp_path / "out"
    g = _make_galaxy()
    _write_layout(images_root, "obj", "1.0_2.0", {"rSDSS": g})
    make_cutouts_for_catalog(
        images_root=images_root,
        catalog=pd.DataFrame({"id": ["obj"]}),
        detect_band="rSDSS", apply_bands=["rSDSS"],
    )

    df = fit_catalog(
        images_root=images_root,
        output_dir=output_dir,
        catalog=pd.DataFrame({
            "id": ["obj"], "A_WORLD": [1.0], "B_WORLD": [0.5], "pa_jplus": [30.0],
        }),
        bands=["rSDSS"],
        workers=1,
        make_summary=False,
        psf_mode="off",
        detect_sigma=0.0,
        # use_clipped_cutouts defaults to False
    )

    assert df.iloc[0]["cutout_source"] == "original"
