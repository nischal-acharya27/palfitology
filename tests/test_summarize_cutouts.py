"""Tests for the summarize-cutouts diagnostic mosaic (V0.5).

Covers:
 - plots.make_clipped_summary writes a PNG of the expected layout.
 - Missing-band panels are rendered as placeholders, not crashes.
 - cutouts.summarize_object_clipped_cutouts returns 'no_clipped' when no
   clipped FITS exist.
 - cutouts.summarize_object_clipped_cutouts returns 'ok' after
   make_cutouts_for_catalog populates the sibling folder.
 - cutouts.summarize_catalog_clipped_cutouts loops correctly over a 2-object
   catalog (one with clipped cutouts, one without).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from palfitology.cutouts import (
    make_cutouts_for_catalog,
    summarize_catalog_clipped_cutouts,
    summarize_object_clipped_cutouts,
)
from palfitology.detect import DetectionResult
from palfitology.plots import make_clipped_summary


# ---------------------------------------------------------------------------
# Helpers (intentionally duplicated from test_clipped_cutouts.py to keep
# each test file self-contained — small enough that the cost is negligible)
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


# ---------------------------------------------------------------------------
# plots.make_clipped_summary
# ---------------------------------------------------------------------------

def test_make_clipped_summary_writes_png(tmp_path: Path):
    """A 12-band raw|clipped grid writes a PNG without raising."""
    bands = ["uJAVA", "J0378", "J0395", "J0410", "J0430", "gSDSS",
             "J0515", "rSDSS", "J0660", "iSDSS", "J0861", "zSDSS"]
    raw = _make_galaxy()
    # Hand-construct a clipped copy: NaN outside a centred 20x20 square.
    clipped = raw.copy().astype(float)
    mask = np.zeros_like(clipped, dtype=bool)
    mask[20:40, 20:40] = True
    clipped[~mask] = np.nan

    band_raw = {b: raw.copy() for b in bands}
    band_clipped = {b: clipped.copy() for b in bands}

    det = DetectionResult(
        status="ok", x0=30.0, y0=30.0,
        pa_deg=0.0, eps=0.3, npix=400,
        background=10.0, background_rms=0.5,
        sigma_threshold=3.0,
    )

    out = tmp_path / "obj_clipped_summary.png"
    make_clipped_summary(
        objectid="obj_test",
        band_raw=band_raw,
        band_clipped=band_clipped,
        bands_order=bands,
        out_path=out,
        detect_result=det,
    )
    assert out.is_file()
    assert out.stat().st_size > 1000  # not an empty file


def test_make_clipped_summary_handles_missing_panels(tmp_path: Path):
    """Bands present in bands_order but absent from either dict get a placeholder."""
    bands = ["rSDSS", "gSDSS"]
    raw = _make_galaxy()

    band_raw = {"rSDSS": raw, "gSDSS": None}  # gSDSS raw missing
    band_clipped = {"rSDSS": None, "gSDSS": raw}  # rSDSS clipped missing
    out = tmp_path / "missing_panels.png"
    # Should write the figure regardless of None panels.
    make_clipped_summary(
        objectid="obj_partial",
        band_raw=band_raw,
        band_clipped=band_clipped,
        bands_order=bands,
        out_path=out,
        detect_result=None,
    )
    assert out.is_file()


def test_make_clipped_summary_accepts_list_of_paths(tmp_path: Path):
    """Multi-output path list is honoured (mirrors make_summary_mosaic)."""
    bands = ["rSDSS"]
    raw = _make_galaxy()
    out1 = tmp_path / "a.png"
    out2 = tmp_path / "b.png"
    make_clipped_summary(
        objectid="obj_multi",
        band_raw={"rSDSS": raw},
        band_clipped={"rSDSS": raw},
        bands_order=bands,
        out_path=[out1, out2],
        detect_result=None,
    )
    assert out1.is_file()
    assert out2.is_file()


# ---------------------------------------------------------------------------
# cutouts.summarize_object_clipped_cutouts
# ---------------------------------------------------------------------------

def test_summarize_object_no_clipped_returns_no_clipped(tmp_path: Path):
    """An object with only raw cutouts (no clipped sibling) returns 'no_clipped'
    and writes no file."""
    obj_dir = _write_layout(
        tmp_path / "images", "obj_no_clip", "1.0_2.0",
        {"rSDSS": _make_galaxy()},
    )
    out = tmp_path / "out.png"
    status = summarize_object_clipped_cutouts(
        image_dir=obj_dir,
        bands=["rSDSS", "gSDSS"],
        out_path=out,
    )
    assert status == "no_clipped"
    assert not out.is_file()


def test_summarize_object_ok_after_make_cutouts(tmp_path: Path):
    """End-to-end: make_cutouts -> summarize writes the PNG and returns 'ok'."""
    images_root = tmp_path / "images"
    g = _make_galaxy()
    _write_layout(images_root, "obj_good", "1.0_2.0",
                  {"rSDSS": g, "gSDSS": g + 1.0})
    catalog = pd.DataFrame({"id": ["obj_good"]})
    make_cutouts_for_catalog(
        images_root=images_root, catalog=catalog,
        detect_band="rSDSS", apply_bands=["rSDSS", "gSDSS"],
    )

    out = tmp_path / "obj_good_summary.png"
    status = summarize_object_clipped_cutouts(
        image_dir=images_root / "obj_good",
        bands=["rSDSS", "gSDSS"],
        out_path=out,
    )
    assert status == "ok"
    assert out.is_file()


# ---------------------------------------------------------------------------
# cutouts.summarize_catalog_clipped_cutouts
# ---------------------------------------------------------------------------

def test_summarize_catalog_writes_one_png_per_object(tmp_path: Path):
    """Two-object catalog: one with clipped cutouts (-> ok PNG), one without (-> no_clipped, no file)."""
    images_root = tmp_path / "images"
    g = _make_galaxy()
    _write_layout(images_root, "withclip", "1.0_2.0", {"rSDSS": g, "gSDSS": g + 1})
    _write_layout(images_root, "noclip", "3.0_4.0", {"rSDSS": g})
    catalog = pd.DataFrame({"id": ["withclip", "noclip"]})
    # Only the first object gets clipped cutouts.
    make_cutouts_for_catalog(
        images_root=images_root,
        catalog=catalog.iloc[:1],
        detect_band="rSDSS",
        apply_bands=["rSDSS", "gSDSS"],
    )

    out_dir = tmp_path / "summaries"
    rows = summarize_catalog_clipped_cutouts(
        images_root=images_root,
        catalog=catalog,
        bands=["rSDSS", "gSDSS"],
        out_dir=out_dir,
    )

    by_id = {r["id"]: r for r in rows}
    assert by_id["withclip"]["status"] == "ok"
    assert Path(by_id["withclip"]["out_path"]).is_file()
    assert by_id["noclip"]["status"] == "no_clipped"
    assert by_id["noclip"]["out_path"] == ""


def test_summarize_catalog_handles_missing_object(tmp_path: Path):
    images_root = tmp_path / "images"
    images_root.mkdir()
    catalog = pd.DataFrame({"id": ["ghost"]})
    rows = summarize_catalog_clipped_cutouts(
        images_root=images_root,
        catalog=catalog,
        bands=["rSDSS"],
        out_dir=tmp_path / "summaries",
    )
    assert len(rows) == 1
    assert rows[0]["status"] == "missing_object"
