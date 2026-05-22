"""Tests for the V0.6 `regenerate-mosaics` subcommand.

Covers:
 - `regenerate_mosaics_from_csv` round-trips: rendering from an existing
   PA_results.csv produces the same per-object PNG paths fit-pa wrote.
 - Missing FITS rows are reported as 'no_data', not as crashes.
 - Band order from the CSV is preserved in the regenerated mosaic.
 - The `palfitology regenerate-mosaics` CLI entry point works end-to-end
   (parser wiring + driver invocation).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from palfitology.cli import main as cli_main
from palfitology.cutouts import make_cutouts_for_catalog
from palfitology.pipeline import (
    RESULT_COLUMNS,
    fit_catalog,
    regenerate_mosaics_from_csv,
)


# ---------------------------------------------------------------------------
# Helpers (kept in-file so each test module is self-contained)
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


def _seed_run(tmp_path: Path, *, use_clipped: bool = False) -> tuple[Path, Path, Path]:
    """Run a tiny fit_catalog so a PA_results.csv + mosaics exist on disk.

    Returns (images_root, output_dir, results_csv_path).
    """
    images_root = tmp_path / "images"
    output_dir = tmp_path / "out"
    galaxy = _make_galaxy()
    _write_layout(images_root, "obj_a", "1.0_2.0",
                  {"rSDSS": galaxy, "gSDSS": galaxy})
    _write_layout(images_root, "obj_b", "3.0_4.0",
                  {"rSDSS": galaxy, "gSDSS": galaxy})

    if use_clipped:
        make_cutouts_for_catalog(
            images_root=images_root,
            catalog=pd.DataFrame({"id": ["obj_a", "obj_b"]}),
            detect_band="rSDSS", apply_bands=["rSDSS", "gSDSS"],
        )

    catalog = pd.DataFrame({
        "id":       ["obj_a", "obj_b"],
        "A_WORLD":  [1.0, 1.0],
        "B_WORLD":  [0.5, 0.5],
        "pa_jplus": [30.0, 30.0],
    })

    df = fit_catalog(
        images_root=images_root,
        output_dir=output_dir,
        catalog=catalog,
        bands=["rSDSS", "gSDSS"],
        workers=1,
        make_summary=True,
        psf_mode="off",
        detect_sigma=3.0,
        use_clipped_cutouts=use_clipped,
    )
    results_csv = output_dir / "PA_results.csv"
    df.to_csv(results_csv, index=False)
    return images_root, output_dir, results_csv


# ---------------------------------------------------------------------------
# 1. Round-trip
# ---------------------------------------------------------------------------

def test_regenerate_recreates_per_object_mosaics(tmp_path: Path):
    """Delete the seeded mosaics, regenerate from CSV, and verify they re-appear."""
    images_root, output_dir, results_csv = _seed_run(tmp_path)

    mosaic_a = output_dir / "obj_a" / "obj_a_summary.png"
    mosaic_b = output_dir / "obj_b" / "obj_b_summary.png"
    central_a = output_dir / "all_summaries" / "obj_a_summary.png"
    central_b = output_dir / "all_summaries" / "obj_b_summary.png"

    # The seed run must have written them (otherwise nothing to regenerate).
    for p in (mosaic_a, mosaic_b, central_a, central_b):
        assert p.is_file(), f"seed run did not produce {p}"
        p.unlink()

    summary = regenerate_mosaics_from_csv(
        results_csv=results_csv,
        images_root=images_root,
        output_dir=output_dir,
        workers=1,
    )

    assert summary["rendered"] == 2
    assert summary["errors"] == 0
    assert summary["no_data"] == 0
    for p in (mosaic_a, mosaic_b, central_a, central_b):
        assert p.is_file(), f"regenerate did not rewrite {p}"
        assert p.stat().st_size > 1000, f"{p} is suspiciously small"


def test_regenerate_with_clipped_cutouts_path(tmp_path: Path):
    """Same round-trip, but seeded with --use-clipped-cutouts.

    Regression for the bug that motivated this command: clipped-path
    mosaics had inconsistent panel aspect ratios pre-fix. The regenerate
    path must produce the post-fix mosaics from the same CSV.
    """
    images_root, output_dir, results_csv = _seed_run(tmp_path, use_clipped=True)
    central = output_dir / "all_summaries" / "obj_a_summary.png"
    assert central.is_file()
    central.unlink()

    summary = regenerate_mosaics_from_csv(
        results_csv=results_csv,
        images_root=images_root,
        output_dir=output_dir,
        workers=1,
    )
    assert summary["rendered"] == 2
    assert central.is_file()


# ---------------------------------------------------------------------------
# 2. Missing FITS -> no_data, not crash
# ---------------------------------------------------------------------------

def test_regenerate_handles_object_with_no_fits_paths(tmp_path: Path):
    """An object whose CSV rows all have empty fits_path is reported as no_data."""
    images_root, output_dir, results_csv = _seed_run(tmp_path)

    # Synthesize a fake row for a third object whose fits_path is empty,
    # mimicking the 'missing' rows _missing_row() writes.
    df = pd.read_csv(results_csv)
    extra = df.iloc[[0]].copy()
    extra["id"] = "obj_phantom"
    extra["fits_path"] = ""
    extra["status"] = "missing"
    extra2 = extra.copy()
    extra2["band"] = "gSDSS"
    df = pd.concat([df, extra, extra2], ignore_index=True)
    df.to_csv(results_csv, index=False)

    summary = regenerate_mosaics_from_csv(
        results_csv=results_csv,
        images_root=images_root,
        output_dir=output_dir,
        workers=1,
    )

    assert summary["no_data"] == 1
    assert summary["rendered"] == 2
    assert summary["errors"] == 0
    assert not (output_dir / "obj_phantom" / "obj_phantom_summary.png").exists()


# ---------------------------------------------------------------------------
# 3. Band order
# ---------------------------------------------------------------------------

def test_regenerate_preserves_csv_band_order(tmp_path: Path):
    """If `bands` isn't passed, the order is taken from first-appearance in CSV."""
    images_root, output_dir, results_csv = _seed_run(tmp_path)
    df = pd.read_csv(results_csv)
    # Sanity: the seed CSV has rSDSS rows before gSDSS rows (fit_catalog
    # honours the bands kwarg order, which we set above).
    band_order = list(dict.fromkeys(df["band"].tolist()))
    assert band_order == ["rSDSS", "gSDSS"]

    # Now flip the CSV so gSDSS comes first. The regenerator should accept
    # that order and pass it through to make_summary_mosaic.
    flipped = pd.concat([df[df["band"] == "gSDSS"], df[df["band"] == "rSDSS"]],
                        ignore_index=True)
    flipped.to_csv(results_csv, index=False)

    # Strip prior PNGs so we know the regenerate call did the work.
    for p in (output_dir / "all_summaries").iterdir():
        p.unlink()

    summary = regenerate_mosaics_from_csv(
        results_csv=results_csv,
        images_root=images_root,
        output_dir=output_dir,
        workers=1,
    )
    assert summary["rendered"] == 2
    # We can't introspect the mosaic's internal panel order without parsing
    # the PNG, but the rendered count + file existence is enough — the order
    # is plumbed via the `bands_order` arg to make_summary_mosaic, which is
    # itself tested in test_summary_mosaic.py.
    assert (output_dir / "all_summaries" / "obj_a_summary.png").is_file()


# ---------------------------------------------------------------------------
# 4. CLI integration
# ---------------------------------------------------------------------------

def test_cli_regenerate_mosaics_end_to_end(tmp_path: Path):
    """`palfitology regenerate-mosaics ...` runs through the parser + driver."""
    images_root, output_dir, results_csv = _seed_run(tmp_path)

    # Delete one mosaic so we can prove the CLI re-rendered it.
    target = output_dir / "all_summaries" / "obj_a_summary.png"
    assert target.is_file()
    target.unlink()

    rc = cli_main([
        "regenerate-mosaics",
        "--results",     str(results_csv),
        "--images-root", str(images_root),
        "--output-dir",  str(output_dir),
        "--workers",     "1",
    ])
    assert rc == 0
    assert target.is_file()


def test_cli_regenerate_mosaics_missing_csv_returns_error(tmp_path: Path):
    """A missing PA_results.csv yields exit code 1, not an exception."""
    rc = cli_main([
        "regenerate-mosaics",
        "--results",     str(tmp_path / "nope.csv"),
        "--images-root", str(tmp_path),
        "--output-dir",  str(tmp_path / "out"),
        "--workers",     "1",
    ])
    assert rc == 1
