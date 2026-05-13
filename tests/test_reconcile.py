"""Tests for palfitology.reconcile."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from palfitology.reconcile import (
    _wrap_pa_0_180,
    circular_diff_deg,
    plot_reconciliation,
    reconcile,
)


def test_wrap_pa_handles_negative_and_large():
    assert _wrap_pa_0_180(-10.0) == 170.0
    assert _wrap_pa_0_180(200.0) == 20.0
    assert _wrap_pa_0_180(0.0) == 0.0


def test_wrap_pa_handles_nan():
    assert np.isnan(_wrap_pa_0_180(float("nan")))


def test_circular_diff_zero_when_identical():
    assert circular_diff_deg(45.0, 45.0) == 0.0


def test_circular_diff_handles_wraparound():
    # 5 deg and 175 deg are only 10 deg apart on the undirected axis.
    assert abs(circular_diff_deg(5.0, 175.0) - 10.0) < 1e-9


def test_circular_diff_caps_at_90():
    # Two perpendicular axes => 90 deg apart, the max.
    assert abs(circular_diff_deg(0.0, 90.0) - 90.0) < 1e-9


def test_circular_diff_nan_propagates():
    assert np.isnan(circular_diff_deg(float("nan"), 30.0))


def test_reconcile_writes_expected_columns(tmp_path: Path):
    # Set up a minimal fitted_pa_images/<id>/PA_fits.csv layout.
    fitted = tmp_path / "fitted_pa_images"
    obj_dir = fitted / "92801-11428"
    obj_dir.mkdir(parents=True)

    pd.DataFrame([
        {"id": "92801-11428", "band": "rSDSS", "est_pa": 15.0, "status": "ok",
         "est_sma": 50.0, "est_ell": 0.7, "fits_path": "/tmp/x.fits"},
        {"id": "92801-11428", "band": "iSDSS", "est_pa": 14.5, "status": "ok",
         "est_sma": 51.0, "est_ell": 0.7, "fits_path": "/tmp/x.fits"},
        {"id": "92801-11428", "band": "uJAVA", "est_pa": 100.0, "status": "weak",
         "est_sma": 1.0, "est_ell": 0.9, "fits_path": "/tmp/x.fits"},
    ]).to_csv(obj_dir / "PA_fits.csv", index=False)

    catalog_path = tmp_path / "cat.csv"
    catalog_path.write_text(
        "id,A_WORLD,B_WORLD,pa_jplus\n"
        "92801-11428,0.013,0.011,15.5\n"
    )

    output_path = fitted / "PA_reconciliation.csv"
    out = reconcile(
        fitted_dir=fitted,
        catalog_path=catalog_path,
        output_path=output_path,
    )

    assert output_path.is_file()
    assert list(out["id"]) == ["92801-11428"]

    # Median over ok rows only: median(15.0, 14.5) = 14.75
    assert abs(out["pa_median_ok"].iloc[0] - 14.75) < 1e-9

    # pa_jplus = 15.5, our median = 14.75 -> diff ~0.75
    assert abs(out["pa_diff_median"].iloc[0] - 0.75) < 1e-9

    # Status counts
    assert out["n_bands_ok"].iloc[0] == 2
    assert out["n_bands_weak"].iloc[0] == 1
    assert out["n_bands_missing"].iloc[0] == 0

    # Per-band columns exist
    assert "pa_rSDSS" in out.columns
    assert "pa_diff_iSDSS" in out.columns


def test_plot_reconciliation_writes_png(tmp_path: Path):
    """plot_reconciliation should produce a non-empty PNG without crashing."""
    table = pd.DataFrame({
        "id": ["a", "b", "c"],
        "pa_jplus": [10.0, 45.0, 90.0],
        "pa_jplus_norm": [10.0, 45.0, 90.0],
        "pa_rSDSS": [11.0, 44.0, 175.0],            # last one wraps near 0
        "pa_diff_rSDSS": [1.0, 1.0, 5.0],
        "pa_median_ok": [10.5, 44.5, 175.0],
        "pa_diff_median": [0.5, 0.5, 5.0],
    })
    out = tmp_path / "scatter.png"
    plot_reconciliation(table=table, band="rSDSS", out_path=out)
    assert out.is_file()
    assert out.stat().st_size > 1000  # plausible PNG, not zero bytes
