"""Tests for palfitology.galfit.

Covers:
 - PA convention conversion (photutils +x -> GALFIT +y) and wrapping
 - ellipticity -> axis-ratio mapping and clamping
 - build_params_for_object: center 0->1-based shift, skips, R_e fallback
 - build_all_params join on the science band
 - render_feedme content (priors land in the right lines)
 - run_galfit graceful failure when the binary is absent
 - CLI integration with --no-run
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from astropy.io import fits

from palfitology.galfit import (
    DEFAULT_MAGZP,
    DEFAULT_PIXSCALE,
    GalfitParams,
    build_all_params,
    build_params_for_object,
    ell_to_axis_ratio,
    pa_photutils_to_galfit,
    render_feedme,
    run_galfit,
    write_feedme_files,
)


# ---------------------------------------------------------------------------
# Convention conversions
# ---------------------------------------------------------------------------

def test_pa_conversion_adds_90_and_wraps():
    assert pa_photutils_to_galfit(0.0) == pytest.approx(90.0)
    # 100 + 90 = 190 -> wraps to -170
    assert pa_photutils_to_galfit(100.0) == pytest.approx(-170.0)
    assert pa_photutils_to_galfit(90.0) == pytest.approx(180.0)


def test_pa_conversion_nan():
    assert np.isnan(pa_photutils_to_galfit(float("nan")))


def test_ell_to_axis_ratio_basic():
    assert ell_to_axis_ratio(0.0) == pytest.approx(1.0)
    assert ell_to_axis_ratio(0.4) == pytest.approx(0.6)


def test_ell_to_axis_ratio_clamps_and_defaults():
    assert ell_to_axis_ratio(2.0) == pytest.approx(0.05)   # q would be -1
    assert ell_to_axis_ratio(-1.0) == pytest.approx(1.0)   # q would be 2
    assert ell_to_axis_ratio(float("nan")) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_cutout(tmp_path) -> Path:
    """A 60x40 (ny x nx) FITS image so nx=40, ny=60 are distinguishable."""
    path = tmp_path / "obj1_rSDSS.fits"
    fits.PrimaryHDU(data=np.zeros((60, 40), dtype="float32")).writeto(path)
    return path


def _consensus_row(oid="obj1", pa=30.0, ell=0.4, status="ok"):
    return pd.Series(
        {"id": oid, "pa_consensus": pa, "ell_consensus": ell, "status": status}
    )


def _science_row(oid="obj1", fits_path="", x0=19.0, y0=29.0, sma=8.0):
    return pd.Series(
        {"id": oid, "band": "rSDSS", "fits_path": fits_path,
         "est_pa": 31.0, "est_sma": sma, "est_ell": 0.4, "x0": x0, "y0": y0}
    )


# ---------------------------------------------------------------------------
# build_params_for_object
# ---------------------------------------------------------------------------

def test_build_params_happy_path(fake_cutout, tmp_path):
    crow = _consensus_row()
    srow = _science_row(fits_path=str(fake_cutout), x0=19.0, y0=29.0, sma=8.0)
    gp = build_params_for_object(
        "obj1", crow, srow, output_dir=tmp_path / "out",
    )
    assert gp is not None
    assert gp.img_nx == 40 and gp.img_ny == 60
    # 0-based -> 1-based
    assert gp.x0 == pytest.approx(20.0)
    assert gp.y0 == pytest.approx(30.0)
    # PA 30 -> 120 ; q = 1-0.4 = 0.6
    assert gp.pa_galfit == pytest.approx(120.0)
    assert gp.axis_ratio == pytest.approx(0.6)
    # R_e seeded from sma
    assert gp.r_eff == pytest.approx(8.0)
    assert gp.magzp == DEFAULT_MAGZP
    assert gp.pixscale == DEFAULT_PIXSCALE


def test_build_params_r_eff_fallback(fake_cutout, tmp_path):
    srow = _science_row(fits_path=str(fake_cutout), sma=float("nan"))
    gp = build_params_for_object(
        "obj1", _consensus_row(), srow, output_dir=tmp_path,
    )
    assert gp.r_eff == pytest.approx(40 / 4.0)  # nx/4


def test_build_params_skips_failed_consensus(fake_cutout, tmp_path):
    crow = _consensus_row(status="failed")
    srow = _science_row(fits_path=str(fake_cutout))
    assert build_params_for_object("obj1", crow, srow, output_dir=tmp_path) is None


def test_build_params_skips_missing_science_row(tmp_path):
    assert build_params_for_object(
        "obj1", _consensus_row(), None, output_dir=tmp_path
    ) is None


def test_build_params_skips_empty_fits_path(tmp_path):
    srow = _science_row(fits_path="")
    assert build_params_for_object(
        "obj1", _consensus_row(), srow, output_dir=tmp_path
    ) is None


# ---------------------------------------------------------------------------
# build_all_params join
# ---------------------------------------------------------------------------

def test_build_all_params_joins_on_science_band(fake_cutout, tmp_path):
    consensus_df = pd.DataFrame([
        {"id": "obj1", "pa_consensus": 30.0, "ell_consensus": 0.4, "status": "ok"},
    ])
    results_df = pd.DataFrame([
        {"id": "obj1", "band": "gSDSS", "fits_path": "ignore.fits",
         "est_pa": 1, "est_sma": 5, "est_ell": 0.3, "x0": 1, "y0": 1},
        {"id": "obj1", "band": "rSDSS", "fits_path": str(fake_cutout),
         "est_pa": 31, "est_sma": 8, "est_ell": 0.4, "x0": 19, "y0": 29},
    ])
    params = build_all_params(consensus_df, results_df, output_dir=tmp_path)
    assert len(params) == 1
    assert params[0].input_image == fake_cutout


def test_build_all_params_missing_column_raises(tmp_path):
    consensus_df = pd.DataFrame([{"id": "x"}])
    results_df = pd.DataFrame([{"id": "x", "band": "rSDSS",
                                "fits_path": "", "x0": 1, "y0": 1}])
    with pytest.raises(ValueError):
        build_all_params(consensus_df, results_df, output_dir=tmp_path)


# ---------------------------------------------------------------------------
# render_feedme + write
# ---------------------------------------------------------------------------

def _params(tmp_path) -> GalfitParams:
    return GalfitParams(
        object_id="obj1",
        input_image=Path("in.fits"),
        output_image=Path("out.fits"),
        x0=20.0, y0=30.0, mag=18.0, r_eff=8.0, sersic_n=2.5,
        axis_ratio=0.6, pa_galfit=120.0,
        magzp=23.0, pixscale=0.2627, img_nx=40, img_ny=60,
    )


def test_render_feedme_contains_priors(tmp_path):
    text = render_feedme(_params(tmp_path))
    assert "sersic" in text
    assert "1 40 1 60" in text          # fit region from nx,ny
    assert "0.6000" in text             # axis ratio
    assert "120.000" in text            # PA
    assert "20.000 30.000" in text      # center
    assert "23.0000" in text            # magzp


def test_write_feedme_files(tmp_path):
    out = tmp_path / "galfit_inputs"
    written = write_feedme_files([_params(tmp_path)], out)
    assert len(written) == 1
    assert written[0].name == "obj1.feedme"
    assert written[0].read_text().strip().endswith("=" * 80)


# ---------------------------------------------------------------------------
# run_galfit
# ---------------------------------------------------------------------------

def test_run_galfit_missing_binary(tmp_path):
    feedme = tmp_path / "x.feedme"
    feedme.write_text("dummy")
    ok, msg = run_galfit(feedme, galfit_bin="definitely-not-a-real-binary-xyz")
    assert ok is False
    assert "not found" in msg


# ---------------------------------------------------------------------------
# CLI integration (no GALFIT binary needed)
# ---------------------------------------------------------------------------

def test_cli_galfit_no_run(fake_cutout, tmp_path, monkeypatch):
    from palfitology.cli import main

    fitted = tmp_path / "fitted_pa_images"
    fitted.mkdir()
    pd.DataFrame([
        {"id": "obj1", "pa_consensus": 30.0, "ell_consensus": 0.4,
         "status": "ok"},
    ]).to_csv(fitted / "PA_consensus.csv", index=False)
    pd.DataFrame([
        {"id": "obj1", "band": "rSDSS", "fits_path": str(fake_cutout),
         "est_pa": 31, "est_sma": 8, "est_ell": 0.4, "x0": 19, "y0": 29},
    ]).to_csv(fitted / "PA_results.csv", index=False)

    monkeypatch.chdir(tmp_path)
    rc = main([
        "galfit", "--fitted-dir", str(fitted),
        "--output-dir", str(tmp_path / "galfit_inputs"), "--no-run",
    ])
    assert rc == 0
    assert (tmp_path / "galfit_inputs" / "obj1.feedme").is_file()
