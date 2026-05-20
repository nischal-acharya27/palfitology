"""Tests for palfitology.consensus.

Covers:
 - circular mean correctness, including the 0/180 wrap edge case
 - circular std behaviour at R=1 (no spread) and R<1
 - ellipticity damping in the weight rule
 - per-object consensus_for_object happy path + degenerate inputs
 - consensus_for_catalog across multiple objects
 - outlier detection threshold
 - min_bands low-confidence flag
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from palfitology.consensus import (
    CONSENSUS_COLUMNS,
    DEFAULT_OUTLIER_K,
    circular_diff_deg,
    circular_mean_deg,
    circular_std_deg,
    consensus_for_catalog,
    consensus_for_object,
)


# ---------------------------------------------------------------------------
# Circular statistics
# ---------------------------------------------------------------------------

def test_circular_mean_identical_angles():
    a = np.array([42.0, 42.0, 42.0])
    mean, r = circular_mean_deg(a)
    assert abs(mean - 42.0) < 1e-6
    assert r == pytest.approx(1.0)


def test_circular_mean_wraps_across_zero():
    # PAs at 5 and 175 should average near 0/180, not at 90.
    # On the [0, 180) half-circle, 5 and 175 are only 10 deg apart.
    a = np.array([5.0, 175.0])
    mean, r = circular_mean_deg(a)
    # Mean should be near 0 (or equivalently 180), NOT 90.
    diff_to_zero = min(abs(mean), abs(180 - mean))
    assert diff_to_zero < 1.0, f"got mean={mean}, expected near 0 or 180"
    assert r > 0.9, "tight cluster should have R close to 1"


def test_circular_mean_orthogonal_pair_low_R():
    # PAs at 0 and 90 are maximally far apart on the half-circle.
    mean, r = circular_mean_deg(np.array([0.0, 90.0]))
    assert r < 0.01, f"orthogonal PAs should have near-zero R; got {r}"


def test_circular_mean_weighted_skews_toward_heavy_band():
    # One PA at 10, one at 80; heavy weight on the 10 -> mean near 10.
    a = np.array([10.0, 80.0])
    w = np.array([100.0, 1.0])
    mean, _ = circular_mean_deg(a, w)
    assert abs(mean - 10.0) < 2.0


def test_circular_mean_empty_returns_nan():
    mean, r = circular_mean_deg(np.array([np.nan, np.nan]))
    assert np.isnan(mean)
    assert r == 0.0


def test_circular_mean_zero_weights_returns_nan():
    mean, r = circular_mean_deg(np.array([10.0, 20.0]), np.array([0.0, 0.0]))
    assert np.isnan(mean)


def test_circular_diff_deg_wrap():
    assert circular_diff_deg(5.0, 175.0) == pytest.approx(10.0, abs=1e-6)
    assert circular_diff_deg(0.0, 90.0) == pytest.approx(90.0, abs=1e-6)
    assert circular_diff_deg(0.0, 89.999) < 90.0
    assert np.isnan(circular_diff_deg(np.nan, 10.0))


def test_circular_std_is_zero_when_R_is_one():
    # With perfectly identical angles R should be ~1; std should be ~0
    # (sub-microdegree numerical floor, not exact zero).
    s = circular_std_deg(np.array([42.0, 42.0, 42.0]))
    assert s < 1e-5


def test_circular_std_grows_with_spread():
    tight  = circular_std_deg(np.array([40.0, 42.0, 44.0]))
    loose  = circular_std_deg(np.array([20.0, 50.0, 80.0]))
    assert loose > tight


# ---------------------------------------------------------------------------
# consensus_for_object
# ---------------------------------------------------------------------------

def _mk_rows(rows):
    return pd.DataFrame(rows, columns=["id", "band", "est_pa", "est_ell", "pa_err", "status"])


def test_object_consensus_happy_path():
    df = _mk_rows([
        ("X1", "rSDSS", 45.0, 0.6, 0.5, "ok"),
        ("X1", "iSDSS", 46.0, 0.55, 0.5, "ok"),
        ("X1", "gSDSS", 44.0, 0.6, 0.5, "ok"),
        ("X1", "zSDSS", 45.5, 0.58, 0.5, "ok"),
    ])
    row = consensus_for_object(df)
    assert row["id"] == "X1"
    assert row["status"] == "ok"
    assert row["n_bands_used"] == 4
    assert abs(row["pa_consensus"] - 45.0) < 1.0
    assert row["resultant_length"] > 0.99
    assert row["n_outliers"] == 0


def test_object_consensus_drops_imputed_and_missing():
    df = _mk_rows([
        ("X1", "rSDSS", 45.0, 0.6, 0.5, "ok"),
        ("X1", "iSDSS", 46.0, 0.6, 0.5, "ok"),
        ("X1", "uJAVA", 30.0, 0.3, 0.5, "ok"),  # legit OK band, must count
        ("X1", "J0378", 10.0, np.nan, np.nan, "imputed"),  # drop
        ("X1", "J0395", np.nan, np.nan, np.nan, "missing"),  # drop
    ])
    row = consensus_for_object(df)
    assert row["n_bands_used"] == 3
    assert row["status"] == "ok"


def test_object_consensus_includes_weak():
    df = _mk_rows([
        ("X1", "rSDSS", 40.0, 0.6, 0.5, "ok"),
        ("X1", "iSDSS", 41.0, 0.6, 0.5, "ok"),
        ("X1", "gSDSS", 42.0, 0.4, 1.0, "weak"),
    ])
    row = consensus_for_object(df)
    assert row["n_bands_used"] == 3, "weak rows should contribute"


def test_object_consensus_low_confidence_below_min_bands():
    df = _mk_rows([
        ("X1", "rSDSS", 40.0, 0.6, 0.5, "ok"),
        ("X1", "iSDSS", 41.0, 0.6, 0.5, "ok"),  # only 2 bands
    ])
    row = consensus_for_object(df, min_bands=3)
    assert row["status"] == "low_confidence"
    assert row["n_bands_used"] == 2


def test_object_consensus_round_galaxy_band_contributes_nothing():
    # gSDSS has ell ~ 0; should be effectively zero-weight.
    df = _mk_rows([
        ("X1", "rSDSS", 60.0, 0.6,  0.5, "ok"),
        ("X1", "iSDSS", 60.0, 0.6,  0.5, "ok"),
        ("X1", "gSDSS",  5.0, 0.001, 0.5, "ok"),  # round, would otherwise pull mean down
        ("X1", "zSDSS", 60.0, 0.6,  0.5, "ok"),
    ])
    row = consensus_for_object(df)
    assert abs(row["pa_consensus"] - 60.0) < 1.0, "round band should not skew the mean"


def test_object_consensus_all_imputed_returns_failed():
    df = _mk_rows([
        ("X1", "rSDSS", np.nan, np.nan, np.nan, "imputed"),
        ("X1", "iSDSS", np.nan, np.nan, np.nan, "imputed"),
    ])
    row = consensus_for_object(df)
    assert row["status"] == "failed"
    assert row["n_bands_used"] == 0
    assert np.isnan(row["pa_consensus"])


def test_object_consensus_zero_weight_inputs_failed():
    # ok status but pa_err == 0 or ell == 0 -> weight 0
    df = _mk_rows([
        ("X1", "rSDSS", 40.0, 0.0,  0.5, "ok"),  # ell=0
        ("X1", "iSDSS", 41.0, 0.5,  0.0, "ok"),  # pa_err=0
    ])
    row = consensus_for_object(df)
    assert row["status"] == "failed"


def test_object_consensus_outlier_flagged():
    # Three bands agree on PA=40, one wild at PA=130.
    # The wild band is well above both the 2-sigma threshold AND the 5-deg floor.
    df = _mk_rows([
        ("X1", "rSDSS", 40.0, 0.6, 0.5, "ok"),
        ("X1", "iSDSS", 41.0, 0.6, 0.5, "ok"),
        ("X1", "gSDSS", 39.0, 0.6, 0.5, "ok"),
        ("X1", "J0378", 130.0, 0.6, 0.5, "ok"),
    ])
    row = consensus_for_object(df, outlier_k=DEFAULT_OUTLIER_K)
    assert "J0378" in row["outlier_bands"].split(",")
    assert row["n_outliers"] >= 1


def test_object_consensus_tight_cluster_suppressed_by_floor():
    # Five bands at PA ~ 42, one at PA = 45. Within the sample, the 45 deg
    # band is statistically far (low circ_std on the others), but only 3 deg
    # from the mean -- below the default 5 deg floor. Should NOT be flagged.
    df = _mk_rows([
        ("X1", "rSDSS", 42.0, 0.6, 0.5, "ok"),
        ("X1", "iSDSS", 42.0, 0.6, 0.5, "ok"),
        ("X1", "gSDSS", 42.0, 0.6, 0.5, "ok"),
        ("X1", "zSDSS", 42.0, 0.6, 0.5, "ok"),
        ("X1", "J0660", 42.0, 0.6, 0.5, "ok"),
        ("X1", "J0395", 45.0, 0.6, 0.5, "ok"),  # 3 deg from mean
    ])
    row = consensus_for_object(df, outlier_k=DEFAULT_OUTLIER_K, min_outlier_deg=5.0)
    assert row["n_outliers"] == 0, (
        f"3-deg deviation should NOT exceed the 5-deg floor; got "
        f"outliers={row['outlier_bands']}"
    )


def test_object_consensus_floor_can_be_disabled():
    # With min_outlier_deg=0, the old behaviour is restored: tight clusters
    # generate spurious outliers.
    df = _mk_rows([
        ("X1", "rSDSS", 42.0, 0.6, 0.5, "ok"),
        ("X1", "iSDSS", 42.0, 0.6, 0.5, "ok"),
        ("X1", "gSDSS", 42.0, 0.6, 0.5, "ok"),
        ("X1", "zSDSS", 42.0, 0.6, 0.5, "ok"),
        ("X1", "J0660", 42.0, 0.6, 0.5, "ok"),
        ("X1", "J0395", 45.0, 0.6, 0.5, "ok"),
    ])
    row = consensus_for_object(df, outlier_k=DEFAULT_OUTLIER_K, min_outlier_deg=0.0)
    assert "J0395" in row["outlier_bands"].split(",")


def test_object_consensus_pa_err_propagation_falls_with_n():
    # Doubling the number of identical bands should roughly half the mean uncertainty.
    rows_1 = _mk_rows([
        ("X", "rSDSS", 45.0, 0.5, 1.0, "ok"),
        ("X", "iSDSS", 45.0, 0.5, 1.0, "ok"),
    ])
    rows_2 = _mk_rows([
        ("X", "rSDSS", 45.0, 0.5, 1.0, "ok"),
        ("X", "iSDSS", 45.0, 0.5, 1.0, "ok"),
        ("X", "gSDSS", 45.0, 0.5, 1.0, "ok"),
        ("X", "zSDSS", 45.0, 0.5, 1.0, "ok"),
        ("X", "J0660", 45.0, 0.5, 1.0, "ok"),
        ("X", "J0515", 45.0, 0.5, 1.0, "ok"),
        ("X", "J0410", 45.0, 0.5, 1.0, "ok"),
        ("X", "J0430", 45.0, 0.5, 1.0, "ok"),
    ])
    r1 = consensus_for_object(rows_1)
    r2 = consensus_for_object(rows_2)
    assert r2["pa_consensus_err"] < r1["pa_consensus_err"]


# ---------------------------------------------------------------------------
# consensus_for_catalog
# ---------------------------------------------------------------------------

def test_catalog_consensus_multiple_objects():
    df = _mk_rows([
        ("A", "rSDSS", 30.0, 0.6, 0.5, "ok"),
        ("A", "iSDSS", 31.0, 0.6, 0.5, "ok"),
        ("A", "gSDSS", 29.0, 0.6, 0.5, "ok"),
        ("B", "rSDSS", 120.0, 0.5, 0.5, "ok"),
        ("B", "iSDSS", 121.0, 0.5, 0.5, "ok"),
        ("B", "gSDSS", 119.0, 0.5, 0.5, "ok"),
    ])
    out = consensus_for_catalog(df)
    assert list(out.columns) == CONSENSUS_COLUMNS
    assert len(out) == 2
    rowA = out[out["id"] == "A"].iloc[0]
    rowB = out[out["id"] == "B"].iloc[0]
    assert abs(rowA["pa_consensus"] - 30.0) < 1.0
    assert abs(rowB["pa_consensus"] - 120.0) < 1.0


def test_catalog_consensus_missing_columns_raises():
    df = pd.DataFrame({"id": ["A"], "band": ["rSDSS"]})  # missing est_pa, etc.
    with pytest.raises(ValueError, match="missing required columns"):
        consensus_for_catalog(df)
