"""Tests for palfitology.selection.

We don't have photutils involved here -- we fake the iso_table API with a
small `MockTable` since `select_isophote` only reads three columns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from palfitology.selection import select_isophote


@dataclass
class _Col:
    """Mimic astropy Column's `.data` attribute."""
    data: np.ndarray

    def __getitem__(self, i):
        return self.data[i]


class MockTable:
    """Tiny stand-in for astropy.Table with the three columns we read."""

    def __init__(self, sma, pa_err, ndata=None):
        self._cols = {
            "sma": _Col(np.asarray(sma, dtype=float)),
            "pa_err": _Col(np.asarray(pa_err, dtype=float)),
        }
        if ndata is not None:
            self._cols["ndata"] = _Col(np.asarray(ndata, dtype=float))

    def __getitem__(self, name):
        return self._cols[name]

    @property
    def colnames(self):
        return list(self._cols.keys())


def test_picks_lowest_score_above_floor():
    # sma >= 5 floor (half-width=100, frac=0.05 -> 5 px; abs floor 3 -> 5 px wins).
    # Two surviving rows: sma=10 pa_err=2 (score=0.2), sma=50 pa_err=5 (score=0.1).
    t = MockTable(
        sma=[0.5, 10, 50],
        pa_err=[0.0, 2.0, 5.0],
        ndata=[10, 10, 10],
    )
    idx, weak = select_isophote(t, image_half_width=100.0)
    assert idx == 2
    assert weak is False


def test_drops_pa_err_zero_dummy_row():
    # The pa_err=0 row would otherwise win by score (0/anything = 0).
    t = MockTable(
        sma=[20, 30],
        pa_err=[0.0, 2.0],
        ndata=[10, 10],
    )
    idx, weak = select_isophote(t, image_half_width=100.0)
    assert idx == 1


def test_face_on_collapse_returns_weak_outermost():
    # All rows are below the SMA floor (5 px); none should be "strong".
    # Weak fallback should return the largest-sma sane row.
    t = MockTable(
        sma=[0.5, 1.0, 2.0, 3.0],
        pa_err=[0.0, 0.3, 0.4, 0.5],
        ndata=[10, 10, 10, 10],
    )
    idx, weak = select_isophote(t, image_half_width=100.0)
    assert weak is True
    assert idx == 3  # outermost sane row


def test_drops_undersampled_isophotes():
    # ndata < 5 rows should be ignored even if they'd score well.
    t = MockTable(
        sma=[10, 50],
        pa_err=[0.1, 5.0],
        ndata=[3, 10],   # first row under-sampled -> dropped
    )
    idx, _ = select_isophote(t, image_half_width=100.0)
    assert idx == 1


def test_returns_none_when_nothing_is_sane():
    # All pa_err non-finite -> nothing usable.
    t = MockTable(
        sma=[10, 20],
        pa_err=[np.nan, np.inf],
        ndata=[10, 10],
    )
    idx, weak = select_isophote(t, image_half_width=100.0)
    assert idx is None
    assert weak is False


def test_floor_is_max_of_abs_and_frac():
    # half_width=1000, frac=0.05 -> 50 px > abs floor 3 px. Floor = 50.
    t = MockTable(
        sma=[20, 60],
        pa_err=[0.5, 1.0],
        ndata=[10, 10],
    )
    idx, weak = select_isophote(t, image_half_width=1000.0, min_sma_abs=3.0, min_sma_frac=0.05)
    assert idx == 1  # only row above the 50px floor
    assert weak is False


def test_floor_overridable():
    # If we drop the fractional floor, the smaller-sma row can become eligible.
    t = MockTable(
        sma=[20, 60],
        pa_err=[0.5, 1.0],
        ndata=[10, 10],
    )
    idx, weak = select_isophote(t, image_half_width=1000.0, min_sma_abs=3.0, min_sma_frac=0.0)
    # Now both are sane; lower score wins (0.5/20 = 0.025 vs 1.0/60 ≈ 0.0167)
    assert idx == 1
