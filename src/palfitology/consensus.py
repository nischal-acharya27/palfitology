"""Cross-band PA consensus from a per-(object, band) results table.

V0.3 of palfitology. Reads the master ``PA_results.csv`` produced by
``palfitology fit-pa`` and computes one consensus row per object:

  * ``pa_consensus``       weighted circular mean of band PAs, in [0, 180)
  * ``pa_consensus_err``   propagated 1-sigma uncertainty on pa_consensus
  * ``ell_consensus``      weighted mean ellipticity
  * ``resultant_length``   R in [0, 1], how tightly bands agree
  * ``n_bands_used``       number of bands contributing to the mean
  * ``n_outliers``         number of bands flagged as discrepant
  * ``outlier_bands``      comma-separated list of those band names
  * ``status``             ok | low_confidence | failed

Weighting rule:

    w_i = est_ell_i^2 / pa_err_i^2

Round galaxies (ell ~ 0) contribute negligible weight even if pa_err is
small, because the PA of a near-circular isophote is intrinsically
ill-defined.

Bands eligible to contribute: status in {"ok", "weak"}. ``imputed`` and
``missing`` rows are dropped before averaging. ``psf_mode`` is not used as
a filter -- a 'deconv->raw_fallback' row is a successful fit on the raw
cutout and its pa_err is already representative.

Outlier rule: a band is flagged when the circular distance between its
PA and the consensus exceeds ``outlier_k * circ_std`` (default k=2).
Outlier flags are informational; they do not retroactively change the
consensus value in this version.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "circular_mean_deg",
    "circular_diff_deg",
    "circular_std_deg",
    "consensus_for_object",
    "consensus_for_catalog",
    "CONSENSUS_COLUMNS",
    "DEFAULT_OUTLIER_K",
    "DEFAULT_MIN_BANDS",
]


DEFAULT_OUTLIER_K = 2.0
DEFAULT_MIN_BANDS = 3


# ---------------------------------------------------------------------------
# Circular statistics (PA lives on the half-circle [0, 180))
# ---------------------------------------------------------------------------

def circular_mean_deg(
    angles_deg: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> tuple[float, float]:
    """Weighted circular mean of PAs on the half-circle [0, 180).

    Returns ``(mean_deg, R)`` where ``R`` is the resultant length in [0, 1].
    R close to 1 means the PAs are tightly clustered; R close to 0 means
    they're uniformly spread.

    PAs are direction-only (PA == PA + 180), so we double-angle them onto
    the full circle, take a vector mean, then halve back. This is the
    standard trick for axial / orientation data (Mardia & Jupp 2000, Sec 9.2).

    Returns ``(NaN, 0.0)`` on degenerate input (no finite angles or all
    weights zero).
    """
    a = np.asarray(angles_deg, dtype=float)
    if weights is None:
        w = np.ones_like(a)
    else:
        w = np.asarray(weights, dtype=float)
    if a.shape != w.shape:
        raise ValueError(
            f"angles and weights shape mismatch: {a.shape} vs {w.shape}"
        )

    mask = np.isfinite(a) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        return float("nan"), 0.0

    a = a[mask] % 180.0
    w = w[mask]
    if w.sum() <= 0.0:
        return float("nan"), 0.0

    # Double-angle, weight, take vector mean, halve back.
    theta = np.deg2rad(2.0 * a)
    s = np.sum(w * np.sin(theta))
    c = np.sum(w * np.cos(theta))
    r = np.hypot(s, c) / w.sum()
    if r == 0.0:
        # Perfectly antipodal -> mean undefined.
        return float("nan"), 0.0

    mean_double = np.arctan2(s, c)  # [-pi, pi]
    mean = (np.rad2deg(mean_double) / 2.0) % 180.0
    return float(mean), float(r)


def circular_diff_deg(a: float, b: float) -> float:
    """Smallest angular separation between two PAs on [0, 180), in [0, 90]."""
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    d = abs((a - b) % 180.0)
    return float(min(d, 180.0 - d))


def circular_std_deg(
    angles_deg: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Circular standard deviation on [0, 180), in degrees.

    Defined as ``sqrt(-2 * ln(R)) / 2``, mapped back from the doubled
    circle. Returns NaN for degenerate input; 0.0 when bands are
    perfectly aligned (R == 1).
    """
    _, r = circular_mean_deg(angles_deg, weights)
    if not np.isfinite(r) or r <= 0.0:
        return float("nan")
    if r >= 1.0:
        return 0.0
    std_doubled = np.sqrt(-2.0 * np.log(r))
    return float(np.rad2deg(std_doubled) / 2.0)


# ---------------------------------------------------------------------------
# Per-object consensus
# ---------------------------------------------------------------------------

def _compute_weights(est_ell: np.ndarray, pa_err: np.ndarray) -> np.ndarray:
    """w = est_ell^2 / pa_err^2; non-finite or non-positive -> 0."""
    e = np.asarray(est_ell, dtype=float)
    s = np.asarray(pa_err, dtype=float)
    w = np.zeros_like(e)
    ok = np.isfinite(e) & np.isfinite(s) & (s > 0) & (e > 0)
    w[ok] = (e[ok] ** 2) / (s[ok] ** 2)
    return w


def _empty_row(oid: str, status: str) -> dict:
    return {
        "id": oid,
        "pa_consensus": float("nan"),
        "pa_consensus_err": float("nan"),
        "ell_consensus": float("nan"),
        "resultant_length": 0.0,
        "n_bands_used": 0,
        "n_outliers": 0,
        "outlier_bands": "",
        "status": status,
    }


def consensus_for_object(
    rows: pd.DataFrame,
    outlier_k: float = DEFAULT_OUTLIER_K,
    min_bands: int = DEFAULT_MIN_BANDS,
) -> dict:
    """Compute the consensus row for one object's per-band table.

    Parameters
    ----------
    rows : pd.DataFrame
        Per-band rows for one object id. Must have columns
        ``band, est_pa, est_ell, pa_err, status``.
    outlier_k : float
        Bands whose circular distance from the consensus exceeds
        ``outlier_k * circ_std`` are flagged.
    min_bands : int
        If fewer than ``min_bands`` bands contribute, status='low_confidence'.
        If zero bands contribute, status='failed'.
    """
    oid = str(rows["id"].iloc[0]) if "id" in rows.columns and len(rows) else ""

    eligible = rows[rows["status"].isin(["ok", "weak"])].copy()
    if len(eligible) == 0:
        return _empty_row(oid, "failed")

    angles = eligible["est_pa"].to_numpy(dtype=float)
    ells   = eligible["est_ell"].to_numpy(dtype=float)
    errs   = eligible["pa_err"].to_numpy(dtype=float)
    bands  = eligible["band"].to_numpy()

    weights = _compute_weights(ells, errs)

    valid = weights > 0
    if not np.any(valid):
        return _empty_row(oid, "failed")

    angles  = angles[valid]
    ells    = ells[valid]
    errs    = errs[valid]
    bands   = bands[valid]
    weights = weights[valid]

    pa_mean, R = circular_mean_deg(angles, weights)
    cstd = circular_std_deg(angles, weights)

    # Weighted mean ellipticity (linear average).
    ell_mean = float(np.average(ells, weights=weights))

    # Propagated 1-sigma uncertainty on the weighted mean.
    # var(mean) = sum(w_i^2 * sigma_i^2) / (sum w_i)^2
    sum_w = float(weights.sum())
    var_mean = float(np.sum((weights ** 2) * (errs ** 2)) / (sum_w ** 2))
    pa_err_mean = float(np.sqrt(var_mean))

    # Outlier detection
    outlier_bands: list[str] = []
    if np.isfinite(cstd) and cstd > 0 and len(angles) >= 2:
        threshold = outlier_k * cstd
        deltas = np.array([circular_diff_deg(a, pa_mean) for a in angles])
        is_out = deltas > threshold
        outlier_bands = [str(b) for b, out in zip(bands, is_out) if out]

    n_used = int(len(angles))
    status = "ok" if n_used >= min_bands else "low_confidence"

    return {
        "id": oid,
        "pa_consensus": pa_mean,
        "pa_consensus_err": pa_err_mean,
        "ell_consensus": ell_mean,
        "resultant_length": float(R),
        "n_bands_used": n_used,
        "n_outliers": int(len(outlier_bands)),
        "outlier_bands": ",".join(outlier_bands),
        "status": status,
    }


CONSENSUS_COLUMNS = [
    "id",
    "pa_consensus",
    "pa_consensus_err",
    "ell_consensus",
    "resultant_length",
    "n_bands_used",
    "n_outliers",
    "outlier_bands",
    "status",
]


def consensus_for_catalog(
    results_df: pd.DataFrame,
    outlier_k: float = DEFAULT_OUTLIER_K,
    min_bands: int = DEFAULT_MIN_BANDS,
) -> pd.DataFrame:
    """Apply consensus_for_object across every id in a long-form PA_results table.

    Parameters
    ----------
    results_df : pd.DataFrame
        Long-form table with one row per (id, band). Must contain
        ``id, band, est_pa, est_ell, pa_err, status``.

    Returns
    -------
    pd.DataFrame
        One row per object, columns documented at ``CONSENSUS_COLUMNS``.
        Sorted by id.
    """
    required = {"id", "band", "est_pa", "est_ell", "pa_err", "status"}
    missing = required - set(results_df.columns)
    if missing:
        raise ValueError(
            f"PA_results DataFrame is missing required columns: {sorted(missing)}"
        )

    out_rows: list[dict] = []
    for _, group in results_df.groupby("id", sort=True):
        out_rows.append(
            consensus_for_object(group, outlier_k=outlier_k, min_bands=min_bands)
        )

    return pd.DataFrame(out_rows, columns=CONSENSUS_COLUMNS)


# ---------------------------------------------------------------------------
# CLI hook
# ---------------------------------------------------------------------------

def _add_consensus_subparser(subparsers):
    """Wire `palfitology consensus` into the CLI."""
    p = subparsers.add_parser(
        "consensus",
        help="Compute the cross-band PA consensus from PA_results.csv.",
        description=(
            "Read fitted_pa_images/PA_results.csv, compute one weighted "
            "circular-mean PA per object across all 12 bands, and write "
            "fitted_pa_images/PA_consensus.csv. Bands with low ellipticity "
            "or high pa_err contribute proportionally less. Bands more than "
            "k * circular_std from the consensus are flagged as outliers."
        ),
    )
    p.add_argument(
        "--fitted-dir", type=Path, default=None,
        help="fitted_pa_images/ folder (default: ./fitted_pa_images).",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Output CSV path (default: <fitted-dir>/PA_consensus.csv).",
    )
    p.add_argument(
        "--outlier-k", type=float, default=DEFAULT_OUTLIER_K,
        help=f"Outlier threshold in circular sigmas (default: {DEFAULT_OUTLIER_K}).",
    )
    p.add_argument(
        "--min-bands", type=int, default=DEFAULT_MIN_BANDS,
        help=(
            f"Mark consensus 'low_confidence' below this many bands "
            f"(default: {DEFAULT_MIN_BANDS})."
        ),
    )
    p.set_defaults(func=_cmd_consensus)
    return p


def _cmd_consensus(args) -> int:
    cwd = Path.cwd()
    fitted_dir = args.fitted_dir or (cwd / "fitted_pa_images")
    if not fitted_dir.is_dir():
        logger.error(f"Fitted-images folder not found: {fitted_dir}")
        return 1

    results_path = fitted_dir / "PA_results.csv"
    if not results_path.is_file():
        logger.error(
            f"PA_results.csv not found at {results_path}. "
            f"Run `palfitology fit-pa` first."
        )
        return 1

    output_path = args.output or (fitted_dir / "PA_consensus.csv")

    logger.info(f"Reading {results_path}")
    df = pd.read_csv(results_path)

    logger.info(
        f"Computing consensus (outlier_k={args.outlier_k}, "
        f"min_bands={args.min_bands}) over {df['id'].nunique()} objects, "
        f"{len(df)} rows"
    )
    out = consensus_for_catalog(
        df, outlier_k=args.outlier_k, min_bands=args.min_bands
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    logger.info(f"Wrote {len(out)} rows to {output_path}")

    if "status" in out.columns:
        counts = out["status"].value_counts().to_dict()
        logger.info(f"status distribution: {counts}")
    if "n_outliers" in out.columns:
        with_outliers = int((out["n_outliers"] > 0).sum())
        logger.info(
            f"objects with >=1 outlier band: {with_outliers}/{len(out)}"
        )
    return 0
