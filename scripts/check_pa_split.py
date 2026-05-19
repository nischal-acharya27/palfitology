"""PA convention split: tile-clustering + ellipticity-dependence diagnostics.

Reads ``fitted_pa_images/PA_reconciliation.csv`` and
``fitted_pa_images/PA_results.csv`` from the current working directory and
prints two diagnostic tables, both keyed off rSDSS (the deepest broadband and
the cleanest reference).

Step 1 -- Is the flip per-tile?
    For every object, decide whether ``180 - pa_jplus`` matches our fitted
    rSDSS PA better than ``pa_jplus`` itself. Aggregate by tile
    (the part of the id before the dash). If different tiles have
    consistently 0% or 100% flip fractions, the convention split is driven
    by heterogeneous CDELT2 / WCS orientation between cutout tiles.

Step 2 -- Is the flip ellipticity-dependent?
    For nearly round galaxies (ellipticity ~ 0), PA is intrinsically
    ambiguous and the fitter will pick whichever 180-degree direction the
    noise happens to favor. If the 'doesn't-flip' minority concentrates in
    the lowest ellipticity bins, that's the geometry, not a code bug.

Run from the directory that contains ``fitted_pa_images/``.

Usage:
    python ~/palfitology/scripts/check_pa_split.py

Optional:
    python ~/palfitology/scripts/check_pa_split.py --band iSDSS
    python ~/palfitology/scripts/check_pa_split.py --reconciliation-path some/PA_reconciliation.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def circ_diff_mod_180(a: pd.Series, b: pd.Series) -> pd.Series:
    """Smallest angular distance between a, b on the half-circle [0, 180)."""
    d = (a - b) % 180.0
    return np.minimum(d, 180.0 - d)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--reconciliation-path",
        type=Path,
        default=Path("fitted_pa_images/PA_reconciliation.csv"),
    )
    ap.add_argument(
        "--results-path",
        type=Path,
        default=Path("fitted_pa_images/PA_results.csv"),
    )
    ap.add_argument(
        "--band",
        default="rSDSS",
        help="Band column to use (default: rSDSS).",
    )
    args = ap.parse_args()

    # ---------- Load ----------
    if not args.reconciliation_path.is_file():
        print(f"ERROR: {args.reconciliation_path} not found.", file=sys.stderr)
        return 1
    if not args.results_path.is_file():
        print(f"ERROR: {args.results_path} not found.", file=sys.stderr)
        return 1

    rec = pd.read_csv(args.reconciliation_path)
    res = pd.read_csv(args.results_path)

    band_col = f"pa_{args.band}"
    if band_col not in rec.columns:
        print(f"ERROR: column {band_col!r} not in reconciliation CSV.", file=sys.stderr)
        return 1
    if "pa_jplus" not in rec.columns:
        print("ERROR: pa_jplus column missing from reconciliation CSV.", file=sys.stderr)
        return 1

    # ---------- Common: compute needs_flip on the chosen band ----------
    d_direct = circ_diff_mod_180(rec[band_col], rec["pa_jplus"])
    d_flip = circ_diff_mod_180(rec[band_col], 180.0 - rec["pa_jplus"])
    rec["needs_flip"] = (d_flip < d_direct).astype(int)
    rec["valid"] = rec[band_col].notna() & rec["pa_jplus"].notna()

    valid = rec[rec["valid"]].copy()
    n_valid = len(valid)
    print(f"Using band: {args.band}  ({band_col})")
    print(f"Rows valid: {n_valid} / {len(rec)}")
    print(f"Overall flip-wins fraction: {valid['needs_flip'].mean() * 100:.1f}%")
    print()

    # =====================================================================
    # Step 1 -- per-tile flip fraction
    # =====================================================================
    print("=" * 64)
    print("Step 1 -- per-tile flip fraction (rSDSS)")
    print("=" * 64)

    valid["tile"] = valid["id"].astype(str).str.split("-").str[0]
    per_tile = (
        valid.groupby("tile")["needs_flip"]
        .agg(["sum", "count", "mean"])
        .rename(columns={"sum": "n_flip", "count": "n", "mean": "frac_flip"})
        .sort_values("n", ascending=False)
    )

    n_tiles = len(per_tile)
    print(f"Tiles with data: {n_tiles}")
    print()
    print("Distribution of per-tile flip fractions:")
    print(per_tile["frac_flip"].describe().to_string())
    print()

    bins = pd.cut(
        per_tile["frac_flip"],
        bins=[-0.001, 0.05, 0.25, 0.5, 0.75, 0.95, 1.001],
        labels=["<5%", "5-25%", "25-50%", "50-75%", "75-95%", ">=95%"],
    )
    print("Tiles bucketed by flip fraction:")
    counts = bins.value_counts().sort_index()
    for label, c in counts.items():
        pct = c / n_tiles * 100
        bar = "#" * int(round(pct / 2))
        print(f"  {str(label):>8s}  n={c:4d}  ({pct:5.1f}%)  {bar}")
    print()
    print("Top 5 most-flipped tiles (rows in ascending tile size):")
    print(per_tile.sort_values("frac_flip", ascending=False).head(5).to_string())
    print()
    print("Top 5 least-flipped tiles:")
    print(per_tile.sort_values("frac_flip", ascending=True).head(5).to_string())
    print()

    # Interpretation hint
    extremes = ((per_tile["frac_flip"] < 0.05) | (per_tile["frac_flip"] > 0.95)).sum()
    extreme_pct = extremes / n_tiles * 100
    print(f"Tiles whose flip fraction is bimodal-extreme (<5% or >=95%): "
          f"{extremes}/{n_tiles} ({extreme_pct:.1f}%)")
    if extreme_pct >= 70:
        print("  --> Suggestive of per-tile WCS heterogeneity (CDELT2 sign varies).")
    elif extreme_pct < 30:
        print("  --> Flip distribution is smeared across tiles, not bimodal.")
        print("      The split is NOT primarily explained by WCS heterogeneity.")
    else:
        print("  --> Mixed signal; both mechanisms may contribute.")
    print()

    # =====================================================================
    # Step 2 -- flip fraction by galaxy ellipticity
    # =====================================================================
    print("=" * 64)
    print("Step 2 -- flip fraction vs galaxy ellipticity (rSDSS)")
    print("=" * 64)

    if "est_ell" not in res.columns or "status" not in res.columns:
        print("WARNING: est_ell / status missing in PA_results.csv -- skipping Step 2.",
              file=sys.stderr)
        return 0

    ell = (
        res[res["status"] == "ok"]
        .groupby("id")["est_ell"]
        .median()
        .rename("med_ell")
    )
    valid = valid.merge(ell, left_on="id", right_index=True, how="left")
    valid_e = valid[valid["med_ell"].notna()].copy()

    print(f"Rows with finite median ellipticity: {len(valid_e)} / {len(valid)}")
    print()
    print("Ellipticity distribution:")
    print(valid_e["med_ell"].describe().to_string())
    print()

    valid_e["ell_bin"] = pd.cut(
        valid_e["med_ell"],
        bins=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
        labels=["[0.00,0.05)", "[0.05,0.10)", "[0.10,0.20)",
                "[0.20,0.30)", "[0.30,0.50)", "[0.50,1.00]"],
    )
    grouped = valid_e.groupby("ell_bin", observed=True)["needs_flip"].agg(
        ["mean", "count"]
    )
    grouped["frac_flip_pct"] = grouped["mean"] * 100
    grouped = grouped[["count", "frac_flip_pct"]]
    print("flip fraction by ellipticity bin (lower ell = rounder galaxy):")
    print()
    for ellb, row in grouped.iterrows():
        pct = row["frac_flip_pct"]
        bar = "#" * int(round(pct / 2))
        n = int(row["count"])
        print(f"  ell {str(ellb):>14s}  n={n:5d}  flip={pct:5.1f}%  {bar}")
    print()

    # Interpretation hint
    if len(grouped) >= 3:
        low = grouped.iloc[0]["frac_flip_pct"]
        high = grouped.iloc[-1]["frac_flip_pct"]
        delta = high - low
        print(f"Δ (highest-ell bin) - (lowest-ell bin) = {delta:+.1f} pp")
        if delta > 15:
            print("  --> Flip strongly rises with ellipticity.")
            print("      The non-flippers concentrate in low-ellipticity (round) galaxies,")
            print("      where PA is intrinsically noise-dominated. That's geometry, not a bug.")
        elif abs(delta) < 5:
            print("  --> Flip fraction is flat across ellipticity.")
            print("      Ellipticity does not explain the 70/30 split; look elsewhere (Step 1).")
        else:
            print("  --> Modest ellipticity trend; ellipticity is a partial explanation only.")
    print()

    # =====================================================================
    # Summary verdict
    # =====================================================================
    print("=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"  Overall flip rate (rSDSS): {valid['needs_flip'].mean()*100:.1f}%")
    print(f"  Tile-extreme fraction:     {extreme_pct:.1f}%  (>=70% suggests WCS heterogeneity)")
    if "med_ell" in valid.columns:
        low_ell = valid[valid["med_ell"] < 0.1]
        high_ell = valid[valid["med_ell"] >= 0.3]
        if len(low_ell) and len(high_ell):
            print(f"  Low-ell flip rate (ell<0.1):  {low_ell['needs_flip'].mean()*100:.1f}%  "
                  f"({len(low_ell)} obj)")
            print(f"  High-ell flip rate (ell>=0.3): {high_ell['needs_flip'].mean()*100:.1f}%  "
                  f"({len(high_ell)} obj)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
