"""Diagnostic: does the new pa_corr transform actually align catalog and fitted PAs?

Reads ``fitted_pa_images/PA_reconciliation.csv`` and compares each band's
fitted PA (``pa_<band>``) against three candidate catalog transforms:

  * NEW (current code):   pa_corr = 180 - (pa_jplus + (pa_jplus<0)*180)
  * OLD (v0.2 code):      (90 - pa_jplus) % 180
  * NAIVE (no transform): pa_jplus % 180

For each band it reports the median circular |Δ| under each transform.
The winning transform has the smallest median |Δ|, ideally < 10 deg on
the deep broadbands.

Run from the cwd that contains fitted_pa_images/.
Usage:
    python ~/palfitology/scripts/check_pa_transform.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def circ_diff(a: pd.Series, b: pd.Series) -> pd.Series:
    d = (a - b) % 180.0
    return np.minimum(d, 180.0 - d)


def main() -> int:
    p = Path("fitted_pa_images/PA_reconciliation.csv")
    if not p.is_file():
        print(f"ERROR: {p} not found.", file=sys.stderr)
        return 1
    df = pd.read_csv(p)

    if "pa_jplus" not in df.columns:
        print("ERROR: pa_jplus column missing.", file=sys.stderr)
        return 1

    # Compute all three candidates.
    pj = df["pa_jplus"].astype(float)
    pj_new   = 180.0 - (pj + (pj < 0) * 180.0)         # current code
    pj_old   = (90.0 - pj) % 180.0                     # v0.2 code
    pj_naive = pj % 180.0                              # no transform at all

    bands = [c.replace("pa_", "", 1) for c in df.columns
             if c.startswith("pa_") and not c.startswith("pa_diff_")
             and c not in ("pa_jplus", "pa_jplus_norm", "pa_median_ok")]

    print(f"Loaded {len(df)} objects from {p}")
    print()
    print(f"{'band':6s}  {'n':>5s}  {'med|d| new':>10s}  {'med|d| old':>10s}  {'med|d| naive':>12s}  best")
    print("-" * 64)

    summary = {"new": [], "old": [], "naive": []}
    for b in bands:
        col = f"pa_{b}"
        if col not in df.columns:
            continue
        est = df[col].astype(float)
        m = est.notna() & pj.notna()
        n = int(m.sum())
        if n == 0:
            continue

        d_new   = float(np.nanmedian(circ_diff(est[m], pj_new[m])))
        d_old   = float(np.nanmedian(circ_diff(est[m], pj_old[m])))
        d_naive = float(np.nanmedian(circ_diff(est[m], pj_naive[m])))

        which = min([("new", d_new), ("old", d_old), ("naive", d_naive)], key=lambda x: x[1])
        winner = which[0]
        summary[winner].append(b)

        print(f"{b:6s}  {n:>5d}  {d_new:>9.1f}°  {d_old:>9.1f}°  {d_naive:>11.1f}°  {winner}")

    print()
    print("Winning transform per band:")
    for k, v in summary.items():
        print(f"  {k:5s}: {len(v):2d} bands  -> {', '.join(v) if v else '(none)'}")

    # Final verdict
    print()
    n_total = sum(len(v) for v in summary.values())
    if n_total == 0:
        print("No bands had data to compare.")
        return 0
    new_share = len(summary["new"]) / n_total * 100
    if new_share >= 80:
        print(f"VERDICT: the new transform wins on {new_share:.0f}% of bands. Ship it.")
    elif new_share >= 50:
        print(f"VERDICT: the new transform wins on {new_share:.0f}% of bands -- better than "
              "the old one but not uniformly. Investigate the bands where another transform wins.")
    else:
        print(f"VERDICT: the new transform wins on only {new_share:.0f}% of bands. "
              "Either the convention varies across bands (unlikely), or there is a "
              "subtler issue. Investigate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
