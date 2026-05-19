"""Quick PA-flip diagnostic.

Compares each band's est_pa against pa_jplus directly vs. against (180 -
pa_jplus). Both are mod-180 circular differences in [0, 90]. Prints, per
band, the median |Δ| in each frame and the fraction of rows for which the
"flipped" version is a better match.

Interpretation:
- 'flip wins' near 0% in every band -> est_pa and pa_jplus already agree on
  convention; the catalog and our fit live in the same frame.
- 'flip wins' near 100% in every band -> a single global 180° offset; trivial
  to fix by negating one side.
- 'flip wins' near 50% scattered -> the original split observation; this is
  the WCS-heterogeneity story and needs a WCS-aware fix in fit.py.
- Mixed numbers per band -> band-specific issue worth digging into.

Run from the directory that contains fitted_pa_images/PA_reconciliation.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def circ_diff_mod_180(a: pd.Series, b: pd.Series) -> pd.Series:
    """Smallest angular distance between a, b on the half-circle [0, 180)."""
    d = (a - b) % 180.0
    return np.minimum(d, 180.0 - d)


def main() -> None:
    path = Path("fitted_pa_images/PA_reconciliation.csv")
    if not path.is_file():
        raise SystemExit(
            f"{path} not found. Run `palfitology reconcile --plot --catalog <cat>.csv` first."
        )

    df = pd.read_csv(path)
    # The reconciliation CSV produced by `palfitology reconcile` uses
    # 'pa_<band>' for the per-band fitted PA and 'pa_diff_<band>' for the
    # already-computed difference. We use the raw per-band column and
    # recompute the difference in two frames here.
    band_cols = [
        c for c in df.columns
        if c.startswith("pa_") and not c.startswith("pa_diff_")
        and c not in ("pa_jplus", "pa_jplus_norm", "pa_median_ok")
    ]
    if not band_cols:
        raise SystemExit("No pa_<band> columns in reconciliation CSV.")
    if "pa_jplus" not in df.columns:
        raise SystemExit("pa_jplus column missing.")

    print(f"Loaded {len(df)} rows from {path}")
    print()
    print(f"{'band':6s}  {'n':>6s}  {'med|d|direct':>13s}  "
          f"{'med|d|flipped':>14s}  {'flip wins %':>11s}")
    print("-" * 60)

    pa_jplus = df["pa_jplus"]
    for col in band_cols:
        band = col.replace("pa_", "", 1)
        est = df[col]

        d_dir = circ_diff_mod_180(est, pa_jplus)
        d_flip = circ_diff_mod_180(est, 180.0 - pa_jplus)

        mask = (~d_dir.isna()) & (~d_flip.isna())
        n = int(mask.sum())
        if n == 0:
            continue

        med_dir = float(np.nanmedian(d_dir[mask]))
        med_flip = float(np.nanmedian(d_flip[mask]))
        flip_better_pct = float((d_flip[mask] < d_dir[mask]).mean() * 100)

        print(f"{band:6s}  {n:>6d}  {med_dir:>12.1f}°  {med_flip:>13.1f}°  "
              f"{flip_better_pct:>10.1f}%")


if __name__ == "__main__":
    main()
