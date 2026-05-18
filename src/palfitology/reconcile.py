"""Cross-match fitted PAs against the input catalog.

Walks `fitted_pa_images/<id>/PA_fits.csv` for every object that has been
processed, joins against the catalog on `id`, and writes a single wide
summary CSV with the catalog `pa_jplus` and our per-band PAs side by side.

Position angle is direction-only (a 180° flip is the same axis), so we
compare using the circular angular separation:

    diff = min(|a - b| mod 180, 180 - |a - b| mod 180)

which is bounded in [0, 90].

This is intentionally a thin module: the heavy lifting is already in
`catalog.py`, and the output is a CSV the user can open in pandas / Excel
to spot disagreements.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from . import ALL_BANDS
from .catalog import load_catalog

logger = logging.getLogger(__name__)


def plot_reconciliation(
    table: pd.DataFrame,
    band: str,
    out_path: Path,
    title_suffix: str = "",
) -> None:
    """Scatter-plot catalog pa_jplus vs our fitted PA for one band (or median).

    ``band`` can be a canonical band name like ``rSDSS`` or the special string
    ``"median"`` for the per-object cross-band median.

    The plot includes a 1:1 line plus wrap-aware ``y = x ± 180`` dashed lines,
    since position angle is direction-only and a 180° flip is the same axis.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if band == "median":
        y_col = "pa_median_ok"
        y_label = "Our PA (per-object median of OK bands)"
    else:
        y_col = f"pa_{band}"
        y_label = f"Our PA ({band})"
        if y_col not in table.columns:
            raise ValueError(
                f"Column {y_col!r} not in reconciliation table. "
                f"Bands available: {[c[3:] for c in table.columns if c.startswith('pa_')]}"
            )

    x = table["pa_jplus_norm"].to_numpy()
    y = table[y_col].to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n_plotted = int(mask.sum())

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(x, y, s=18, alpha=0.75, edgecolors="black", linewidths=0.4)

    # 1:1 line and the two PA-wrap "shadow" lines.
    ax.plot([0, 180], [0, 180], "-", color="C3", lw=1.0, label="y = x (1:1)")
    ax.plot([0, 180], [180, 360], "--", color="C3", lw=0.7, alpha=0.6,
            label="y = x ± 180 (wrap-equivalent)")
    ax.plot([0, 180], [-180, 0], "--", color="C3", lw=0.7, alpha=0.6)

    ax.set_xlim(0, 180)
    ax.set_ylim(0, 180)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Catalog pa_jplus  (wrapped to [0, 180°))")
    ax.set_ylabel(y_label + "  (wrapped to [0, 180°))")

    title = f"PA reconciliation: catalog vs {band}"
    if title_suffix:
        title += f"  —  {title_suffix}"
    ax.set_title(title)

    # Median circular diff annotation (lower-right corner).
    diff_col = "pa_diff_median" if band == "median" else f"pa_diff_{band}"
    if diff_col in table.columns:
        diffs = table[diff_col].dropna().to_numpy()
        if len(diffs) > 0:
            ax.text(
                0.98, 0.03,
                f"N = {n_plotted}\n"
                f"median |Δ| = {np.median(diffs):.2f}°\n"
                f"68th pct |Δ| = {np.percentile(diffs, 68):.2f}°\n"
                f"max |Δ| = {np.max(diffs):.2f}°",
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7", pad=4),
            )

    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
    ax.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info(f"Wrote reconciliation scatter to {out_path}")


def _wrap_pa_0_180(deg: float) -> float:
    """Wrap a PA in degrees into [0, 180). Returns NaN on NaN input."""
    if deg is None or not np.isfinite(deg):
        return float("nan")
    return float(deg % 180.0)


def transform_pa_jplus(deg: float) -> float:
    """Convert J-PLUS catalog pa_jplus into photutils' image frame.

    photutils returns PA in degrees, CCW from +x (image-frame). J-PLUS /
    SExtractor THETA_J2000 is measured east of north (sky-frame), so the
    reflection-plus-shift rule (90 - pa_jplus) mod 180 moves it into
    photutils' frame. Empirically verified on the user's data: this collapses
    the reconciliation scatter onto y=x, whereas a plain `% 180` produces two
    parallel branches.

    NaN propagates.
    """
    if deg is None or not np.isfinite(deg):
        return float("nan")
    return float((90.0 - float(deg)) % 180.0)


def circular_diff_deg(a: float, b: float) -> float:
    """Smallest angular separation between two undirected axes, in [0, 90]."""
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    d = abs((a - b) % 180.0)
    return float(min(d, 180.0 - d))


def _gather_per_object_fits(fitted_dir: Path) -> pd.DataFrame:
    """Concatenate every fitted_pa_images/<id>/PA_fits.csv into one long DataFrame."""
    csvs = sorted(fitted_dir.glob("*/PA_fits.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No <id>/PA_fits.csv files found under {fitted_dir}. "
            f"Has `palfitology fit-pa` been run yet?"
        )
    frames = []
    for p in csvs:
        try:
            frames.append(pd.read_csv(p))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Skipping unreadable {p}: {e}")
    out = pd.concat(frames, ignore_index=True)
    logger.info(f"Loaded {len(out)} rows across {len(frames)} per-object CSVs from {fitted_dir}")
    return out


def reconcile(
    fitted_dir: Path,
    catalog_path: Path,
    output_path: Path,
    bands: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Build the wide reconciliation table and write it to ``output_path``.

    Parameters
    ----------
    fitted_dir : Path
        Top-level fitted_pa_images/ folder.
    catalog_path : Path
        Original catalog CSV.
    output_path : Path
        Where to write the summary CSV.
    bands : sequence of str, optional
        Bands to include as columns. Defaults to the canonical 12.

    Returns
    -------
    pd.DataFrame
        One row per object. Columns:
            id, pa_jplus, pa_<band> ... pa_median_ok, pa_diff_<band>,
            pa_diff_median, n_bands_ok, n_bands_weak, n_bands_missing
    """
    bands = list(bands) if bands is not None else list(ALL_BANDS)

    long_df = _gather_per_object_fits(fitted_dir)
    catalog = load_catalog(catalog_path)

    # Wrap our fitted PAs into [0, 180) since PA is direction-only.
    long_df["pa_norm"] = long_df["est_pa"].apply(_wrap_pa_0_180)

    # Wide table: one row per id, one column per band's PA.
    wide_pa = long_df.pivot_table(
        index="id", columns="band", values="pa_norm", aggfunc="first"
    ).rename(columns=lambda b: f"pa_{b}")

    # Per-object median of OK-only PAs. Using circular median is overkill for
    # PAs that are already in [0, 180): a plain median is fine as long as the
    # PAs cluster. We do a quick wrap-aware fallback: if the spread is large
    # (>90 deg), shift values < 90 up by 180 before taking the median, which
    # avoids the wraparound trap.
    medians = {}
    n_ok = {}
    n_weak = {}
    n_missing = {}
    for oid, group in long_df.groupby("id"):
        ok_vals = group.loc[group["status"] == "ok", "pa_norm"].dropna().to_numpy()
        n_ok[oid] = int(len(ok_vals))
        n_weak[oid] = int((group["status"] == "weak").sum())
        n_missing[oid] = int((group["status"] == "missing").sum())
        if len(ok_vals) == 0:
            medians[oid] = float("nan")
            continue
        if len(ok_vals) >= 2 and (ok_vals.max() - ok_vals.min()) > 90.0:
            shifted = np.where(ok_vals < 90.0, ok_vals + 180.0, ok_vals)
            medians[oid] = float(np.median(shifted) % 180.0)
        else:
            medians[oid] = float(np.median(ok_vals))

    # Convert catalog pa_jplus (sky-frame, N-through-E) into photutils' image
    # frame (CCW from +x) via (90 - x) mod 180. This is the empirically
    # correct mapping; the scatter then clusters on the y=x diagonal.
    cat = catalog[["id", "pa_jplus"]].copy()
    cat["pa_jplus_norm"] = cat["pa_jplus"].apply(transform_pa_jplus)

    out = (
        cat.set_index("id")
        .join(wide_pa, how="inner")
        .reset_index()
    )

    out["pa_median_ok"] = out["id"].map(medians)
    out["n_bands_ok"] = out["id"].map(n_ok).fillna(0).astype(int)
    out["n_bands_weak"] = out["id"].map(n_weak).fillna(0).astype(int)
    out["n_bands_missing"] = out["id"].map(n_missing).fillna(0).astype(int)

    # Circular diffs: our band vs catalog, and median vs catalog.
    for b in bands:
        col = f"pa_{b}"
        if col in out.columns:
            out[f"pa_diff_{b}"] = [
                circular_diff_deg(a, c)
                for a, c in zip(out[col], out["pa_jplus_norm"])
            ]
    out["pa_diff_median"] = [
        circular_diff_deg(a, c)
        for a, c in zip(out["pa_median_ok"], out["pa_jplus_norm"])
    ]

    # Friendly column order: id, catalog, our median + counts, then per-band.
    base_cols = [
        "id",
        "pa_jplus",
        "pa_jplus_norm",
        "pa_median_ok",
        "pa_diff_median",
        "n_bands_ok",
        "n_bands_weak",
        "n_bands_missing",
    ]
    per_band = []
    for b in bands:
        if f"pa_{b}" in out.columns:
            per_band.append(f"pa_{b}")
        if f"pa_diff_{b}" in out.columns:
            per_band.append(f"pa_diff_{b}")
    final_cols = base_cols + per_band

    out = out[final_cols].sort_values("id").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    logger.info(f"Wrote reconciliation table with {len(out)} rows to {output_path}")
    return out


def _add_reconcile_subparser(subparsers):
    """Wire `palfitology reconcile` into the CLI."""
    p = subparsers.add_parser(
        "reconcile",
        help="Compare fitted PAs against the input catalog's pa_jplus.",
        description=(
            "Walk fitted_pa_images/<id>/PA_fits.csv files, join against the "
            "input catalog on id, and write one wide CSV with the catalog "
            "pa_jplus alongside our per-band fitted PAs and circular diffs."
        ),
    )
    p.add_argument("--fitted-dir", type=Path, default=None,
                   help="fitted_pa_images/ folder (default: ./fitted_pa_images).")
    p.add_argument("--catalog", type=Path, default=None,
                   help="Catalog CSV. If omitted, auto-discover a single .csv in cwd.")
    p.add_argument("--output", type=Path, default=None,
                   help="Output summary CSV (default: ./fitted_pa_images/PA_reconciliation.csv).")
    p.add_argument("--bands", nargs="+", default=None,
                   help="Bands to include as columns (default: the canonical 12).")
    p.add_argument(
        "--plot",
        nargs="*",
        default=None,
        metavar="BAND",
        help=(
            "Also write scatter plots of catalog pa_jplus vs our fitted PA "
            "for the given band(s). Pass band names (e.g. --plot rSDSS iSDSS) "
            "or the special value 'median' for the per-object cross-band "
            "median. With no arguments (just --plot), defaults to "
            "'rSDSS iSDSS median'. Output PNGs go next to the summary CSV."
        ),
    )
    p.set_defaults(func=_cmd_reconcile)
    return p


def _cmd_reconcile(args) -> int:
    from .catalog import auto_discover_catalog

    cwd = Path.cwd()
    fitted_dir = args.fitted_dir or (cwd / "fitted_pa_images")
    if not fitted_dir.is_dir():
        logger.error(f"Fitted-images folder not found: {fitted_dir}")
        return 1

    if args.catalog is None:
        try:
            args.catalog = auto_discover_catalog(cwd)
        except (FileNotFoundError, ValueError) as e:
            logger.error(str(e))
            return 1
        logger.info(f"Auto-discovered catalog: {args.catalog.name}")

    output_path = args.output or (fitted_dir / "PA_reconciliation.csv")

    try:
        table = reconcile(
            fitted_dir=fitted_dir,
            catalog_path=args.catalog,
            output_path=output_path,
            bands=args.bands,
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    # Optional scatter plots.
    if args.plot is not None:
        plot_bands = args.plot if len(args.plot) > 0 else ["rSDSS", "iSDSS", "median"]
        for b in plot_bands:
            png_path = output_path.with_name(
                output_path.stem + f"_{b}.png"
            )
            try:
                plot_reconciliation(table=table, band=b, out_path=png_path)
            except ValueError as e:
                logger.warning(f"Skipping plot for {b}: {e}")

    return 0
