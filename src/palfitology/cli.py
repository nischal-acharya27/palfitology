"""Command-line interface for palfitology.

Exposes subcommands, one per pipeline stage:

    palfitology fit-pa        -- per-band isophotal PA fit + diagnostics
    palfitology make-cutouts  -- sigma-clipped cutouts (mask from rSDSS, applied
                                  to one or more bands) written next to originals
    palfitology download      -- (planned) fetch J-PLUS cutouts + PSFs
    palfitology consensus     -- (planned) cross-band PA consensus + flagging
    palfitology galfit        -- (planned) emit GALFIT input files

Entry point is `main`; the `palfitology` console script (from
`pyproject.toml`) calls it.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# IMPORTANT: pin BLAS / OpenMP to 1 thread BEFORE the heavy imports happen.
# When the user later spawns a pool of workers, each worker also runs this
# block via the package-init path, but doing it here in the parent too
# prevents the parent from importing numpy with 40 threads baked in.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

from . import ALL_BANDS, __version__  # noqa: E402 -- after BLAS pin
from .detect import DEFAULT_CLIP_DILATE, DEFAULT_DETECT_BAND, DEFAULT_DETECT_SIGMA  # noqa: E402
from .catalog import (  # noqa: E402
    auto_discover_catalog,
    filter_to_existing_image_dirs,
    load_catalog,
)
from .consensus import _add_consensus_subparser  # noqa: E402
from .cutouts import make_cutouts_for_catalog  # noqa: E402
from .pipeline import fit_catalog  # noqa: E402
from .reconcile import _add_reconcile_subparser  # noqa: E402

logger = logging.getLogger("palfitology")


def _add_fit_pa_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "fit-pa",
        help="Fit isophotal position angle for every (object, band) in a catalog.",
        description=(
            "Run the per-band isophotal PA fit on every object in the catalog, "
            "writing per-band PNGs, a 3x4 summary mosaic per object, and a "
            "PA_results.csv with one row per (object, band)."
        ),
    )
    p.add_argument("--images-root", type=Path, default=None,
                   help="Folder containing one subfolder per object (default: ./images).")
    p.add_argument("--catalog", type=Path, default=None,
                   help="Input catalog CSV. If omitted, auto-discover a single .csv in cwd.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Where to write outputs (default: ./fitted_pa_images).")
    p.add_argument("--bands", nargs="+", default=ALL_BANDS,
                   help=f"List of bands to fit (default: all 12 = {' '.join(ALL_BANDS)}).")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N objects (0 = all).")
    p.add_argument("--all", action="store_true",
                   help="Shortcut for --limit 0.")
    p.add_argument("--min-sma-abs", type=float, default=3.0,
                   help="Absolute lower bound on selected SMA, in pixels (default: 3).")
    p.add_argument("--min-sma-frac", type=float, default=0.05,
                   help="Fractional lower bound on selected SMA (default: 0.05).")
    p.add_argument("--keep-best-of", type=int, default=8,
                   help="Collect up to N strong fits before picking the best (default: 8).")
    p.add_argument("--workers", type=int, default=1,
                   help="Number of parallel worker processes. 0 = os.cpu_count().")
    p.add_argument("--no-summary", action="store_true",
                   help="Skip the per-object 3x4 summary mosaic.")
    p.add_argument("--psf-mode", choices=["auto", "on", "off"], default="auto",
                   help=(
                       "PSF preprocessing mode (default: auto). "
                       "'auto' deconvolves only when PSF FWHM >= --psf-gate * R_EFF; "
                       "'on' always deconvolves when a PSF is available; "
                       "'off' disables PSF preprocessing (v0.1.0 behaviour)."
                   ))
    p.add_argument("--psf-gate", type=float, default=0.2,
                   help=(
                       "Threshold ratio (PSF FWHM / R_EFF) above which the "
                       "auto mode deconvolves (default: 0.2). Lower = more "
                       "aggressive deconvolution."
                   ))
    p.add_argument("--detect-sigma", type=float, default=DEFAULT_DETECT_SIGMA,
                   help=(
                       f"Sigma threshold for source detection in the detect-band image "
                       f"(default: {DEFAULT_DETECT_SIGMA}). Set to 0 to disable detection "
                       f"and revert to catalog-prior seeding (v0.3 behaviour)."
                   ))
    p.add_argument("--detect-band", type=str, default=DEFAULT_DETECT_BAND,
                   help=(
                       f"Band used as the detection master image (default: {DEFAULT_DETECT_BAND}). "
                       f"Its sigma-clipped mask seeds the ellipse geometry for all bands."
                   ))
    p.add_argument("--debug", action="store_true",
                   help="Verbose per-attempt logging.")
    p.set_defaults(func=_cmd_fit_pa)


def _cmd_fit_pa(args: argparse.Namespace) -> int:
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.workers == 0:
        args.workers = os.cpu_count() or 1

    cwd = Path.cwd()

    images_root = args.images_root or (cwd / "images")
    output_dir = args.output_dir or (cwd / "fitted_pa_images")

    if args.catalog is None:
        try:
            args.catalog = auto_discover_catalog(cwd)
        except (FileNotFoundError, ValueError) as e:
            logger.error(str(e))
            return 1
        logger.info(f"Auto-discovered catalog: {args.catalog.name}")

    if not args.catalog.is_file():
        logger.error(f"Catalog not found: {args.catalog}")
        return 1
    if not images_root.is_dir():
        logger.error(f"Images root not found: {images_root}")
        return 1

    unknown_bands = [b for b in args.bands if b not in ALL_BANDS]
    if unknown_bands:
        logger.warning(
            f"Bands not in the canonical J-PLUS list: {unknown_bands}. "
            f"palfitology will still look for <band>_cutout.fits files matching those names."
        )

    try:
        df = load_catalog(args.catalog)
    except ValueError as e:
        logger.error(str(e))
        return 1

    df = filter_to_existing_image_dirs(df, images_root)

    limit = 0 if args.all else args.limit
    if limit and limit > 0:
        df = df.head(limit)
        logger.info(f"Limiting to first {limit} objects for this run")

    results = fit_catalog(
        images_root=images_root,
        output_dir=output_dir,
        catalog=df,
        bands=args.bands,
        min_sma_abs=args.min_sma_abs,
        min_sma_frac=args.min_sma_frac,
        keep_best_of=args.keep_best_of,
        workers=args.workers,
        make_summary=not args.no_summary,
        psf_mode=args.psf_mode,
        psf_gate=args.psf_gate,
        detect_sigma=args.detect_sigma,
        detect_band=args.detect_band,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_out = output_dir / "PA_results.csv"
    results.to_csv(csv_out, index=False)
    logger.info(f"Wrote {len(results)} rows to {csv_out}")
    return 0


def _add_make_cutouts_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "make-cutouts",
        help="Generate sigma-clipped FITS cutouts from a detect band (default rSDSS).",
        description=(
            "Run sigma-clipped detection on the detect-band (default rSDSS) "
            "image for every object in the catalog, then apply the resulting "
            "mask to one or more bands to produce new FITS cutouts where "
            "pixels outside the detected galaxy are NaN.  The new cutouts are "
            "written to a sibling 'clipped_cutouts_<ra>_<dec>/' folder next "
            "to the originals; the originals are never modified.  Run with "
            "--apply-bands rSDSS for the first sanity check, then expand to "
            "the full 12-band list once the masks look right."
        ),
    )
    p.add_argument("--images-root", type=Path, default=None,
                   help="Folder containing one subfolder per object (default: ./images).")
    p.add_argument("--catalog", type=Path, default=None,
                   help="Input catalog CSV. If omitted, auto-discover a single .csv in cwd.")
    p.add_argument("--detect-band", type=str, default=DEFAULT_DETECT_BAND,
                   help=f"Band used to build the mask (default: {DEFAULT_DETECT_BAND}).")
    p.add_argument("--apply-bands", nargs="+", default=None,
                   help=(
                       "Bands the mask is applied to (default: --detect-band only). "
                       "Pass 'all' to apply to every canonical J-PLUS band, or "
                       "list explicit band names."
                   ))
    p.add_argument("--detect-sigma", type=float, default=DEFAULT_DETECT_SIGMA,
                   help=f"Sigma threshold above background (default: {DEFAULT_DETECT_SIGMA}).")
    p.add_argument("--dilate", type=int, default=DEFAULT_CLIP_DILATE,
                   help=(
                       f"Dilate the binary mask by N pixels before applying "
                       f"(default: {DEFAULT_CLIP_DILATE}). Use 1-3 to give the "
                       f"isophote fitter breathing room around the source edge."
                   ))
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N objects (0 = all).")
    p.add_argument("--all", action="store_true", help="Shortcut for --limit 0.")
    p.add_argument("--no-overwrite", action="store_true",
                   help="Skip objects whose clipped cutout already exists.")
    p.add_argument("--report", type=Path, default=None,
                   help=(
                       "Where to write the per-(id, band) report CSV (default: "
                       "./make_cutouts_report.csv)."
                   ))
    p.add_argument("--debug", action="store_true", help="Verbose logging.")
    p.set_defaults(func=_cmd_make_cutouts)


def _cmd_make_cutouts(args: argparse.Namespace) -> int:
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    cwd = Path.cwd()

    images_root = args.images_root or (cwd / "images")

    if args.catalog is None:
        try:
            args.catalog = auto_discover_catalog(cwd)
        except (FileNotFoundError, ValueError) as e:
            logger.error(str(e))
            return 1
        logger.info(f"Auto-discovered catalog: {args.catalog.name}")

    if not args.catalog.is_file():
        logger.error(f"Catalog not found: {args.catalog}")
        return 1
    if not images_root.is_dir():
        logger.error(f"Images root not found: {images_root}")
        return 1

    # Resolve --apply-bands.
    if args.apply_bands is None:
        apply_bands = [args.detect_band]
        logger.info(
            f"--apply-bands not given -- defaulting to detect-band only "
            f"({args.detect_band}). Pass --apply-bands all once you trust the masks."
        )
    elif len(args.apply_bands) == 1 and args.apply_bands[0].lower() == "all":
        apply_bands = list(ALL_BANDS)
    else:
        apply_bands = list(args.apply_bands)

    unknown_bands = [b for b in apply_bands if b not in ALL_BANDS]
    if unknown_bands:
        logger.warning(
            f"Bands not in the canonical J-PLUS list: {unknown_bands}. "
            f"make-cutouts will still look for <band>_cutout.fits matching those names."
        )

    try:
        df = load_catalog(args.catalog)
    except ValueError as e:
        logger.error(str(e))
        return 1

    df = filter_to_existing_image_dirs(df, images_root)

    limit = 0 if args.all else args.limit
    if limit and limit > 0:
        df = df.head(limit)
        logger.info(f"Limiting to first {limit} objects for this run")

    logger.info(
        f"make-cutouts: {len(df)} objects, detect-band={args.detect_band}, "
        f"apply-bands={apply_bands}, sigma={args.detect_sigma}, dilate={args.dilate}"
    )

    reports = make_cutouts_for_catalog(
        images_root=images_root,
        catalog=df,
        detect_band=args.detect_band,
        apply_bands=apply_bands,
        sigma_threshold=args.detect_sigma,
        dilate=args.dilate,
        overwrite=not args.no_overwrite,
    )

    # Summarise + write the report CSV.
    import pandas as pd  # local import keeps top-of-file slim

    report_df = pd.DataFrame([r.__dict__ for r in reports])
    report_path = args.report or (cwd / "make_cutouts_report.csv")
    report_df.to_csv(report_path, index=False)

    n_ok = int((report_df["status"] == "ok").sum()) if len(report_df) else 0
    n_total = len(report_df)
    logger.info(
        f"make-cutouts complete: {n_ok}/{n_total} (id, band) entries written. "
        f"Report -> {report_path}"
    )
    return 0


def _stub_subparser(subparsers: argparse._SubParsersAction, name: str, status: str) -> None:
    p = subparsers.add_parser(name, help=f"({status}) -- not yet implemented")
    p.set_defaults(func=lambda _args: _stub_run(name))


def _stub_run(name: str) -> int:
    logger.error(
        f"The '{name}' subcommand is planned but not yet implemented. "
        f"See the project roadmap in docs/pipeline.md."
    )
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="palfitology",
        description="PSF-aware isophotal PA fitting and GALFIT prep for J-PLUS cutouts.",
    )
    parser.add_argument("--version", action="version", version=f"palfitology {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_fit_pa_subparser(subparsers)
    _add_make_cutouts_subparser(subparsers)
    _add_reconcile_subparser(subparsers)
    _add_consensus_subparser(subparsers)
    _stub_subparser(subparsers, "download", "planned")
    _stub_subparser(subparsers, "galfit", "planned")

    return parser


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
