"""Command-line interface for palfitology.

Exposes subcommands, one per pipeline stage:

    palfitology fit-pa     -- per-band isophotal PA fit + diagnostics
    palfitology download   -- (planned) fetch J-PLUS cutouts + PSFs
    palfitology consensus  -- (planned) cross-band PA consensus + flagging
    palfitology galfit     -- (planned) emit GALFIT input files

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
from .detect import DEFAULT_DETECT_BAND, DEFAULT_DETECT_SIGMA  # noqa: E402
from .catalog import (  # noqa: E402
    auto_discover_catalog,
    filter_to_existing_image_dirs,
    load_catalog,
)
from .consensus import _add_consensus_subparser  # noqa: E402
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
