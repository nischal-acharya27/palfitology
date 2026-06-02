"""GALFIT input-file writer and runner.

V0.7 of palfitology. Turns the cross-band PA consensus into ready-to-run
single-Sersic GALFIT input files (``.feedme``) and, optionally, invokes the
GALFIT binary on each one.

Data flow
---------
``palfitology fit-pa``      -> ``fitted_pa_images/PA_results.csv``   (one row per object,band)
``palfitology consensus``   -> ``fitted_pa_images/PA_consensus.csv`` (one row per object)
``palfitology galfit``      -> ``galfit_inputs/<id>.feedme`` (+ runs GALFIT)

For each object we build a single-Sersic model whose geometry priors come
from the consensus:

    * position angle    PA  = pa_consensus            (degrees)
    * axis ratio        q   = 1 - ell_consensus       (in (0, 1])
    * center (x, y)     from the science-band fit (``x0``/``y0`` in PA_results)

The remaining Sersic parameters -- integrated magnitude, effective radius,
and the Sersic index -- are seeded with conservative starting guesses and
left free for GALFIT to optimise. The science image GALFIT actually fits is
the ``fits_path`` recorded for the science band in PA_results.

GALFIT's PA convention is degrees CCW, with PA=0 pointing up (the +y axis)
and measured from +y toward -x. photutils returns PA measured CCW from the
+x axis. The two differ by exactly 90 degrees, so we add 90 and wrap into
GALFIT's accepted (-180, 180] range. This is the one place in the pipeline
that converts between the two conventions; everywhere upstream PA is the
photutils/[0,180) convention documented in the pipeline-conventions memory.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "GalfitParams",
    "DEFAULT_PIXSCALE",
    "DEFAULT_MAGZP",
    "DEFAULT_SCIENCE_BAND",
    "DEFAULT_SERSIC_N",
    "DEFAULT_MAG_GUESS",
    "pa_photutils_to_galfit",
    "ell_to_axis_ratio",
    "build_params_for_object",
    "build_all_params",
    "render_feedme",
    "write_feedme_files",
    "run_galfit",
]


# J-PLUS / T80Cam pixel scale (arcsec/pixel). The same value is hard-coded in
# the AutomatedImageDownloads scripts that produce the cutouts.
DEFAULT_PIXSCALE = 0.2627

# A neutral AB zeropoint default. Real J-PLUS magzps are band-dependent and
# image-dependent; users should pass --magzp for science-grade runs. We keep a
# default only so the file is well-formed and GALFIT can converge.
DEFAULT_MAGZP = 23.0

# Band whose cutout GALFIT fits, and whose fitted center seeds the model.
DEFAULT_SCIENCE_BAND = "rSDSS"

DEFAULT_SERSIC_N = 2.5     # between exponential (1) and de Vaucouleurs (4)
DEFAULT_MAG_GUESS = 18.0   # rough; GALFIT refines it freely


# ---------------------------------------------------------------------------
# Convention conversions
# ---------------------------------------------------------------------------

def pa_photutils_to_galfit(pa_deg: float) -> float:
    """Convert a photutils PA (CCW from +x, [0,180)) to GALFIT's convention.

    GALFIT measures PA from +y (up) toward -x, so it is offset by +90 deg
    from the photutils +x reference. The result is wrapped into (-180, 180].
    Returns NaN for non-finite input.
    """
    if not np.isfinite(pa_deg):
        return float("nan")
    g = (pa_deg + 90.0) % 360.0
    if g > 180.0:
        g -= 360.0
    return float(g)


def ell_to_axis_ratio(ell: float) -> float:
    """Convert ellipticity e = 1 - b/a to axis ratio q = b/a in (0, 1].

    Clamps into a numerically safe range so GALFIT never gets q<=0 or q>1.
    Returns 1.0 (circular) for non-finite input -- a round prior is the
    least-committal default.
    """
    if not np.isfinite(ell):
        return 1.0
    q = 1.0 - float(ell)
    return float(min(1.0, max(0.05, q)))


# ---------------------------------------------------------------------------
# Per-object parameters
# ---------------------------------------------------------------------------

@dataclass
class GalfitParams:
    """Everything needed to render one object's single-Sersic .feedme."""

    object_id: str
    input_image: Path           # science cutout GALFIT fits
    output_image: Path          # GALFIT imgblock output (.fits)
    x0: float                   # center, 1-based GALFIT pixel coords
    y0: float
    mag: float                  # starting integrated magnitude (free)
    r_eff: float                # starting effective radius in px (free)
    sersic_n: float             # starting Sersic index (free)
    axis_ratio: float           # q = b/a, GALFIT (free)
    pa_galfit: float            # PA in GALFIT convention (free)
    magzp: float
    pixscale: float
    img_nx: int                 # fitting region size (px)
    img_ny: int
    psf_image: Optional[Path] = None
    convbox: int = 100


def _image_shape(fits_path: Path) -> tuple[int, int]:
    """Return (nx, ny) for a FITS image, or (0, 0) if it can't be read."""
    try:
        from astropy.io import fits  # local import: keeps astropy optional

        with fits.open(fits_path) as hdul:
            for hdu in hdul:
                if getattr(hdu, "data", None) is not None and hdu.data.ndim == 2:
                    ny, nx = hdu.data.shape
                    return int(nx), int(ny)
    except Exception as exc:  # pragma: no cover - depends on file contents
        logger.warning(f"Could not read image shape from {fits_path}: {exc}")
    return 0, 0


def build_params_for_object(
    object_id: str,
    consensus_row: pd.Series,
    science_row: Optional[pd.Series],
    *,
    output_dir: Path,
    magzp: float = DEFAULT_MAGZP,
    pixscale: float = DEFAULT_PIXSCALE,
    sersic_n: float = DEFAULT_SERSIC_N,
    mag_guess: float = DEFAULT_MAG_GUESS,
    psf_image: Optional[Path] = None,
) -> Optional[GalfitParams]:
    """Assemble GalfitParams from a consensus row and the science-band fit row.

    Returns None (and logs) when the object can't be turned into a usable
    model -- e.g. no science-band cutout, or a failed/empty consensus.
    """
    status = str(consensus_row.get("status", ""))
    if status == "failed":
        logger.warning(f"[{object_id}] consensus failed; skipping GALFIT input.")
        return None

    if science_row is None:
        logger.warning(
            f"[{object_id}] no science-band row in PA_results; skipping."
        )
        return None

    fits_path = str(science_row.get("fits_path", "") or "")
    if not fits_path:
        logger.warning(f"[{object_id}] science-band fits_path is empty; skipping.")
        return None
    input_image = Path(fits_path)

    nx, ny = _image_shape(input_image)
    if nx == 0 or ny == 0:
        logger.warning(
            f"[{object_id}] could not read {input_image}; skipping."
        )
        return None

    # photutils centers are 0-based; GALFIT pixel coords are 1-based.
    x0 = float(science_row.get("x0", nx / 2.0))
    y0 = float(science_row.get("y0", ny / 2.0))
    if not np.isfinite(x0):
        x0 = nx / 2.0
    if not np.isfinite(y0):
        y0 = ny / 2.0
    x0 += 1.0
    y0 += 1.0

    pa = pa_photutils_to_galfit(
        float(consensus_row.get("pa_consensus", float("nan")))
    )
    if not np.isfinite(pa):
        pa = 0.0
    q = ell_to_axis_ratio(float(consensus_row.get("ell_consensus", float("nan"))))

    # Effective-radius prior: prefer the science-band fitted SMA if present,
    # else a quarter of the image (matches the fit-pa sma_guess convention).
    sma = science_row.get("est_sma", float("nan"))
    sma = float(sma) if sma is not None else float("nan")
    r_eff = sma if np.isfinite(sma) and sma > 0 else nx / 4.0

    output_image = output_dir / f"{object_id}_imgblock.fits"

    return GalfitParams(
        object_id=object_id,
        input_image=input_image,
        output_image=output_image,
        x0=x0,
        y0=y0,
        mag=float(mag_guess),
        r_eff=float(r_eff),
        sersic_n=float(sersic_n),
        axis_ratio=q,
        pa_galfit=pa,
        magzp=float(magzp),
        pixscale=float(pixscale),
        img_nx=nx,
        img_ny=ny,
        psf_image=psf_image,
    )


def build_all_params(
    consensus_df: pd.DataFrame,
    results_df: pd.DataFrame,
    *,
    output_dir: Path,
    science_band: str = DEFAULT_SCIENCE_BAND,
    magzp: float = DEFAULT_MAGZP,
    pixscale: float = DEFAULT_PIXSCALE,
    sersic_n: float = DEFAULT_SERSIC_N,
    mag_guess: float = DEFAULT_MAG_GUESS,
) -> list[GalfitParams]:
    """Join consensus + per-band results into one GalfitParams list."""
    for col in ("id", "pa_consensus", "ell_consensus"):
        if col not in consensus_df.columns:
            raise ValueError(f"PA_consensus is missing required column '{col}'.")
    for col in ("id", "band", "fits_path", "x0", "y0"):
        if col not in results_df.columns:
            raise ValueError(f"PA_results is missing required column '{col}'.")

    sci = results_df[results_df["band"].astype(str) == science_band]
    sci_by_id = {str(r["id"]): r for _, r in sci.iterrows()}

    params: list[GalfitParams] = []
    for _, crow in consensus_df.iterrows():
        oid = str(crow["id"])
        srow = sci_by_id.get(oid)
        gp = build_params_for_object(
            oid, crow, srow,
            output_dir=output_dir,
            magzp=magzp,
            pixscale=pixscale,
            sersic_n=sersic_n,
            mag_guess=mag_guess,
        )
        if gp is not None:
            params.append(gp)
    return params


# ---------------------------------------------------------------------------
# .feedme rendering
# ---------------------------------------------------------------------------

def render_feedme(p: GalfitParams) -> str:
    """Render a complete single-Sersic GALFIT input file as a string."""
    psf_line = str(p.psf_image) if p.psf_image is not None else "none"
    ps = f"{p.pixscale:.4f}"

    return f"""\
================================================================================
# GALFIT input for {p.object_id}  (palfitology single-Sersic, consensus priors)

A) {p.input_image}            # Input data image (science cutout)
B) {p.output_image}           # Output data image block
C) none                       # Sigma image (GALFIT computes from data)
D) {psf_line}                 # Input PSF image
E) 1                          # PSF fine-sampling factor
F) none                       # Bad-pixel mask
G) none                       # Parameter constraint file
H) 1 {p.img_nx} 1 {p.img_ny}  # Image region to fit (xmin xmax ymin ymax)
I) {p.convbox} {p.convbox}    # Size of convolution box (px)
J) {p.magzp:.4f}              # Magnitude photometric zeropoint
K) {ps} {ps}                  # Plate scale (dx dy, arcsec/px)
O) regular                    # Display type
P) 0                          # 0=optimize, 1=model, 2=imgblock, 3=subcomps

# Component 1: Sersic
 0) sersic                    #  Component type
 1) {p.x0:.3f} {p.y0:.3f} 1 1 #  Center (x y), free
 3) {p.mag:.3f}     1         #  Integrated magnitude, free
 4) {p.r_eff:.3f}   1         #  Effective radius R_e (px), free
 5) {p.sersic_n:.3f} 1        #  Sersic index n, free
 9) {p.axis_ratio:.4f} 1      #  Axis ratio (b/a), free  [consensus prior]
10) {p.pa_galfit:.3f} 1       #  Position angle (deg), free  [consensus prior]
 Z) 0                         #  Output option (0=residual)

================================================================================
"""


def write_feedme_files(
    params: list[GalfitParams],
    output_dir: Path,
) -> list[Path]:
    """Write one .feedme per GalfitParams. Returns the paths written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for p in params:
        path = output_dir / f"{p.object_id}.feedme"
        path.write_text(render_feedme(p))
        written.append(path)
    logger.info(f"Wrote {len(written)} GALFIT input file(s) to {output_dir}")
    return written


# ---------------------------------------------------------------------------
# Running GALFIT
# ---------------------------------------------------------------------------

def run_galfit(
    feedme: Path,
    galfit_bin: str = "galfit",
    cwd: Optional[Path] = None,
    timeout: Optional[float] = None,
) -> tuple[bool, str]:
    """Invoke the GALFIT binary on one .feedme.

    Returns ``(ok, combined_output)``. ``ok`` is False (without raising) when
    the binary is missing, exits non-zero, or times out, so a batch run can
    continue past a single object's failure.
    """
    exe = shutil.which(galfit_bin)
    if exe is None:
        msg = f"GALFIT binary '{galfit_bin}' not found on PATH."
        logger.error(msg)
        return False, msg

    try:
        proc = subprocess.run(
            [exe, str(feedme)],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        msg = f"GALFIT timed out on {feedme.name}"
        logger.error(msg)
        return False, msg

    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        logger.error(f"GALFIT exited {proc.returncode} on {feedme.name}")
        return False, output
    return True, output


# ---------------------------------------------------------------------------
# CLI hook
# ---------------------------------------------------------------------------

def _add_galfit_subparser(subparsers):
    """Wire `palfitology galfit` into the CLI."""
    p = subparsers.add_parser(
        "galfit",
        help="Write (and optionally run) GALFIT input files from the consensus.",
        description=(
            "Read fitted_pa_images/PA_consensus.csv and PA_results.csv, build "
            "one single-Sersic GALFIT input file per object with the consensus "
            "PA and axis ratio as geometry priors, write them to galfit_inputs/, "
            "and (unless --no-run) invoke the GALFIT binary on each."
        ),
    )
    p.add_argument(
        "--fitted-dir", type=Path, default=None,
        help="fitted_pa_images/ folder holding the two input CSVs "
             "(default: ./fitted_pa_images).",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where to write .feedme files and GALFIT outputs "
             "(default: ./galfit_inputs).",
    )
    p.add_argument(
        "--science-band", type=str, default=DEFAULT_SCIENCE_BAND,
        help=f"Band whose cutout GALFIT fits (default: {DEFAULT_SCIENCE_BAND}).",
    )
    p.add_argument(
        "--magzp", type=float, default=DEFAULT_MAGZP,
        help=f"Magnitude photometric zeropoint (default: {DEFAULT_MAGZP}).",
    )
    p.add_argument(
        "--pixscale", type=float, default=DEFAULT_PIXSCALE,
        help=f"Plate scale in arcsec/px (default: {DEFAULT_PIXSCALE}, J-PLUS).",
    )
    p.add_argument(
        "--sersic-n", type=float, default=DEFAULT_SERSIC_N,
        help=f"Starting Sersic index, left free (default: {DEFAULT_SERSIC_N}).",
    )
    p.add_argument(
        "--mag-guess", type=float, default=DEFAULT_MAG_GUESS,
        help=f"Starting integrated magnitude, left free "
             f"(default: {DEFAULT_MAG_GUESS}).",
    )
    p.add_argument(
        "--galfit-bin", type=str, default="galfit",
        help="GALFIT executable name or path (default: 'galfit').",
    )
    p.add_argument(
        "--no-run", action="store_true",
        help="Write the .feedme files but do not invoke the GALFIT binary.",
    )
    p.add_argument(
        "--timeout", type=float, default=None,
        help="Per-object GALFIT timeout in seconds (default: none).",
    )
    p.set_defaults(func=_cmd_galfit)
    return p


def _cmd_galfit(args) -> int:
    cwd = Path.cwd()
    fitted_dir = args.fitted_dir or (cwd / "fitted_pa_images")
    output_dir = args.output_dir or (cwd / "galfit_inputs")

    if not fitted_dir.is_dir():
        logger.error(f"Fitted-images folder not found: {fitted_dir}")
        return 1

    consensus_path = fitted_dir / "PA_consensus.csv"
    results_path = fitted_dir / "PA_results.csv"
    if not consensus_path.is_file():
        logger.error(
            f"PA_consensus.csv not found at {consensus_path}. "
            f"Run `palfitology consensus` first."
        )
        return 1
    if not results_path.is_file():
        logger.error(
            f"PA_results.csv not found at {results_path}. "
            f"Run `palfitology fit-pa` first."
        )
        return 1

    consensus_df = pd.read_csv(consensus_path)
    results_df = pd.read_csv(results_path)

    try:
        params = build_all_params(
            consensus_df, results_df,
            output_dir=output_dir,
            science_band=args.science_band,
            magzp=args.magzp,
            pixscale=args.pixscale,
            sersic_n=args.sersic_n,
            mag_guess=args.mag_guess,
        )
    except ValueError as exc:
        logger.error(str(exc))
        return 1

    if not params:
        logger.error(
            "No objects produced a usable GALFIT model "
            "(all consensus rows failed or lacked a science-band cutout)."
        )
        return 1

    feedmes = write_feedme_files(params, output_dir)

    if args.no_run:
        logger.info("--no-run set; wrote input files only.")
        return 0

    n_ok = 0
    n_fail = 0
    for feedme in feedmes:
        ok, _out = run_galfit(
            feedme, galfit_bin=args.galfit_bin,
            cwd=output_dir, timeout=args.timeout,
        )
        if ok:
            n_ok += 1
        else:
            n_fail += 1
    logger.info(f"GALFIT runs: {n_ok} ok, {n_fail} failed.")
    return 0 if n_fail == 0 else 1
