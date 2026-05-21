"""End-to-end PA-fitting pipeline orchestrator.

`fit_catalog` is the high-level entry point: given a catalog dataframe and
config, it spawns workers, fits every (object, band) in parallel, renders
per-object mosaics incrementally as each object's last band finishes, and
returns a results dataframe.

The output folder structure created by this module:

    output_dir/
        <id>/
            <band>_PA_fit.png      (one per band per object)
            <id>_summary.png       (3x4 mosaic of all bands)
            PA_fits.csv            (per-band rows for this object)
        all_summaries/
            <id>_summary.png       (copy of every mosaic for easy browsing)
        PA_results.csv             (per-band rows for all objects)
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from astropy.io import fits

from .detect import DEFAULT_DETECT_BAND, DEFAULT_DETECT_SIGMA, DetectionResult, detect_source
from .fit import FitCandidate, fit_pa_with_fallbacks
from .images import locate_band_fits, locate_band_psf
from .plots import make_band_plot, make_summary_mosaic
from .psf import DEFAULT_PSF_GATE, preprocess_for_fit

logger = logging.getLogger(__name__)


RESULT_COLUMNS = [
    "id", "band", "fits_path", "est_pa", "est_sma", "est_ell", "x0", "y0",
    "pa_err", "selection_score", "fit_config", "smoothing_sigma",
    "used_weak_fallback", "n_configs_tried", "is_imputed", "status",
    "psf_mode", "psf_fwhm_pixels",
    # V0.4 detection columns (populated from the r-band detection pass)
    "detect_status", "detect_npix", "detect_sigma",
]


# ---------------------------------------------------------------------------
# Per-(object, band) worker
# ---------------------------------------------------------------------------

def _missing_row(objectid: str, band: str, eps_prior: float, pa_prior: float,
                 status: str) -> Dict[str, Any]:
    return {
        "id": objectid, "band": band, "fits_path": "",
        "est_pa": np.nan, "est_sma": np.nan, "est_ell": np.nan,
        "x0": np.nan, "y0": np.nan,
        "pa_err": np.nan, "selection_score": np.nan,
        "fit_config": status, "smoothing_sigma": np.nan,
        "used_weak_fallback": 0, "n_configs_tried": 0,
        "is_imputed": 0, "status": status,
        "psf_mode": "none", "psf_fwhm_pixels": np.nan,
        "detect_status": "unknown", "detect_npix": 0, "detect_sigma": np.nan,
    }


def process_one_band(task: Dict[str, Any]) -> Dict[str, Any]:
    """Fit one (object, band) pair, write its diagnostic PNG, return its CSV row.

    This is the function dispatched to each worker process. It always returns
    a row (with status='missing' if no cutout was found), so the pipeline can
    track per-object completion deterministically.

    PSF flow (v0.2+):
      1. Open the cutout.
      2. Call ``preprocess_for_fit`` to optionally Wiener-deconvolve the cutout
         using the per-band PSF. The 'auto' mode skips deconvolution when the
         PSF is small compared to the galaxy.
      3. Fit on the (possibly deconvolved) image. If the fit fails AND we
         deconvolved, retry on the raw cutout as a safety net. The final
         ``psf_mode`` in the CSV reflects what was actually used:
         ``raw``, ``deconv``, ``deconv->raw_fallback``, ``missing_psf``, or
         ``off``.

    Detection flow (v0.4+):
      The task dict may carry ``detect_result``: a ``DetectionResult`` computed
      from the r-band (or whichever ``detect_band``) image **before** this
      worker was dispatched.  When present and status=='ok', its moment-derived
      ``pa_deg`` and ``eps`` override the catalog priors fed into
      ``fit_pa_with_fallbacks``.  The catalog priors are still included as
      fallback configs inside the ladder, so we never lose that information.
      When ``detect_result`` is absent or 'no_detection', behaviour is
      identical to v0.3.
    """
    objectid = task["objectid"]
    band = task["band"]
    images_root = Path(task["images_root"])
    output_dir = Path(task["output_dir"])
    eps_prior = task["eps_prior"]
    pa_prior = task["pa_prior"]
    min_sma_abs = task["min_sma_abs"]
    min_sma_frac = task["min_sma_frac"]
    keep_best_of = task["keep_best_of"]
    psf_mode = task.get("psf_mode", "auto")
    psf_gate = task.get("psf_gate", DEFAULT_PSF_GATE)
    r_eff_pixels = task.get("r_eff_pixels", float("nan"))
    detect_result: Optional[DetectionResult] = task.get("detect_result", None)

    obj_out_dir = output_dir / objectid
    obj_out_dir.mkdir(parents=True, exist_ok=True)

    image_path = images_root / objectid
    if not image_path.is_dir():
        logger.warning(f"[{objectid}/{band}] object directory missing, skipping")
        return _missing_row(objectid, band, eps_prior, pa_prior, "missing")

    fits_file = locate_band_fits(image_path, band)
    if fits_file is None:
        logger.debug(f"[{objectid}/{band}] no cutout, skipping band")
        return _missing_row(objectid, band, eps_prior, pa_prior, "missing")

    try:
        with fits.open(fits_file) as hdul:
            data = hdul[0].data.astype(float)
    except Exception as e:  # noqa: BLE001
        logger.error(f"[{objectid}/{band}] failed to open FITS: {e}")
        return _missing_row(objectid, band, eps_prior, pa_prior, "missing")

    # ------------------------------------------------------------------
    # PSF preprocessing (V0.2)
    # ------------------------------------------------------------------
    psf_path = locate_band_psf(image_path, band) if psf_mode != "off" else None
    psf_info = preprocess_for_fit(
        image=data,
        psf_path=psf_path,
        r_eff_pixels=float(r_eff_pixels),
        mode=psf_mode,
        gate=psf_gate,
    )
    fit_image = psf_info.image_for_fit
    final_psf_mode = psf_info.mode
    psf_fwhm = psf_info.psf_fwhm_pixels

    # ------------------------------------------------------------------
    # Detection-seeded priors (V0.4)
    # ------------------------------------------------------------------
    # If the r-band (or configured detect-band) detection succeeded, use its
    # moment-derived PA and eps as the leading prior instead of the catalog
    # values.  The catalog priors are still tried inside the fallback ladder
    # (as config "catalog_prior"), so no information is discarded.
    detect_status = "skipped"
    detect_npix = 0
    detect_sigma_val = float("nan")
    if detect_result is not None:
        detect_status = detect_result.status
        detect_npix = detect_result.npix
        detect_sigma_val = detect_result.sigma_threshold
        if detect_result.status == "ok":
            # Override the priors with moment-based values for this band's fit.
            fit_pa_prior = detect_result.pa_deg
            fit_eps_prior = detect_result.eps
            logger.debug(
                f"[{objectid}/{band}] detection ok: "
                f"seeding fit with pa={fit_pa_prior:.1f}° eps={fit_eps_prior:.3f} "
                f"(catalog was pa={pa_prior:.1f}° eps={eps_prior:.3f})"
            )
        else:
            fit_pa_prior = pa_prior
            fit_eps_prior = eps_prior
            logger.debug(
                f"[{objectid}/{band}] detection {detect_result.status}: "
                f"falling back to catalog priors"
            )
    else:
        fit_pa_prior = pa_prior
        fit_eps_prior = eps_prior

    cand, n_tried = fit_pa_with_fallbacks(
        data=fit_image,
        eps_prior=fit_eps_prior,
        pa_prior=fit_pa_prior,
        min_sma_abs=min_sma_abs,
        min_sma_frac=min_sma_frac,
        keep_best_of=keep_best_of,
    )

    # Safety net: if we deconvolved and the fit failed, retry on the raw
    # cutout. This protects against deconvolution amplifying noise in a way
    # that breaks the isophote fitter.
    if cand is None and final_psf_mode == "deconv":
        logger.info(
            f"[{objectid}/{band}] deconvolved fit failed -- retrying on raw cutout"
        )
        cand_raw, n_tried_raw = fit_pa_with_fallbacks(
            data=data,
            eps_prior=eps_prior,
            pa_prior=pa_prior,
            min_sma_abs=min_sma_abs,
            min_sma_frac=min_sma_frac,
            keep_best_of=keep_best_of,
        )
        n_tried += n_tried_raw
        if cand_raw is not None:
            cand = cand_raw
            fit_image = data
            final_psf_mode = "deconv->raw_fallback"

    is_imputed = cand is None
    if is_imputed:
        logger.warning(
            f"[{objectid}/{band}] all {n_tried} fits failed -- imputing"
        )
        y_shape, x_shape = data.shape
        imputed_priors = (
            pa_prior if np.isfinite(pa_prior) else float("nan"),
            float("nan"),
            max(0.0, min(1.0, 1.0 - eps_prior)) if np.isfinite(eps_prior) else float("nan"),
            x_shape / 2.0,
            y_shape / 2.0,
        )
        status = "imputed"
    else:
        imputed_priors = None
        status = "weak" if cand.weak else "ok"
        logger.info(
            f"[{objectid}/{band}] {status.upper()}: PA={cand.pa_deg:.2f} "
            f"SMA={cand.sma:.2f} ell={cand.ell:.3f} score={cand.score:.4f}"
        )

    png_path = obj_out_dir / f"{band}_PA_fit.png"
    try:
        make_band_plot(
            data=data, objectid=objectid, band=band, cand=cand,
            out_path=png_path, is_imputed=is_imputed,
            fallback_priors=imputed_priors,
            detect_result=detect_result,
        )
    except Exception as e:  # noqa: BLE001
        logger.error(f"[{objectid}/{band}] plotting failed: {e}")

    if cand is not None:
        return {
            "id": objectid, "band": band, "fits_path": str(fits_file),
            "est_pa": cand.pa_deg, "est_sma": cand.sma, "est_ell": cand.ell,
            "x0": cand.x0, "y0": cand.y0,
            "pa_err": cand.pa_err, "selection_score": cand.score,
            "fit_config": cand.config_tag, "smoothing_sigma": cand.smoothing,
            "used_weak_fallback": int(cand.weak),
            "n_configs_tried": n_tried,
            "is_imputed": 0, "status": status,
            "psf_mode": final_psf_mode, "psf_fwhm_pixels": psf_fwhm,
            "detect_status": detect_status,
            "detect_npix": detect_npix,
            "detect_sigma": detect_sigma_val,
        }
    y_shape, x_shape = data.shape
    return {
        "id": objectid, "band": band, "fits_path": str(fits_file),
        "est_pa": pa_prior if np.isfinite(pa_prior) else np.nan,
        "est_sma": np.nan,
        "est_ell": (
            max(0.0, min(1.0, 1.0 - eps_prior))
            if np.isfinite(eps_prior) else np.nan
        ),
        "x0": x_shape / 2.0, "y0": y_shape / 2.0,
        "pa_err": np.nan, "selection_score": np.nan,
        "fit_config": "imputed", "smoothing_sigma": np.nan,
        "used_weak_fallback": 0, "n_configs_tried": n_tried,
        "is_imputed": 1, "status": status,
        "psf_mode": final_psf_mode, "psf_fwhm_pixels": psf_fwhm,
        "detect_status": detect_status,
        "detect_npix": detect_npix,
        "detect_sigma": detect_sigma_val,
    }


def _worker_init() -> None:
    """Pin BLAS threads in each worker so 40 processes don't oversubscribe a node."""
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, "1")
    logging.basicConfig(
        level=logger.level,
        format="%(asctime)s - [pid %(process)d] - %(levelname)s - %(message)s",
        force=True,
    )


# ---------------------------------------------------------------------------
# Per-object mosaic finalizer
# ---------------------------------------------------------------------------

def _render_object_summary(
    objectid: str,
    rows: List[Dict[str, Any]],
    bands_order: List[str],
    output_dir: Path,
    all_summaries_dir: Path,
    detect_result: Optional[DetectionResult] = None,
) -> None:
    """Write one object's PA_fits.csv and 3x4 mosaic to per-object + central folders."""
    obj_dir = output_dir / objectid
    obj_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(
        obj_dir / "PA_fits.csv", index=False
    )

    band_data: Dict[str, np.ndarray] = {}
    band_cands: Dict[str, Optional[FitCandidate]] = {}
    band_statuses: Dict[str, str] = {}
    for r in rows:
        band = r["band"]
        status = r["status"]
        band_statuses[band] = status
        fits_path = r["fits_path"]
        if not fits_path:
            continue
        try:
            with fits.open(fits_path) as hdul:
                band_data[band] = hdul[0].data.astype(float)
        except Exception:  # noqa: BLE001
            continue
        if status in ("ok", "weak"):
            band_cands[band] = FitCandidate(
                pa_deg=r["est_pa"], sma=r["est_sma"], ell=r["est_ell"],
                x0=r["x0"], y0=r["y0"], pa_err=r["pa_err"],
                score=r["selection_score"],
                config_tag=r["fit_config"],
                smoothing=r["smoothing_sigma"] or 0.0,
                weak=(status == "weak"),
            )
        else:
            band_cands[band] = None

    try:
        make_summary_mosaic(
            objectid=objectid,
            band_data=band_data,
            band_cands=band_cands,
            band_statuses=band_statuses,
            bands_order=bands_order,
            out_path=[
                obj_dir / f"{objectid}_summary.png",
                all_summaries_dir / f"{objectid}_summary.png",
            ],
            detect_result=detect_result,
        )
    except Exception as e:  # noqa: BLE001
        logger.error(f"[{objectid}] summary mosaic failed: {e}")


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def _run_detection_for_object(
    objectid: str,
    images_root: Path,
    detect_band: str,
    detect_sigma: float,
) -> Optional[DetectionResult]:
    """Open the detect-band cutout for one object and run sigma-clip detection.

    Returns None if the cutout cannot be opened (missing file, I/O error).
    A ``DetectionResult`` with status='no_detection' is still returned when the
    file exists but no signal is found above threshold.
    """
    image_path = images_root / objectid
    fits_file = locate_band_fits(image_path, detect_band)
    if fits_file is None:
        logger.debug(
            f"[{objectid}] detect-band '{detect_band}' cutout not found -- "
            f"detection skipped"
        )
        return None
    try:
        from astropy.io import fits as astropy_fits
        with astropy_fits.open(fits_file) as hdul:
            data = hdul[0].data.astype(float)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[{objectid}] failed to open detect-band FITS: {e}")
        return None
    return detect_source(data, sigma_threshold=detect_sigma)


def fit_catalog(
    images_root: Path,
    output_dir: Path,
    catalog: pd.DataFrame,
    bands: List[str],
    min_sma_abs: float = 3.0,
    min_sma_frac: float = 0.05,
    keep_best_of: int = 8,
    workers: int = 1,
    make_summary: bool = True,
    psf_mode: str = "auto",
    psf_gate: float = DEFAULT_PSF_GATE,
    detect_sigma: float = DEFAULT_DETECT_SIGMA,
    detect_band: str = DEFAULT_DETECT_BAND,
) -> pd.DataFrame:
    """Fit every (object, band) pair in `catalog` and write outputs to `output_dir`.

    Returns a dataframe with one row per (id, band), columns documented at
    `RESULT_COLUMNS`.

    Parameters
    ----------
    psf_mode
        ``'auto'`` (default) -- deconvolve only when the PSF FWHM is at least
        ``psf_gate`` * R_EFF (R_EFF in pixels, taken from the catalog's
        ``R_EFF`` column when present).
        ``'on'`` -- always deconvolve when a PSF file is available.
        ``'off'`` -- never deconvolve (v0.1.0 behaviour).
    psf_gate
        Threshold ratio used in ``'auto'`` mode. See ``palfitology.psf``.
    detect_sigma
        Sigma threshold for source detection in the ``detect_band`` image.
        Default 3.0 (SExtractor-style).  Set to 0 to disable detection
        entirely (reverts to catalog-prior seeding as in v0.3).
    detect_band
        Which band image is used as the detection master.  Default 'rSDSS'.
        Must be a band present in ``bands``; if not found for a given object,
        detection is silently skipped and the fit falls back to catalog priors.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    has_r_eff = "R_EFF" in catalog.columns
    if psf_mode in ("auto", "on") and not has_r_eff:
        if psf_mode == "auto":
            logger.warning(
                "psf-mode=auto requested but catalog has no R_EFF column -- "
                "the gate falls back to 'raw' (no deconvolution) for every row. "
                "Add an R_EFF column (in pixels) to enable deconvolution."
            )

    # ------------------------------------------------------------------
    # V0.4: pre-compute r-band (or detect_band) detection for every object.
    # This runs in the parent process before workers are spawned so the
    # DetectionResult can be serialised into each task dict.
    # ------------------------------------------------------------------
    use_detection = detect_sigma > 0.0
    detection_by_obj: Dict[str, Optional[DetectionResult]] = {}
    if use_detection:
        logger.info(
            f"Running {detect_band} sigma-clip detection "
            f"(sigma={detect_sigma}) for {len(catalog)} objects ..."
        )
        for _, row in catalog.iterrows():
            oid = str(row["id"])
            detection_by_obj[oid] = _run_detection_for_object(
                oid, images_root, detect_band, detect_sigma
            )
        n_ok = sum(
            1 for d in detection_by_obj.values()
            if d is not None and d.status == "ok"
        )
        logger.info(
            f"Detection complete: {n_ok}/{len(catalog)} objects detected "
            f"in {detect_band} (sigma={detect_sigma})"
        )

    tasks: List[Dict[str, Any]] = []
    expected_bands_by_obj: Dict[str, set] = {}
    for _, row in catalog.iterrows():
        a_world = row.get("A_WORLD", np.nan)
        b_world = row.get("B_WORLD", np.nan)
        eps_prior = (
            float(b_world) / float(a_world)
            if np.isfinite(a_world) and a_world > 0 and np.isfinite(b_world)
            else 0.3
        )
        pa_prior = float(row.get("pa_jplus", 30.0))
        r_eff_pixels = float(row.get("R_EFF", np.nan)) if has_r_eff else float("nan")
        objectid = str(row["id"])
        det_result = detection_by_obj.get(objectid) if use_detection else None
        expected_bands_by_obj.setdefault(objectid, set())
        for band in bands:
            expected_bands_by_obj[objectid].add(band)
            tasks.append({
                "objectid": objectid,
                "band": band,
                "images_root": str(images_root),
                "output_dir": str(output_dir),
                "eps_prior": eps_prior,
                "pa_prior": pa_prior,
                "min_sma_abs": min_sma_abs,
                "min_sma_frac": min_sma_frac,
                "keep_best_of": keep_best_of,
                "psf_mode": psf_mode,
                "psf_gate": psf_gate,
                "r_eff_pixels": r_eff_pixels,
                "detect_result": det_result,
            })

    logger.info(
        f"Submitting {len(tasks)} fits ({len(catalog)} objects × {len(bands)} bands) "
        f"to {workers} worker(s)"
    )

    all_summaries_dir = output_dir / "all_summaries"
    if make_summary:
        all_summaries_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    pending_rows: Dict[str, List[Dict[str, Any]]] = {oid: [] for oid in expected_bands_by_obj}
    seen_bands: Dict[str, set] = {oid: set() for oid in expected_bands_by_obj}

    def _maybe_finalize(objectid: str) -> None:
        if not make_summary:
            return
        if seen_bands[objectid] == expected_bands_by_obj[objectid]:
            logger.info(f"[{objectid}] all bands complete -- rendering mosaic")
            _render_object_summary(
                objectid=objectid,
                rows=pending_rows[objectid],
                bands_order=bands,
                output_dir=output_dir,
                all_summaries_dir=all_summaries_dir,
                detect_result=detection_by_obj.get(objectid) if use_detection else None,
            )
            pending_rows[objectid] = []
            seen_bands[objectid] = set()

    def _ingest(row: Dict[str, Any]) -> None:
        results.append(row)
        oid = row["id"]
        if oid in expected_bands_by_obj and row["band"] in expected_bands_by_obj[oid]:
            seen_bands[oid].add(row["band"])
            pending_rows[oid].append(row)
            _maybe_finalize(oid)

    if workers <= 1:
        for task in tasks:
            _ingest(process_one_band(task))
    else:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_worker_init
        ) as pool:
            futures = {
                pool.submit(process_one_band, t): (t["objectid"], t["band"])
                for t in tasks
            }
            for fut in as_completed(futures):
                obj, band = futures[fut]
                try:
                    row = fut.result()
                except Exception as e:  # noqa: BLE001
                    logger.error(f"[{obj}/{band}] worker crashed: {e}")
                    row = _missing_row(obj, band, 0.3, 30.0, "missing")
                _ingest(row)

    # Safety net: any object that never reached full completion (e.g. a worker
    # crash where the row couldn't be synthesized) gets a partial mosaic.
    if make_summary:
        for oid, rows in pending_rows.items():
            if rows:
                logger.warning(
                    f"[{oid}] only {len(rows)}/{len(expected_bands_by_obj[oid])} "
                    f"bands recorded -- rendering partial mosaic"
                )
                _render_object_summary(
                    objectid=oid,
                    rows=rows,
                    bands_order=bands,
                    output_dir=output_dir,
                    all_summaries_dir=all_summaries_dir,
                    detect_result=detection_by_obj.get(oid) if use_detection else None,
                )

    # Stable order in the final dataframe: catalog order, then band order.
    cat_order = {str(row["id"]): i for i, row in catalog.reset_index(drop=True).iterrows()}
    band_order = {b: i for i, b in enumerate(bands)}
    results.sort(key=lambda r: (cat_order.get(r["id"], 1 << 30),
                                band_order.get(r["band"], 1 << 30)))

    return pd.DataFrame(results, columns=RESULT_COLUMNS)
