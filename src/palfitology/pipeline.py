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

from .fit import FitCandidate, fit_pa_with_fallbacks
from .images import locate_band_fits
from .plots import make_band_plot, make_summary_mosaic

logger = logging.getLogger(__name__)


RESULT_COLUMNS = [
    "id", "band", "fits_path", "est_pa", "est_sma", "est_ell", "x0", "y0",
    "pa_err", "selection_score", "fit_config", "smoothing_sigma",
    "used_weak_fallback", "n_configs_tried", "is_imputed", "status",
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
    }


def process_one_band(task: Dict[str, Any]) -> Dict[str, Any]:
    """Fit one (object, band) pair, write its diagnostic PNG, return its CSV row.

    This is the function dispatched to each worker process. It always returns
    a row (with status='missing' if no cutout was found), so the pipeline can
    track per-object completion deterministically.
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

    cand, n_tried = fit_pa_with_fallbacks(
        data=data,
        eps_prior=eps_prior,
        pa_prior=pa_prior,
        min_sma_abs=min_sma_abs,
        min_sma_frac=min_sma_frac,
        keep_best_of=keep_best_of,
    )

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
        )
    except Exception as e:  # noqa: BLE001
        logger.error(f"[{objectid}] summary mosaic failed: {e}")


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

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
) -> pd.DataFrame:
    """Fit every (object, band) pair in `catalog` and write outputs to `output_dir`.

    Returns a dataframe with one row per (id, band), columns documented at
    `RESULT_COLUMNS`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

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
        objectid = str(row["id"])
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
                )

    # Stable order in the final dataframe: catalog order, then band order.
    cat_order = {str(row["id"]): i for i, row in catalog.reset_index(drop=True).iterrows()}
    band_order = {b: i for i, b in enumerate(bands)}
    results.sort(key=lambda r: (cat_order.get(r["id"], 1 << 30),
                                band_order.get(r["band"], 1 << 30)))

    return pd.DataFrame(results, columns=RESULT_COLUMNS)
