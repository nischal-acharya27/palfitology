# Pipeline structure

palfitology is organised as four CLI subcommands, one per stage. Each stage
is independent: it reads from the filesystem, writes to a known output
location, and the next stage picks up from there.

| stage | subcommand | status | inputs | outputs |
|-------|------------|--------|--------|---------|
| 1 | `palfitology download`  | planned | catalog CSV                       | `images/<id>/{fits_images_*, psfs_*}/` |
| 2 | `palfitology fit-pa`    | working | `images/`, catalog CSV            | `fitted_pa_images/<id>/*_PA_fit.png`, `PA_results.csv` |
| 3 | `palfitology consensus` | planned | `PA_results.csv`                  | `consensus.csv` with per-object PA/ell + flags |
| 4 | `palfitology galfit`    | planned | `consensus.csv`, `images/`, PSFs  | one GALFIT input block per object       |

## Stage 2 — fit-pa

This is the only fully implemented stage today.

**Inputs**

A catalog CSV (auto-discovered as the unique `.csv` in cwd, or specified
with `--catalog`) and an `images/` folder where each object has its own
subfolder containing `fits_images_<ra>_<dec>/<band>_cutout.fits` files.

**Per-(object, band) work**

For every `(object, band)` pair, palfitology:

1. Opens the cutout FITS.
2. Runs `fit_pa_with_fallbacks` (`fit.py`) — the V6 fallback ladder against
   the V6 face-on-safe selection rule (`selection.py`).
3. Writes a diagnostic PNG (`plots.make_band_plot`) to
   `fitted_pa_images/<id>/<band>_PA_fit.png`.
4. Returns a row dict that becomes one line of `PA_results.csv`.

These tasks are dispatched to a `ProcessPoolExecutor` so up to `--workers`
of them run simultaneously.

**Per-object finalization (incremental)**

The parent process tracks which bands each object has finished. The moment an
object's last band returns, the parent:

1. Writes `fitted_pa_images/<id>/PA_fits.csv` (per-object rows).
2. Renders a 3×4 mosaic of all bands with ellipses overlaid
   (`plots.make_summary_mosaic`) and saves it both to
   `fitted_pa_images/<id>/<id>_summary.png` and
   `fitted_pa_images/all_summaries/<id>_summary.png`.

This means mosaics start appearing in `all_summaries/` immediately, not only
after the full catalog finishes.

**Outputs**

```
fitted_pa_images/
    <id>/
        <band>_PA_fit.png   * 12
        <id>_summary.png
        PA_fits.csv
    all_summaries/
        <id>_summary.png    * N
    PA_results.csv          (catalog-ordered, band-ordered)
```

**CSV schema**

`PA_results.csv` has one row per `(id, band)` with these columns:

- `id`, `band`, `fits_path`
- `est_pa`, `est_sma`, `est_ell`, `x0`, `y0`
- `pa_err`, `selection_score`
- `fit_config`, `smoothing_sigma`, `used_weak_fallback`, `n_configs_tried`
- `is_imputed`, `status` (`ok` / `weak` / `imputed` / `missing`)

## Roadmap

Next planned upgrades (from the working notebook):

1. **PSF-aware fit** (v0.2) — Wiener deconvolution against the per-band PSF
   before the photutils fit, gated on PSF FWHM vs catalog `R_EFF`.
2. **Cross-band consensus** (v0.3) — per-object PA/ell from a weighted
   combination of the 12 bands, with outlier flagging.
3. **GALFIT priors writer** (v0.4) — emit GALFIT input blocks from the
   consensus values, closing the loop with the existing `GalfitM/` pipeline.
4. **Download integration** (v0.5) — port the existing
   `AutomatedImageDownloadsV2` script into `palfitology download`.
