# Design notes

This document captures the reasoning behind a few non-obvious choices in
palfitology that have come up multiple times in the project's history. If
you're modifying the fit logic, read this first.

## Why a fallback ladder instead of a single fit?

`photutils.isophote.Ellipse.fit_image()` is sensitive to its initial geometry
guess. For most galaxies the catalog priors (ellipticity = B_WORLD/A_WORLD,
PA = pa_jplus) are good enough, but in marginal cases — low S/N, contamination
from neighbours, near-face-on orientation — the optimiser falls into a local
minimum that produces a meaningless isophote chain.

The fallback ladder runs the same fit with progressively more diverse starting
points: catalog priors → orthogonal/perpendicular variants → 50 reproducible
Monte-Carlo seeds spread across the (eps, pa) parameter space. The first pass
runs at full image resolution; if no strong fit emerges, the ladder is rerun
on a Gaussian-smoothed copy (sigma=2 px) to suppress high-frequency noise.

The cost is real (up to ~57 photutils calls per band) but each photutils call
is cheap, and on a healthy galaxy the very first config converges and we exit
immediately.

## Why the "second-min pa_err" rule was wrong

Early versions (V4/V5) picked the isophote with the second-smallest pa_err
from each fit table. This works for normal disks but fails for face-on
galaxies, because photutils generates many sub-pixel near-center isophotes
with artificially low pa_err (no flux gradient = no apparent uncertainty).
The rule latched onto those degenerate rows, returning sma ≈ 0.5 px ellipses
that bore no relation to the galaxy.

The V6 fix:

1. **SMA floor.** Candidate isophotes must have `sma ≥ max(3 px, 0.05 ×
   image_half_width)`. This filters out the near-center degenerate points.
2. **Sanity mask.** Drop non-finite pa_err, the dummy pa_err==0 row that
   photutils prepends, and rows with `ndata < 5` (under-sampled isophotes).
3. **Combined score.** Among the surviving rows, pick the one with the
   smallest `pa_err / sma`. This penalises both high-uncertainty fits and
   tiny-SMA fits simultaneously.
4. **Weak fallback.** If no row clears the SMA floor, return the
   outermost surviving row tagged `weak=True`, so downstream code can
   surface it in the CSV/PNG.

See `src/palfitology/selection.py` for the implementation.

## Initial SMA guess: x_shape / 4

Cutouts are pre-centered on the catalog target. A typical galaxy fills
roughly half the cutout, so its true semi-major axis is around `x_shape / 4`.
Starting photutils there (rather than at `x_shape / 2`, the cutout edge) gives
much better convergence on the first config.

## Parallelism: per-(object, band), not per-object

V7 onwards fits one (object, band) pair per worker task. With 12 bands and N
objects, that's 12N tasks fed to a `ProcessPoolExecutor`. The granularity
matters because:

- A small catalog (~20 objects) wouldn't saturate a 40-core node if the unit
  of work were "fit all 12 bands of one object". 12N tasks → workers always
  busy until the queue drains.
- Bands within one object aren't ordered by difficulty; rSDSS may finish in
  a second while uJAVA chews through 50 MC seeds. Per-band tasks let the
  fast bands free up workers for the slow ones.

The per-object summary mosaic is the one piece that can't be parallelised
naively (it needs all 12 bands of one object), so it runs incrementally in
the parent process the moment an object's last band returns.

## BLAS thread pinning

numpy/scipy/photutils use OpenMP/BLAS internally. On a 40-core node with 40
worker processes, each worker would otherwise try to grab 40 threads → 1600
threads competing for 40 cores. The pin-to-1 environment variables
(`OMP_NUM_THREADS=1` etc.) must be set **before** numpy is imported, in both
the parent process and every worker. This is wired in two places:

- `cli.py` sets the env vars at import time, before any heavy import.
- `pipeline._worker_init` re-sets them inside each spawned worker.

The `--no-pin-threads` flag is provided for the case where you run with
`--workers 1` and want numpy to use all cores within the single process.
