# palfitology

PSF-aware isophotal position-angle fitting and GALFIT prep for J-PLUS multi-band
galaxy cutouts.

`palfitology` is a Python package and command-line tool that turns a catalog of
galaxies and a folder of FITS cutouts into:

- per-band PA, semi-major axis (SMA), and ellipticity estimates,
- per-object diagnostic PNGs (one per band) and a 3×4 mosaic summarising all
  twelve J-PLUS / SDSS filters,
- a results CSV with one row per `(object, band)`,
- cross-band consensus values and GALFIT-ready single-Sersic input files.

The pipeline is built around `photutils.isophote.Ellipse` with a robust,
face-on-safe isophote selection rule (see `docs/design.md`) and runs many
galaxies in parallel on a multi-core node.

## Documentation

- **[Architecture map](https://nischal-acharya27.github.io/palfitology/)** — interactive
  single-page diagram of every module, its dependencies, the data flows, ADRs,
  CSV schemas, and the V0.2/V0.3 cluster run results. Click any module for its
  file path, public symbols, and tests. `Cmd/Ctrl + K` opens a command palette.
- **[`docs/architecture.json`](https://nischal-acharya27.github.io/palfitology/architecture.json)** —
  machine-readable architecture spec the page above consumes; suitable for AI
  coding agents and other tooling.
- **[`docs/design.md`](docs/design.md)** — rationale behind the isophote selection rule.
- **[`docs/pipeline.md`](docs/pipeline.md)** — per-stage walkthrough of the pipeline.

## Status

Early alpha. Working subcommands: `palfitology fit-pa`,
`palfitology reconcile`, `palfitology consensus`, `palfitology make-cutouts`,
`palfitology summarize-cutouts`, `palfitology regenerate-mosaics`,
`palfitology galfit`.

**v0.2** — PSF-aware preprocessing (Wiener deconvolution gated on PSF
FWHM vs catalog `R_EFF`). Shipped, cluster-validated on 243k rows.

**v0.3** — Cross-band PA consensus (weighted circular mean across the
12 bands with `est_ell² / pa_err²` weights and a two-clause outlier rule).
Shipped, cluster-validated on 20k objects.

**v0.4** — *next*: r-band sigma-cutoff detection. The rSDSS detection mask
will seed the isophote-fit geometry for every band, replacing the catalog
priors as the initial guess. This is the next planned upgrade.

**v0.7** — GALFIT input writer. `palfitology galfit` turns each object's
consensus PA and ellipticity into a single-Sersic `.feedme` (PA and axis
ratio as geometry priors; magnitude, effective radius, and Sersic index left
free) and optionally runs the GALFIT binary on each. Shipped.

**Download** — planned: `AutomatedImageDownloadsV2` port
(`palfitology download`).

## Install

From source (recommended while in development):

```bash
git clone https://github.com/nischal-acharya27/palfitology.git
cd palfitology
pip install -e .
```

Once a release is published to PyPI:

```bash
pip install palfitology
```

## Quick start

Suppose you have a project folder laid out like this:

```
my-project/
    catalog.csv
    images/
        <id_1>/
            fits_images_<ra>_<dec>/
                rSDSS_cutout.fits
                iSDSS_cutout.fits
                ... (12 bands)
            psfs_<ra>_<dec>/
                psf_<ra>_<dec>_rSDSS.fits
                ... (12 PSFs)
        <id_2>/
            ...
```

Then run, from inside `my-project/`:

```bash
palfitology fit-pa --all --workers 40
```

This will create `fitted_pa_images/` with one subfolder per object, each
containing twelve `<band>_PA_fit.png` plots and a `<id>_summary.png` mosaic,
plus a central `all_summaries/` folder collecting every mosaic. A
`PA_results.csv` at the top of `fitted_pa_images/` has one row per
`(object, band)` with the fitted parameters.

For a quick sanity check on five objects with two bands:

```bash
palfitology fit-pa --limit 5 --bands rSDSS iSDSS --workers 4
```

## From PA fit to GALFIT

The full per-object morphology flow is three commands:

```bash
palfitology fit-pa --all --workers 40      # -> fitted_pa_images/PA_results.csv
palfitology consensus                       # -> fitted_pa_images/PA_consensus.csv
palfitology galfit --magzp 23.5             # -> galfit_inputs/<id>.feedme  (+ runs GALFIT)
```

`palfitology galfit` joins `PA_consensus.csv` (per-object PA and ellipticity)
with `PA_results.csv` (per-band centers and the science-band cutout path) and
writes one single-Sersic GALFIT input file per object into `galfit_inputs/`.
The consensus drives two geometry priors:

- **position angle** — `pa_consensus`, converted from the photutils
  (CCW-from-+x) convention to GALFIT's (+90°, wrapped to (-180, 180]).
- **axis ratio** — `q = 1 - ell_consensus`, clamped to (0.05, 1].

Magnitude, effective radius, and the Sersic index are seeded with starting
guesses and left free for GALFIT to optimise. By default the binary is
invoked on each file; pass `--no-run` to write inputs only. Useful flags:

```bash
palfitology galfit \
    --science-band rSDSS \   # band whose cutout GALFIT fits (default rSDSS)
    --magzp 23.5 \           # photometric zeropoint (J-PLUS, band-dependent)
    --pixscale 0.2627 \      # arcsec/px (J-PLUS/T80Cam default)
    --galfit-bin galfit \    # executable name or path
    --no-run                 # write .feedme files only
```

## Catalog format

The catalog CSV must contain these columns:

| column     | meaning                                              |
|------------|------------------------------------------------------|
| `id`       | object id matching the `images/<id>/` folder name    |
| `A_WORLD`  | catalog semi-major axis (used only to derive priors) |
| `B_WORLD`  | catalog semi-minor axis                              |
| `pa_jplus` | catalog position angle in degrees                    |

If `id` is missing but `TILE_ID` and `NUMBER` are present (as in raw J-PLUS
ADQL exports), `palfitology` will synthesize `id` as `"<TILE_ID>-<NUMBER>"`
automatically. Comment lines starting with `#` (such as the SQL preamble in
ADQL dumps) are skipped.

## Cluster usage (SGE)

A reference SGE submission script lives in `scripts/run_palfitology.sh`. The
common pattern, for a 40-slot job:

```bash
#$ -N palfitology
#$ -pe parallel 40
#$ -cwd
#$ -j y
#$ -e output/error_$JOB_ID
#$ -o output/output_$JOB_ID

source ~/anaconda3/etc/profile.d/conda.sh
conda activate palfitology
export LD_PRELOAD=$CONDA_PREFIX/lib/libgomp.so.1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

palfitology fit-pa --all --workers $NSLOTS
```

The `LD_PRELOAD=libgomp.so.1` line works around a static-TLS dlopen failure
that scipy.special triggers on some clusters.

## Pipeline structure (current + planned)

The `palfitology` CLI is organised as subcommands, one per pipeline stage:

| subcommand              | status   | purpose                                                          |
|-------------------------|----------|------------------------------------------------------------------|
| `palfitology fit-pa`    | working  | Per-band isophotal PA fit + diagnostics                          |
| `palfitology reconcile` | working  | Cross-match fitted PAs against catalog `pa_jplus` (with scatter plots) |
| `palfitology consensus` | working  | Cross-band weighted circular-mean consensus + outlier flagging   |
| `palfitology galfit`    | working  | Emit (and optionally run) single-Sersic GALFIT input files       |
| `palfitology download`  | planned  | Fetch J-PLUS cutouts and PSFs for catalog entries                |

## Development

```bash
pip install -e ".[dev]"
pytest
```

See `docs/design.md` for the rationale behind the isophote selection rule and
`docs/pipeline.md` for the per-stage walkthrough.

## License

MIT. See `LICENSE`.
