# palfitology

PSF-aware isophotal position-angle fitting and GALFIT prep for J-PLUS multi-band
galaxy cutouts.

`palfitology` is a Python package and command-line tool that turns a catalog of
galaxies and a folder of FITS cutouts into:

- per-band PA, semi-major axis (SMA), and ellipticity estimates,
- per-object diagnostic PNGs (one per band) and a 3×4 mosaic summarising all
  twelve J-PLUS / SDSS filters,
- a results CSV with one row per `(object, band)`,
- (planned) cross-band consensus values and GALFIT-ready input files.

The pipeline is built around `photutils.isophote.Ellipse` with a robust,
face-on-safe isophote selection rule (see `docs/design.md`) and runs many
galaxies in parallel on a multi-core node.

## Status

Early alpha. The library refactor is in progress. The current canonical command
is `palfitology fit-pa`; PSF support and consensus are planned for the next
iterations.

## Install

From source (recommended while in development):

```bash
git clone https://github.com/<your-username>/palfitology.git
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
| `palfitology download`  | planned  | Fetch J-PLUS cutouts and PSFs for catalog entries                |
| `palfitology fit-pa`    | working  | Per-band isophotal PA fit + diagnostics                          |
| `palfitology consensus` | planned  | Cross-band PA/ell consensus + outlier flagging                   |
| `palfitology galfit`    | planned  | Emit GALFIT-ready input files from consensus values              |

## Development

```bash
pip install -e ".[dev]"
pytest
```

See `docs/design.md` for the rationale behind the isophote selection rule and
`docs/pipeline.md` for the per-stage walkthrough.

## License

MIT. See `LICENSE`.
