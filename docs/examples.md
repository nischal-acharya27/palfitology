# Examples

## End-to-end run on a small catalog

Suppose you have a project folder with a small catalog and 17 objects already
downloaded:

```
my-run/
    catalog.csv
    images/
        92801-11428/
            fits_images_202.4698_47.1955/
                rSDSS_cutout.fits
                iSDSS_cutout.fits
                ...
            psfs_202.4698_47.1955/
                psf_202.4698_47.1955_rSDSS.fits
                ...
        93140-3576/
            ...
```

From inside `my-run/`:

```bash
palfitology fit-pa --all --workers 8
```

After ~1 minute (depending on catalog size and number of cores), you'll have:

```
fitted_pa_images/
    92801-11428/
        rSDSS_PA_fit.png   ... iSDSS_PA_fit.png ... J0660_PA_fit.png ...
        92801-11428_summary.png
        PA_fits.csv
    93140-3576/
        ...
    all_summaries/
        92801-11428_summary.png
        93140-3576_summary.png
        ...
    PA_results.csv
```

Open `fitted_pa_images/all_summaries/` and flip through the 3×4 mosaics —
that's the fastest way to spot bad fits.

## Sanity-check run on 5 objects with 2 bands

For a quick visual check while iterating:

```bash
palfitology fit-pa --limit 5 --bands rSDSS iSDSS --workers 4
```

## Tuning the SMA floor

If you're seeing weak fits where you'd expect strong ones (or vice versa),
tune the SMA floor:

```bash
# More aggressive: require sma >= 5 px or 8% of half-width
palfitology fit-pa --all --workers 8 --min-sma-abs 5 --min-sma-frac 0.08

# More lenient
palfitology fit-pa --all --workers 8 --min-sma-abs 2 --min-sma-frac 0.03
```

## Running on an SGE cluster

```bash
qsub scripts/run_palfitology.sh
```

This grabs 40 slots from the `parallel` PE and calls
`palfitology fit-pa --all --workers $NSLOTS`. Watch progress with:

```bash
qstat -u $USER
tail -f output/output_<JOB_ID>
```

The log will stream lines like:

```
2026-05-12 ... INFO - [92801-11428/rSDSS] OK: PA=12.34 SMA=58.31 ell=0.796 score=0.0067
2026-05-12 ... INFO - [92801-11428] all bands complete -- rendering mosaic
```

so you know mosaics are landing in `all_summaries/` incrementally.

## Embedding in Python

If you don't want the CLI, the public API mirrors the subcommands:

```python
from pathlib import Path
import pandas as pd
from palfitology import ALL_BANDS, fit_catalog
from palfitology.catalog import load_catalog, filter_to_existing_image_dirs

cat = load_catalog(Path("catalog.csv"))
cat = filter_to_existing_image_dirs(cat, Path("images"))

results = fit_catalog(
    images_root=Path("images"),
    output_dir=Path("fitted_pa_images"),
    catalog=cat,
    bands=ALL_BANDS,
    workers=8,
)
print(results[["id", "band", "est_pa", "est_sma", "status"]])
```

The returned `DataFrame` is the same as `PA_results.csv` on disk.
