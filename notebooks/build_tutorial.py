"""Build tutorial.ipynb programmatically.

Run once locally:
    python notebooks/build_tutorial.py

Produces notebooks/tutorial.ipynb with four executed sections:
  1. one galaxy, one band
  2. one galaxy, all 12 bands
  3. multiple galaxies, one band
  4. multiple galaxies, all 12 bands

Synthetic FITS cutouts and PSFs are generated under a temp dir so the
notebook can be re-executed without cluster data. Replace the synthetic
data path with your real `images/` and the notebook works the same way.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


def nb_md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in textwrap.dedent(text).strip("\n").splitlines()],
    }


def nb_code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in textwrap.dedent(text).strip("\n").splitlines()],
    }


CELLS: list[dict] = []

# ---------------------------------------------------------------------------
CELLS.append(nb_md("""
# palfitology · tutorial

This notebook walks through the **PA fitting pipeline** end-to-end on
small synthetic data so you can see exactly what each stage produces.

We cover four scenarios in order of increasing scale:

1. **One galaxy, one band** — the smallest possible unit. Just the fit
   function + a single diagnostic plot.
2. **One galaxy, all 12 bands** — same galaxy across all J-PLUS bands.
3. **Multiple galaxies, one band** — same band across a small catalog.
4. **Multiple galaxies, all bands** — the full pipeline as you'd run it
   on the cluster.

Each scenario produces an overlay showing **two ellipses**:

* **red** = our photutils-fit PA, semi-major axis, ellipticity
* **cyan** = catalog `pa_jplus` (after the new convention transform),
  using our fitted SMA and ellipticity so only the angle differs

When the two overlap, the catalog and our fit agree on direction.
The text overlay reports both PA values, our `pa_err`, `est_sma`,
`est_ell`, and the circular difference.
"""))

# ---------------------------------------------------------------------------
CELLS.append(nb_md("""
## 0 · Setup

Imports plus a small synthetic-data factory. Skip ahead to section 1 if
you don't care how the fake galaxies are made.
"""))

CELLS.append(nb_code("""
from __future__ import annotations

import os
# Pin BLAS BEFORE importing numpy -- matches what palfitology.cli does.
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse as MplEllipse
from astropy.io import fits
from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval

import palfitology
from palfitology.fit import fit_pa_with_fallbacks
from palfitology.pipeline import fit_catalog
from palfitology.reconcile import transform_pa_jplus, circular_diff_deg
from palfitology import ALL_BANDS

print(f"palfitology v{palfitology.__version__}")
print(f"bands: {len(ALL_BANDS)} -- {ALL_BANDS[:4]} ...")
"""))

CELLS.append(nb_code("""
def make_synthetic_galaxy(shape=(120, 120), pa_deg=42.0, sma=28.0, ell=0.55,
                         noise=0.05, seed=0):
    \"\"\"Render a 2D elliptical Gaussian galaxy with PA, SMA, ellipticity.

    pa_deg is measured CCW from +x in the image frame -- the same
    convention photutils uses, so we can compare directly.
    \"\"\"
    rng = np.random.default_rng(seed)
    ny, nx = shape
    y, x = np.mgrid[0:ny, 0:nx]
    x0, y0 = nx/2, ny/2
    pa = np.deg2rad(pa_deg)
    cos_p, sin_p = np.cos(pa), np.sin(pa)
    # Rotate coords into the galaxy's principal frame
    xr = (x - x0) * cos_p + (y - y0) * sin_p
    yr = -(x - x0) * sin_p + (y - y0) * cos_p
    a = sma
    b = sma * (1.0 - ell)
    arg = (xr / a)**2 + (yr / b)**2
    img = np.exp(-0.5 * arg) * 100.0
    img += rng.normal(0, noise * 100.0, size=shape)
    return img

def make_synthetic_psf(shape=(25, 25), fwhm_px=2.7):
    \"\"\"Circular Gaussian PSF, peak-normalised.\"\"\"
    ny, nx = shape
    y, x = np.mgrid[0:ny, 0:nx]
    sigma = fwhm_px / (2 * np.sqrt(2 * np.log(2)))
    g = np.exp(-((x - nx/2)**2 + (y - ny/2)**2) / (2 * sigma**2))
    return g / g.sum()

# Quick sanity render
fig, ax = plt.subplots(figsize=(4.2, 4.2))
img = make_synthetic_galaxy(pa_deg=42.0)
norm = ImageNormalize(img, interval=ZScaleInterval(), stretch=AsinhStretch())
ax.imshow(img, origin='lower', cmap='gray_r', norm=norm)
ax.set_title("synthetic galaxy · PA=42° · SMA=28 px · ell=0.55")
ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
"""))

# ---------------------------------------------------------------------------
CELLS.append(nb_md("""
## 1 · One galaxy, one band

The smallest unit of work. We call `fit_pa_with_fallbacks` directly on
the image array and then visualise the result.

The function takes priors from the catalog (`eps_prior = B_WORLD/A_WORLD`,
`pa_prior = pa_jplus`) and runs an isophotal fit ladder with ~57
initialisations. It returns the best-scoring `FitCandidate`.
"""))

CELLS.append(nb_code("""
TRUE_PA   = 42.0       # what we put into the synthetic galaxy
TRUE_SMA  = 28.0
TRUE_ELL  = 0.55

# Catalog-style priors
PA_JPLUS_RAW = -48.0   # as if the catalog reported SExtractor THETA_IMAGE
A_WORLD = 0.013
B_WORLD = 0.006        # B/A = 0.46 -> ell_prior = 0.54

img = make_synthetic_galaxy(pa_deg=TRUE_PA, sma=TRUE_SMA, ell=TRUE_ELL, seed=1)

cand, n_tried = fit_pa_with_fallbacks(
    data=img,
    eps_prior=B_WORLD / A_WORLD,
    pa_prior=PA_JPLUS_RAW,
    keep_best_of=2,   # tutorial: stop early; default in production is 8
)
print(f"tried {n_tried} configs; best fit config = {cand.config_tag!r}")
print(f"  PA  = {cand.pa_deg:7.2f} deg  (truth {TRUE_PA})")
print(f"  SMA = {cand.sma:7.2f} px   (truth {TRUE_SMA})")
print(f"  ell = {cand.ell:7.3f}      (truth {TRUE_ELL})")
print(f"  pa_err = {cand.pa_err:.3f} deg   score = {cand.score:.4f}")
"""))

CELLS.append(nb_code("""
def plot_overlay(ax, img, cand, pa_jplus, title=\"\"):
    \"\"\"Render image + two ellipses (photutils red, catalog cyan) + caption.\"\"\"
    norm = ImageNormalize(img, interval=ZScaleInterval(), stretch=AsinhStretch())
    ax.imshow(img, origin='lower', cmap='gray_r', norm=norm)

    if cand is None or not np.isfinite(cand.pa_deg):
        ax.text(0.5, 0.5, "no fit", ha='center', va='center',
                transform=ax.transAxes, color='red')
        return

    # --- ellipse 1: our photutils fit (red) ---
    sma, smb = cand.sma, cand.sma * (1 - cand.ell)
    ax.add_patch(MplEllipse(
        (cand.x0, cand.y0), width=2*sma, height=2*smb,
        angle=cand.pa_deg, edgecolor='red', facecolor='none', lw=2.0,
        label='photutils'))
    # SMA line for the photutils PA
    pa_rad = np.deg2rad(cand.pa_deg)
    ax.plot([cand.x0 - sma*np.cos(pa_rad), cand.x0 + sma*np.cos(pa_rad)],
            [cand.y0 - sma*np.sin(pa_rad), cand.y0 + sma*np.sin(pa_rad)],
            '-', color='orange', lw=1.0)
    ax.plot([cand.x0], [cand.y0], '+', color='cyan', ms=10, mew=1.6)

    # --- ellipse 2: catalog PA (after transform) + OUR sma/ell (cyan) ---
    pa_corr = transform_pa_jplus(pa_jplus)
    if np.isfinite(pa_corr):
        ax.add_patch(MplEllipse(
            (cand.x0, cand.y0), width=2*sma, height=2*smb,
            angle=pa_corr, edgecolor='cyan', facecolor='none', lw=1.5,
            linestyle='--', label='catalog'))
        diff = circular_diff_deg(cand.pa_deg, pa_corr)
    else:
        diff = float('nan')

    txt = '\\n'.join([
        f\"photutils  PA = {cand.pa_deg:7.2f}°  err {cand.pa_err:5.2f}°\",
        f\"catalog    PA = {pa_jplus:+7.2f}°  -> pa_corr = {pa_corr:7.2f}°\",
        f\"|Δ| = {diff:5.2f}°\",
        f\"SMA = {cand.sma:6.2f} px    ell = {cand.ell:.3f}\",
        f\"config = {cand.config_tag}\",
    ])
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va='top', ha='left',
            color='white', fontsize=8.5, family='monospace',
            bbox=dict(facecolor='black', alpha=0.55, pad=4, edgecolor='none'))
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc='lower right', fontsize=8, framealpha=0.8)

fig, ax = plt.subplots(figsize=(6.0, 6.0))
plot_overlay(ax, img, cand, PA_JPLUS_RAW, title=\"one galaxy · one band (rSDSS-like)\")
fig.tight_layout()
"""))

# ---------------------------------------------------------------------------
CELLS.append(nb_md("""
## 2 · One galaxy, all 12 bands

Now we vary noise level and PSF FWHM across the 12 bands to mimic
J-PLUS depth ordering (the bluest narrowbands are noisiest). All bands
fit the same underlying galaxy, so the PAs should agree.
"""))

CELLS.append(nb_code("""
# Per-band noise multipliers, roughly matching real depth ordering.
BAND_NOISE = {
    'uJAVA': 0.18, 'J0378': 0.16, 'J0395': 0.15, 'J0410': 0.12,
    'J0430': 0.11, 'gSDSS': 0.06, 'J0515': 0.10, 'rSDSS': 0.05,
    'J0660': 0.09, 'iSDSS': 0.05, 'J0861': 0.08, 'zSDSS': 0.06,
}

band_fits = {}
band_imgs = {}
TUTORIAL_BANDS = [b for b in ALL_BANDS if b in ("uJAVA","J0395","gSDSS","rSDSS","J0660","iSDSS","J0861","zSDSS")]
for band in TUTORIAL_BANDS:
    img_b = make_synthetic_galaxy(pa_deg=TRUE_PA, sma=TRUE_SMA, ell=TRUE_ELL,
                                  noise=BAND_NOISE[band], seed=hash(band) % 1000)
    band_imgs[band] = img_b
    cand_b, _ = fit_pa_with_fallbacks(
        data=img_b, eps_prior=B_WORLD/A_WORLD, pa_prior=PA_JPLUS_RAW,
        keep_best_of=2,
    )
    band_fits[band] = cand_b

# 3x4 mosaic
fig, axes = plt.subplots(2, 4, figsize=(13, 6.8))
for ax, band in zip(axes.flat, TUTORIAL_BANDS):
    plot_overlay(ax, band_imgs[band], band_fits[band], PA_JPLUS_RAW, title=band)
fig.suptitle(\"one galaxy · all 12 bands · red=photutils  cyan(dashed)=catalog\",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
"""))

# ---------------------------------------------------------------------------
CELLS.append(nb_md("""
## 3 · Multiple galaxies, one band

Three galaxies with different intrinsic PAs (and different catalog
priors), all fitted in band `rSDSS`. This is what `palfitology fit-pa
--bands rSDSS` does internally for every catalog row.
"""))

CELLS.append(nb_code("""
GALAXIES = [
    # id          true_pa  true_sma  true_ell   pa_jplus_raw
    ('GAL-001',    18.0,    30.0,    0.55,       -72.0),  # pa_corr = 72
    ('GAL-002',    90.0,    24.0,    0.45,         0.0),  # pa_corr = 180 ~ 0
    ('GAL-003',   135.0,    32.0,    0.62,       -45.0),  # pa_corr = 45
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
for ax, (gid, pa_true, sma_true, ell_true, pa_cat) in zip(axes, GALAXIES):
    im = make_synthetic_galaxy(pa_deg=pa_true, sma=sma_true,
                               ell=ell_true, seed=hash(gid) % 1000)
    c, _ = fit_pa_with_fallbacks(
        data=im,
        eps_prior=(1 - ell_true),
        pa_prior=pa_cat,
        keep_best_of=2,
    )
    plot_overlay(ax, im, c, pa_cat,
                 title=f\"{gid}  true PA={pa_true}°  (rSDSS)\")
fig.tight_layout()
"""))

# ---------------------------------------------------------------------------
CELLS.append(nb_md("""
## 4 · Multiple galaxies, all bands · the full pipeline

This is what you run on the cluster. We:

1. Generate synthetic data on disk that mimics `images/<id>/...`
2. Build a small catalog CSV
3. Call `fit_catalog()` (the same entry point `palfitology fit-pa`
   uses) to produce `PA_results.csv` and per-object mosaics
4. Show the master CSV and a couple of mosaics

If you have real data on the cluster, replace the synthetic-data setup
with paths to your `images/` folder and `<catalog>.csv` and rerun this
cell. The pipeline machinery is identical.
"""))

CELLS.append(nb_code("""
import pandas as pd

# ----- 1. set up a temp project directory -----
tmp = Path(tempfile.mkdtemp(prefix=\"palfit_tutorial_\"))
images_root = tmp / \"images\"
output_dir  = tmp / \"fitted_pa_images\"
images_root.mkdir(parents=True, exist_ok=True)

DEMO_BANDS = [\"gSDSS\", \"rSDSS\", \"iSDSS\", \"zSDSS\"]   # 4 bands for tutorial speed

def write_object_data(oid, pa_true, sma_true, ell_true, ra=180.0, dec=0.0):
    obj_dir = images_root / oid
    fits_dir = obj_dir / f\"fits_images_{ra:.4f}_{dec:.4f}\"
    psf_dir  = obj_dir / f\"psfs_{ra:.4f}_{dec:.4f}\"
    fits_dir.mkdir(parents=True, exist_ok=True)
    psf_dir.mkdir(parents=True, exist_ok=True)
    for band in DEMO_BANDS:
        img_b = make_synthetic_galaxy(pa_deg=pa_true, sma=sma_true,
                                      ell=ell_true,
                                      noise=BAND_NOISE[band],
                                      seed=hash((oid, band)) % 10000)
        fits.PrimaryHDU(data=img_b.astype(np.float32)).writeto(
            fits_dir / f\"{band}_cutout.fits\", overwrite=True)
        psf_b = make_synthetic_psf(fwhm_px=2.6 + 0.5 * np.random.rand())
        # PSFEx layout: primary HDU header-only, image in extension 1
        hdul = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data=psf_b.astype(np.float32))])
        hdul.writeto(psf_dir / f\"psf_{ra:.4f}_{dec:.4f}_{band}.fits\", overwrite=True)

galaxies = [
    ('GAL-001',  18.0, 30.0, 0.55, -72.0),
    ('GAL-002',  90.0, 24.0, 0.45,   0.0),
    ('GAL-003', 135.0, 32.0, 0.62, -45.0),
]
for gid, pa_t, sma_t, ell_t, _ in galaxies:
    write_object_data(gid, pa_t, sma_t, ell_t)

# ----- 2. tiny catalog -----
cat_path = tmp / \"catalog.csv\"
pd.DataFrame([
    {\"id\": g[0], \"A_WORLD\": 0.013, \"B_WORLD\": 0.013*(1-g[3]),
     \"pa_jplus\": g[4], \"R_EFF\": g[2]}
    for g in galaxies
]).to_csv(cat_path, index=False)
catalog_df = pd.read_csv(cat_path)
print(\"catalog:\")
catalog_df
"""))

CELLS.append(nb_code("""
# ----- 3. run the pipeline -----
results = fit_catalog(
    images_root=images_root,
    output_dir=output_dir,
    catalog=catalog_df,
    bands=DEMO_BANDS,
    workers=1,             # set to NSLOTS on the cluster
    keep_best_of=2,        # demo: stop after 2 strong fits; default 8
    psf_mode=\"auto\",
    psf_gate=0.2,
    make_summary=True,
)
print(f\"fit_catalog produced {len(results)} rows\")
results.head()
"""))

CELLS.append(nb_code("""
# ----- 4. inspect a mosaic -----
from IPython.display import Image as IpyImage, display

for gid in ['GAL-001', 'GAL-002', 'GAL-003']:
    mosaic = output_dir / gid / f\"{gid}_summary.png\"
    if mosaic.exists():
        print(f\"--- {gid} mosaic ---\")
        display(IpyImage(filename=str(mosaic)))
"""))

CELLS.append(nb_code("""
# ----- 5. summary stats -----
print(\"status by band:\")
print(results.groupby(['band','status']).size().unstack(fill_value=0))
print()
print(\"psf_mode distribution:\")
print(results['psf_mode'].value_counts())
"""))

# ---------------------------------------------------------------------------
CELLS.append(nb_md("""
## 5 · Next steps

Now you'd typically run:

```bash
palfitology reconcile --plot          # PA scatter against catalog pa_jplus
palfitology consensus                  # one PA per object across the 12 bands
```

For a real cluster run, see `scripts/submit_all.sh` and the architecture
map at <https://nisach02.github.io/palfitology/>.
"""))


# ---------------------------------------------------------------------------
NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.x",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT = Path(__file__).resolve().parent / "tutorial.ipynb"
OUT.write_text(json.dumps(NOTEBOOK, indent=1))
print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(CELLS)} cells)")
