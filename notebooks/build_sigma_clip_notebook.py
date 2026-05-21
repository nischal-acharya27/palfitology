"""Programmatically build notebooks/sigma_clip_and_fit.ipynb.

Run this script once to (re)generate the notebook.  Kept as a .py file
because nbformat-generated notebooks are easier to maintain than hand-edited
JSON.  Re-run after API changes.

    python notebooks/build_sigma_clip_notebook.py

The notebook walks through:

1. Sigma-clipping one galaxy and overlaying the 12-band PA-fit ellipses on
   the clipped images.
2. Sigma-clipping multiple galaxies and summarising them with a per-object
   raw|clipped diagnostic mosaic.
3. Building an RGB composite from iSDSS / rSDSS / gSDSS.

Uses the ``%matplotlib inline`` backend so figures render directly in
JupyterLab / nbclassic / VS Code.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

OUT = Path(__file__).parent / "sigma_clip_and_fit.ipynb"


def md(src: str) -> dict:
    return new_markdown_cell(source=src.strip("\n"))


def code(src: str) -> dict:
    return new_code_cell(source=src.strip("\n"))


cells = []

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
cells.append(md(r"""
# Sigma-clip + Position-Angle fitting demo

This notebook demonstrates the V0.6 palfitology workflow end-to-end:

1. **One galaxy** — build 12-band synthetic cutouts, sigma-clip on rSDSS,
   apply the mask to every band, then fit the position angle on the clipped
   images.  Show the 12-band ellipse-overlay mosaic.
2. **Multiple galaxies** — repeat for a small synthetic catalog of three
   galaxies with different orientations, and produce a `summarize-cutouts`
   style raw|clipped diagnostic per object.
3. **RGB composite** — build a colour image from the iSDSS / rSDSS / gSDSS
   bands (the J-PLUS bands closest to true red / green / blue).

This notebook uses **synthetic** galaxies so it can be run with no external
data dependencies.  Once you've eyeballed the workflow, run the same
commands as a CLI on your real `images/` folder:

```bash
palfitology make-cutouts        --detect-sigma 3.0 --apply-bands all
palfitology fit-pa              --use-clipped-cutouts
palfitology summarize-cutouts   # per-object raw|clipped diagnostic PNGs
```
"""))

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
cells.append(md("## 1. Setup"))

cells.append(code(r"""
# Inline backend so all matplotlib figures appear right under the cells.
%matplotlib inline

import os
import sys
import shutil
from pathlib import Path

# Make BLAS / OpenMP single-threaded BEFORE numpy is imported so the per-band
# parallelism in fit_catalog doesn't oversubscribe the kernel.
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval

# Editable-install of the package; assume the notebook lives in
# <repo>/notebooks/.  Adjust sys.path so the kernel finds palfitology even
# without `pip install -e .`.
repo_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(repo_root / "src"))

from palfitology import ALL_BANDS, fit_catalog
from palfitology.cutouts import (
    make_cutouts_for_catalog,
    summarize_object_clipped_cutouts,
)
from palfitology.detect import detect_source, make_clipped_cutout
from palfitology.fit import fit_pa_with_fallbacks
from palfitology.images import locate_band_fits

print("palfitology imports ok; ALL_BANDS =", ALL_BANDS)
"""))

# ---------------------------------------------------------------------------
# Build the synthetic image set
# ---------------------------------------------------------------------------
cells.append(md(r"""
## 2. Build a synthetic 12-band image set

We render edge-on Gaussian "galaxies" with a per-object PA and a per-band
brightness scale that mimics the J-PLUS bands' relative throughput.  These
land on disk in the same `fits_images_<ra>_<dec>/<band>_cutout.fits` layout
the palfitology pipeline expects.
"""))

cells.append(code(r"""
def synth_galaxy(shape=(81, 81), sigma_x=14.0, sigma_y=4.0, amp=120.0,
                 bg=10.0, noise=0.6, pa_deg=0.0, seed=0):
    "Render a rotated Gaussian galaxy on a noisy background."
    ny, nx = shape
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    ys, xs = np.mgrid[0:ny, 0:nx]
    theta = np.radians(pa_deg)
    x_rot =  (xs - cx) * np.cos(theta) + (ys - cy) * np.sin(theta)
    y_rot = -(xs - cx) * np.sin(theta) + (ys - cy) * np.cos(theta)
    g = amp * np.exp(-(x_rot ** 2 / (2 * sigma_x ** 2)
                       + y_rot ** 2 / (2 * sigma_y ** 2)))
    rng = np.random.default_rng(seed)
    return bg + g + rng.normal(0, noise, size=shape)


# Work in a temporary directory so the notebook is self-contained and
# re-runnable.  Use a folder *inside* the repo so paths stay readable.
DEMO_DIR = Path("demo_data").resolve()
# ignore_errors=True so prior runs that left unwritable files (e.g. a
# sandboxed kernel) don't break re-execution.
if DEMO_DIR.exists():
    shutil.rmtree(DEMO_DIR, ignore_errors=True)
DEMO_DIR.mkdir(parents=True, exist_ok=True)
print("Demo workspace:", DEMO_DIR)


def write_synthetic_object(images_root: Path, objectid: str, ra_dec: str,
                            pa_deg: float, seed: int):
    "Write 12 synthetic FITS cutouts mirroring the J-PLUS layout."
    fits_dir = images_root / objectid / f"fits_images_{ra_dec}"
    fits_dir.mkdir(parents=True, exist_ok=True)
    base = synth_galaxy(pa_deg=pa_deg, seed=seed)
    for i, band in enumerate(ALL_BANDS):
        # Per-band brightness scale (roughly mimicking J-PLUS throughput).
        scale = 0.7 + 0.05 * i
        data = base * scale
        fits.PrimaryHDU(data=data.astype(np.float32)).writeto(
            fits_dir / f"{band}_cutout.fits", overwrite=True
        )


images_root = DEMO_DIR / "images"
images_root.mkdir(exist_ok=True)

write_synthetic_object(images_root, "obj_A", "10.0_20.0", pa_deg=0.0,  seed=1)
print("Wrote 12 bands for obj_A at PA=0° (galaxy elongated along x).")
print(sorted((images_root / "obj_A" / "fits_images_10.0_20.0").glob("*.fits"))[:3], "...")
"""))

# ---------------------------------------------------------------------------
# Single-object: detect + clip on rSDSS
# ---------------------------------------------------------------------------
cells.append(md(r"""
## 3. One galaxy — sigma-clip on the rSDSS band

Step 1 of the V0.6 workflow: use the rSDSS cutout to derive a sigma-clipped
detection mask, then apply the **same** mask to every other band.  We do
this in-process here so the notebook can plot the mask and the clipped
arrays directly.
"""))

cells.append(code(r"""
# Open the rSDSS cutout for obj_A.
r_path = locate_band_fits(images_root / "obj_A", "rSDSS")
with fits.open(r_path) as hdul:
    r_data = hdul[0].data.astype(float)

# Sigma-clip detection: returns the clipped array, the binary mask, and the
# DetectionResult dataclass with centroid + moments.
r_clipped, mask, det = make_clipped_cutout(r_data, sigma_threshold=3.0)

print(f"detection status   : {det.status}")
print(f"pixels above 3σ    : {det.npix}")
print(f"centroid (x, y)    : ({det.x0:.2f}, {det.y0:.2f})")
print(f"moment-derived PA  : {det.pa_deg:.2f}°")
print(f"moment-derived eps : {det.eps:.3f}")
print(f"background ± rms   : {det.background:.3g} ± {det.background_rms:.3g}")
"""))

cells.append(code(r"""
# Visualise raw | mask | clipped, side by side.
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
norm = ImageNormalize(r_data, interval=ZScaleInterval(), stretch=AsinhStretch())

axes[0].imshow(r_data, origin="lower", cmap="gray_r", norm=norm)
axes[0].set_title("rSDSS raw")
axes[0].set_xticks([]); axes[0].set_yticks([])

axes[1].imshow(r_data, origin="lower", cmap="gray_r", norm=norm)
axes[1].contour(mask.astype(int), levels=[0.5], colors=["lime"], linewidths=1.5)
axes[1].plot(det.x0, det.y0, "r+", ms=14, mew=2)
axes[1].set_title(f"Detection mask  (σ=3, npix={det.npix})")
axes[1].set_xticks([]); axes[1].set_yticks([])

cmap_nan = plt.cm.gray_r.copy()
cmap_nan.set_bad("crimson")
axes[2].imshow(r_clipped, origin="lower", cmap=cmap_nan, norm=norm)
axes[2].set_title("rSDSS clipped  (NaN → crimson)")
axes[2].set_xticks([]); axes[2].set_yticks([])

fig.tight_layout()
plt.show()
"""))

# ---------------------------------------------------------------------------
# Apply mask to all 12 bands + fit PA on each
# ---------------------------------------------------------------------------
cells.append(md(r"""
## 4. One galaxy — apply the rSDSS mask to all 12 bands and fit PA

We persist the clipped cutouts to disk via `make_cutouts_for_catalog`, then
fit the position angle on each band's clipped image.  Because the isophote
fitter doesn't handle NaN cleanly, we fill the masked region with the
background mean before fitting — that gives photutils a smooth field
outside the source while still suppressing any contaminating neighbours.

This is the recommended workflow until a future palfitology release adds
NaN-aware fitting at the CLI level.
"""))

cells.append(code(r"""
# Build the clipped FITS files in the sibling 'clipped_cutouts_*/' folder.
catalog = pd.DataFrame({
    "id": ["obj_A"],
    "A_WORLD": [1.0],
    "B_WORLD": [0.3],
    "pa_jplus": [0.0],
})
reports = make_cutouts_for_catalog(
    images_root=images_root,
    catalog=catalog,
    detect_band="rSDSS",
    apply_bands=list(ALL_BANDS),
    sigma_threshold=3.0,
)
print(f"wrote {sum(1 for r in reports if r.status == 'ok')} clipped FITS files")
print("first three:", [r.out_path for r in reports[:3]])
"""))

cells.append(code(r"""
# For each band, load the clipped FITS, fill NaN with the background, fit PA.
# We use keep_best_of=2 here so the notebook finishes quickly in CI; the
# default (8) gives slightly tighter fits but takes ~5x longer.
KEEP_BEST_OF = 2

band_results = []
band_images = {}   # the filled image we actually fitted (for plotting)
band_clipped = {}  # the raw NaN-clipped image (for the mosaic overlay)
band_cands = {}

for band in ALL_BANDS:
    clipped_path = next(
        (images_root / "obj_A").glob(f"clipped_cutouts_*/{band}_cutout.fits")
    )
    with fits.open(clipped_path) as hdul:
        data_nan = hdul[0].data.astype(float)

    # Fill NaN with the rSDSS background so photutils can fit.  We use the
    # detection's background; for real data the background varies per band,
    # so re-estimate per-band if you need higher precision.
    finite = np.isfinite(data_nan)
    bg_band = float(np.nanmedian(data_nan[finite])) if finite.any() else 0.0
    data_filled = np.where(finite, data_nan, bg_band)

    cand, n_tried = fit_pa_with_fallbacks(
        data=data_filled,
        eps_prior=0.7,
        pa_prior=0.0,
        min_sma_abs=3.0,
        min_sma_frac=0.05,
        keep_best_of=KEEP_BEST_OF,
    )
    band_clipped[band] = data_nan
    band_images[band] = data_filled
    band_cands[band] = cand
    band_results.append({
        "band": band,
        "fit_status": "ok" if cand is not None and not cand.weak else (
            "weak" if cand is not None else "imputed"
        ),
        "PA_deg": cand.pa_deg if cand is not None else np.nan,
        "ell":    cand.ell    if cand is not None else np.nan,
        "SMA":    cand.sma    if cand is not None else np.nan,
        "PA_err": cand.pa_err if cand is not None else np.nan,
        "n_tried": n_tried,
    })

results_df = pd.DataFrame(band_results)
results_df
"""))

# ---------------------------------------------------------------------------
# 12-band ellipse-overlay mosaic on the clipped images
# ---------------------------------------------------------------------------
cells.append(md(r"""
## 5. One galaxy — 12-band ellipse-overlay mosaic on the clipped images

This is the headline diagnostic: each panel shows the clipped cutout (NaN
in crimson, source in grayscale) with the fitted ellipse overlaid.  The
ellipse should hug the bright source in every band; if any band's ellipse
looks misaligned that's the one to investigate.
"""))

cells.append(code(r"""
from matplotlib.patches import Ellipse as MplEllipse

ncols = 4
nrows = 3  # 12 J-PLUS bands -> 3 x 4 grid
fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.4 * nrows))
cmap_nan = plt.cm.gray_r.copy()
cmap_nan.set_bad("crimson")

for i, band in enumerate(ALL_BANDS):
    ax = axes[i // ncols, i % ncols]
    clipped = band_clipped[band]
    filled  = band_images[band]
    cand    = band_cands[band]

    # Stretch from the filled image so the source dynamic range is preserved.
    norm = ImageNormalize(filled, interval=ZScaleInterval(), stretch=AsinhStretch())
    ax.imshow(clipped, origin="lower", cmap=cmap_nan, norm=norm)

    if cand is not None and np.all(np.isfinite([cand.pa_deg, cand.sma, cand.ell, cand.x0, cand.y0])):
        smb = cand.sma * (1.0 - cand.ell)
        ax.add_patch(MplEllipse(
            (cand.x0, cand.y0),
            width=2 * cand.sma,
            height=2 * smb,
            angle=cand.pa_deg,
            edgecolor="red" if not cand.weak else "orange",
            facecolor="none",
            lw=1.5,
        ))
        ax.plot(cand.x0, cand.y0, "+", color="cyan", ms=8, mew=1.4)
        title = f"{band}\nPA={cand.pa_deg:.1f}°  ell={cand.ell:.2f}"
    else:
        title = f"{band}\n(no fit)"
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

fig.suptitle("obj_A — 12-band ellipse overlays on the sigma-clipped cutouts",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
plt.show()
"""))

# ---------------------------------------------------------------------------
# Multi-galaxy section
# ---------------------------------------------------------------------------
cells.append(md(r"""
## 6. Multiple galaxies — batch sigma-clip + fit

The CLI handles this with `palfitology make-cutouts` + `fit-pa
--use-clipped-cutouts`.  Inside the notebook we drive the same Python API
directly so we can plot results inline.

We render three synthetic galaxies at PA = 0°, 35°, and 75° and check that
the pipeline recovers each.
"""))

cells.append(code(r"""
# Two new objects in addition to obj_A.
write_synthetic_object(images_root, "obj_B", "30.0_40.0", pa_deg=35.0, seed=2)
write_synthetic_object(images_root, "obj_C", "50.0_60.0", pa_deg=75.0, seed=3)

multi_catalog = pd.DataFrame({
    "id":       ["obj_A", "obj_B", "obj_C"],
    "A_WORLD":  [1.0, 1.0, 1.0],
    "B_WORLD":  [0.3, 0.3, 0.3],
    "pa_jplus": [0.0, 35.0, 75.0],
})

# Re-run make-cutouts to cover the two new objects.
reports = make_cutouts_for_catalog(
    images_root=images_root,
    catalog=multi_catalog,
    detect_band="rSDSS",
    apply_bands=list(ALL_BANDS),
    sigma_threshold=3.0,
)
print(f"clipped FITS written: {sum(1 for r in reports if r.status == 'ok')} / {len(reports)}")
"""))

cells.append(code(r"""
# Now fit PA on every (object, band) using the clipped cutouts.  We do the
# fitting ourselves (rather than calling fit_catalog) so we can fill NaN
# with the per-band median first, which photutils needs.
#
# To keep the notebook quick we limit this section to four representative
# bands (one broadband + one narrowband per blue/red half).  Swap in
# ``ALL_BANDS`` if you want the full 12-band grid for each object.
MULTI_BANDS = ["gSDSS", "J0515", "rSDSS", "iSDSS"]
MULTI_KEEP_BEST_OF = 1  # one strong fit per band is enough for the demo

all_rows = []
overlay_data = {}  # objectid -> {band: (clipped_array, cand)}

for objectid in multi_catalog["id"]:
    overlay_data[objectid] = {}
    for band in MULTI_BANDS:
        path = next((images_root / objectid).glob(f"clipped_cutouts_*/{band}_cutout.fits"))
        with fits.open(path) as hdul:
            data_nan = hdul[0].data.astype(float)
        finite = np.isfinite(data_nan)
        bg_band = float(np.nanmedian(data_nan[finite])) if finite.any() else 0.0
        data_filled = np.where(finite, data_nan, bg_band)
        cand, _ = fit_pa_with_fallbacks(
            data=data_filled,
            eps_prior=0.7, pa_prior=0.0,
            min_sma_abs=3.0, min_sma_frac=0.05,
            keep_best_of=MULTI_KEEP_BEST_OF,
        )
        overlay_data[objectid][band] = (data_nan, data_filled, cand)
        all_rows.append({
            "id": objectid, "band": band,
            "PA_deg": cand.pa_deg if cand is not None else np.nan,
            "ell":    cand.ell    if cand is not None else np.nan,
            "fit_ok": cand is not None and not cand.weak,
        })

multi_results = pd.DataFrame(all_rows)
# Per-object median PA across all bands (wrapped to [0, 180)).
multi_results["PA_wrap"] = multi_results["PA_deg"] % 180
summary = (
    multi_results.groupby("id")
    .agg(n_fit=("fit_ok", "sum"),
         PA_median=("PA_wrap", "median"),
         PA_std=("PA_wrap", "std"),
         ell_median=("ell", "median"))
    .join(multi_catalog.set_index("id")["pa_jplus"].rename("PA_true"))
)
summary["PA_resid"] = (summary["PA_median"] - summary["PA_true"] + 90) % 180 - 90
summary
"""))

# ---------------------------------------------------------------------------
# Per-object ellipse mosaic loop
# ---------------------------------------------------------------------------
cells.append(md(r"""
## 7. Multiple galaxies — per-object ellipse mosaics

One figure per object so you can scroll through and spot-check each.  We
show the four representative bands defined in `MULTI_BANDS` above; swap in
the full `ALL_BANDS` list for the 12-band grid (slower).
"""))

cells.append(code(r"""
cmap_nan = plt.cm.gray_r.copy()
cmap_nan.set_bad("crimson")

for objectid in multi_catalog["id"]:
    fig, axes = plt.subplots(1, len(MULTI_BANDS), figsize=(3.4 * len(MULTI_BANDS), 3.6))
    if len(MULTI_BANDS) == 1:
        axes = [axes]
    for i, band in enumerate(MULTI_BANDS):
        ax = axes[i]
        clipped, filled, cand = overlay_data[objectid][band]
        norm = ImageNormalize(filled, interval=ZScaleInterval(), stretch=AsinhStretch())
        ax.imshow(clipped, origin="lower", cmap=cmap_nan, norm=norm)
        if cand is not None and np.all(np.isfinite([cand.pa_deg, cand.sma, cand.ell, cand.x0, cand.y0])):
            smb = cand.sma * (1.0 - cand.ell)
            ax.add_patch(MplEllipse(
                (cand.x0, cand.y0),
                width=2 * cand.sma, height=2 * smb,
                angle=cand.pa_deg,
                edgecolor="red" if not cand.weak else "orange",
                facecolor="none", lw=1.5,
            ))
            title = f"{band}  PA={cand.pa_deg:.1f}°"
        else:
            title = f"{band}  (no fit)"
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    true_pa = float(multi_catalog.loc[multi_catalog.id == objectid, "pa_jplus"].iloc[0])
    fig.suptitle(f"{objectid}  —  true PA = {true_pa:.1f}°", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()
"""))

# ---------------------------------------------------------------------------
# Per-object raw|clipped diagnostic
# ---------------------------------------------------------------------------
cells.append(md(r"""
## 8. Per-object raw|clipped diagnostic

The same mosaic produced by `palfitology summarize-cutouts` from the CLI:
each band shows raw vs. clipped side-by-side so you can see the mask
boundary directly.  We render obj_A only to keep the notebook compact —
loop over `multi_catalog["id"]` to do all of them.
"""))

cells.append(code(r"""
out_png = DEMO_DIR / "obj_A_clipped_summary.png"
status = summarize_object_clipped_cutouts(
    image_dir=images_root / "obj_A",
    bands=list(ALL_BANDS),
    out_path=out_png,
)
print(f"summary status: {status}")

# Display the saved PNG inline.
from IPython.display import Image as IPyImage
IPyImage(filename=str(out_png))
"""))

# ---------------------------------------------------------------------------
# RGB composite
# ---------------------------------------------------------------------------
cells.append(md(r"""
## 9. RGB composite

For J-PLUS the three broad SDSS bands closest to true R / G / B are:

| Channel | Band   | Why |
|---------|--------|------|
| Red     | iSDSS  | Effective wavelength ≈ 7480 Å, the reddest broad band before zSDSS turns near-IR. |
| Green   | rSDSS  | ≈ 6230 Å — sits right in the visual green-yellow. |
| Blue    | gSDSS  | ≈ 4750 Å — broad blue band, cleaner than uJAVA's UV throughput. |

We use `astropy.visualization.make_lupton_rgb` (the Lupton-Blanton-Szalay
arcsinh stretch popularized by SDSS) so the dynamic range across the three
bands is balanced.
"""))

cells.append(code(r"""
from astropy.visualization import make_lupton_rgb

def load_clipped(objectid: str, band: str) -> np.ndarray:
    path = next((images_root / objectid).glob(f"clipped_cutouts_*/{band}_cutout.fits"))
    with fits.open(path) as hdul:
        arr = hdul[0].data.astype(float)
    # Replace NaN with 0 so the RGB stretch isn't dominated by black holes.
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return arr


fig, axes = plt.subplots(1, len(multi_catalog), figsize=(4.5 * len(multi_catalog), 4.5))
if len(multi_catalog) == 1:
    axes = [axes]

for ax, objectid in zip(axes, multi_catalog["id"]):
    r_img = load_clipped(objectid, "iSDSS")
    g_img = load_clipped(objectid, "rSDSS")
    b_img = load_clipped(objectid, "gSDSS")

    # The Lupton stretch needs a sensible 'Q' (asinh softness) and 'stretch'
    # (saturation point).  These defaults work well for J-PLUS-like data;
    # tweak per dataset.
    rgb = make_lupton_rgb(r_img, g_img, b_img, Q=8, stretch=30)
    ax.imshow(rgb, origin="lower")
    ax.set_title(f"{objectid}\nRGB = iSDSS / rSDSS / gSDSS", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

fig.tight_layout()
plt.show()
"""))

# ---------------------------------------------------------------------------
# Closeout
# ---------------------------------------------------------------------------
cells.append(md(r"""
## 10. Next steps

- Replace the synthetic galaxies with your real J-PLUS cutouts by pointing
  `images_root` at your data folder and dropping the `write_synthetic_object`
  calls.
- For batch runs the CLI is faster:
  ```bash
  palfitology make-cutouts        --detect-sigma 3.0 --apply-bands all
  palfitology fit-pa              --use-clipped-cutouts --workers 0
  palfitology summarize-cutouts   --limit 20
  ```
- Open follow-up: the NaN-fill step in Sections 4 and 6 is currently done in
  the notebook because `palfitology fit-pa --use-clipped-cutouts` passes
  NaN-bearing arrays straight to `photutils.isophote`, which often fails
  to converge.  A future release should fill NaN with the background mean
  inside the worker so the CLI workflow Just Works.
"""))

# ---------------------------------------------------------------------------
# Assemble + write
# ---------------------------------------------------------------------------
nb = new_notebook(cells=cells)
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "pygments_lexer": "ipython3",
    },
}

with open(OUT, "w") as f:
    nbf.write(nb, f)
print(f"Wrote {OUT}")
