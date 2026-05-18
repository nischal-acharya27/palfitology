#!/bin/bash
#$ -N palfit_reconcile
#$ -pe parallel 4
#$ -cwd
#$ -j y
#$ -e output/error_$JOB_ID
#$ -o output/output_$JOB_ID

# Held by submit_all.sh until the array job finishes. Concatenates the per-task
# PA_results.csv files, copies the per-object outputs into a single
# fitted_pa_images/ tree, and runs `palfitology reconcile --plot` to make the
# pa_jplus vs est_pa scatter plots.

set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate palfitology
export LD_PRELOAD=$CONDA_PREFIX/lib/libgomp.so.1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p output fitted_pa_images fitted_pa_images/all_summaries

# ---------------------------------------------------------------------------
# Merge per-task outputs into a single fitted_pa_images/ tree
# ---------------------------------------------------------------------------
echo "Merging per-task outputs..."

# 1. Per-object subfolders: copy each task's <id>/ subfolders into the master.
#    -n = don't overwrite, so a re-run won't clobber a more recent good fit.
for part in fitted_pa_images_part_*/; do
    [ -d "$part" ] || continue
    # Object subfolders (skip 'all_summaries' which we handle below).
    find "$part" -mindepth 1 -maxdepth 1 -type d ! -name all_summaries -print0 \
        | xargs -0 -I{} cp -rn {} fitted_pa_images/ 2>/dev/null || true
    # all_summaries/*.png mosaics
    if [ -d "${part}all_summaries" ]; then
        cp -n "${part}all_summaries/"*.png fitted_pa_images/all_summaries/ 2>/dev/null || true
    fi
done

# 2. Concatenate the 8 PA_results.csv files into one.
echo "Concatenating PA_results.csv slices..."
python - <<'PY'
import glob
from pathlib import Path
import pandas as pd

parts = sorted(glob.glob("fitted_pa_images_part_*/PA_results.csv"))
if not parts:
    raise SystemExit("No fitted_pa_images_part_*/PA_results.csv files found.")

frames = [pd.read_csv(p) for p in parts]
merged = pd.concat(frames, ignore_index=True)

# Sanity: drop duplicate (id, band) rows that could appear if chunks
# overlapped due to a re-submission. Keep the most recent (last) write.
before = len(merged)
merged = merged.drop_duplicates(subset=["id", "band"], keep="last")
after = len(merged)
if before != after:
    print(f"Dropped {before - after} duplicate (id, band) rows during merge.")

out = Path("fitted_pa_images/PA_results.csv")
merged.to_csv(out, index=False)
print(f"Wrote {len(merged)} rows to {out}")

# Quick stats for the SGE log.
if "psf_mode" in merged.columns:
    print("\npsf_mode distribution:")
    print(merged["psf_mode"].value_counts().to_string())
if "status" in merged.columns:
    print("\nstatus distribution:")
    print(merged["status"].value_counts().to_string())
PY

# ---------------------------------------------------------------------------
# Reconcile + scatter plots
# ---------------------------------------------------------------------------
echo "Running reconcile --plot..."
palfitology reconcile --plot

echo "Done. Outputs are in fitted_pa_images/"
