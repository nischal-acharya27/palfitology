#!/bin/bash
#$ -N palfit_v0.2
#$ -pe parallel 40
#$ -cwd
#$ -j y
#$ -e output/error_$JOB_ID.$TASK_ID
#$ -o output/output_$JOB_ID.$TASK_ID
#$ -t 1-8
#$ -tc 8

# Array job: 8 tasks total, up to 8 nodes running concurrently (-tc 8). Each
# task is one SGE job slot of 40 cores (-pe parallel 40) and processes one
# pre-split chunk of the catalog. This script assumes `prepare_chunks.sh` has
# already created cat_part_01.csv ... cat_part_08.csv in the cwd.

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
source ~/anaconda3/etc/profile.d/conda.sh
conda activate palfitology

# Static-TLS workaround for the scipy.special dlopen error on hpc-login.
export LD_PRELOAD=$CONDA_PREFIX/lib/libgomp.so.1

# Pin BLAS / OpenMP threads at the shell level so the 40 worker processes
# don't oversubscribe the node. palfitology also does this in cli.py and
# pipeline._worker_init, but setting it here covers any subprocess.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p output

# ---------------------------------------------------------------------------
# Per-task config
# ---------------------------------------------------------------------------
# SGE_TASK_ID runs 1..8. Pad to 2 digits to match the chunk filenames.
TASK_PAD=$(printf "%02d" "$SGE_TASK_ID")
CHUNK_CSV="cat_part_${TASK_PAD}.csv"
TASK_OUT_DIR="fitted_pa_images_part_${TASK_PAD}"

if [ ! -f "$CHUNK_CSV" ]; then
    echo "[task $SGE_TASK_ID] ERROR: chunk file $CHUNK_CSV not found in $(pwd)" >&2
    echo "[task $SGE_TASK_ID] Run prepare_chunks.sh first." >&2
    exit 1
fi

# Where the per-object image folders live. Override on the qsub command line
# with `qsub -v IMAGES_ROOT=/some/path scripts/run_palfitology_array.sh` if
# the data folder moves again.
IMAGES_ROOT="${IMAGES_ROOT:-$HOME/PALFITology_OLD/images}"

if [ ! -d "$IMAGES_ROOT" ]; then
    echo "[task $SGE_TASK_ID] ERROR: IMAGES_ROOT=$IMAGES_ROOT does not exist." >&2
    exit 2
fi

echo "[task $SGE_TASK_ID] catalog=$CHUNK_CSV  images=$IMAGES_ROOT  out=$TASK_OUT_DIR  cores=$NSLOTS"

# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
palfitology fit-pa \
    --catalog "$CHUNK_CSV" \
    --images-root "$IMAGES_ROOT" \
    --output-dir "$TASK_OUT_DIR" \
    --workers "$NSLOTS" \
    --psf-mode auto \
    --psf-gate 0.2

echo "[task $SGE_TASK_ID] done."
