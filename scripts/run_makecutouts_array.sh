#!/bin/bash
#$ -N palfit_clip
#$ -pe parallel 40
#$ -cwd
#$ -j y
#$ -e output/error_$JOB_ID.$TASK_ID
#$ -o output/output_$JOB_ID.$TASK_ID
#$ -t 1-8
#$ -tc 6

# Array job: 8 tasks, up to 6 concurrent (240 slots = full main.q budget at
# 40 cores/task). Each task runs `palfitology make-cutouts` on one
# pre-split chunk of the catalog. Assumes `prepare_chunks.sh` has already
# created cat_part_01.csv ... cat_part_08.csv in the cwd.
#
# Pair with run_fitpa_clipped_array.sh -- submit clip first, then submit
# fit-pa with `-hold_jid <clip_job_id>` so fit-pa only starts once every
# chunk's clipped cutouts are on disk.
#
# Note: at V0.6 `make-cutouts` is single-threaded per process. The 40-core
# pe reservation gives the task a whole node so memory pressure from FITS
# I/O is comfortable, but only 1 core is actually busy per task. Drop to
# `#$ -pe parallel 1` if you want to be polite about slot accounting; the
# tradeoff is potential I/O contention if you co-locate tasks on a node.

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
source ~/anaconda3/etc/profile.d/conda.sh
conda activate palfitology

# Static-TLS workaround for the scipy.special dlopen error on hpc-login.
export LD_PRELOAD=$CONDA_PREFIX/lib/libgomp.so.1

# Pin BLAS / OpenMP threads at the shell level.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p output

# ---------------------------------------------------------------------------
# Per-task config
# ---------------------------------------------------------------------------
TASK_PAD=$(printf "%02d" "$SGE_TASK_ID")
CHUNK_CSV="cat_part_${TASK_PAD}.csv"
IMAGES_ROOT="${IMAGES_ROOT:-$HOME/PALFITology_OLD/images}"

if [ ! -f "$CHUNK_CSV" ]; then
    echo "[task $SGE_TASK_ID] ERROR: chunk file $CHUNK_CSV not found in $(pwd)" >&2
    echo "[task $SGE_TASK_ID] Run prepare_chunks.sh first." >&2
    exit 1
fi

if [ ! -d "$IMAGES_ROOT" ]; then
    echo "[task $SGE_TASK_ID] ERROR: IMAGES_ROOT=$IMAGES_ROOT does not exist." >&2
    exit 2
fi

echo "[task $SGE_TASK_ID] make-cutouts  catalog=$CHUNK_CSV  images=$IMAGES_ROOT"

# ---------------------------------------------------------------------------
# Write the NaN-clipped FITS siblings for EVERY band, not just the detect
# band. Without --apply-bands all, make-cutouts only writes the rSDSS
# clipped sibling and fit-pa --use-clipped-cutouts then falls back to raw
# cutouts for the other 11 bands. That's exactly the "rSDSS is clipped,
# other 11 are not" symptom we want to avoid.
# Output: <images>/<id>/clipped_cutouts_<ra>_<dec>/<band>_cutout.fits for
# each band in the J-PLUS canonical list.
# ---------------------------------------------------------------------------
palfitology make-cutouts \
    --catalog "$CHUNK_CSV" \
    --images-root "$IMAGES_ROOT" \
    --apply-bands all \
    --all

echo "[task $SGE_TASK_ID] make-cutouts done."
