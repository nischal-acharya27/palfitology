#!/bin/bash
#$ -N palfit_clipped
#$ -pe parallel 40
#$ -cwd
#$ -j y
#$ -e output/error_$JOB_ID.$TASK_ID
#$ -o output/output_$JOB_ID.$TASK_ID
#$ -t 1-8
#$ -tc 6

# Array job: 8 tasks, up to 6 concurrent (240 slots = full main.q at 40
# cores/task). Each task runs `palfitology fit-pa --use-clipped-cutouts`
# on one pre-split catalog chunk and writes per-task results to
# fitted_pa_clipped_part_<NN>/.
#
# Submit after run_makecutouts_array.sh:
#     CLIP_JOB=$(qsub -terse scripts/run_makecutouts_array.sh | cut -d. -f1)
#     qsub -hold_jid $CLIP_JOB scripts/run_fitpa_clipped_array.sh
#
# `-hold_jid` on an array waits for *all* tasks of the held job to finish
# before any task here starts -- which is what we want because each fit-pa
# task reads clipped FITS that the corresponding make-cutouts task wrote.

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
source ~/anaconda3/etc/profile.d/conda.sh
conda activate palfitology

# Static-TLS workaround.
export LD_PRELOAD=$CONDA_PREFIX/lib/libgomp.so.1

# Pin BLAS / OpenMP at the shell level so 40 workers don't oversubscribe.
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
TASK_OUT_DIR="fitted_pa_clipped_part_${TASK_PAD}"
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

echo "[task $SGE_TASK_ID] fit-pa  catalog=$CHUNK_CSV  images=$IMAGES_ROOT  out=$TASK_OUT_DIR  cores=$NSLOTS"

# ---------------------------------------------------------------------------
# Fit PA on the clipped cutouts. Falls back to original cutouts where the
# clipped sibling is absent (records cutout_source='original' in the CSV).
# ---------------------------------------------------------------------------
palfitology fit-pa \
    --catalog "$CHUNK_CSV" \
    --images-root "$IMAGES_ROOT" \
    --output-dir "$TASK_OUT_DIR" \
    --workers "$NSLOTS" \
    --use-clipped-cutouts \
    --psf-mode auto \
    --psf-gate 0.2

echo "[task $SGE_TASK_ID] fit-pa done."
