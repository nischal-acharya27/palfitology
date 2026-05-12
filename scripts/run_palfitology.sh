#!/bin/bash
#$ -N palfitology
#$ -pe parallel 40
#$ -cwd
#$ -j y
#$ -e output/error_$JOB_ID
#$ -o output/output_$JOB_ID

# Activate the conda environment with palfitology installed.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate palfitology

# Static-TLS workaround for the scipy.special dlopen error on this node.
export LD_PRELOAD=$CONDA_PREFIX/lib/libgomp.so.1

# Pin BLAS / OpenMP to 1 thread per worker so 40 workers don't oversubscribe
# the node. palfitology also does this internally, but setting it here covers
# any subprocess it spawns.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p output
palfitology fit-pa --all --workers $NSLOTS
