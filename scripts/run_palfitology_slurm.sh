#!/bin/bash
#SBATCH --job-name=palfitology
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=04:00:00
#SBATCH --output=output/slurm_%j.out
#SBATCH --error=output/slurm_%j.err

# Activate the env containing palfitology.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate palfitology

# Pin BLAS to one thread per worker to avoid oversubscribing the node.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Some HPC sites need this to avoid static-TLS dlopen failures from scipy.
export LD_PRELOAD=$CONDA_PREFIX/lib/libgomp.so.1

mkdir -p output
palfitology fit-pa --all --workers $SLURM_CPUS_PER_TASK
