#!/bin/bash -l

# SGE Job Directives
# --------------------------------------------------------------------------------
# Project name
#$ -P cancergrp

# Hard time limit (hh:mm:ss). Adjust as needed.
#$ -l h_rt=12:00:00

# Job name
#$ -N UniprotSequenceMapper

# Merge stdout and stderr into a single file (.o#jobID).
#$ -j y

# Email notification: job ends (e)
#$ -m e

# Your email address for notifications
#$ -M phitro@bu.edu

# Request 1 core for this single-threaded Python script.
#$ -pe omp 1

# Request 16GB total memory for the job.
#$ -l mem_per_core=8G

# Setup Conda Environment
# --------------------------------------------------------------------------------
echo "Loading Conda environment..."
module load miniconda
conda activate jupyter_env

# Display active environment for debugging
echo "Conda environment active: $CONDA_DEFAULT_ENV (prefix: $CONDA_PREFIX)"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# Prepare Working Directory
# --------------------------------------------------------------------------------
echo "Changing to submission directory: $SGE_O_WORKDIR"
cd $SGE_O_WORKDIR

# Ensure necessary data and checkpoint directories exist
mkdir -p data

# Run Python Script
# --------------------------------------------------------------------------------
echo "Job started on host: $(hostname)"
echo "Requested CPU cores (NSLOTS): $NSLOTS" # Will be 1 with -pe omp 1

python get-sequences-for-interacting-proteins.py

# Post-job Cleanup
# --------------------------------------------------------------------------------
echo "Python script finished."

# Deactivate the Conda environment
if [ -n "$CONDA_PREFIX" ]; then
    conda deactivate
    echo "Conda environment deactivated."
fi