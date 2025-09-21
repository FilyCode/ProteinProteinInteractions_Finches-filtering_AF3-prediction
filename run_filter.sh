#!/bin/bash -l

# --------------------------------------------------------------------------------
# SGE Job Directives
# --------------------------------------------------------------------------------
# Project name
#$ -P cancergrp

# Hard time limit: hh:mm:ss., Default is 12h so if need to run longer set it here.
# For OMP jobs max is 720 hours (30 days)
#$ -l h_rt=30:00:00

# Job name
#$ -N ViralFinchFilter

# Merge stdout and stderr into a single file (.o#jobID).
#$ -j y

# Email notification (job begins (b), ends (e), is aborted (a), suspended (s), or never (n))
# 'be' for both start and end.
#$ -m e

#$ -M phitro@bu.edu

# Request parallel environment for shared memory applications (like multiprocessing.Pool).
#$ -pe omp 32

# Request memory per core. Total memory = N * mem_per_core.
#$ -l mem_per_core=4G

# --------------------------------------------------------------------------------
# Set up Conda environment
# --------------------------------------------------------------------------------

echo "Loading Conda environment..."
module load miniconda

# Activate specific Conda environment
conda activate finches-env

# Verify environment is active (optional, good for debugging)
echo "Conda environment active: $CONDA_DEFAULT_ENV (prefix: $CONDA_PREFIX)"
echo "Current Python environment: $(which python)"
echo "Python version: $(python --version)"

# --------------------------------------------------------------------------------
# Run Python script
# --------------------------------------------------------------------------------

echo "Job started on host: $(hostname)"
echo "Requested number of CPU cores (NSLOTS): $NSLOTS"

# Run Python script with arguments.
python filter-viral-human-pairs-with-finches.py \
    --num_processes $NSLOTS \
    --input_file_virus /projectnb/cancergrp/Philipp/data/VP_pos_selec_enriched_hits.csv \
    --input_file_human /projectnb/cancergrp/Philipp/data/Human-proteom_GCF_000001405.40.csv \
    --output_file /projectnb/cancergrp/Philipp/results/finches_interaction_results_chunked.parquet \
    --resume

# --------------------------------------------------------------------------------
# Post-job cleanup
# --------------------------------------------------------------------------------

echo "Python script finished."

# Deactivate the Conda environment. 
if [ -n "$CONDA_PREFIX" ]; then
    conda deactivate
    echo "Conda environment deactivated."
fi
