#!/bin/bash -l

# --------------------------------------------------------------------------------
# SGE Job Directives
# --------------------------------------------------------------------------------
# Project name
#$ -P cancergrp

# Hard time limit (hh:mm:ss). Adjust as needed.
#$ -l h_rt=48:00:00

# Job name
#$ -N PeptideProperties

# Merge stdout and stderr into a single file (.o#jobID)
#$ -j y

# Email notification: job ends (e)
#$ -m e

# Your email address for notifications
#$ -M phitro@bu.edu

# Request parallel environment for shared memory applications
#$ -pe omp 4

# Request memory per core. Total memory = N_cores * mem_per_core.
#$ -l mem_per_core=8G

# --------------------------------------------------------------------------------
# Set up Conda environment
# --------------------------------------------------------------------------------

echo "Job started on host: $(hostname)"
echo "Loading Conda environment..."
module load miniconda

# Activate the specific Conda environment created earlier
conda activate my_analysis_env

# Verify environment is active (optional, good for debugging)
echo "Conda environment active: $CONDA_DEFAULT_ENV (prefix: $CONDA_PREFIX)"
echo "Current Python executable: $(which python)"
echo "Python version: $(python --version)"

# --------------------------------------------------------------------------------
# Define paths and arguments
# --------------------------------------------------------------------------------
# IMPORTANT: Adjust PROJECT_DIR to the actual path where you store your scripts and data.
PROJECT_DIR="/projectnb/cancergrp/Philipp" 

OUTPUT_DIR="${PROJECT_DIR}/results/RITA_peptides"
PYTHON_SCRIPT="${PROJECT_DIR}/calculate_peptide_properties.py"

# Create the output directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_DIR")"

# --------------------------------------------------------------------------------
# Run Python script
# --------------------------------------------------------------------------------

echo "Requested number of CPU cores (NSLOTS): $NSLOTS"
echo "Running Python script: $PYTHON_SCRIPT"

# Execute the Python script with all arguments
python "$PYTHON_SCRIPT" 

# Check the exit status of the python script
if [ $? -eq 0 ]; then
    echo "Python script finished successfully."
else
    echo "Python script failed with error code $?."
    exit 1 # Exit with error status
fi

# --------------------------------------------------------------------------------
# Post-job cleanup
# --------------------------------------------------------------------------------

# Deactivate the Conda environment.
if [ -n "$CONDA_PREFIX" ]; then
    conda deactivate
    echo "Conda environment deactivated."
fi

echo "Job completed."
