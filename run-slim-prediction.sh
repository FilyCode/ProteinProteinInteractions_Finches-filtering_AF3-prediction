#!/bin/bash -l

# Before running the script make sure to have created the conda environment
# as well as checked all the necessary tools to load and their paths
# create  SLiMSuite/settings/defaults.ini containing:
## defaults.ini - SLiMSuite Global Settings for External Programs
## --- External Program Paths ---
## BLAST+ expects the directory containing the executables (e.g., blastp)
#blastpath={path}/blast+/2.12.0/install/bin/
#blast+path={path}/blast+/2.12.0/install/bin/
#
## Alignment tools expect the full path to the executable
#clustalw={path}/clustal/clustal-omega/1.2.1/bin/clustalo
#clustalo={path}/clustal/clustal-omega/1.2.1/bin/clustalo
#muscle={path}/muscle/3.8.31/bin/muscle         
#
## IUPRED is not needed as 'masking=F' is set in the Python script for peptide analysis.
#
## --- System Settings (for Linux/Unix environments) ---
#win32=F # Set to False for Linux/Unix systems


# --------------------------------------------------------------------------------
# SGE Job Directives
# --------------------------------------------------------------------------------
# Project name
#$ -P cancergrp

# Hard time limit (hh:mm:ss). Adjust as needed.
#$ -l h_rt=04:00:00

# Job name
#$ -N SlimPrediction

# Merge stdout and stderr into a single file (.o#jobID)
#$ -j y

# Email notification: job ends (e)
#$ -m e

# Your email address for notifications
#$ -M phitro@bu.edu

# Request parallel environment for shared memory applications
#$ -pe omp 16

# Request memory per core. Total memory = N_cores * mem_per_core.
#$ -l mem_per_core=1G



# --------------------------------------------------------------------------------
# Load required system modules.
# --------------------------------------------------------------------------------
module load blast+           # Loads blast+/2.12.0 (or whatever is default 'blast+')
module load clustalomega     # Loads clustalomega/1.2.1
module load muscle           # Loads muscle/3.8.31

# Verify loading of modules
echo "Currently loaded modules: $(module list)"

# --------------------------------------------------------------------------------
# Set up Conda environment
# --------------------------------------------------------------------------------

echo "Job started on host: $(hostname)"
echo "Loading Conda environment..."
module load miniconda

# Activate the specific Conda environment created earlier
conda activate slimsuite_env

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
PYTHON_SCRIPT="${PROJECT_DIR}/find-slims-for-different-peptide-groups.py"

# Create the output directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_DIR")"

# SLiMFinder's 'forks' option needs to be passed through the Python script.
NUM_FORKS=$NSLOTS
if [ -z "$NUM_FORKS" ]; then
    echo "WARNING: NSLOTS not set by SGE, defaulting NUM_FORKS to 1."
    NUM_FORKS=1
fi

# --------------------------------------------------------------------------------
# Run Python script
# --------------------------------------------------------------------------------

echo "Requested number of CPU cores (NSLOTS): $NSLOTS"
echo "Running Python script: $PYTHON_SCRIPT"

# Execute the Python script with all arguments
python "$PYTHON_SCRIPT" --num_forks "$NUM_FORKS"

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
