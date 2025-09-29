#!/bin/bash -l

# --------------------------------------------------------------------------------
# SGE Job Directives
#
# These directives instruct the Grid Engine (SGE) on how to allocate resources
# and manage this array job for AlphaFold3 data pipeline (feature generation).
# --------------------------------------------------------------------------------

#$ -P cancergrp                    # Specify the project name for accounting and resource allocation.
#$ -l h_rt=08:00:00                # Set the hard time limit for each array task (hh:mm:ss).
                                   # Adjust based on your expected feature generation runtime.
#$ -N af3_data_pipeline_array      # Assign a descriptive name to this job array.
#$ -j y                            # Merge standard output (stdout) and standard error (stderr) into a single file.
#$ -o af3_data_pipeline_array.out  # Define the output file for the merged stdout/stderr for all array tasks.
#$ -m e                            # Email notification (job begins (b), ends (e), is aborted (a), suspended (s), or never (n))

# Resource Requests for Data Pipeline:
# AlphaFold3 feature generation benefits from multiple CPU cores for MSA generation.
#$ -pe omp 16                      # Request a parallel environment for shared memory (OpenMP) applications,
                                  # specifying the number of CPU cores (e.g., for multiprocessing.Pool).
#$ -l mem_per_core=8G              # Request memory per core. Total memory will be N_cores * mem_per_core (16 * 4G = 64G).
                                  # This is generally sufficient for most feature generation tasks.

# --------------------------------------------------------------------------------
# Environment Setup and Working Directory Configuration
#
# This section prepares the environment by loading necessary modules and setting
# up critical paths for the AlphaFold3 data pipeline.
# --------------------------------------------------------------------------------

echo "------------------------------------------------------------"
echo "Job started on host: $(hostname)"
echo "Initial working directory: $(pwd)"
echo "Job ID: ${JOB_ID}, Array Task ID: ${SGE_TASK_ID}"
echo "------------------------------------------------------------"

# IMPORTANT: Define the absolute path to your project's base directory.
# All relative paths for inputs/outputs in this script will be resolved against this base.
PROJECT_BASE_DIR="/projectnb/cancergrp/Philipp/alphafold3_pipeline" 

# Change to the project base directory. This is good practice to ensure consistent
# relative path interpretation, though absolute paths are preferred for robustness.
cd "$PROJECT_BASE_DIR" || { 
    echo "CRITICAL Error: Failed to change to project directory: $PROJECT_BASE_DIR" >&2
    exit 1 
}

echo "Changed working directory to: $(pwd)"
echo "Loading AlphaFold3 module..."

# Load the AlphaFold3 module with the explicit version.
# This makes the 'run_alphafold.sh' wrapper script available in the PATH.
module load alphafold3/3.0.0 || { 
    echo "CRITICAL Error: Failed to load alphafold3/3.0.0 module." >&2
    exit 1 
}
echo "AlphaFold3 module loaded."

# --- IMPORTANT: Specify the absolute path to your AlphaFold3 model weights ---
# This directory should contain the downloaded AlphaFold3 model parameters.
# It's defined relative to the PROJECT_BASE_DIR for easy management.
RELATIVE_MODEL_DIR="alphafold3_models"
export MODEL_DIR="$PROJECT_BASE_DIR/$RELATIVE_MODEL_DIR"
echo "AlphaFold3 model directory set to: $MODEL_DIR"

# JAX compilation cache - SCC documentation recommends using $SCRATCH or $TMPDIR.
# This helps speed up subsequent runs by caching compiled XLA computations.
export JAX_COMPILATION_CACHE_DIR="${SCRATCH:-$TMPDIR}/jax_comp_cache"
# Ensure the JAX compilation cache directory exists.
mkdir -p "${JAX_COMPILATION_CACHE_DIR}" || {
    echo "CRITICAL Error: Failed to create JAX compilation cache directory: ${JAX_COMPILATION_CACHE_DIR}" >&2
    exit 1
}
echo "JAX compilation cache directory set to: ${JAX_COMPILATION_CACHE_DIR}"

# --------------------------------------------------------------------------------
# AlphaFold3 Data Pipeline Execution
#
# This section handles fetching input, preparing output paths (including nested
# directory structures), implementing resume logic, and finally executing the
# AlphaFold3 data pipeline.
# --------------------------------------------------------------------------------

echo "------------------------------------------------------------"
echo "Starting AlphaFold3 Data Pipeline for task ${SGE_TASK_ID}"
echo "------------------------------------------------------------"

# Get the relative JSON input file path for this array task from 'json_list.txt'.
# Each line in 'json_list.txt' should correspond to an SGE_TASK_ID.
RELATIVE_JSON_PATH=$(sed -n "${SGE_TASK_ID}p" json_list.txt)

if [ -z "$RELATIVE_JSON_PATH" ]; then
    echo "CRITICAL Error: No JSON file path found for array task ID ${SGE_TASK_ID} in json_list.txt." >&2
    exit 1
fi

# Construct the ABSOLUTE path to the JSON input file.
ABS_JSON_INPUT_FILE="$PROJECT_BASE_DIR/$RELATIVE_JSON_PATH"

echo "Task ID ${SGE_TASK_ID}: Relative JSON path from json_list.txt: $RELATIVE_JSON_PATH"
echo "Task ID ${SGE_TASK_ID}: Absolute JSON path being used: $ABS_JSON_INPUT_FILE"

# Debugging: Verify the input file exists and has correct permissions.
echo "Task ID ${SGE_TASK_ID}: Verifying input JSON file existence and permissions:"
ls -l "$ABS_JSON_INPUT_FILE"

if [ ! -f "$ABS_JSON_INPUT_FILE" ]; then
    echo "CRITICAL Error: Task ID ${SGE_TASK_ID}: Input JSON file not found at expected ABSOLUTE path: $ABS_JSON_INPUT_FILE" >&2
    echo "Please ensure 'json_list.txt' entries are correct and 'PROJECT_BASE_DIR' is accurately set." >&2
    exit 1
fi

# Extract the base job name from the input JSON filename (e.g., "my_protein.json" -> "my_protein").
# Convert the job name to lowercase to match AlphaFold's default output directory naming convention.
JOB_NAME=$(basename "$RELATIVE_JSON_PATH" .json | tr '[:upper:]' '[:lower:]')

# Define the base output directory where all data pipeline JSONs will reside.
RELATIVE_DATA_JSON_OUTPUT_BASE_DIR="af3_data_json"
ABS_DATA_JSON_OUTPUT_ROOT_DIR="$PROJECT_BASE_DIR/$RELATIVE_DATA_JSON_OUTPUT_BASE_DIR"

# Extract the subdirectory path from the RELATIVE_JSON_PATH,
# removing the "af3_inputs/" prefix.
SUBDIR_PATH=$(dirname "$RELATIVE_JSON_PATH" | sed 's|^af3_inputs/||')

# AlphaFold expects the *parent* directory for its output. It will create a subdir named by $JOB_NAME
ABS_AF3_OUTPUT_PARENT_DIR="$ABS_DATA_JSON_OUTPUT_ROOT_DIR/$SUBDIR_PATH"

# Construct the FULL, specific output directory path for this job.
# This ensures that the output structure mirrors the input structure under 'af3_inputs'.
# Example: PROJECT_BASE_DIR/af3_data_json/positive-controls/my_protein/
EXPECTED_AF3_CREATED_JOB_DIR_INNER="$ABS_AF3_OUTPUT_PARENT_DIR/$JOB_NAME"
EXPECTED_OUTPUT_FILE="$EXPECTED_AF3_CREATED_JOB_DIR_INNER/${JOB_NAME}_data.json"

# --- Resume Logic for Data Pipeline ---
# Check if the expected output file already exists and is non-empty.
# This prevents redundant calculations for previously completed tasks.

if [ -f "$EXPECTED_OUTPUT_FILE" ] && [ -s "$EXPECTED_OUTPUT_FILE" ]; then
    echo "Task ID ${SGE_TASK_ID}: Output file for '$JOB_NAME' already exists and is non-empty: $EXPECTED_OUTPUT_FILE."
    echo "Task ID ${SGE_TASK_ID}: Skipping data pipeline calculation for this task."
    exit 0 # Exit successfully as this task is already completed.
fi
# --- End Resume Logic ---

# Ensure the full, nested output directory path exists with proper permissions
# BEFORE AlphaFold3 attempts to write any files into it.
mkdir -p "$EXPECTED_AF3_CREATED_JOB_DIR_INNER" || { 
    echo "CRITICAL Error: Failed to create output directory: $EXPECTED_AF3_CREATED_JOB_DIR_INNER" >&2
    exit 1 
}
# Set group write permissions to allow collaboration if needed.
chmod 775 "$EXPECTED_AF3_CREATED_JOB_DIR_INNER" 

echo "Task ID ${SGE_TASK_ID}: Processing input: $ABS_JSON_INPUT_FILE"
echo "Task ID ${SGE_TASK_ID}: Output data pipeline results to: $EXPECTED_AF3_CREATED_JOB_DIR_INNER/"

# Run the AlphaFold3 data pipeline using SCC's wrapper script.
# IMPORTANT: Pass ABSOLUTE paths to --json_path and --output_dir.
# --output_dir now points to the specific nested directory for this job's outputs.
# --norun_inference ensures only the data pipeline (feature generation) is run.
$(which run_alphafold.sh) \
    --json_path="$ABS_JSON_INPUT_FILE" \
    --output_dir="$ABS_AF3_OUTPUT_PARENT_DIR" \
    --model_dir="$MODEL_DIR" \
    --norun_inference \
    --jax_compilation_cache_dir="$JAX_COMPILATION_CACHE_DIR"

# Optional: Add a check here to ensure the expected output file was actually created after the run.
if [ ! -f "$EXPECTED_OUTPUT_FILE" ] || [ ! -s "$EXPECTED_OUTPUT_FILE" ]; then
    echo "WARNING: Task ID ${SGE_TASK_ID}: Data pipeline finished for '$JOB_NAME', but expected output file '$EXPECTED_OUTPUT_FILE' is missing or empty!" >&2
    # Consider uncommenting the line below if a missing or empty output file
    # should be treated as a critical failure for the job task.
    # exit 1 
fi

echo "------------------------------------------------------------"
echo "Finished AlphaFold3 Data Pipeline for task ${SGE_TASK_ID}"
echo "------------------------------------------------------------"


# ----- Submit job like this ---------
# First, ensure you have a 'json_list.txt' file in your project's base directory.
# This file should contain one relative JSON input path per line.
#
# Example 'json_list.txt':
# af3_inputs/positive-controls/my_protein_1.json
# af3_inputs/experiment_X/condition_Y/another_protein.json
#
# Calculate the number of tasks based on the number of lines in json_list.txt.
# NUM_TASKS=$(cat json_list.txt | wc -l)
#
# Then, submit the array job:
# qsub -t 1-"${NUM_TASKS}" 02_run_data_pipeline.sh
# ------------------------------------