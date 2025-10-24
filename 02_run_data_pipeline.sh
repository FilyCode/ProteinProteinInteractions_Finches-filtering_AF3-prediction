#!/bin/bash -l

# --------------------------------------------------------------------------------
# SGE Job Directives
#
# These directives instruct the Grid Engine (SGE) on how to allocate resources
# and manage this array job for AlphaFold3 data pipeline (feature generation).
# --------------------------------------------------------------------------------

#$ -P cancergrp                    # Specify the project name for accounting and resource allocation.
#$ -l h_rt=24:00:00                # Set the hard time limit for each array task (hh:mm:ss).
                                   # Adjust based on your expected feature generation runtime.
#$ -N af3_data_pipeline_array      # Assign a descriptive name to this job array.
#$ -j y                            # Merge standard output (stdout) and standard error (stderr) into a single file.
#$ -o af3_data_pipeline_array.out  # Define the output file for the merged stdout/stderr for all array tasks.
##$ -m e                            # Email notification (job begins (b), ends (e), is aborted (a), suspended (s), or never (n))

# Resource Requests for Data Pipeline:
# AlphaFold3 feature generation benefits from multiple CPU cores for MSA generation.
#$ -pe omp 16                      # Request a parallel environment for shared memory (OpenMP) applications,
                                   # specifying the number of CPU cores (e.g., for multiprocessing.Pool).
#$ -l mem_per_core=1G              # Request memory per core. Total memory will be N_cores * mem_per_core (16 * 4G = 64G).
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

# --- IMPORTANT: Validate SGE_TASK_ID for array jobs ---
# Check if SGE_TASK_ID is empty OR if it's not a positive integer
if [ -z "${SGE_TASK_ID}" ] || ! [[ "${SGE_TASK_ID}" =~ ^[1-9][0-9]*$ ]]; then
    echo "CRITICAL Error: SGE_TASK_ID is either undefined/empty or not a valid positive number." >&2
    echo "This script is designed for SGE array jobs. Please ensure you submit it using:" >&2
    echo "  1. unset SGE_TASK_ID (to clear any previous value)" >&2
    echo "  2. NUM_TASKS=\$(wc -l < json_list.txt)" >&2
    echo "  3. qsub -t 1-\${NUM_TASKS} 02_run_data_pipeline.sh" >&2
    exit 1
fi

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
# Global Completion Tracking and Work Management Setup
# (Adapted from 03_run_inference.sh for Data Pipeline)
# This section implements logic to track overall job progress and allows early
# termination of the array job if all tasks are globally completed.
# --------------------------------------------------------------------------------

# --- Lock file directory setup ---
# Use a common lock directory for all tasks of this job array.
# This directory will hold shared state for global completion tracking.
LOCK_DIR="$PROJECT_BASE_DIR/af3_data_pipeline_locks/$JOB_ID"
mkdir -p "$LOCK_DIR" || {
    echo "CRITICAL Error: Failed to create lock directory: $LOCK_DIR" >&2
    exit 1
}
echo "Shared lock directory for data pipeline: $LOCK_DIR"

# --- Global completion tracking setup ---
# File to store the total number of tasks from json_list.txt
TOTAL_TASKS_FILE="$LOCK_DIR/total_items.count"
# File to store the count of successfully completed tasks
COMPLETED_TASKS_COUNT_FILE="$LOCK_DIR/completed_items.count"
# Lock file for atomic updates to count files (using a unique FD 200 for flock)
COUNT_LOCK_FILE="$LOCK_DIR/count.lock"
# Define QDEL_SENT_FLAG and QDEL_CONTROL_LOCK
QDEL_SENT_FLAG="$LOCK_DIR/qdel_sent.flag"
QDEL_CONTROL_LOCK="$LOCK_DIR/qdel_control.lock"

# Calculate total tasks once, and store it for all array tasks to read.
# Uses flock for atomic initialization, ensuring only one task performs this.
(
    flock -xn -w 0 200 || sleep 5 # Try non-blocking lock (timeout 0). If it fails, another task has it, wait briefly.
    if [ ! -f "$TOTAL_TASKS_FILE" ]; then
        TOTAL_ITEMS_TO_PROCESS_TEMP=$(wc -l < json_list.txt)
        echo "$TOTAL_ITEMS_TO_PROCESS_TEMP" > "$TOTAL_TASKS_FILE"
        echo "Task ID ${SGE_TASK_ID}: Initialized total tasks count to $TOTAL_ITEMS_TO_PROCESS_TEMP."
    fi
) 200>"$COUNT_LOCK_FILE" # Using FD 200 for the flock lock file

# All tasks will read the pre-calculated total tasks count
TOTAL_ITEMS_TO_PROCESS=$(cat "$TOTAL_TASKS_FILE")
echo "Task ID ${SGE_TASK_ID}: Total items to process: $TOTAL_ITEMS_TO_PROCESS"

# Function to atomically read the completed tasks count
get_completed_count() {
    (
        flock -s 200 # Acquire a shared lock to read the count
        cat "$COMPLETED_TASKS_COUNT_FILE" 2>/dev/null || echo 0
    ) 200>"$COUNT_LOCK_FILE"
}

# Function to attempt sending qdel if all tasks are confirmed completed
trigger_qdel_if_all_done() {
    local current_completion_val="$1" # Argument is the current completed count for logging

    (
        # Acquire an exclusive lock on QDEL_CONTROL_LOCK to ensure only one task tries to issue qdel
        flock -xn -w 5 201 || {
            return 1; # Another task is handling qdel, or lock contention.
        }

        if [ ! -f "$QDEL_SENT_FLAG" ]; then # Only send qdel if not already sent by ANY task
            echo "Task ID ${SGE_TASK_ID}: Global completion detected ($current_completion_val/$TOTAL_ITEMS_TO_PROCESS). Issuing qdel -f $JOB_ID for job array $JOB_ID." >&2
            touch "$QDEL_SENT_FLAG" # Mark that qdel is being handled to prevent others from trying
            qdel -f "$JOB_ID" # Forcefully delete all tasks in this array
            if [ $? -eq 0 ]; then
                echo "Task ID ${SGE_TASK_ID}: Successfully sent qdel for job array $JOB_ID." >&2
            else
                echo "ERROR: Task ID ${SGE_TASK_ID}: Failed to send qdel for job array $JOB_ID. Check permissions or SGE status." >&2
            fi
            return 0 # Qdel initiated
        else
            return 1 # Qdel already initiated
        fi
    ) 201>"$QDEL_CONTROL_LOCK" # Using FD 201 for the qdel control lock file
}

# Function to atomically increment the completed tasks count
increment_completed_count() {
    (
        flock -x 200 # Acquire an exclusive lock on the count file
        local CURRENT_COUNT=$(cat "$COMPLETED_TASKS_COUNT_FILE" 2>/dev/null || echo 0)
        local NEW_COUNT=$((CURRENT_COUNT + 1))
        echo "$NEW_COUNT" > "$COMPLETED_TASKS_COUNT_FILE"
        echo "Task ID ${SGE_TASK_ID}: Incremented completed tasks count to $NEW_COUNT."

        # If this increment makes it globally complete, trigger qdel
        if [ "$NEW_COUNT" -ge "$TOTAL_ITEMS_TO_PROCESS" ]; then
            trigger_qdel_if_all_done "$NEW_COUNT"
        fi
    ) 200>"$COUNT_LOCK_FILE"
}

# Calculate initial completed count and store it atomically
(
    flock -xn -w 0 200 || sleep 5 # Try non-blocking lock, if it fails, another task has it, wait briefly.
    if [ ! -f "$COMPLETED_TASKS_COUNT_FILE" ]; then
        echo "Task ID ${SGE_TASK_ID}: Performing initial scan for already completed tasks to initialize global count..."
        INITIAL_COMPLETED_COUNT=0

        # Define the base output directory for data pipeline JSONs, consistent with later usage.
        RELATIVE_DATA_JSON_OUTPUT_BASE_DIR_LOCAL="af3_data_json"

        # Iterate through json_list.txt to count already existing completed outputs
        while IFS= read -r RELATIVE_ORIGINAL_JSON_PATH_FOR_INIT; do
            if [ -z "$RELATIVE_ORIGINAL_JSON_PATH_FOR_INIT" ]; then
                continue
            fi

            ORIGINAL_JOB_NAME_FOR_INIT=$(basename "$RELATIVE_ORIGINAL_JSON_PATH_FOR_INIT" .json | tr '[:upper:]' '[:lower:]')
            SUBDIR_PATH_FOR_INIT=$(dirname "$RELATIVE_ORIGINAL_JSON_PATH_FOR_INIT" | sed 's|^af3_inputs/||')

            # Construct the expected output path for the data pipeline: JOB_NAME_data.json
            ABS_AF3_OUTPUT_PARENT_DIR_FOR_INIT="$PROJECT_BASE_DIR/$RELATIVE_DATA_JSON_OUTPUT_BASE_DIR_LOCAL/$SUBDIR_PATH_FOR_INIT"
            EXPECTED_AF3_CREATED_JOB_DIR_INNER_FOR_INIT="$ABS_AF3_OUTPUT_PARENT_DIR_FOR_INIT/$ORIGINAL_JOB_NAME_FOR_INIT"
            EXPECTED_OUTPUT_FILE_FOR_INIT="$EXPECTED_AF3_CREATED_JOB_DIR_INNER_FOR_INIT/${ORIGINAL_JOB_NAME_FOR_INIT}_data.json"

            if [ -f "$EXPECTED_OUTPUT_FILE_FOR_INIT" ] && [ -s "$EXPECTED_OUTPUT_FILE_FOR_INIT" ]; then
                INITIAL_COMPLETED_COUNT=$((INITIAL_COMPLETED_COUNT + 1))
            fi
        done < json_list.txt

        echo "$INITIAL_COMPLETED_COUNT" > "$COMPLETED_TASKS_COUNT_FILE"
        echo "Task ID ${SGE_TASK_ID}: Initialized completed tasks count to $INITIAL_COMPLETED_COUNT (found $INITIAL_COMPLETED_COUNT pre-existing outputs)."
    fi

    # Check immediately after initial scan if all tasks are completed.
    # If this task is the first to detect global completion, it should try to send qdel.
    # Removed 'local' from here
    CURRENT_COMPLETED_COUNT=$(cat "$COMPLETED_TASKS_COUNT_FILE") # Read again after potential update
    if [ "$CURRENT_COMPLETED_COUNT" -ge "$TOTAL_ITEMS_TO_PROCESS" ]; then
        trigger_qdel_if_all_done "$CURRENT_COMPLETED_COUNT" # Pass current count for logging
    fi

) 200>"$COUNT_LOCK_FILE"

# At this point, the script can check if a qdel signal has been sent *before*
# even attempting to find its own task, providing an even earlier exit.
if [ -f "$QDEL_SENT_FLAG" ]; then
    echo "Task ID ${SGE_TASK_ID}: Detected global job termination signal (QDEL_SENT_FLAG exists) after initial scan. Exiting gracefully."
    exit 0 # Exit immediately without further work
fi

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

# Extract the subdirectory path from the RELATIVE_JSON_PATH, removing the "af3_inputs/" prefix.
SUBDIR_PATH=$(dirname "$RELATIVE_JSON_PATH" | sed 's|^af3_inputs/||')

# AlphaFold expects the *parent* directory for its output. It will create a subdir named by $JOB_NAME
ABS_AF3_OUTPUT_PARENT_DIR="$ABS_DATA_JSON_OUTPUT_ROOT_DIR/$SUBDIR_PATH"

# Construct the FULL, specific output directory path for this job.
# This ensures that the output structure mirrors the input structure under 'af3_inputs'.
# Example: PROJECT_BASE_DIR/af3_data_json/positive-controls/my_protein/
EXPECTED_AF3_CREATED_JOB_DIR_INNER="$ABS_AF3_OUTPUT_PARENT_DIR/$JOB_NAME"
EXPECTED_OUTPUT_FILE="$EXPECTED_AF3_CREATED_JOB_DIR_INNER/${JOB_NAME}_data.json"

# Resume Logic for Data Pipeline
# Check if the expected output file already exists and is non-empty.
# This prevents redundant calculations for previously completed tasks.
if [ -f "$EXPECTED_OUTPUT_FILE" ] && [ -s "$EXPECTED_OUTPUT_FILE" ]; then
    echo "Task ID ${SGE_TASK_ID}: Output file for '$JOB_NAME' already exists and is non-empty: $EXPECTED_OUTPUT_FILE."
    echo "Task ID ${SGE_TASK_ID}: Skipping data pipeline calculation for this task and incrementing global count."
    increment_completed_count # Mark this task as completed in the global counter
    exit 0 # Exit successfully as this task is already completed.
fi

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

# Check if the expected output file was actually created and is non-empty after the run.
if [ ! -f "$EXPECTED_OUTPUT_FILE" ] || [ ! -s "$EXPECTED_OUTPUT_FILE" ]; then
    echo "WARNING: Task ID ${SGE_TASK_ID}: Data pipeline finished for '$JOB_NAME', but expected output file '$EXPECTED_OUTPUT_FILE' is missing or empty!" >&2
    # If a failure occurred and the output is missing/empty, we do NOT increment the count.
    # The task will be considered incomplete, and the array will not trigger qdel prematurely.
    exit 1 # Treat missing/empty output as a critical failure for this task
else
    echo "Task ID ${SGE_TASK_ID}: Successfully completed data pipeline for '$JOB_NAME'."
    increment_completed_count # Mark this task as completed in the global counter
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