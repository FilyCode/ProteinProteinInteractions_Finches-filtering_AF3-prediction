#!/bin/bash -l

# --------------------------------------------------------------------------------
# SGE Job Directives
# --------------------------------------------------------------------------------

#$ -P cancergrp                  # Specify the project name.
#$ -N af3_inference_array        # Assign a descriptive name to this job array.
#$ -l h_rt=24:00:00              # Set the hard time limit for each array task (hh:mm:ss).
                                 # This value will be used to dynamically set polling limits.
#$ -j y                          # Merge stdout and stderr.
#$ -o af3_inference_array.out    # Output file for merged stdout/stderr for all array tasks.
#$ -m e                            # Email notification (job begins (b), ends (e), is aborted (a), suspended (s), or never (n))

# Resource Requests for Inference:
#$ -pe omp 4                     # Request 4 CPU cores.
#$ -l mem_per_core=8G            # Request 8GB of memory per CPU core (32GB total).
#$ -l gpus=1                     # Request x GPU.
#$ -l gpu_type=A100              # Specify GPU model.
#$ -l gpu_memory=80G             # Specify GPU memory.
##$ -l gpu_c=8.6                  # Optional: Specify minimum GPU compute capability.


# --------------------------------------------------------------------------------
# Environment Setup and Working Directory Configuration
# --------------------------------------------------------------------------------

echo "------------------------------------------------------------"
echo "Starting AlphaFold3 Inference job array task ${SGE_TASK_ID}"
echo "Job ID: ${JOB_ID}, Array Task ID: ${SGE_TASK_ID}"
echo "Running on host: $(hostname)"
echo "Initial working directory: $(pwd)"
echo "------------------------------------------------------------"

# IMPORTANT: Define the absolute path to your project's base directory.
PROJECT_BASE_DIR="/projectnb/cancergrp/Philipp/alphafold3_pipeline" 

cd "$PROJECT_BASE_DIR" || { 
    echo "CRITICAL Error: Failed to change to project directory: $PROJECT_BASE_DIR" >&2
    exit 1 
}

echo "Changed working directory to: $(pwd)"
echo "Loading AlphaFold3 module..."

module load alphafold3/3.0.0 || { 
    echo "CRITICAL Error: Failed to load alphafold3/3.0.0 module." >&2
    exit 1 
}
echo "AlphaFold3 module loaded."

RELATIVE_MODEL_DIR="alphafold3_models"
export MODEL_DIR="$PROJECT_BASE_DIR/$RELATIVE_MODEL_DIR"
echo "AlphaFold3 model directory set to: $MODEL_DIR"

export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_CLIENT_MEM_FRACTION=0.95
echo "GPU memory preallocation enabled with fraction: ${XLA_CLIENT_MEM_FRACTION}"

export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
echo "XLA_FLAGS set to: ${XLA_FLAGS}"

export JAX_COMPILATION_CACHE_DIR="${SCRATCH:-$TMPDIR}/jax_comp_cache"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}" || {
    echo "CRITICAL Error: Failed to create JAX compilation cache directory: ${JAX_COMPILATION_CACHE_DIR}" >&2
    exit 1
}
echo "JAX compilation cache directory set to: ${JAX_COMPILATION_CACHE_DIR}"

# --------------------------------------------------------------------------------
# Dynamic Polling Limits Calculation and Work Management Setup
# --------------------------------------------------------------------------------

# Define polling interval (e.g., 5 minutes)
POLLING_INTERVAL_SECONDS=300 # 5 minutes

# Define a safety buffer: minimum time required for actual inference work
# (e.g., 1 hour, to ensure job doesn't exit right after polling succeeds)
SAFETY_BUFFER_SECONDS=$((1 * 3600)) # 1 hour in seconds

# Extract the hard runtime limit (h_rt) from the SGE directive in this script.
JOB_TIME_STR_FULL=$(grep "^#\$ -l h_rt=" "$0" | head -1 | cut -d '=' -f 2)
JOB_TIME_STR=$(echo "$JOB_TIME_STR_FULL" | cut -d '#' -f 1 | xargs)

if [ -z "$JOB_TIME_STR" ]; then
    echo "CRITICAL Error: Could not extract h_rt from job script. Ensure '#$ -l h_rt=HH:MM:SS' is correctly defined." >&2
    exit 1
fi

# Parse HH:MM:SS into total seconds
IFS=':' read -r HH MM SS <<< "$JOB_TIME_STR"
TOTAL_JOB_SECONDS=$(( 10#$HH * 3600 + 10#$MM * 60 + 10#$SS ))

START_TIME=$(date +%s)
echo "Job h_rt: $JOB_TIME_STR ($TOTAL_JOB_SECONDS s)"
echo "Safety buffer for work: $(($SAFETY_BUFFER_SECONDS/3600)) hours"
echo "Polling interval: $(($POLLING_INTERVAL_SECONDS/60)) minutes"

# --- Lock file directory setup ---
# Use a common lock directory for all tasks of this job array.
# This allows tasks to see what others are processing or have picked up.
LOCK_DIR="$PROJECT_BASE_DIR/af3_inference_locks/$JOB_ID"
mkdir -p "$LOCK_DIR" || {
    echo "CRITICAL Error: Failed to create lock directory: $LOCK_DIR" >&2
    exit 1
}
echo "Shared lock directory: $LOCK_DIR"

# Trap to ensure lock files are cleaned up if the script exits or is killed
# Important: This removes *this specific task's* lock file, not the main lock dir.
CURRENT_JOB_LOCK="" # Placeholder for the lock file path currently held by this task
trap 'if [ -n "$CURRENT_JOB_LOCK" ] && [ -d "$CURRENT_JOB_LOCK" ]; then rmdir "$CURRENT_JOB_LOCK" 2>/dev/null; echo "Task ${SGE_TASK_ID}: Cleaned up lock file $CURRENT_JOB_LOCK"; fi' EXIT HUP INT TERM

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
        TOTAL_ITEMS_TO_PROCESS_TEMP=$(wc -l < json_list.txt) # Removed 'local'
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
# This function is called by the first task to detect global completion.
trigger_qdel_if_all_done() {
    local current_completion_val="$1" # Argument is the current completed count for logging

    (
        # Acquire an exclusive lock on QDEL_CONTROL_LOCK to ensure only one task tries to issue qdel
        # Use a short timeout (e.g., 5 seconds) to prevent blocking indefinitely if another task is already handling it
        flock -xn -w 5 201 || { 
            echo "Task ID ${SGE_TASK_ID}: Another task is handling qdel, or lock contention. Skipping qdel attempt." >&2; 
            return 1; 
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
            echo "Task ID ${SGE_TASK_ID}: Qdel already sent by another task. Exiting this qdel trigger." >&2
            return 1 # Qdel already initiated
        fi
    ) 201>"$QDEL_CONTROL_LOCK" # Using FD 201 for the qdel control lock file
}


# Calculate initial completed count and store it atomically
(
    flock -xn -w 0 200 || sleep 5 # Try non-blocking lock, if it fails, another task has it, wait briefly.
    if [ ! -f "$COMPLETED_TASKS_COUNT_FILE" ]; then
        echo "Task ID ${SGE_TASK_ID}: Performing initial scan for already completed tasks to initialize global count..."
        INITIAL_COMPLETED_COUNT=0
        
        # Variables defined in a subshell are implicitly local.
        # No need for 'local' keyword, as it can cause issues in some shells/environments.
        RELATIVE_OUTPUT_BASE_DIR_FOR_INIT="af3_outputs"
        PROJECT_BASE_DIR_FOR_INIT="$PROJECT_BASE_DIR" 
        
        # Iterate through json_list.txt to count already existing completed outputs
        while IFS= read -r RELATIVE_ORIGINAL_JSON_PATH_FOR_INIT; do
            if [ -z "$RELATIVE_ORIGINAL_JSON_PATH_FOR_INIT" ]; then
                continue
            fi
            ORIGINAL_JOB_NAME_FOR_INIT=$(basename "$RELATIVE_ORIGINAL_JSON_PATH_FOR_INIT" .json | tr '[:upper:]' '[:lower:]')
            SUBDIR_PATH_FOR_INIT=$(dirname "$RELATIVE_ORIGINAL_JSON_PATH_FOR_INIT" | sed 's|^af3_inputs/||')
            
            # These variables are correctly defined and used here now.
            ABS_AF3_BASE_OUTPUT_DIR_FOR_INIT="$PROJECT_BASE_DIR_FOR_INIT/$RELATIVE_OUTPUT_BASE_DIR_FOR_INIT/$SUBDIR_PATH_FOR_INIT"
            ABS_JOB_SPECIFIC_OUTPUT_DIR_FOR_INIT="$ABS_AF3_BASE_OUTPUT_DIR_FOR_INIT/$ORIGINAL_JOB_NAME_FOR_INIT"
            EXPECTED_FINAL_OUTPUT_FILE_FOR_INIT="$ABS_JOB_SPECIFIC_OUTPUT_DIR_FOR_INIT/${ORIGINAL_JOB_NAME_FOR_INIT}_model.cif" # Typo fixed: ORIGINAL_JOB_NAME_FOR_INIT

            # --- DEBUGGING: Print the path being checked ---
            # Uncomment the line below for detailed debugging of path checks during initial scan
            # echo "Task ID ${SGE_TASK_ID}: DEBUG-INIT-SCAN: Checking for file: '$EXPECTED_FINAL_OUTPUT_FILE_FOR_INIT'" >&2
            # ----------------------------------------------

            if [ -f "$EXPECTED_FINAL_OUTPUT_FILE_FOR_INIT" ] && [ -s "$EXPECTED_FINAL_OUTPUT_FILE_FOR_INIT" ]; then
                INITIAL_COMPLETED_COUNT=$((INITIAL_COMPLETED_COUNT + 1))
                # echo "Task ID ${SGE_TASK_ID}: DEBUG-INIT-SCAN: Found pre-existing completed output for: $ORIGINAL_JOB_NAME_FOR_INIT (Count: $INITIAL_COMPLETED_COUNT)" >&2
            fi
        done < json_list.txt

        echo "$INITIAL_COMPLETED_COUNT" > "$COMPLETED_TASKS_COUNT_FILE"
        echo "Task ID ${SGE_TASK_ID}: Initialized completed tasks count to $INITIAL_COMPLETED_COUNT (found $INITIAL_COMPLETED_COUNT pre-existing outputs)."
    fi

    # Check immediately after initial scan if all tasks are completed.
    # If this task is the first to detect global completion, it should try to send qdel.
    CURRENT_COMPLETED_COUNT=$(cat "$COMPLETED_TASKS_COUNT_FILE") # Read again after potential update
    if [ "$CURRENT_COMPLETED_COUNT" -ge "$TOTAL_ITEMS_TO_PROCESS" ]; then
        trigger_qdel_if_all_done "$CURRENT_COMPLETED_COUNT" # Pass current count for logging
    fi

) 200>"$COUNT_LOCK_FILE"


# Function to atomically increment the completed tasks count
increment_completed_count() {
    (
        flock -x 200 # Acquire an exclusive lock on the count file
        CURRENT_COUNT=$(cat "$COMPLETED_TASKS_COUNT_FILE" 2>/dev/null || echo 0)
        NEW_COUNT=$((CURRENT_COUNT + 1))
        echo "$NEW_COUNT" > "$COMPLETED_TASKS_COUNT_FILE"
        echo "Task ID ${SGE_TASK_ID}: Incremented completed tasks count to $NEW_COUNT."

        # If this increment makes it globally complete, trigger qdel
        if [ "$NEW_COUNT" -ge "$TOTAL_ITEMS_TO_PROCESS" ]; then
            trigger_qdel_if_all_done "$NEW_COUNT"
        fi
    ) 200>"$COUNT_LOCK_FILE"
}



# --------------------------------------------------------------------------------
# AlphaFold3 Inference Execution (Dynamic Task Picking)
# --------------------------------------------------------------------------------

echo "------------------------------------------------------------"
echo "Task ID ${SGE_TASK_ID}: Starting AlphaFold3 Inference work loop."
echo "------------------------------------------------------------"

RELATIVE_OUTPUT_BASE_DIR="af3_outputs"
RELATIVE_AUGMENTED_JSON_BASE_DIR="af3_data_json"

# Main loop for picking and processing tasks
while true; do
    # First, check if a global termination has been signaled by another task (e.g., via qdel)
    if [ -f "$QDEL_SENT_FLAG" ]; then
        echo "Task ID ${SGE_TASK_ID}: Detected global job termination signal (QDEL_SENT_FLAG exists). Exiting gracefully."
        exit 0 # Exit immediately without further work or polling
    fi
    
    CURRENT_TIME=$(date +%s)
    ELAPSED_SECONDS=$((CURRENT_TIME - START_TIME))
    TIME_REMAINING_SECONDS=$((TOTAL_JOB_SECONDS - ELAPSED_SECONDS))

    # --- Early exit condition 1: Job time limit approaching ---
    if [ "$TIME_REMAINING_SECONDS" -le "$SAFETY_BUFFER_SECONDS" ]; then
        echo "Task ID ${SGE_TASK_ID}: Remaining time ($TIME_REMAINING_SECONDS s) is below safety buffer ($SAFETY_BUFFER_SECONDS s). Exiting."
        break # Exit the loop, job is running out of time
    fi

    # --- Early exit condition 2: All tasks globally completed ---
    # Check this *before* scanning for new tasks, and also before sleeping.
    COMPLETED_COUNT=$(get_completed_count)
    if [ "$COMPLETED_COUNT" -ge "$TOTAL_ITEMS_TO_PROCESS" ]; then
        echo "Task ID ${SGE_TASK_ID}: All tasks (${TOTAL_ITEMS_TO_PROCESS}) globally completed (current completed: $COMPLETED_COUNT). Exiting gracefully."
        break # All work is done, exit
    fi

    echo "Task ID ${SGE_TASK_ID}: Scanning for available tasks. (Completed: $COMPLETED_COUNT/$TOTAL_ITEMS_TO_PROCESS, Remaining: $((TIME_REMAINING_SECONDS/3600))h $(( (TIME_REMAINING_SECONDS%3600)/60 ))m)"
    
    PICKED_JOB_NAME=""
    PICKED_ABS_AUGMENTED_JSON_FILE=""
    PICKED_ABS_INFERENCE_OUTPUT_ROOT_DIR=""
    PICKED_ABS_JOB_OUTPUT_DIR=""
    PICKED_EXPECTED_FINAL_OUTPUT_FILE=""

    TASK_FOUND_IN_THIS_SCAN=false # Flag to track if this specific scan found work

    # Iterate through all possible jobs from the master list
    while IFS= read -r RELATIVE_ORIGINAL_JSON_PATH; do
        if [ -z "$RELATIVE_ORIGINAL_JSON_PATH" ]; then
            continue # Skip empty lines
        fi

        ORIGINAL_JOB_NAME=$(basename "$RELATIVE_ORIGINAL_JSON_PATH" .json | tr '[:upper:]' '[:lower:]')
        SUBDIR_PATH=$(dirname "$RELATIVE_ORIGINAL_JSON_PATH" | sed 's|^af3_inputs/||')

        ABS_INFERENCE_OUTPUT_ROOT_DIR="$PROJECT_BASE_DIR/$RELATIVE_OUTPUT_BASE_DIR/$SUBDIR_PATH"
        ABS_JOB_OUTPUT_DIR="$ABS_INFERENCE_OUTPUT_ROOT_DIR/$ORIGINAL_JOB_NAME"
        EXPECTED_FINAL_OUTPUT_FILE="$ABS_JOB_OUTPUT_DIR/${ORIGINAL_JOB_NAME}_model.cif"
        ABS_AUGMENTED_JSON_FILE="$PROJECT_BASE_DIR/$RELATIVE_AUGMENTED_JSON_BASE_DIR/$SUBDIR_PATH/${ORIGINAL_JOB_NAME}/${ORIGINAL_JOB_NAME}_data.json"

        # Define the unique lock path for this specific job
        JOB_LOCK_PATH="$LOCK_DIR/$ORIGINAL_JOB_NAME.lock"

        # --- Work stealing logic ---
        # 1. Check if final inference output already exists (task completed)
        if [ -f "$EXPECTED_FINAL_OUTPUT_FILE" ] && [ -s "$EXPECTED_FINAL_OUTPUT_FILE" ]; then
            continue # Already done, skip
        fi

        # 2. Check if augmented JSON file (data pipeline output) is ready
        if [ ! -f "$ABS_AUGMENTED_JSON_FILE" ] || [ ! -s "$ABS_AUGMENTED_JSON_FILE" ]; then
            continue # Input data not ready, skip for now
        fi

        # 3. Attempt to acquire a lock for this task
        if mkdir "$JOB_LOCK_PATH" 2>/dev/null; then
            echo "Task ID ${SGE_TASK_ID}: Acquired lock for '$ORIGINAL_JOB_NAME' at $JOB_LOCK_PATH"
            PICKED_JOB_NAME="$ORIGINAL_JOB_NAME"
            PICKED_ABS_AUGMENTED_JSON_FILE="$ABS_AUGMENTED_JSON_FILE"
            PICKED_ABS_INFERENCE_OUTPUT_ROOT_DIR="$ABS_INFERENCE_OUTPUT_ROOT_DIR"
            PICKED_ABS_JOB_OUTPUT_DIR="$ABS_JOB_OUTPUT_DIR"
            PICKED_EXPECTED_FINAL_OUTPUT_FILE="$EXPECTED_FINAL_OUTPUT_FILE"
            CURRENT_JOB_LOCK="$JOB_LOCK_PATH" # Store the current lock for trap cleanup
            TASK_FOUND_IN_THIS_SCAN=true
            break # Found and picked a task, exit the list scanning loop
        else
            # Lock already exists, another task is processing or has claimed it
            continue
        fi
    done < json_list.txt

    # --- Process the picked task, or wait if no work was found ---
    if [ -n "$PICKED_JOB_NAME" ]; then
        echo "Task ID ${SGE_TASK_ID}: Processing '$PICKED_JOB_NAME'."
        echo "Task ID ${SGE_TASK_ID}: Absolute augmented JSON path being used for inference: $PICKED_ABS_AUGMENTED_JSON_FILE"
        
        # Ensure the full, nested output directory path exists with proper permissions
        mkdir -p "$PICKED_ABS_JOB_OUTPUT_DIR" || { 
            echo "CRITICAL Error: Task ID ${SGE_TASK_ID}: Failed to create output directory: $PICKED_ABS_JOB_OUTPUT_DIR" >&2
            rmdir "$CURRENT_JOB_LOCK" 2>/dev/null # Release lock on critical failure
            CURRENT_JOB_LOCK=""
            continue # Try to pick another task
        }
        chmod 775 "$PICKED_ABS_JOB_OUTPUT_DIR" 

        echo "Task ID ${SGE_TASK_ID}: Outputting inference results to: $PICKED_ABS_JOB_OUTPUT_DIR"

        # Run AlphaFold3 inference using SCC's wrapper script.
        $(which run_alphafold.sh) \
            --json_path="$PICKED_ABS_AUGMENTED_JSON_FILE" \
            --output_dir="$PICKED_ABS_INFERENCE_OUTPUT_ROOT_DIR" \
            --model_dir="$MODEL_DIR" \
            --norun_data_pipeline \
            --jax_compilation_cache_dir="$JAX_COMPILATION_CACHE_DIR"

        INFERENCE_EXIT_CODE=$?

        # Post-inference polling loop to ensure file system is consistent
        POST_INFERENCE_POLLING_ATTEMPTS=6   # Number of times to check
        POST_INFERENCE_POLLING_INTERVAL=10 # Seconds between checks (total 60 seconds wait)
        OUTPUT_FILE_READY=false
        for (( k=1; k<=$POST_INFERENCE_POLLING_ATTEMPTS; k++ )); do
            if [ -f "$PICKED_EXPECTED_FINAL_OUTPUT_FILE" ] && [ -s "$PICKED_EXPECTED_FINAL_OUTPUT_FILE" ]; then
                OUTPUT_FILE_READY=true
                echo "Task ID ${SGE_TASK_ID}: Final output file '$PICKED_EXPECTED_FINAL_OUTPUT_FILE' found and is non-empty after $((k * POST_INFERENCE_POLLING_INTERVAL)) seconds post-run."
                break
            fi
            echo "Task ID ${SGE_TASK_ID}: Waiting for final output file '$PICKED_EXPECTED_FINAL_OUTPUT_FILE' to become available (attempt $k/$POST_INFERENCE_POLLING_ATTEMPTS)..." >&2
            sleep "$POST_INFERENCE_POLLING_INTERVAL"
        done
        
        # Final check: Verify that the expected output file was created and is non-empty after the run.
        # This now includes the result of the post-inference polling.
        if [ "$INFERENCE_EXIT_CODE" -ne 0 ] || [ "$OUTPUT_FILE_READY" != true ]; then
            echo "WARNING: Task ID ${SGE_TASK_ID}: Inference for '$PICKED_JOB_NAME' failed or produced missing/empty output '$PICKED_EXPECTED_FINAL_OUTPUT_FILE' (Exit code: $INFERENCE_EXIT_CODE)!" >&2
            # Importantly, do NOT remove the lock if it failed, so another task doesn't pick it up immediately.
            # A human will need to inspect and potentially delete the lock or move the task to a failed queue.
            # If you want it to be picked again on failure, uncomment: rmdir "$CURRENT_JOB_LOCK" 2>/dev/null; CURRENT_JOB_LOCK=""
        else
            echo "Task ID ${SGE_TASK_ID}: Successfully completed inference for '$PICKED_JOB_NAME'."
            increment_completed_count # Increment global counter on success
            rmdir "$CURRENT_JOB_LOCK" 2>/dev/null # Release lock on successful completion
            CURRENT_JOB_LOCK=""
        fi
        
    else
        # No work found in this scan. Check for overall completion before sleeping.
        # This double-check is crucial to prevent unnecessary long sleeps if all work finishes
        # right after this task's previous `COMPLETED_COUNT` check.
        COMPLETED_COUNT=$(get_completed_count)
        if [ "$COMPLETED_COUNT" -ge "$TOTAL_ITEMS_TO_PROCESS" ]; then
            echo "Task ID ${SGE_TASK_ID}: No new tasks ready, and all tasks (${TOTAL_ITEMS_TO_PROCESS}) globally completed (current completed: $COMPLETED_COUNT). Exiting gracefully."
            break # All work is done, exit
        else
            echo "Task ID ${SGE_TASK_ID}: No new tasks ready for processing. Waiting ${POLLING_INTERVAL_SECONDS} seconds..."
            sleep "$POLLING_INTERVAL_SECONDS"
        fi
    fi

done

echo "------------------------------------------------------------"
echo "Task ID ${SGE_TASK_ID}: Exiting. No more tasks or time limit reached."
echo "------------------------------------------------------------"

# ----- Submit job like this ---------
# Before submitting, ensure you have:
# 1. A 'json_list.txt' file in your 'PROJECT_BASE_DIR' with one relative JSON path per line.
#    Example 'json_list.txt' content (paths relative to PROJECT_BASE_DIR):
#    af3_inputs/positive-controls/my_protein_1.json
#    af3_inputs/experiment_X/condition_Y/another_protein.json
#
# 2. The 'af3_data_pipeline_array' job has been successfully completed for all tasks
#    or is currently running.
#
# Calculate the number of tasks based on the number of lines in 'json_list.txt':
# NUM_TASKS=$(cat json_list.txt | wc -l)
#
# Submit the task:
# qsub -t 1-"${NUM_TASKS}" 03_run_inference.sh
# ------------------------------------
