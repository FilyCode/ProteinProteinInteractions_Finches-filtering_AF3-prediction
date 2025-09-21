#!/bin/bash -l

#$ -P cancergrp             
#$ -N af3_data_prep         # Job name
#$ -l h_rt=12:00:00         # Hard run time limit
#$ -j y                     # Merge stdout and stderr into a single output file
#$ -o af3_data_prep_array.out # Output file for the merged stdout/stderr for all array tasks
#$ -t 1-$(cat json_list.txt | wc -l) # Submit as an array job, one task per line in json_list.txt
#$ -pe omp 16               # Request 16 CPU cores (slots) for shared memory parallelization (AF3 recommends >=8)
#$ -l mem_per_core=4G       # Request 4GB of memory per core (16 cores * 4GB/core = 64GB total, adjust if >64GB needed)

echo "Starting AlphaFold3 Data Pipeline job array task ${SGE_TASK_ID} on $(hostname)"

# Load the AlphaFold3 module. This makes 'run_alphafold.sh' available in PATH.
module load alphafold3/3.0.0

# --- IMPORTANT: Specify the path to your AlphaFold3 model weights ---
# These are NOT included with the module and must be obtained from DeepMind.
export MODEL_DIR="/projectnb/cancergrp/Philipp/alphafold3_pipeline/alphafold3_models"

# JAX compilation cache - SCC documentation recommends using $TMPDIR by default,
# but allows overriding. Using /scratch for persistence.
export JAX_COMPILATION_CACHE_DIR="${SCRATCH:-$TMPDIR}/jax_comp_cache"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}"

# Get the JSON input file for this array task
JSON_INPUT_FILE=$(sed -n "${SGE_TASK_ID}p" json_list.txt)
if [ -z "$JSON_INPUT_FILE" ]; then
    echo "Error: No JSON file found for array task ID ${SGE_TASK_ID}"
    exit 1
fi

# Extract the job name from the input JSON (used for output directory structure)
JOB_NAME=$(basename "$JSON_INPUT_FILE" .json)
DATA_JSON_OUTPUT_BASE_DIR="af3_data_json" # Base directory for augmented JSONs
mkdir -p "$DATA_JSON_OUTPUT_BASE_DIR"

# The data pipeline will create a subdirectory named after the job, and put the output JSON inside.
# E.g., af3_data_json/pair_prot1_vs_prot2/pair_prot1_vs_prot2.json (or _data.json, check actual output)
# The `run_alphafold.sh` output directory is the parent for the job-specific output.
# The `run_alphafold.sh` outputs the augmented JSON as <job_name>.json OR <job_name>_data.json
# based on its internal logic. The docs specify `_data.json` but in my previous checks I've seen just `.json`
# when `--output_dir` is used for the parent. We will check the output structure from the actual run.

echo "Processing input: $JSON_INPUT_FILE (Task ID: ${SGE_TASK_ID})"
echo "Output data pipeline results to: $DATA_JSON_OUTPUT_BASE_DIR/${JOB_NAME}/"

# Run the AlphaFold3 data pipeline only using SCC's wrapper script
$(which run_alphafold.sh) \
    --json_path="$JSON_INPUT_FILE" \
    --output_dir="$DATA_JSON_OUTPUT_BASE_DIR" \
    --model_dir="$MODEL_DIR" \
    --norun_inference \
    --jax_compilation_cache_dir="$JAX_COMPILATION_CACHE_DIR"

echo "Finished AlphaFold3 Data Pipeline for task ${SGE_TASK_ID}"


# ----- Submit job like this ---------
# DATA_PIPELINE_HOLD_NAME="af3_data_pipeline_stage" # A simple name for job dependency
# qsub -N "$DATA_PIPELINE_HOLD_NAME" 02_run_data_pipeline.sh
# echo "Data pipeline job submitted. Its name for 'hold_jid' is: $DATA_PIPELINE_HOLD_NAME"
# echo "$DATA_PIPELINE_HOLD_NAME" > data_prep_hold_name.txt