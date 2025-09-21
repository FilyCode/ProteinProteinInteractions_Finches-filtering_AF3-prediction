#!/bin/bash -l

#$ -P cancergrp             
#$ -N af3_inference         # Job name
#$ -l h_rt=48:00:00         # Hard run time limit: 48 hours (2 days, max for GPU jobs on SCC)
#$ -j y                     # Merge stdout and stderr
#$ -o af3_inference_array.out # Output file for merged stdout/stderr for all array tasks
#$ -t 1-$(cat json_list.txt | wc -l) # Submit as an array job
#$ -pe omp 4                # Request 4 CPU cores (AF3 recommends 4-8)
#$ -l mem_per_core=8G       # Request 8GB of memory per core (4 cores * 8GB/core = 32GB total, as recommended for GPU inference)
#$ -l gpus=1                # Request 1 GPU (SCC recommends against multiple unless absolutely necessary for memory/speed)
#$ -l gpu_type=A100         # IMPORTANT: Specify GPU model, e.g., A100 (run 'qgpus' to see options on SCC)
#$ -l gpu_memory=80G        # IMPORTANT: Specify GPU memory, e.g., 80G (match your requested gpu_type)
#$ -l gpu_c=8.6             # IMPORTANT: Specify minimum GPU capability, e.g., 8.6 for A100 (AF3 recommends >=6.0)

# We need to explicitly hold this job until the data pipeline completes
#$ -hold_jid af3_data_pipeline_stage # IMPORTANT: Match the -N name from the data pipeline job.

echo "Starting AlphaFold3 Inference job array task ${SGE_TASK_ID} on $(hostname) with GPU"

# Load the AlphaFold3 module. This makes 'run_alphafold.sh' available in PATH.
module load alphafold3/3.0.0

# --- IMPORTANT: Specify the path to your AlphaFold3 model weights ---
# These are NOT included with the module and must be obtained from DeepMind.
export MODEL_DIR="/projectnb/cancergrp/Philipp/alphafold3_models"

# GPU Memory Optimization Flags - Recommended for A100 80GB.
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_CLIENT_MEM_FRACTION=0.95

# XLA Compilation Time Workaround - SCC's wrapper script may already set this.
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"

# JAX Persistent Compilation Cache (same as in data pipeline)
export JAX_COMPILATION_CACHE_DIR="${SCRATCH:-$TMPDIR}/jax_comp_cache"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}"

# Get the augmented JSON input file for this array task
AUGMENTED_JSON_BASE_DIR="af3_data_json"
ORIGINAL_JSON_PATH=$(sed -n "${SGE_TASK_ID}p" json_list.txt)
ORIGINAL_JOB_NAME=$(basename "$ORIGINAL_JSON_PATH" .json)

# Construct the path to the augmented JSON. Based on examples, it's typically in a subdir
# named after the job, and the file itself has "_data.json" or similar.
# The AF3 documentation states that the `run_alphafold.py` (which `run_alphafold.sh` wraps)
# will generate `<job_name>_data.json` when the data pipeline is run with `--norun_inference`.
AUGMENTED_JSON_FILE="$AUGMENTED_JSON_BASE_DIR/${ORIGINAL_JOB_NAME}/${ORIGINAL_JOB_NAME}_data.json"

if [ ! -f "$AUGMENTED_JSON_FILE" ]; then
    echo "Error: Augmented JSON file not found: $AUGMENTED_JSON_FILE"
    echo "Expected path: $AUGMENTED_JSON_FILE"
    echo "Please check the output structure of the data pipeline for task ${SGE_TASK_ID}."
    exit 1
fi

OUTPUT_BASE_DIR="af3_outputs"
JOB_OUTPUT_DIR="$OUTPUT_BASE_DIR/$ORIGINAL_JOB_NAME" # Each job gets its own subdir
mkdir -p "$JOB_OUTPUT_DIR"

echo "Processing augmented input: $AUGMENTED_JSON_FILE (Task ID: ${SGE_TASK_ID})"
echo "Outputting results to: $JOB_OUTPUT_DIR"

# Run AlphaFold3 inference only using SCC's wrapper script
$(which run_alphafold.sh) \
    --json_path="$AUGMENTED_JSON_FILE" \
    --output_dir="$JOB_OUTPUT_DIR" \
    --model_dir="$MODEL_DIR" \
    --norun_data_pipeline \
    --jax_compilation_cache_dir="$JAX_COMPILATION_CACHE_DIR" \
    --num_samples=5

echo "Finished AlphaFold3 Inference for task ${SGE_TASK_ID}"


# ----- Submit job like this ---------
# DATA_PIPELINE_HOLD_NAME=$(cat data_prep_hold_name.txt)
# qsub -hold_jid "$DATA_PIPELINE_HOLD_NAME" 03_run_inference.sh
# echo "Inference job submitted, will start after data pipeline job named '$DATA_PIPELINE_HOLD_NAME' completes."
