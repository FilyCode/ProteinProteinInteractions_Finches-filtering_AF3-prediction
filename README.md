# Protein-Protein Interaction Prediction: Finches-filtered AlphaFold3 Pipeline

This repository provides a comprehensive pipeline for predicting viral-human protein-protein interactions (PPIs). It leverages `finches` for identifying potential interactions involving intrinsically disordered regions (IDRs) as a filtering step, followed by detailed structural prediction using AlphaFold3 (AF3) for selected pairs. The pipeline is optimized for execution on the Boston University Shared Computing Cluster (BU SCC) using the SGE job scheduler.

## Overview

Predicting protein-protein interactions, particularly between viral and human proteins, is crucial for understanding host-pathogen dynamics. This pipeline addresses the computational intensity of structural prediction methods like AlphaFold3 by introducing an initial filtering step using `finches`. `finches` rapidly assesses the propensity for interactions involving IDRs, reducing the number of pairs that require full AF3 modeling.

## Workflow

The pipeline consists of the following sequential stages:

1.  **Initial Data Preparation:** Convert raw FASTA protein sequences (viral and human) into a structured `.csv` format suitable for downstream processing.
2.  **Finches IDR Interaction Filtering:** Calculate interaction propensity (epsilon values) for all viral-human protein pairs using `finches`. This step identifies pairs with predicted IDR-mediated attraction or repulsion.
3.  **Analyze Finches Results:** Process and visualize the `finches` output to select promising protein pairs for detailed structural prediction.
4.  **AlphaFold3 Input Preparation:** Generate `alphafold_input.json` files for the selected protein pairs, structuring them for AlphaFold3.
5.  **AlphaFold3 Data Pipeline:** Run AlphaFold3's data preparation pipeline for the selected pairs (CPU-intensive).
6.  **AlphaFold3 Inference:** Perform AlphaFold3's structural inference to predict the 3D complex structures of the interacting pairs (GPU-intensive).
7.  **AlphaFold3 Post-processing:** Analyze and extract relevant metrics (e.g., pLDDT, pae) from the predicted AlphaFold3 structures.

## Directory Structure

**Expected data directories** (create these, or adjust scripts):

*   `data/`: Input FASTA files, and CSV files from `transform-fasta-data-to-csv.ipynb`.
*   `results/`: Finches output (e.g., `finches_interaction_results_chunked.parquet`).
*   `af3_data_json/`: Intermediate directory for AlphaFold3 data pipeline outputs.
*   `af3_outputs/`: Final directory for AlphaFold3 PDB structures and metadata.

## Setup and Prerequisites

### Conda Environments

This pipeline requires at least two separate Conda environments due to dependencies:

1.  **`finches-env`**: For `finches` calculations and general data handling (pandas, numpy, pyarrow, tqdm).
    ```bash
    # Example environment creation (adjust as needed)
    conda create -n finches-env python=3.9
    conda activate finches-env
    pip install pandas numpy finches tqdm pyarrow
    ```
2.  **`alphafold3`**: For AlphaFold3 execution. On BU SCC, this is typically provided via a module.

### AlphaFold3 Model Weights

AlphaFold3 model weights **must be downloaded separately from DeepMind** and placed in an accessible directory on the SCC.
You will need to set the `MODEL_DIR` variable in `02_run_data_pipeline.sh` and `03_run_inference.sh` to point to this location.
Example: `export MODEL_DIR="/projectnb/cancergrp/Philipp/alphafold3_models"`

### BU SCC Specifics

The `.sh` scripts (`run_filter.sh`, `02_run_data_pipeline.sh`, `03_run_inference.sh`) are designed for the SGE job scheduler on BU SCC.

*   **Module Loading:** Ensure `module load miniconda` (for finches) and `module load alphafold3/3.0.0` (for AF3) are correctly configured.
*   **Project Name:** Update `#$ -P myproject` to your actual SCC project name.
*   **Resource Requests:** Adjust `h_rt`, `pe omp`, `mem_per_core`, `gpus`, `gpu_type`, `gpu_memory` as needed based on your job requirements and SCC policies.
*   **SCC AlphaFold3 Examples:** The AF3 scripts assume you have copied the SCC AlphaFold3 example files, especially `run_alphafold.sh` wrapper script, and established the `$SCC_ALPHAFOLD3_EXAMPLES` environment variable.

## Usage

### 1. Initial Data Preparation

Use `transform-fasta-data-to-csv.ipynb` to convert your viral and human protein FASTA files into CSV format.

*   **Input:** Viral and human FASTA files.
*   **Output:** `VP_pos_selec_enriched_hits.csv` (viral) and `Human-proteom_GCF_000001405.40.csv` (human) or similarly named CSV files in your `data/` directory. Ensure `ID_COLUMN` and `SEQUENCE_COLUMN` match those expected by `filter-viral-human-pairs-with-finches.py`.

### 2. Finches IDR Interaction Filtering

This step calculates interaction propensities for all viral-human protein pairs.

1.  **Edit `run_filter.sh`:**
    *   Ensure `conda activate finches-env` is correct.
    *   Update the `--input_file_virus`, `--input_file_human`, and `--output_file` paths to match your data and desired output location (e.g., in `results/`).
    *   Adjust `--num_processes $NSLOTS` if you want a fixed number of cores, otherwise `$NSLOTS` dynamically uses the number of cores requested via `#$ -pe omp`.
    *   Consider using `--resume` for long runs to avoid recalculating existing pairs.

2.  **Submit the job to SCC:**
    ```bash
    qsub run_filter.sh
    ```
    *   Monitor the job using `qstat -u $(whoami)`.
    *   Output will be `finches_interaction_results_chunked.parquet` (or your specified output file).

### 3. Analyze Finches Results

Use `finches-results-analysis.ipynb` to load and analyze the `.parquet` output from the `finches` filtering step. This notebook will guide you through:

*   Loading the interaction results.
*   Filtering for significant attraction/repulsion.
*   Identifying candidate protein pairs for AF3 prediction.

### 4. AlphaFold3 Input Preparation

Use `01_prepare_inputs.py` to generate the necessary AlphaFold3 input JSON files for the filtered pairs.

*   **Input:** The filtered `finches` results (e.g., a CSV of selected pairs). The script expects a `json_list.txt` file which contains paths to input JSON files.
*   **Output:** A series of `alphafold_input.json` files, typically placed in an `input/` directory, and `json_list.txt` listing these files.

### 5. AlphaFold3 Data Pipeline (SCC)

This stage performs the data preparation for AF3. It's a CPU-only array job.

1.  **Copy SCC AlphaFold3 Examples:** If you haven't already, ensure the example files are copied:
    ```bash
    module load alphafold3/3.0.0
    mkdir -p af3_scc_examples
    cp -r "$SCC_ALPHAFOLD3_EXAMPLES"/* af3_scc_examples/
    # Ensure json_list.txt is in the directory where you submit the job from
    ```
2.  **Edit `02_run_data_pipeline.sh`:**
    *   Update `#$ -P myproject` to your SCC project.
    *   Set `export MODEL_DIR="/path/to/your/alphafold3_models"`.
    *   Confirm the path to `json_list.txt` is correct for the `-t` argument.
    *   Adjust `h_rt`, `pe omp`, `mem_per_core` as needed.
3.  **Submit the data pipeline job:**
    ```bash
    DATA_PIPELINE_HOLD_NAME="af3_data_pipeline_stage" # Or your desired name
    qsub -N "$DATA_PIPELINE_HOLD_NAME" 02_run_data_pipeline.sh
    echo "$DATA_PIPELINE_HOLD_NAME" > data_prep_hold_name.txt # Save name for hold_jid
    ```
    *   This will create augmented JSON files in `af3_data_json/<JOB_NAME>/<JOB_NAME>_data.json`.

### 6. AlphaFold3 Inference (SCC)

This stage performs the structural prediction using GPUs. It's an array job that depends on the data pipeline completing.

1.  **Edit `03_run_inference.sh`:**
    *   Update `#$ -P myproject` to your SCC project.
    *   Set `export MODEL_DIR="/path/to/your/alphafold3_models"` (same as above).
    *   Ensure `#$ -hold_jid af3_data_pipeline_stage` matches the `-N` name from your `02_run_data_pipeline.sh` submission.
    *   Confirm the path to `json_list.txt` is correct.
    *   Verify `AUGMENTED_JSON_FILE` path logic matches the output from the data pipeline (usually `_data.json`).
    *   Adjust `h_rt`, `pe omp`, `mem_per_core`, `gpus`, `gpu_type`, `gpu_memory`, `gpu_c` to match your GPU requirements and SCC availability (A100-80G is recommended if available).
2.  **Submit the inference job:**
    ```bash
    DATA_PIPELINE_HOLD_NAME=$(cat data_prep_hold_name.txt) # Retrieve job name
    qsub -hold_jid "$DATA_PIPELINE_HOLD_NAME" 03_run_inference.sh
    ```
    *   Results (PDBs, JSONs) will be stored in `af3_outputs/<JOB_NAME>/`.

### 7. AlphaFold3 Post-processing

After AF3 inference, use `04_post_processing_results.ipynb` to analyze the predicted structures. This notebook will help you:

*   Load predicted structures and associated metadata.
*   Extract key quality metrics (pLDDT, pae).
*   Filter and rank interaction predictions based on confidence scores.

## Output

The pipeline generates several key outputs:

*   `finches_interaction_results_chunked.parquet`: A parquet file containing calculated epsilon values, mean epsilon, std dev, and interaction type for all viral-human protein pairs.
*   `af3_data_json/<JOB_NAME>/<JOB_NAME>_data.json`: Augmented JSON files per predicted pair, generated by the AF3 data pipeline.
*   `af3_outputs/<JOB_NAME>/*.pdb`: Predicted PDB files for the protein complexes.
*   `af3_outputs/<JOB_NAME>/*.json`: AlphaFold3 prediction metadata (e.g., confidence scores).
*   Analysis notebooks will output further filtered lists, visualizations, and summary tables.

## Authorship

This pipeline and its associated scripts were solely developed by Philipp Trollmann during their first PhD rotation in Dr. Pinghua Liu's lab at Boston University.
