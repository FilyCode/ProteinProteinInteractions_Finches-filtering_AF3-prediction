# Protein-Protein Interaction Prediction: Finches-filtered AlphaFold3 Pipeline

This repository provides a comprehensive pipeline for predicting viral-human protein-protein interactions (PPIs). It leverages [`finches`](https://github.com/idptools/finches) for identifying potential interactions involving intrinsically disordered regions (IDRs) as a filtering step, followed by detailed structural prediction using AlphaFold3 (AF3) for selected pairs. The pipeline is fully optimized for execution on the Boston University Shared Computing Cluster (BU SCC) using the SGE job scheduler and robust for large-scale array jobs with parallelization.


## Overview

Predicting protein-protein interactions, particularly between viral and human proteins, is crucial for understanding host-pathogen dynamics. This pipeline addresses the computational intensity of methods like AlphaFold3 by introducing an initial filtering step using Finches.


## Workflow

The pipeline consists of the following sequential stages:

1. **Initial Data Preparation**
    - Retrieve protein sequences for a list of PPI pairs (e.g., literature reference; provide interacting pairs as Ensembl IDs in a TSV)
    - Use UniProt API to fetch canonical sequences for each unique Ensembl gene  
      (`get-sequences-for-interacting-proteins.py`, optionally via `run-sequence-api-requests.sh`)
    - **Output:** CSV file with PPI pairs and their sequences

2. **Protein Tiling & Dataset Expansion (Optional)**
    - Split proteins into overlapping peptides (tiles) for systematic mapping (e.g., 48AA segment, 24AA overlap, configurable; `split-protein-into-overlapping-peptides.ipynb`)
    - **Output:** CSVs for full pairs, tile-to-full, and tile-to-tile pairs

3. **Negative Control Generation**
    - Generate negative controls: shuffled (within-dataset) pairs, and random amino acid sequence pairs, matched for length distribution (`get-negative-control-proteins.ipynb`)
    - **Output:** Corresponding negative control CSVs for each positive dataset

4. **Finches IDR Interaction Filtering**
    - Calculate interaction propensity (epsilon values) for all pairs using Finches on all dataset and control variants  
      (`calculate-existing-combinations-with-finches.py`, various run scripts)
    - **Output:** Parquet results for each analysis (supports resume)

5. **Analyze Finches Results**
    - Process and visualize Finches output to select promising protein pairs  
      (`finches-results-analysis.ipynb`, `finches-results-analysis-pos-controll.ipynb`)

6. **AlphaFold3 Input Preparation**
    - Generate input JSON files for selected protein pairs for AF3  
      (`01_prepare_inputs.py`)
    - **Output:** `af3_inputs/…/*.json` and master `json_list.txt`

7. **AlphaFold3 Data Pipeline**
    - Highly parallelized feature generation (array jobs), avoids redundant computation via resume logic  
      (`02_run_data_pipeline.sh`)
    - **Output:** `af3_data_json/.../*.json` feature files

8. **AlphaFold3 Inference**
    - Massively parallel with dynamic work allocation: inference jobs poll for ready feature outputs and claim them via lockfiles; implements robust auto-cancellation ("self-cleaning" job array) when all tasks finish.  
      (`03_run_inference.sh`)
    - **Output:** `af3_outputs/.../*.cif` model files and metadata

9. **AlphaFold3 Post-processing**
    - Analyze structures and metrics  
      (`04_post_processing_results.ipynb`)


## Directory Structure

Expected directories (customize as needed):

data/                    # Raw input lists, sequence datasets, controls
af3_inputs/              # AlphaFold3 input JSONs, organized by experiment/control type
af3_data_json/           # AlphaFold3 data pipeline output JSONs (features)
af3_outputs/             # AlphaFold3 final structure files (.cif, .json, metadata)
results/                 # Finches epsilon results and summary stats
data/positive_controls/  # Positive control datasets
data/negative_controls_shuffled/  # Shuffled negative controls
data/negative_controls_random/    # Random sequence controls
af3_inference_locks/     # Lock directories for polling in inference


## Setup and Prerequisites

### Conda Environments

You need multiple environments:

1. **finches-env** (for finches):

    ```bash
    conda env create -f finches_env.yml
    conda activate finches-env
    ```

2. **jupyter-env** (for notebooks, sequence retrieval/prep/analysis):

    ```bash
    conda env create -f jupyter_env.yml
    conda activate jupyter-env
    python -m ipykernel install --user --name=jupyter-env
    ```

3. **AlphaFold3** (provided as a module on BU SCC):

    ```bash
    module load alphafold3/3.0.0
    ```

### AlphaFold3 Model Weights

Download model weights from DeepMind, set `MODEL_DIR` variable in AlphaFold scripts.

### BU SCC Specifics

- Scripts use SGE; most support multicore jobs (`-pe omp N`)
- Robust resume/redundancy skipping for large jobs
- New scripts for polling, locking, and auto-cancellation (job arrays clean themselves up as jobs complete)
- Update `#$ -P myproject` for your SCC project

---

## Usage

### 1. Retrieve Sequences for Literature PPIs

Prepare a `.tsv` file listing Ensembl pairs.  
Run:

    
    qsub run-sequence-api-requests.sh
    

Output: 
    
    data/interacting-proteins-full-dataset_Lit-BM.csv
    

### 2. Tiling Proteins into Overlapping Peptides (optional)

Run the notebook:
    
    # Open in Jupyter, run all cells
    split-protein-into-overlapping-peptides.ipynb
    
Outputs:    

    data/interacting-proteins_tiles-to-full-protein.csv
    data/interacting-proteins_tiles-to-tiles.csv    

3. Generate Negative Controls

Run:
    
    # Open in Jupyter, run all cells
    get-negative-control-proteins.ipynb    

Outputs:
    
    data/negative_controls_shuffled/...
    data/negative_controls_random/...    

4. Finches Interaction Calculation (all data/control variants)

Submit job(s):

    qsub run-finches-controls.sh
    For main dataset: qsub run-finches-calculations.sh


Outputs: Parquet results per variant

5. Analyze Finches Results

Run:

    # Open in Jupyter, run all cells
    finches-results-analysis.ipynb
    finches-results-analysis-pos-controll.ipynb

6. Prepare AlphaFold3 Inputs

Run:

    python 01_prepare_inputs.py

Outputs:

    JSON files in af3_inputs/, plus json_list.txt
    
7. AlphaFold3 Data Pipeline

Feature generation, highly parallel (CPU array job):

    NUM_TASKS=$(cat json_list.txt | wc -l)
    qsub -t 1-"${NUM_TASKS}" 02_run_data_pipeline.sh

Output:

    af3_data_json/…/…_data.json
    
8. AlphaFold3 Inference

GPU array job; dynamically picks available feature files on-the-fly (work stealing/polling/locking):

    qsub -t 1-"${NUM_TASKS}" 03_run_inference.sh

Output:

    af3_outputs/…/…_model.cif files
    
9. Post-processing

Run:

    # Open in Jupyter, run all cells
    04_post_processing_results.ipynb
    
Output

    Sequence datasets for all PPI pairs and variants
    Finches epsilon calculations for all data/controls
    AlphaFold3 input JSONs, output structures, and metrics
    Systematic analysis notebooks for scoring, experimental design, control validation


## Authorship

This pipeline and its associated scripts were solely developed by Philipp Trollmann during his second PhD rotation in Dr. Juan Fuxman Bass's lab at Boston University.

---

For further details, see the individual scripts or reach out!

All scripts are heavily annotated. If you use/adapt this pipeline, please cite the repository and attribute appropriately.

---
