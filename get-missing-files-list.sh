#!/bin/bash -l

# IMPORTANT: Define the absolute path to your project's base directory.
PROJECT_BASE_DIR="/projectnb/cancergrp/Philipp/alphafold3_pipeline"

echo "------------------------------------------------------------"
echo "Identifying missing AlphaFold3 data pipeline outputs..."
echo "Project directory: $PROJECT_BASE_DIR"
echo "------------------------------------------------------------"

cd "$PROJECT_BASE_DIR" || {
    echo "CRITICAL Error: Failed to change to project directory: $PROJECT_BASE_DIR" >&2
    exit 1
}

# Define paths relative to PROJECT_BASE_DIR
AF3_INPUTS_DIR="af3_inputs"
AF3_DATA_JSON_DIR="af3_data_json"
MISSING_JSON_LIST_FILE="json_list.txt"

# 1. Generate a comprehensive list of all *expected* input JSONs
# This assumes your input JSONs are under af3_inputs/**/*.json
echo "Scanning for all input JSON files..."
find "$AF3_INPUTS_DIR" -name "*.json" | sort > all_input_jsons.tmp

# Initialize the new list for missing files
> "$MISSING_JSON_LIST_FILE" # Clear any previous content

# 2. Iterate through each input JSON and check if its corresponding output exists and is non-empty
NUM_TOTAL_INPUTS=0
NUM_MISSING=0

while IFS= read -r input_relative_path; do
    NUM_TOTAL_INPUTS=$((NUM_TOTAL_INPUTS + 1))

    # Extract the base job name (e.g., "VT_20379_AVA07189.1_vs_NP_001027451.1")
    # Convert to lowercase as AlphaFold3 outputs use lowercase job names for directories.
    JOB_NAME_LOWER=$(basename "$input_relative_path" .json | tr '[:upper:]' '[:lower:]')

    # Determine the subdirectory structure from the input path (e.g., "cleaved_VP1_with_primary_P53_interactors")
    # This removes the "af3_inputs/" prefix.
    # dirname returns "af3_inputs/cleaved_VP1_with_primary_P53_interactors"
    # sed removes "af3_inputs/" leaving "cleaved_VP1_with_primary_P53_interactors"
    SUBDIR_PATH=$(dirname "$input_relative_path" | sed "s|^$AF3_INPUTS_DIR/||")

    # Construct the expected absolute path to the output _data.json file
    EXPECTED_OUTPUT_FILE="$PROJECT_BASE_DIR/$AF3_DATA_JSON_DIR/${SUBDIR_PATH}/${JOB_NAME_LOWER}/${JOB_NAME_LOWER}_data.json"

    # Check if the expected output file exists and is non-empty
    if [ ! -s "$EXPECTED_OUTPUT_FILE" ]; then
        echo "$input_relative_path" >> "$MISSING_JSON_LIST_FILE"
        NUM_MISSING=$((NUM_MISSING + 1))
    fi
done < all_input_jsons.tmp

# 3. Report findings and provide instructions
echo "------------------------------------------------------------"
echo "Total input JSON files found: $NUM_TOTAL_INPUTS"
echo "Found $NUM_MISSING missing AlphaFold3 data pipeline outputs."
echo "------------------------------------------------------------"

if [ "$NUM_MISSING" -gt 0 ]; then
    echo "A new list of missing files has been created: $PROJECT_BASE_DIR/$MISSING_JSON_LIST_FILE"
    echo "To run these missing tasks, you need to first replace your main 'json_list.txt' with this new list."
    echo ""
    echo "   mv $MISSING_JSON_LIST_FILE json_list.txt"
    echo ""
    echo "Then, submit a new SGE job array using this (now updated) 'json_list.txt':"
    echo ""
    echo "   NUM_TASKS_TO_RUN=$(wc -l < json_list.txt)"
    echo "   qsub -t 1-\${NUM_TASKS_TO_RUN} 02_run_data_pipeline.sh"
    echo ""
    echo "Remember to revert 'json_list.txt' to its original, full version if you intend to run a full sweep later."
else
    echo "All expected outputs are present and non-empty. No new run needed."
fi

# Clean up temporary files
rm all_input_jsons.tmp
echo "Temporary files cleaned up."
echo "Script finished."