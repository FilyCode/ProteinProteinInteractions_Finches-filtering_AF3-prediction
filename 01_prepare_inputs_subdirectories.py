# 01_prepare_inputs.py
import pandas as pd
import json
import os
import random

input_csv = "inputs/strong_significant_viral_tiles_with_primary_P53_interactors.csv"
output_dir = "af3_inputs"
json_list_file = "json_list.txt"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Read the filtered hits
df = pd.read_csv(input_csv)

json_paths = []
for index, row in df.iterrows():
    s1_id = row["ID_A_Interactor"].upper()
    s1_sequence = row["Sequence_A"]
    s2_id = row["ID_B_Interactor"].upper()
    s2_sequence = row["Sequence_B"]

    # Check if s1_sequence is null (NaN/None), empty string, or 'NOT_FOUND'
    # OR if s2_sequence is null (NaN/None), empty string, or 'NOT_FOUND'
    if (pd.isna(s1_sequence) or s1_sequence == '' or s1_sequence == 'NOT_FOUND' or
        pd.isna(s2_sequence) or s2_sequence == '' or s2_sequence == 'NOT_FOUND'):
        print(f"Skipping row {index} due to invalid sequence for ID_A: '{s1_id}' or ID_B: '{s2_id}'")
        continue

    # AlphaFold3 requires simple uppercase letters for entity IDs
    af3_entity_id_1 = "A"
    af3_entity_id_2 = "B"

    # Generate a unique job name
    job_name = f"{s1_id}_vs_{s2_id}".replace(" ", "_").replace("/", "-") # Sanitize for directory names

    # AlphaFold3 JSON format for two proteins
    input_json = {
        "name": job_name,
        "modelSeeds": [random.randint(1, 1000000)],  # Use a random seed for each job
        "sequences": [
            {"protein": {"id": af3_entity_id_1, "sequence": s1_sequence}},
            {"protein": {"id": af3_entity_id_2, "sequence": s2_sequence}},
        ],
        "dialect": "alphafold3",
        "version": 1,
    }

    output_dir_sub = f"{output_dir}/{s1_id}"
    os.makedirs(output_dir_sub, exist_ok=True)

    output_json_path = os.path.join(output_dir_sub, f"{job_name}.json")
    with open(output_json_path, "w") as f:
        json.dump(input_json, f, indent=2)

    json_paths.append(output_json_path)
    print(f"Generated {output_json_path}")

# Write the list of JSON paths for the Batch job
with open(json_list_file, "w") as f:
    for path in json_paths:
        f.write(f"{path}\n")

print(f"\nAll input JSON files generated in subdirectories in '{output_dir}/'.")
print(f"List of JSON paths saved to '{json_list_file}'.")
print(f"Total protein pairs to process: {len(json_paths)}")