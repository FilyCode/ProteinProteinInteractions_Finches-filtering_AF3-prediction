import pandas as pd
import numpy as np
import finches
from finches import Mpipi_frontend, CALVADOS_frontend
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pyarrow as pa 
import pyarrow.parquet as pq
import os
import warnings
import argparse
from typing import Dict, List, Any, Tuple # Added for type hinting from previous script

# This line will ignore FutureWarning messages coming specifically
# from the 'finches.forcefields.calvados' module.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="finches\.forcefields\.calvados" # Using regex to match the module path
)

# Configuration 
ID_A_COLUMN = "ID_A_Interactor"
SEQUENCE_A_COLUMN = "Sequence_A"
ID_B_COLUMN = "ID_B_Interactor"
SEQUENCE_B_COLUMN = "Sequence_B"

parser = argparse.ArgumentParser(description="Filter viral-human pairs with finches.")
parser.add_argument('--num_processes', type=int, default=cpu_count(),
                    help='Number of processes to use for parallelization. Defaults to system CPU count.')
# Add your other arguments here, e.g., input/output file paths
parser.add_argument('--input_file', type=str, required=True, help='Path to the protein input file.')
parser.add_argument('--output_file', type=str, required=True, help='Path for the output file.')
parser.add_argument('--resume', action='store_true', help='If specified, checks the output file for existing combinations and skips their calculation. New results are appended.')

args = parser.parse_args()

NUM_PROCESSES = args.num_processes

# Size of chunks to write to file
CHUNK_SIZE = 1000 

# Input file name
INPUT_FILENAME = args.input_file

# Output file name
OUTPUT_FILENAME = args.output_file
os.makedirs(os.path.dirname(OUTPUT_FILENAME), exist_ok=True)

# Column names for the final DataFrame
# Define them explicitly to ensure consistent order and types for chunks
COLUMN_NAMES = [
    's1_id', 's2_id',
    'mf_s1_s2', 'mf_s2_s1',
    'cf_s1_s2', 'cf_s2_s1',
    'mean_epsilon', 'std_epsilon',
    'interaction_type'
]
# Define dtypes for numeric columns to force float32
NUMERIC_DTYPES = {
    'mf_s1_s2': np.float32,
    'mf_s2_s1': np.float32,
    'cf_s1_s2': np.float32,
    'cf_s2_s1': np.float32,
    'mean_epsilon': np.float32,
    'std_epsilon': np.float32
}

# Define the PyArrow Schema explicitly for consistent writing (CRUCIAL FOR RESUME)
# Default to string for IDs and interaction_type, and float32 for numerics
PA_SCHEMA = pa.schema([
    pa.field('s1_id', pa.string()),
    pa.field('s2_id', pa.string()),
    pa.field('mf_s1_s2', pa.float32()),
    pa.field('mf_s2_s1', pa.float32()),
    pa.field('cf_s1_s2', pa.float32()),
    pa.field('cf_s2_s1', pa.float32()),
    pa.field('mean_epsilon', pa.float32()),
    pa.field('std_epsilon', pa.float32()),
    pa.field('interaction_type', pa.string())
])


# Global variables to hold the finches frontend objects within each worker process
# These will be initialized by the 'init_worker' function
_mf_frontend = None
_cf_frontend = None

# Worker initialization function 
def init_worker():
    """
    This function is called once per worker process when the Pool starts.
    It initializes the finches frontend objects for that specific process.
    """
    global _mf_frontend, _cf_frontend
    _mf_frontend = Mpipi_frontend()
    _cf_frontend = CALVADOS_frontend()
    # print(f"Worker process {os.getpid()} initialized finches frontends.") # Optional: for debugging


# Define the calculation function for a single pair 
def calculate_epsilons(pair_data: Tuple[str, str, str, str]) -> Dict[str, Any]:
    """
    Performs the 4 epsilon calculations for a given pair of sequences
    and returns a dictionary of results.
    """
    s1_id, s1_seq, s2_id, s2_seq = pair_data

    # Use the globally (per-process) initialized frontends
    global _mf_frontend, _cf_frontend


    results = {
        's1_id': s1_id,
        's2_id': s2_id,
        'mf_s1_s2': np.nan,
        'mf_s2_s1': np.nan,
        'cf_s1_s2': np.nan,
        'cf_s2_s1': np.nan,
        'mean_epsilon': np.nan,
        'std_epsilon': np.nan,
        'interaction_type': 'N/A'
    }

    try:
        results['mf_s1_s2'] = _mf_frontend.epsilon(s1_seq, s2_seq)
        results['mf_s2_s1'] = _mf_frontend.epsilon(s2_seq, s1_seq)
        results['cf_s1_s2'] = _cf_frontend.epsilon(s1_seq, s2_seq)
        results['cf_s2_s1'] = _cf_frontend.epsilon(s2_seq, s1_seq)

        # Calculate mean and standard deviation
        epsilon_values = [
            results['mf_s1_s2'], results['mf_s2_s1'],
            results['cf_s1_s2'], results['cf_s2_s1']
        ]
        results['mean_epsilon'] = np.mean(epsilon_values)
        results['std_epsilon'] = np.std(epsilon_values)

        # Determine interaction type
        if results['mean_epsilon'] > 0:
            results['interaction_type'] = 'repulsion'
        elif results['mean_epsilon'] < 0:
            results['interaction_type'] = 'attraction'
        else:
            results['interaction_type'] = 'neutral'

    except Exception as e:
        # Catch any errors from finches (e.g., malformed sequence) and log them, then continue with NaNs
        print(f"Warning: Error processing pair ({s1_id}, {s2_id}): {e}")
        # Results are already initialized to NaN, so no further action needed

    return results

# Main execution block for multiprocessing
if __name__ == "__main__":
    # Load DataFrame that already contains the desired combinations
    try:
        interactions_df = pd.read_csv(INPUT_FILENAME)
        print(f"Loaded interactions_df: {interactions_df.shape[0]} combinations")

    except FileNotFoundError as e:
        print(f"Error loading file: {e}. Make sure {INPUT_FILENAME} exists.")
        exit(1)

    # For testing purposes we reduce dataset (optional, uncomment for full run)
    # This keeps the behavior consistent with your previous testing reduction.
    #interactions_df = interactions_df[:200]
    #print(f"Dataset reduced to {interactions_df.shape[0]} combinations for testing.")

    # Pre-processing: Convert sequence columns to string type first to handle NaNs consistently
    interactions_df[SEQUENCE_A_COLUMN] = interactions_df[SEQUENCE_A_COLUMN].astype(str)
    interactions_df[SEQUENCE_B_COLUMN] = interactions_df[SEQUENCE_B_COLUMN].astype(str)

    # Filter out empty strings or 'nan' strings (which occur from NaN conversion to str)
    initial_rows = interactions_df.shape[0]
    interactions_df = interactions_df[
        (interactions_df[SEQUENCE_A_COLUMN].str.strip() != '') &
        (interactions_df[SEQUENCE_A_COLUMN].str.lower() != 'nan') &
        (interactions_df[SEQUENCE_B_COLUMN].str.strip() != '') &
        (interactions_df[SEQUENCE_B_COLUMN].str.lower() != 'nan')
    ]
    
    print(f"After cleaning: interactions_df: {interactions_df.shape[0]} valid combinations (removed {initial_rows - interactions_df.shape[0]} invalid rows)")

    # Prepare the iterable of arguments for the worker function (all possible tasks)
    all_task_args = []
    # Using itertuples() can be slightly more efficient than iterrows()
    for row in interactions_df.itertuples(index=False): # index=False to exclude the DataFrame index
        # Access elements by column name using getattr
        s1_id = getattr(row, ID_A_COLUMN.replace(' ', '_')) # Replace spaces for valid attribute name
        s1_seq = getattr(row, SEQUENCE_A_COLUMN.replace(' ', '_'))
        s2_id = getattr(row, ID_B_COLUMN.replace(' ', '_'))
        s2_seq = getattr(row, SEQUENCE_B_COLUMN.replace(' ', '_'))
        all_task_args.append((s1_id, s1_seq, s2_id, s2_seq))
    total_possible_combinations = len(all_task_args)

    # --- Resume Logic Initialization ---
    completed_pairs = set()
    existing_df_for_rewrite: pd.DataFrame = pd.DataFrame(columns=COLUMN_NAMES) # To hold existing data if resuming
    output_file_exists_before_run = os.path.exists(OUTPUT_FILENAME)
    temp_output_filename = OUTPUT_FILENAME + "_temp.parquet" # Define temp filename

    if args.resume and output_file_exists_before_run:
        print(f"Resume mode enabled. Checking existing file: {OUTPUT_FILENAME}")
        try:
            # Read all existing data. This can be memory-intensive for extremely large files,
            # but is necessary to then write it back with new data.
            temp_existing_df = pd.read_parquet(OUTPUT_FILENAME, engine='pyarrow')

            # Ensure consistent dtypes using the explicit schema for parsing robustness
            # This helps against issues if a column only had NaNs which PyArrow might infer as int/bool
            for field in PA_SCHEMA:
                col_name = field.name
                col_type = field.type
                if col_name in temp_existing_df.columns:
                    if pa.types.is_string(col_type):
                        temp_existing_df[col_name] = temp_existing_df[col_name].astype(object).astype(str)
                    elif pa.types.is_floating(col_type):
                        temp_existing_df[col_name] = pd.to_numeric(temp_existing_df[col_name], errors='coerce').astype(np.float32)
                    # Add other type conversions if necessary (e.g., int)

            # Filter out any rows that don't have required IDs (robustness)
            temp_existing_df = temp_existing_df.dropna(subset=['s1_id', 's2_id'])

            if not temp_existing_df.empty:
                completed_pairs = set(zip(temp_existing_df['s1_id'], temp_existing_df['s2_id']))
                existing_df_for_rewrite = temp_existing_df # Keep the full df in memory for rewriting
                print(f"Found {len(completed_pairs)} already calculated combinations in {OUTPUT_FILENAME}.")
            else:
                print(f"Existing file {OUTPUT_FILENAME} is empty or contains no valid pairs. Starting fresh.")

        except Exception as e:
            print(f"Warning: Could not read existing output file {OUTPUT_FILENAME} for resume: {e}.")
            print("Proceeding to calculate all combinations (output file will be overwritten if it exists).")
            completed_pairs = set() # Reset
            existing_df_for_rewrite = pd.DataFrame(columns=COLUMN_NAMES) # Reset
    elif args.resume and not output_file_exists_before_run:
        print(f"Resume mode enabled but output file {OUTPUT_FILENAME} does not exist. Starting a new calculation.")

    # Filter task_args based on completed_pairs
    tasks_to_perform = [arg for arg in all_task_args if (arg[0], arg[2]) not in completed_pairs]
    total_calculations_to_perform = len(tasks_to_perform)

    # Early Exit if Nothing to Do
    if total_calculations_to_perform == 0:
        if args.resume and not existing_df_for_rewrite.empty:
            print("All relevant combinations already calculated. Output file remains unchanged.")
        else:
            print("No combinations to calculate or existing data to preserve. No output file action taken.")
        exit()

    # Determine the *actual* file path to write to during the main calculation loop
    # If resuming AND there's existing data to rewrite AND new tasks to perform, we use a temp file.
    # Otherwise, we write directly to the OUTPUT_FILENAME.
    writing_to_temp_file = False
    if args.resume and not existing_df_for_rewrite.empty and total_calculations_to_perform > 0:
        target_output_file = temp_output_filename
        writing_to_temp_file = True
        print(f"Resuming with existing data: will write to temporary file {target_output_file}.")
    else:
        target_output_file = OUTPUT_FILENAME
        # If not resuming, or if resuming but existing data is empty, ensure the target_output_file is removed.
        if os.path.exists(target_output_file):
            print(f"Removing existing file: {target_output_file} to start fresh.")
            try:
                os.remove(target_output_file)
            except OSError as e:
                print(f"Error removing file {target_output_file}: {e}. Please ensure file permissions.")
                exit(1)

    # Initialize ParquetWriter
    parquet_writer = None
    try:
        parquet_writer = pq.ParquetWriter(target_output_file, PA_SCHEMA)
        print(f"Initialized ParquetWriter for {target_output_file} with explicit schema.")
    except Exception as e:
        print(f"Error initializing ParquetWriter for {target_output_file}: {e}")
        exit(1)

    # Write Back Existing Data (if applicable)
    if writing_to_temp_file and not existing_df_for_rewrite.empty:
        print(f"Writing back {len(existing_df_for_rewrite)} existing rows to {target_output_file}...")
        for i in tqdm(range(0, len(existing_df_for_rewrite), CHUNK_SIZE), desc="Rewriting existing data"):
            chunk_df = existing_df_for_rewrite.iloc[i : i + CHUNK_SIZE]
            # Convert to PyArrow Table, ensuring it conforms to the predefined schema
            table = pa.Table.from_pandas(chunk_df, schema=PA_SCHEMA, preserve_index=False)
            parquet_writer.write_table(table)
        print("Finished rewriting existing data.")

    print(f"\nStarting calculations for {total_calculations_to_perform} new combinations (out of {total_possible_combinations} total) using {NUM_PROCESSES} processes.")
    print(f"Results will be appended to {target_output_file}")

    results_buffer = [] # Buffer for new results from calculations

    with Pool(processes=NUM_PROCESSES, initializer=init_worker) as pool:
        # Use imap_unordered to process results as they become available
        for result in tqdm(pool.imap_unordered(calculate_epsilons, tasks_to_perform), total=total_calculations_to_perform, desc="Calculating new combinations"):
            if result: # Only process if calculation was successful
                results_buffer.append(result)

                # If buffer reaches CHUNK_SIZE, convert to DataFrame, then PyArrow Table, and write
                if len(results_buffer) >= CHUNK_SIZE:
                    chunk_df = pd.DataFrame(results_buffer, columns=COLUMN_NAMES)
                    # Convert to PyArrow Table, ensuring it conforms to the predefined schema
                    table = pa.Table.from_pandas(chunk_df, schema=PA_SCHEMA, preserve_index=False)
                    parquet_writer.write_table(table)
                    results_buffer = [] # Clear the buffer

    # Write any remaining new results
    if results_buffer:
        chunk_df = pd.DataFrame(results_buffer, columns=COLUMN_NAMES)
        # Convert to PyArrow Table, ensuring it conforms to the predefined schema
        table = pa.Table.from_pandas(chunk_df, schema=PA_SCHEMA, preserve_index=False)
        parquet_writer.write_table(table)
        print(f"Wrote final chunk of {len(results_buffer)} new rows.")

    # Crucial: Close the ParquetWriter to finalize the file
    if parquet_writer is not None:
        parquet_writer.close()
        print(f"\nAll calculations and writing complete to {target_output_file}!")

        # Final rename step if we were writing to a temporary file
        if writing_to_temp_file:
            print(f"Renaming {temp_output_filename} to {OUTPUT_FILENAME}...")
            # Remove the original OUTPUT_FILENAME before renaming the temp file
            if os.path.exists(OUTPUT_FILENAME):
                try:
                    os.remove(OUTPUT_FILENAME)
                    print(f"Removed original file: {OUTPUT_FILENAME}")
                except OSError as e:
                    print(f"Error removing original {OUTPUT_FILENAME} for rename: {e}")
                    print(f"Please manually delete '{OUTPUT_FILENAME}' and rename '{temp_output_filename}' to '{OUTPUT_FILENAME}' to finalize.")
                    exit(1) # Critical failure, cannot complete without user intervention

            try:
                os.rename(temp_output_filename, OUTPUT_FILENAME)
                print(f"Successfully renamed {temp_output_filename} to {OUTPUT_FILENAME}.")
            except OSError as e:
                print(f"Error renaming {temp_output_filename} to {OUTPUT_FILENAME}: {e}")
                print(f"Temporary file '{temp_output_filename}' might still exist. Please check manually.")
                exit(1) # Critical failure

        print(f"Final results are in {OUTPUT_FILENAME}")
    else:
        # This case implies an error during ParquetWriter initialization or no data was processed.
        print(f"Error: ParquetWriter was not initialized or closed properly, or no data was written to {OUTPUT_FILENAME}.")


    # Optional: Verify the file size and first few rows after completion
    if os.path.exists(OUTPUT_FILENAME):
        print(f"\nFinal file size: {os.path.getsize(OUTPUT_FILENAME) / (1024*1024):.2f} MB")
        try:
            # Read a small part of the output file to show the head
            final_df_head = pd.read_parquet(OUTPUT_FILENAME, engine='pyarrow').head(5)
            print("\nFirst 5 rows of the final output file:")
            print(final_df_head)
            print("\nData types of numerical columns (should be float32-like):")
            print(final_df_head[list(NUMERIC_DTYPES.keys())].dtypes)
        except Exception as e:
            print(f"Could not read back the file head for verification: {e}. Is the Parquet file corrupted?")
    else:
         print(f"Output file {OUTPUT_FILENAME} was not created.")