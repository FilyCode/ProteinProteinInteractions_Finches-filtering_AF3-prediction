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
import sys 
from typing import Dict, List, Any, Tuple

# This line will ignore FutureWarning messages coming specifically from the 'finches.forcefields.calvados' module.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="finches\.forcefields\.calvados" # Using regex to match the module path
)

# Column Name Configuration
SEQUENCE_COLUMN = 'Sequence' # column containing the protein sequence string
ID_COLUMN = 'ID'     # column containing a unique identifier for each sequence

parser = argparse.ArgumentParser(description="Filter viral-human pairs with finches.")
parser.add_argument('--num_processes', type=int, default=cpu_count(),
                    help='Number of processes to use for parallelization. Defaults to system CPU count.')
# Add your other arguments here, e.g., input/output file paths
parser.add_argument('--input_file_virus', type=str, required=True, help='Path to the virus protein input file.')
parser.add_argument('--input_file_human', type=str, required=True, help='Path to the human proteom input file.')
parser.add_argument('--output_file', type=str, required=True, help='Path for the output file.')
parser.add_argument('--resume', action='store_true', help='If specified, checks the output file for existing combinations and skips their calculation. New results are appended.')

args = parser.parse_args()

NUM_PROCESSES = args.num_processes

# Size of chunks to write to file
CHUNK_SIZE = 10000

# Input file name
VIRAL_INPUT_FILENAME = args.input_file_virus
HUMAN_INPUT_FILENAME = args.input_file_human

# Output file name (final destination after successful completion)
OUTPUT_FILENAME = args.output_file
os.makedirs(os.path.dirname(OUTPUT_FILENAME), exist_ok=True)

# Column names for the final DataFrame
COLUMN_NAMES = [
    's1_id', 's2_id',
    'mf_s1_s2', 'mf_s2_s1',
    'cf_s1_s2', 'cf_s2_s1',
    'mean_epsilon', 'std_epsilon',
    'interaction_type', 's1_length', 's2_length',
    's1_sequence', 's2_sequence'
]
# Define dtypes for numeric columns to force float32
NUMERIC_DTYPES = {
    'mf_s1_s2': np.float32,
    'mf_s2_s1': np.float32,
    'cf_s1_s2': np.float32,
    'cf_s2_s1': np.float32,
    'mean_epsilon': np.float32,
    'std_epsilon': np.float32,
    's1_length': np.float32,
    's2_length': np.float32
}

# Define the PyArrow Schema explicitly for consistent writing
PA_SCHEMA = pa.schema([
    pa.field('s1_id', pa.string()),
    pa.field('s2_id', pa.string()),
    pa.field('mf_s1_s2', pa.float32()),
    pa.field('mf_s2_s1', pa.float32()),
    pa.field('cf_s1_s2', pa.float32()),
    pa.field('cf_s2_s1', pa.float32()),
    pa.field('mean_epsilon', pa.float32()),
    pa.field('std_epsilon', pa.float32()),
    pa.field('interaction_type', pa.string()),
    pa.field('s1_length', pa.float32()),
    pa.field('s2_length', pa.float32()),
    pa.field('s1_sequence', pa.string()),
    pa.field('s2_sequence', pa.string())
])

# Global variables to hold the finches frontend objects within each worker process
_mf_frontend = None
_cf_frontend = None

def init_worker():
    """
    This function is called once per worker process when the Pool starts.
    It initializes the finches frontend objects for that specific process.
    """
    global _mf_frontend, _cf_frontend
    _mf_frontend = Mpipi_frontend()
    _cf_frontend = CALVADOS_frontend()


def calculate_epsilons(pair_data: Tuple[str, str, str, str]) -> Dict[str, Any]:
    """
    Performs the 4 epsilon calculations for a given pair of sequences
    and returns a dictionary of results.
    """
    s1_id, s1_seq, s2_id, s2_seq = pair_data

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
        'interaction_type': 'N/A',
        's1_length': len(s1_seq),
        's2_length': len(s2_seq),
        's1_sequence': s1_seq,
        's2_sequence': s2_seq
    }

    try:
        results['mf_s1_s2'] = _mf_frontend.epsilon(s1_seq, s2_seq)
        results['mf_s2_s1'] = _mf_frontend.epsilon(s2_seq, s1_seq)
        results['cf_s1_s2'] = _cf_frontend.epsilon(s1_seq, s2_seq)
        results['cf_s2_s1'] = _cf_frontend.epsilon(s2_seq, s1_seq)

        epsilon_values = [
            results['mf_s1_s2'], results['mf_s2_s1'],
            results['cf_s1_s2'], results['cf_s2_s1']
        ]
        results['mean_epsilon'] = np.mean(epsilon_values)
        results['std_epsilon'] = np.std(epsilon_values)

        if results['mean_epsilon'] > 0:
            results['interaction_type'] = 'repulsion'
        elif results['mean_epsilon'] < 0:
            results['interaction_type'] = 'attraction'
        else:
            results['interaction_type'] = 'neutral'

    except Exception as e:
        print(f"Warning: Error processing pair ({s1_id}, {s2_id}): {e}", file=sys.stderr)
        # Results are already initialized to NaN, so no further action needed

    return results


# Main execution block for multiprocessing
if __name__ == "__main__":
    try:
        virus_sequences_df = pd.read_csv(VIRAL_INPUT_FILENAME)
        human_proteom_df = pd.read_csv(HUMAN_INPUT_FILENAME)
        print(f"Loaded virus_sequences_df: {virus_sequences_df.shape[0]} sequences")
        print(f"Loaded human_proteom_df: {human_proteom_df.shape[0]} sequences")

    except FileNotFoundError as e:
        print(f"Error loading files: {e}. Make sure 'data/' directory exists and files are present.", file=sys.stderr)
        sys.exit(1)

    # Pre-processing: Filter out empty or non-string sequences
    virus_sequences_df = virus_sequences_df[
        virus_sequences_df[SEQUENCE_COLUMN].astype(str).str.strip() != ''
    ]
    human_proteom_df = human_proteom_df[
        human_proteom_df[SEQUENCE_COLUMN].astype(str).str.strip() != ''
    ]

    # Convert sequence columns to string type to avoid errors
    virus_sequences_df[SEQUENCE_COLUMN] = virus_sequences_df[SEQUENCE_COLUMN].astype(str)
    human_proteom_df[SEQUENCE_COLUMN] = human_proteom_df[SEQUENCE_COLUMN].astype(str)

    print(f"After cleaning: virus_sequences_df: {virus_sequences_df.shape[0]} sequences")
    print(f"After cleaning: human_proteom_df: {human_proteom_df.shape[0]} sequences")

    virus_proteins = list(zip(virus_sequences_df[ID_COLUMN], virus_sequences_df[SEQUENCE_COLUMN]))
    human_proteins = list(zip(human_proteom_df[ID_COLUMN], human_proteom_df[SEQUENCE_COLUMN]))

    all_task_args = []
    for vp_id, vp_seq in virus_proteins:
        for hp_id, hp_seq in human_proteins:
            all_task_args.append((vp_id, vp_seq, hp_id, hp_seq))
    total_possible_combinations = len(all_task_args)

    # ATOMIC WRITE AND RESUME LOGIC 
    # Define a temporary file name, using the original output path, but with a unique .tmp suffix.
    # This ensures the temporary file is created in the same directory as the final output.
    temp_output_filename = os.path.join(os.path.dirname(OUTPUT_FILENAME),
                                        "." + os.path.basename(OUTPUT_FILENAME) + f".tmp.{os.getpid()}")

    completed_pairs = set()
    existing_df_for_rewrite: pd.DataFrame = pd.DataFrame(columns=COLUMN_NAMES)
    final_output_file_exists = os.path.exists(OUTPUT_FILENAME) # Check for the *final* output file

    if args.resume and final_output_file_exists:
        print(f"Resume mode enabled. Attempting to read existing data from: {OUTPUT_FILENAME}")
        try:
            # Read all existing data from the *final* output file.
            temp_existing_df = pd.read_parquet(OUTPUT_FILENAME, engine='pyarrow')

            # Ensure consistent dtypes using the explicit schema for parsing robustness
            for field in PA_SCHEMA:
                col_name = field.name
                col_type = field.type
                if col_name in temp_existing_df.columns:
                    if pa.types.is_string(col_type):
                        temp_existing_df[col_name] = temp_existing_df[col_name].astype(object).astype(str)
                    elif pa.types.is_floating(col_type):
                        temp_existing_df[col_name] = pd.to_numeric(temp_existing_df[col_name], errors='coerce').astype(np.float32)

            temp_existing_df = temp_existing_df.dropna(subset=['s1_id', 's2_id'])

            if not temp_existing_df.empty:
                completed_pairs = set(zip(temp_existing_df['s1_id'], temp_existing_df['s2_id']))
                existing_df_for_rewrite = temp_existing_df
                print(f"Found {len(completed_pairs)} already calculated combinations in {OUTPUT_FILENAME}.")
            else:
                print(f"Existing file {OUTPUT_FILENAME} is empty or contains no valid pairs. Starting fresh.")

        except Exception as e:
            # IMPORTANT: If the output file is corrupted, move it aside and start fresh.
            print(f"Warning: Could not read existing output file {OUTPUT_FILENAME} for resume: {e}.", file=sys.stderr)
            print(f"It might be corrupted or incomplete. Moving it to {OUTPUT_FILENAME}.bak and starting fresh.", file=sys.stderr)
            try:
                os.rename(OUTPUT_FILENAME, OUTPUT_FILENAME + ".bak")
                print(f"Moved corrupted/incomplete file to {OUTPUT_FILENAME}.bak")
            except OSError as rename_e:
                print(f"Could not move {OUTPUT_FILENAME} to .bak: {rename_e}. It might be unreadable or locked. Please check manually.", file=sys.stderr)
            completed_pairs = set() # Reset so we calculate all
            existing_df_for_rewrite = pd.DataFrame(columns=COLUMN_NAMES) # Reset
    elif args.resume and not final_output_file_exists:
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
        sys.exit(0) # Exit successfully

    parquet_writer = None
    try:
        # Initialize ParquetWriter to the temporary file
        parquet_writer = pq.ParquetWriter(temp_output_filename, PA_SCHEMA)
        print(f"Initialized ParquetWriter for temporary file: {temp_output_filename} with explicit schema.")

        # If resuming, first write back the existing, valid data to the temporary file
        if not existing_df_for_rewrite.empty:
            print(f"Writing back {len(existing_df_for_rewrite)} existing rows to {temp_output_filename}...")
            for i in tqdm(range(0, len(existing_df_for_rewrite), CHUNK_SIZE), desc="Rewriting existing data"):
                chunk_df = existing_df_for_rewrite.iloc[i : i + CHUNK_SIZE]
                table = pa.Table.from_pandas(chunk_df, schema=PA_SCHEMA, preserve_index=False)
                parquet_writer.write_table(table)
            print("Finished rewriting existing data.")

        print(f"\nStarting calculations for {total_calculations_to_perform} new combinations (out of {total_possible_combinations} total) using {NUM_PROCESSES} processes.")
        print(f"Results will be appended to {temp_output_filename}")

        results_buffer = []

        with Pool(processes=NUM_PROCESSES, initializer=init_worker) as pool:
            # Use imap_unordered to process results as they become available
            for result in tqdm(pool.imap_unordered(calculate_epsilons, tasks_to_perform), total=total_calculations_to_perform, desc="Calculating new combinations"):
                if result: # Only process if calculation was successful
                    results_buffer.append(result)

                    # If buffer reaches CHUNK_SIZE, convert to DataFrame, then PyArrow Table, and write
                    if len(results_buffer) >= CHUNK_SIZE:
                        chunk_df = pd.DataFrame(results_buffer, columns=COLUMN_NAMES)
                        table = pa.Table.from_pandas(chunk_df, schema=PA_SCHEMA, preserve_index=False)
                        parquet_writer.write_table(table)
                        results_buffer = [] # Clear the buffer

        # Write any remaining new results
        if results_buffer:
            chunk_df = pd.DataFrame(results_buffer, columns=COLUMN_NAMES)
            table = pa.Table.from_pandas(chunk_df, schema=PA_SCHEMA, preserve_index=False)
            parquet_writer.write_table(table)
            print(f"Wrote final chunk of {len(results_buffer)} new rows to {temp_output_filename}.")

        # Close the ParquetWriter to finalize the temporary file.
        # This writes the footer and makes the file valid.
        parquet_writer.close()
        print(f"\nSuccessfully finalized temporary output file: {temp_output_filename}!")

        # If everything succeeded, perform the atomic rename.
        # First, remove any existing final output file if it's there.
        if os.path.exists(OUTPUT_FILENAME):
            try:
                os.remove(OUTPUT_FILENAME)
                print(f"Removed old output file: {OUTPUT_FILENAME}")
            except OSError as e:
                print(f"Warning: Could not remove old {OUTPUT_FILENAME}: {e}. Attempting rename anyway.", file=sys.stderr)

        try:
            os.rename(temp_output_filename, OUTPUT_FILENAME)
            print(f"Successfully renamed {temp_output_filename} to {OUTPUT_FILENAME}.")
        except OSError as e:
            print(f"ERROR: Could not rename {temp_output_filename} to {OUTPUT_FILENAME}: {e}", file=sys.stderr)
            print(f"The completed results are in '{temp_output_filename}'. Please rename it manually to '{OUTPUT_FILENAME}' to finalize.", file=sys.stderr)
            sys.exit(1) # Critical failure if rename fails

    except Exception as e:
        # This block catches any error that happens *during* the writing process,
        # including if parquet_writer.close() itself fails due to an internal error.
        print(f"An error occurred during the main writing process: {e}", file=sys.stderr)
        if parquet_writer is not None:
            try:
                parquet_writer.close() # Attempt to close the writer even if an error occurred
                print(f"Attempted to close the ParquetWriter for {temp_output_filename} after an error.", file=sys.stderr)
            except Exception as close_e:
                print(f"Error while attempting to close ParquetWriter after initial error: {close_e}", file=sys.stderr)
        print(f"The temporary file '{temp_output_filename}' might be incomplete or corrupted and has not been renamed. "
              "It is left for inspection if needed.", file=sys.stderr)
        sys.exit(1) # Exit with an error code

    # Optional: Verify the file size and first few rows after completion
    if os.path.exists(OUTPUT_FILENAME):
        print(f"\nFinal file size: {os.path.getsize(OUTPUT_FILENAME) / (1024*1024):.2f} MB")
        try:
            final_df_head = pd.read_parquet(OUTPUT_FILENAME, engine='pyarrow').head(5)
            print("\nFirst 5 rows of the final output file:")
            print(final_df_head)
            print("\nData types of numerical columns (should be float32-like):")
            print(final_df_head[list(NUMERIC_DTYPES.keys())].dtypes)
        except Exception as e:
            print(f"Could not read back the file head for verification: {e}. Is the Parquet file corrupted?", file=sys.stderr)
    else:
        print(f"Output file {OUTPUT_FILENAME} was not created or renamed successfully.", file=sys.stderr)

