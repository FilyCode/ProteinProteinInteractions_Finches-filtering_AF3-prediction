import sys
import os
import subprocess
import tempfile
import csv
import pandas as pd
from collections import Counter
from tqdm.auto import tqdm
import argparse
import openpyxl
import math

parser = argparse.ArgumentParser(description="Find SLiMs for different peptide groups.")
parser.add_argument("--num_forks", type=int, default=1,
                    help="Number of CPU forks for SLiMFinder parallel execution.")
args = parser.parse_args()

NUM_FORKS_SLIMFINDER = args.num_forks

# --- SLiMSuite Configuration ---
# IMPORTANT: Adjust this path to where you cloned SLiMSuite
SLIMSUITE_PATH = "/projectnb/cancergrp/Philipp/SLiMSuite" 
SLIMFINDER_SCRIPT = os.path.join(SLIMSUITE_PATH, "tools", "slimfinder.py")

# Ensure the SLiMFinder script exists
if not os.path.exists(SLIMFINDER_SCRIPT):
    print(f"ERROR: SLiMFinder script not found at {SLIMFINDER_SCRIPT}")
    print("Please ensure SLiMSuite is cloned and SLIMSUITE_PATH is set correctly.")
    sys.exit(1)

# Ensure this points to the Python executable within your 'slimsuite_env'
# When you activate your conda environment and run the script, 'python' usually points to it.
PYTHON_EXECUTABLE = "python" 

# --- The existing data and results directories ---
DATA_DIR = "/projectnb/cancergrp/Philipp/data/"
RESULTS_DIR = "/projectnb/cancergrp/Philipp/results/RITA_peptides"
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Load your DataFrames ---
print("Loading data...")
full_library_df = pd.read_csv(f"{DATA_DIR}VP_library_all_sequences.csv")
RITA_exp_df = pd.read_excel(f"{DATA_DIR}RITA_and_ABT_pos_selection_screens.xlsx", sheet_name='RITA')



# --- SLiMSuite Input/Output Helper Functions ---

def create_fasta_from_df(df, id_col, seq_col, output_fasta_path):
    """
    Creates a FASTA file from a DataFrame, ensuring valid headers and sequences.
    Args:
        df (pd.DataFrame): Input DataFrame.
        id_col (str): Name of the column containing identifiers.
        seq_col (str): Name of the column containing amino acid sequences.
        output_fasta_path (str): Path for the output FASTA file.
    """
    print(f"Generating FASTA for {len(df)} peptides...")
    with open(output_fasta_path, 'w') as f:
        # Filter out rows with missing or non-string identifiers/sequences
        valid_rows = df[df[id_col].notna() & df[seq_col].notna()].copy()
        
        for index, row in valid_rows.iterrows():
            identifier = str(row[id_col]).strip()
            # Replace spaces in identifier with underscores for FASTA header compatibility
            identifier = identifier.replace(' ', '_').replace('|', '_').replace('>', '') 
            sequence = str(row[seq_col]).strip()
            
            if identifier and sequence: # Only write non-empty identifiers and sequences
                f.write(f">{identifier}\n{sequence}\n")
            else:
                # print(f"Warning: Skipping row {index} due to invalid identifier or sequence: ID='{identifier}', Seq='{sequence[:20]}...'")
                pass # Suppress warning for every skipped row for cleaner output
    print(f"Generated FASTA file: {output_fasta_path} with {len(valid_rows)} sequences.")

def run_slimfinder(group_name, input_fasta_path, output_base_dir, slimfinder_script, python_executable,
                   slimlen=10, maxwild=2, minocc_perc=0.1, probcut=0.05, maxseq=0, nr_peptides = 0.1, num_forks=1):
    """
    Runs SLiMFinder for a given group of peptides.
    Args:
        group_name (str): Name of the peptide group (for output organization).
        input_fasta_path (str): Path to the input FASTA file for SLiMFinder.
        output_base_dir (str): Base directory for SLiMFinder results.
        slimfinder_script (str): Path to the slimfinder.py script.
        python_executable (str): Python executable to run slimfinder.py with (e.g., 'python').
        slimlen (int): Maximum SLiM length.
        maxwild (int): Maximum wildcards between defined positions.
        minocc_perc (int): Minimum percentage of occurrences for a motif.
        probcut (float): Probability cut-off for returned motifs.
        maxseq (int): Maximum number of sequences to process (0 for no limit).
        nr_peptides (int): Total number of peptides in this group.
    Returns:
        pd.DataFrame: A DataFrame of significant motifs found, or empty if none.
    """
    print(f"\n--- Running SLiMFinder for group: {group_name} ---")

    # Create a dedicated output directory for this SLiMFinder run
    slimfinder_output_dir = os.path.join(output_base_dir, f"slimfinder_output_{group_name.replace(' ', '_')}")
    os.makedirs(slimfinder_output_dir, exist_ok=True)
    
    # Calculate absolute minocc based on percentage and total peptides
    # Ensure minocc is an integer and at least 2 (a motif must occur at least twice to be a motif) and max the total number of peptides
    minocc_abs = max(2, min(int(math.ceil(minocc_perc * nr_peptides)), nr_peptides))

    print(f"  Using minocc_perc={minocc_perc} for {nr_peptides} peptides, resulting in absolute minocc={minocc_abs}")

    # Define the main results file path
    results_csv_path = os.path.join(slimfinder_output_dir, f"{group_name.replace(' ', '_')}_slimfinder_results.csv")
    
    # SLiMFinder command arguments
    command = [
        python_executable, slimfinder_script,
        f"seqin={input_fasta_path}",
        f"resdir={slimfinder_output_dir}/", # SLiMFinder expects trailing slash for resdir
        f"resfile={results_csv_path}",
        f"efilter=F",      # Turn off evolutionary filtering for short peptides
        f"masking=F",      # Turn off masking for short peptides
        f"slimlen={slimlen}",
        f"maxwild={maxwild}",
        f"minocc={minocc_abs}",
        f"probcut={probcut}",
        f"maxseq={maxseq}", # 0 means no limit on sequences; useful if one has many peptides.
        f"v=0",             # Verbosity: 0 for default, -1 for silent, 1 for more debug info
        f"runid={group_name.replace(' ', '_')}", # Add a run ID for easier tracking
        f"forks={num_forks}" # Forks the task to use several CPUs if provided
    ]

    print(f"SLiMFinder command: {' '.join(command)}")

    try:
        # Execute the command, capturing stdout/stderr. check=True raises error for non-zero exit codes.
        # Assuming the environment (conda and modules) is activated *before* running Python.
        process = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"SLiMFinder for {group_name} completed successfully.")
        # Uncomment these lines for detailed debug output from SLiMFinder
        # print("SLiMFinder Stdout:\n", process.stdout) 
        # print("SLiMFinder Stderr:\n", process.stderr) 

        # Read the main results CSV
        if os.path.exists(results_csv_path):
            slim_results_df = pd.read_csv(results_csv_path)
            print(f"Found {len(slim_results_df)} motifs for {group_name}.")
            return slim_results_df
        else:
            print(f"Warning: SLiMFinder results file not found at {results_csv_path}")
            return pd.DataFrame() # Return empty DataFrame if file not found

    except subprocess.CalledProcessError as e:
        print(f"Error running SLiMFinder for {group_name}: {e}")
        print("SLiMFinder Stdout:\n", e.stdout)
        print("SLiMFinder Stderr:\n", e.stderr)
        return pd.DataFrame() # Return empty DataFrame on error
    except Exception as e:
        print(f"An unexpected error occurred while running SLiMFinder for {group_name}: {e}")
        return pd.DataFrame() # Return empty DataFrame on error


# --- Main Script Logic to filter data and run SLiMFinder ---

print("\nFiltering Data ")
full_library_filtered = full_library_df[full_library_df['code'].isin(['VT', 'VP'])].copy()
RITA_exp_filtered = RITA_exp_df[RITA_exp_df['type'].isin(['VT', 'VP'])].copy()

print(f"Number of peptides in full library (VT/VP): {len(full_library_filtered)}")
print(f"Number of peptides used in RITA experiment (VT/VP): {len(RITA_exp_filtered)}")

# Ensure 'identifier' is consistent for RITA_exp_filtered
RITA_exp_filtered['identifier'] = RITA_exp_filtered['tileID']

# Peptides Used vs. Not Used in Experiment (from the VT/VP filtered library)
used_sequences_set = set(RITA_exp_filtered['Aminoacids'].unique())
not_used_peptides_df = full_library_filtered[~full_library_filtered['Aminoacids'].isin(used_sequences_set)].copy()

# Experiment Significant vs. Non-Significant (from VT/VP used in experiment)
RITA_sig = RITA_exp_filtered[RITA_exp_filtered['sig'] == 'Yes'].copy()
RITA_non_sig = RITA_exp_filtered[RITA_exp_filtered['sig'] == 'No'].copy()

# Experiment Upregulated vs. Downregulated Significant (from VT/VP used and significant)
RITA_sig['log2FoldChange'] = pd.to_numeric(RITA_sig['log2FoldChange'], errors='coerce')
RITA_up = RITA_sig[RITA_sig['log2FoldChange'] > 0].copy()
RITA_down = RITA_sig[RITA_sig['log2FoldChange'] < 0].copy()

# Store DataFrames in a dictionary for easy iteration and FASTA creation
peptide_groups_dfs = {
    #'Full_Library_VT_VP': full_library_filtered,
    #'Experiment_Used_VT_VP': RITA_exp_filtered,
    #'Experiment_Not_Used_VT_VP': not_used_peptides_df,
    'Experiment_Significant_VT_VP': RITA_sig,
    'Experiment_NonSignificant_VT_VP': RITA_non_sig,
    'Experiment_Upregulated_VT_VP': RITA_up,
    'Experiment_Downregulated_VT_VP': RITA_down
}


# --- Run SLiMFinder for each peptide group ---
print("\n--- Starting SLiMFinder Analysis for Peptide Groups ---")
slimfinder_results_summary = {}
slimfinder_output_base_dir = os.path.join(RESULTS_DIR, "slimfinder_analyses")
os.makedirs(slimfinder_output_base_dir, exist_ok=True)

for group_name, group_df in tqdm(peptide_groups_dfs.items(), desc="Processing peptide groups for SLiMFinder"):
    if group_df.empty:
        print(f"Skipping SLiMFinder for empty group: {group_name}")
        slimfinder_results_summary[group_name] = pd.DataFrame()
        continue

    # Create a temporary FASTA file for SLiMFinder input
    # Using tempfile.NamedTemporaryFile to manage temporary files better
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=f"_{group_name.replace(' ', '_')}.fasta") as tmp_fasta_file:
        fasta_input_path = tmp_fasta_file.name
    
    create_fasta_from_df(group_df, 'identifier', 'Aminoacids', fasta_input_path)

    # Run SLiMFinder with chosen parameters
    group_slim_results = run_slimfinder(
        group_name=group_name,
        input_fasta_path=fasta_input_path,
        output_base_dir=slimfinder_output_base_dir,
        slimfinder_script=SLIMFINDER_SCRIPT,
        python_executable=PYTHON_EXECUTABLE,
        slimlen=10,     # Max SLiM length
        maxwild=2,      # Max wildcards (e.g., A.B.C)
        minocc_perc=0.1, # Motif must occur in at least 3 unique peptides
        probcut=0.05,   # p-value cutoff for significance (higher allows more motifs)
        maxseq=0,       # 0 means no limit on input sequences (peptides)
        nr_peptides = len(group_df), # Number of peptides in group
        num_forks=NUM_FORKS_SLIMFINDER # Number of forks to create (number of CPUs available)
    )
    slimfinder_results_summary[group_name] = group_slim_results

    # Clean up the temporary FASTA file
    if os.path.exists(fasta_input_path):
        os.remove(fasta_input_path)

print("\n--- SLiMFinder Analysis Complete ---")
print("Summary of significant motifs found per group:")
for group, results_df in slimfinder_results_summary.items():
    num_motifs = len(results_df)
    print(f"- {group}: {num_motifs} significant motifs")
    if num_motifs > 0:
        # Display top 5 motifs (by significance if available, otherwise just first 5)
        if 'Sig' in results_df.columns:
            print("  Top 5 Motifs (by Significance):")
            # Ensure 'Sig' column is numeric, coercing errors to NaN and dropping NaNs before sorting
            results_df['Sig'] = pd.to_numeric(results_df['Sig'], errors='coerce')
            print(results_df[['Pattern', 'Sig', 'Occ', 'Support']].dropna(subset=['Sig']).sort_values(by='Sig').head())
        else:
            print("  Top 5 Motifs:")
            print(results_df[['Pattern', 'Occ', 'Support']].head())
    print("-" * 30)

# Optionally, combine all SLiMFinder results into a single DataFrame
all_slimfinder_results_combined = pd.concat(
    [df.assign(Group=name) for name, df in slimfinder_results_summary.items() if not df.empty]
)
if not all_slimfinder_results_combined.empty:
    combined_slim_results_path = os.path.join(slimfinder_output_base_dir, "all_groups_slimfinder_combined_results.csv")
    all_slimfinder_results_combined.to_csv(combined_slim_results_path, index=False)
    print(f"\nAll SLiMFinder results combined and saved to: {combined_slim_results_path}")

print("\nScript execution finished.")