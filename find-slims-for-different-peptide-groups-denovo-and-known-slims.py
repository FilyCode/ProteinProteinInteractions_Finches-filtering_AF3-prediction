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
SLIMSUITE_PATH = "/projectnb/cancergrp/Philipp/SLiMSuite" 
SLIMFINDER_SCRIPT = os.path.join(SLIMSUITE_PATH, "tools", "slimfinder.py")
SLIMSEARCH_SCRIPT = os.path.join(SLIMSUITE_PATH, "tools", "slimsearch.py")

# Ensure the SLiMFinder script exists
if not os.path.exists(SLIMFINDER_SCRIPT):
    print(f"ERROR: SLiMFinder script not found at {SLIMFINDER_SCRIPT}")
    print("Please ensure SLiMSuite is cloned and SLIMSUITE_PATH is set correctly.")
    sys.exit(1)

# Ensure the SLiMSearch script exists
if not os.path.exists(SLIMSEARCH_SCRIPT):
    print(f"ERROR: SLiMSearch script not found at {SLIMSEARCH_SCRIPT}")
    print("Please ensure SLiMSuite is cloned and SLIMSUITE_PATH is set correctly.")
    sys.exit(1)

# Ensure this points to the Python executable within your 'slimsuite_env'
PYTHON_EXECUTABLE = "python" 

# --- The existing data and results directories ---
DATA_DIR = "/projectnb/cancergrp/Philipp/data/"
RESULTS_DIR = "/projectnb/cancergrp/Philipp/results/RITA_peptides"
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Load your DataFrames ---
print("Loading data...")
full_library_df = pd.read_csv(f"{DATA_DIR}VP_library_all_sequences.csv")
RITA_exp_df = pd.read_excel(f"{DATA_DIR}RITA_and_ABT_pos_selection_screens.xlsx", sheet_name='RITA')

# Define the path to the ELM motifs file provided by SLiMSuite
ELM_MOTIFS_FILE = os.path.join(SLIMSUITE_PATH, "data", "elm2019.motifs") 
if not os.path.exists(ELM_MOTIFS_FILE):
    print(f"ERROR: ELM motifs file not found at {ELM_MOTIFS_FILE}")
    print("Please ensure your SLiMSuite installation contains data/elm2019.motifs (or similar motif file).")
    sys.exit(1)


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
    # Ensure minocc is an integer and at least 5 (a motif must occur at least twice to be a motif) and max the total number of peptides
    minocc_abs = max(5, min(int(math.ceil(minocc_perc * nr_peptides)), nr_peptides))

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
        f"v=1",             # Verbosity: 0 for default, -1 for silent, 1 for more debug info
        f"runid={group_name.replace(' ', '_')}", # Add a run ID for easier tracking
        f"forks={num_forks}" # Forks the task to use several CPUs if provided
    ]

    print(f"SLiMFinder command: {' '.join(command)}")

    try:
        # Execute the command, capturing stdout/stderr. check=True raises error for non-zero exit codes.
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

def run_slimsearch(group_name, input_fasta_path, known_motifs_file_path, motif_filter_pattern, output_base_dir, slimsearch_script, python_executable, max_seq=0):
    """
    Runs SLiMSearch for a given group of peptides against known motifs, with an optional filter.
    Args:
        group_name (str): Name of the peptide group (for output organization).
        input_fasta_path (str): Path to the input FASTA file for SLiMSearch.
        known_motifs_file_path (str): Path to the file containing known SLiMs (e.g., elm2019.motifs).
        motif_filter_pattern (str): Pattern for 'goodmotif' to filter motifs (e.g., "ELM_LIG_*").
        output_base_dir (str): Base directory for SLiMSearch results.
        slimsearch_script (str): Path to the slimsearch.py script.
        python_executable (str): Python executable to run slimsearch.py with.
        max_seq (int): Maximum number of sequences to process (0 for no limit).
    Returns:
        pd.DataFrame: A DataFrame of SLiMSearch summary results, or empty if none.
    """
    print(f"\n--- Running SLiMSearch for group: {group_name} ---")

    # The output directory name can now also include the motif filter for clarity
    sanitized_filter = motif_filter_pattern.replace('*', 'star').replace(',', '_') 
    slimsearch_output_dir = os.path.join(output_base_dir, f"slimsearch_output_{group_name.replace(' ', '_')}_{sanitized_filter}")
    os.makedirs(slimsearch_output_dir, exist_ok=True)

    output_basename = f"{group_name.replace(' ', '_')}_slimsearch_results"
    occ_csv_path = os.path.join(slimsearch_output_dir, f"{output_basename}.csv")
    summary_csv_path = os.path.join(slimsearch_output_dir, f"{output_basename}.summary.csv")

    command = [
        python_executable, slimsearch_script,
        f"seqin={input_fasta_path}",
        f"motifs={known_motifs_file_path}", # Use the predefined ELM file
        f"goodmotif={motif_filter_pattern}", # Apply the filter here
        f"resdir={slimsearch_output_dir}/",
        f"resfile={occ_csv_path}",
        f"runid={group_name.replace(' ', '_')}_{sanitized_filter}",
        f"maxseq={max_seq}",
        "slimchance=T",
        "v=0",
        "force=F", # Set to T if you want to force re-computation ignoring existing pickles
        "masking=F",         # You might want to enable specific masking, but keeping consistent with SLiMFinder run for now
        "efilter=F",         # You might want to enable evolutionary filtering, but keeping consistent with SLiMFinder run for now
    ]

    print(f"SLiMSearch command: {' '.join(command)}")

    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"SLiMSearch for {group_name} with filter '{motif_filter_pattern}' completed successfully.")

        if os.path.exists(summary_csv_path):
            slim_summary_df = pd.read_csv(summary_csv_path)
            print(f"Found {len(slim_summary_df)} known motifs with summary results for {group_name}.")
            return slim_summary_df
        else:
            print(f"Warning: SLiMSearch summary file not found at {summary_csv_path}")
            return pd.DataFrame()

    except subprocess.CalledProcessError as e:
        print(f"Error running SLiMSearch for {group_name} with filter '{motif_filter_pattern}': {e}")
        print("SLiMSearch Stdout:\n", e.stdout)
        print("SLiMSearch Stderr:\n", e.stderr)
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred while running SLiMSearch for {group_name} with filter '{motif_filter_pattern}': {e}")
        return pd.DataFrame()


# --- Main Script Logic to filter data and run SLiMFinder ---

print("\nFiltering Data ")
full_library_filtered = full_library_df[full_library_df['code'].isin(['VT', 'VP'])].copy()
RITA_exp_filtered = RITA_exp_df[RITA_exp_df['type'].isin(['VT', 'VP'])].copy()

# Removing leading M (from sequencing, exists in experimental peptides but not in true viral protein and would bias SLiM search)
RITA_exp_filtered['Aminoacids'] = RITA_exp_filtered['Aminoacids'].str[1:]

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
    #'Full_Library_VT_VP': full_library_filtered, # Uncomment if you want to include these groups
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
        minocc_perc=0.02, # Motif must occur in at least x% of peptides
        probcut=0.1,   # p-value cutoff for significance (higher allows more motifs)
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
        if 'Sig' in results_df.columns:
            print("  Top 5 Motifs (by Significance):")
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


# --- Run SLiMSearch for each peptide group ---
print("\n--- Starting SLiMSearch Analysis for Peptide Groups ---")
slimsearch_results_summary = {}
slimsearch_output_base_dir = os.path.join(RESULTS_DIR, "slimsearch_analyses")
os.makedirs(slimsearch_output_base_dir, exist_ok=True)

# Define the motif filter pattern you want to use.
# You can change this to "ELM_LIG_*", "ELM_DEG_*", "ELM_DOC_*", "ELM_LIG_14-3-3_*", or even "ELM_*" for all ELM motifs.
# For now let's look for all ELM motifs.
MOTIF_FILTER_TO_USE = "ELM_*"

for group_name, group_df in tqdm(peptide_groups_dfs.items(), desc="Processing peptide groups for SLiMSearch"):
    if group_df.empty:
        print(f"Skipping SLiMSearch for empty group: {group_name}")
        slimsearch_results_summary[group_name] = pd.DataFrame()
        continue

    # Create a temporary FASTA file for SLiMSearch input
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=f"_{group_name.replace(' ', '_')}_slimsearch.fasta") as tmp_fasta_file:
        fasta_input_path = tmp_fasta_file.name
    
    create_fasta_from_df(group_df, 'identifier', 'Aminoacids', fasta_input_path)

    group_slim_results = run_slimsearch(
        group_name=group_name,
        input_fasta_path=fasta_input_path,
        known_motifs_file_path=ELM_MOTIFS_FILE, # Point to the SLiMSuite ELM file
        motif_filter_pattern=MOTIF_FILTER_TO_USE, # Apply the filter
        output_base_dir=slimsearch_output_base_dir,
        slimsearch_script=SLIMSEARCH_SCRIPT,
        python_executable=PYTHON_EXECUTABLE,
        max_seq=0
    )
    slimsearch_results_summary[group_name] = group_slim_results

    # Clean up the temporary FASTA file
    if os.path.exists(fasta_input_path):
        os.remove(fasta_input_path)

print("\n--- SLiMSearch Analysis Complete ---")
print("Summary of known motifs found per group:")
for group, results_df in slimsearch_results_summary.items():
    num_motifs = len(results_df)
    print(f"- {group}: {num_motifs} known motifs (summary lines) with filter '{MOTIF_FILTER_TO_USE}'")
    if num_motifs > 0:
        if 'p_UPC' in results_df.columns: 
            print("  Top 5 Motifs (by p_UPC - lower is more significant):")
            results_df['p_UPC'] = pd.to_numeric(results_df['p_UPC'], errors='coerce')
            print(results_df[['Pattern', 'N_UPC', 'E_UPC', 'p_UPC']].dropna(subset=['p_UPC']).sort_values(by='p_UPC').head())
        else:
            print("  Top 5 Motifs (No p_UPC column found):")
            print(results_df.head())
    print("-" * 30)

# Optionally, combine all SLiMSearch results into a single DataFrame
# This block now correctly handles the case where slimsearch_results_summary might be empty
# by only attempting concatenation if there are actual DataFrames to concatenate.
non_empty_slimsearch_dfs = [df.assign(Group=name, MotifFilter=MOTIF_FILTER_TO_USE) 
                            for name, df in slimsearch_results_summary.items() if not df.empty]

if non_empty_slimsearch_dfs: # Check if the list is not empty
    all_slimsearch_results_combined = pd.concat(non_empty_slimsearch_dfs)
    # Adjust filename to include filter for clarity
    sanitized_filter_for_filename = MOTIF_FILTER_TO_USE.replace('*', 'star').replace(',', '_')
    combined_slimsearch_results_path = os.path.join(slimsearch_output_base_dir, f"all_groups_slimsearch_combined_summary_results_{sanitized_filter_for_filename}.csv")
    all_slimsearch_results_combined.to_csv(combined_slimsearch_results_path, index=False)
    print(f"\nAll SLiMSearch summary results combined and saved to: {combined_slimsearch_results_path}")
else:
    print(f"\nNo SLiMSearch results found for any group with filter '{MOTIF_FILTER_TO_USE}'. No combined file generated.")

print("\nScript execution finished.")