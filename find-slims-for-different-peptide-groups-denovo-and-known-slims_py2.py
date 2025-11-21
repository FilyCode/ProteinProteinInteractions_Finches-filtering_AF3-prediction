# Enable Python 3-style print function and division behavior in Python 2.7
from __future__ import print_function
from __future__ import division

import sys
import os
import subprocess
import tempfile
import csv
import pandas as pd
from collections import Counter
from tqdm import tqdm # Changed from tqdm.auto for better Python 2.7 compatibility
import argparse
import openpyxl # Ensure this is compatible with your Python 2.7 env (e.g., openpyxl<=2.6)
import math

parser = argparse.ArgumentParser(description="Find SLiMs for different peptide groups.")
parser.add_argument("--num_forks", type=int, default=1,
                    help="Number of CPU forks for SLiMFinder parallel execution.")
args = parser.parse_args()

NUM_FORKS_SLIMFINDER = args.num_forks

# --- SLiMSuite Configuration ---
SLIMSUITE_PATH = "/projectnb/cancergrp/Philipp/SLiMSuite" 
SLIMFINDER_SCRIPT = os.path.join(SLIMSUITE_PATH, "tools", "slimfinder.py")
SLIMPROB_SCRIPT = os.path.join(SLIMSUITE_PATH, "tools", "slimprob.py")

# Ensure the SLiMFinder script exists
if not os.path.exists(SLIMFINDER_SCRIPT):
    # Converted f-string to % formatting
    print("ERROR: SLiMFinder script not found at %s" % SLIMFINDER_SCRIPT)
    print("Please ensure SLiMSuite is cloned and SLIMSUITE_PATH is set correctly.")
    sys.exit(1)

# Ensure the SLiMSearch script exists
if not os.path.exists(SLIMSEARCH_SCRIPT):
    # Converted f-string to % formatting
    print("ERROR: SLiMSearch script not found at %s" % SLIMSEARCH_SCRIPT)
    print("Please ensure SLiMSuite is cloned and SLIMSUITE_PATH is set correctly.")
    sys.exit(1)

# This PYTHON_EXECUTABLE will be the Python 2.7 interpreter running this entire script
PYTHON_EXECUTABLE = "python" 

# --- The existing data and results directories ---
DATA_DIR = "/projectnb/cancergrp/Philipp/data/"
RESULTS_DIR = "/projectnb/cancergrp/Philipp/results/RITA_peptides"
# Python 2.7 compatible os.makedirs
try:
    os.makedirs(RESULTS_DIR)
except OSError as exc:
    if exc.errno == 17 and os.path.isdir(RESULTS_DIR): # errno 17 is EEXIST
        pass
    else: raise


# --- Load your DataFrames ---
print("Loading data...")
# Converted f-strings to % formatting
full_library_df = pd.read_csv("%sVP_library_all_sequences.csv" % DATA_DIR)
RITA_exp_df = pd.read_excel("%sRITA_and_ABT_pos_selection_screens.xlsx" % DATA_DIR, sheet_name='RITA')

# Define the path to the ELM motifs file provided by SLiMSuite
ELM_MOTIFS_FILE = os.path.join(SLIMSUITE_PATH, "data", "elm2019.motifs") 
if not os.path.exists(ELM_MOTIFS_FILE):
    # Converted f-string to % formatting
    print("ERROR: ELM motifs file not found at %s" % ELM_MOTIFS_FILE)
    print("Please ensure your SLiMSuite installation contains data/elm2019.motifs (or similar motif file).")
    sys.exit(1)


# --- SLiMSuite Input/Output Helper Functions ---

def create_fasta_from_df(df, id_col, seq_col, output_fasta_path):
    """
    Creates a FASTA file from a DataFrame, ensuring valid headers and sequences.
    """
    print("Generating FASTA for %d peptides..." % len(df))
    with open(output_fasta_path, 'w') as f:
        valid_rows = df[df[id_col].notna() & df[seq_col].notna()].copy()
        
        for index, row in valid_rows.iterrows():
            identifier = str(row[id_col]).strip()
            identifier = identifier.replace(' ', '_').replace('|', '_').replace('>', '') 
            sequence = str(row[seq_col]).strip()
            
            if identifier and sequence:
                f.write(">%s\n%s\n" % (identifier, sequence))
            else:
                pass 
    print("Generated FASTA file: %s with %d sequences." % (output_fasta_path, len(valid_rows)))


def run_slimfinder(group_name, input_fasta_path, output_base_dir, slimfinder_script, python_executable,
                   slimlen=10, maxwild=2, minocc_perc=0.1, probcut=0.05, maxseq=0, nr_peptides=0.1, num_forks=1):
    """
    Runs SLiMFinder for a given group of peptides.
    """
    # Converted f-string to % formatting
    print("\n--- Running SLiMFinder for group: %s ---" % group_name)

    slimfinder_output_dir = os.path.join(output_base_dir, "slimfinder_output_%s" % group_name.replace(' ', '_'))
    # Python 2.7 compatible os.makedirs
    try:
        os.makedirs(slimfinder_output_dir)
    except OSError as exc:
        if exc.errno == 17 and os.path.isdir(slimfinder_output_dir):
            pass
        else: raise
    
    # Calculate absolute minocc based on percentage and total peptides
    # Ensure nr_peptides is treated as an int for the min calculation, and math.ceil returns float, so int() conversion is fine.
    minocc_abs = max(5, int(min(math.ceil(minocc_perc * nr_peptides), nr_peptides)))

    # Converted f-string to % formatting
    print("  Using minocc_perc=%s for %d peptides, resulting in absolute minocc=%d" % (minocc_perc, nr_peptides, minocc_abs))

    # Converted f-string to % formatting
    results_csv_path = os.path.join(slimfinder_output_dir, "%s_slimfinder_results.csv" % group_name.replace(' ', '_'))
    
    command = [
        python_executable, slimfinder_script,
        "seqin=%s" % input_fasta_path,
        "resdir=%s/" % slimfinder_output_dir, # SLiMFinder expects trailing slash for resdir
        "resfile=%s" % results_csv_path,
        "efilter=F",      # Turn off evolutionary filtering for short peptides
        "masking=F",      # Turn off masking for short peptides
        "slimlen=%d" % slimlen,
        "maxwild=%d" % maxwild,
        "minocc=%d" % minocc_abs,
        "probcut=%s" % probcut,
        "maxseq=%d" % maxseq, # 0 means no limit on sequences; useful if one has many peptides.
        "v=1",             # Verbosity: 1 for more debug info
        "runid=%s" % group_name.replace(' ', '_'), # Add a run ID for easier tracking
        "forks=%d" % num_forks # Forks the task to use several CPUs if provided
    ]

    # Converted f-string to % formatting
    print("SLiMFinder command: %s" % ' '.join(command))

    try:
        # Replaced subprocess.run with subprocess.Popen for Python 2.7
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout_bytes, stderr_bytes = process.communicate()
        # Decode byte strings to regular strings (assuming UTF-8, common)
        stdout = stdout_bytes.decode('utf-8', errors='ignore') if stdout_bytes else ""
        stderr = stderr_bytes.decode('utf-8', errors='ignore') if stderr_bytes else ""

        if process.returncode != 0:
            # Custom error handling to get Python 3-like CalledProcessError details in Python 2.7
            print("SLiMFinder Stdout for %s (Error):\n%s" % (group_name, stdout))
            print("SLiMFinder Stderr for %s (Error):\n%s" % (group_name, stderr))
            raise subprocess.CalledProcessError(process.returncode, " ".join(command)) # Python 2.7 has less robust CalledProcessError

        # Converted f-string to % formatting
        print("SLiMFinder for %s completed successfully." % group_name)
        
        if stdout:
            # Converted f-string to % formatting
            print("SLiMFinder Stdout for %s:\n%s" % (group_name, stdout))
        if stderr:
            # Converted f-string to % formatting
            print("SLiMFinder Stderr for %s:\n%s" % (group_name, stderr))

        if os.path.exists(results_csv_path):
            slim_results_df = pd.read_csv(results_csv_path)
            # Converted f-string to % formatting
            print("Found %d motifs for %s." % (len(slim_results_df), group_name))
            return slim_results_df
        else:
            # Converted f-string to % formatting
            print("Warning: SLiMFinder results file not found at %s" % results_csv_path)
            return pd.DataFrame() # Return empty DataFrame if file not found

    except subprocess.CalledProcessError as e:
        # Converted f-string to % formatting
        print("Error running SLiMFinder for %s: %s" % (group_name, e))
        # Note: In Python 2.7, CalledProcessError might not have .stdout/.stderr attributes automatically
        # The stdout/stderr were printed above before re-raising the error
        return pd.DataFrame() # Return empty DataFrame on error
    except Exception as e:
        # Converted f-string to % formatting
        print("An unexpected error occurred while running SLiMFinder for %s: %s" % (group_name, e))
        return pd.DataFrame() # Return empty DataFrame on error

def run_slimsearch(group_name, input_fasta_path, known_motifs_file_path, motif_filter_pattern, output_base_dir, slimsearch_script, python_executable, max_seq=0):
    """
    Runs SLiMSearch for a given group of peptides against known motifs, with an optional filter.
    """
    # Converted f-string to % formatting
    print("\n--- Running SLiMSearch for group: %s ---" % group_name)

    sanitized_filter = motif_filter_pattern.replace('*', 'star').replace(',', '_') 
    slimsearch_output_dir = os.path.join(output_base_dir, "slimsearch_output_%s_%s" % (group_name.replace(' ', '_'), sanitized_filter))
    # Python 2.7 compatible os.makedirs
    try:
        os.makedirs(slimsearch_output_dir)
    except OSError as exc:
        if exc.errno == 17 and os.path.isdir(slimsearch_output_dir):
            pass
        else: raise

    # Converted f-string to % formatting
    output_basename = "%s_slimsearch_results" % group_name.replace(' ', '_')
    occ_csv_path = os.path.join(slimsearch_output_dir, "%s.csv" % output_basename)
    summary_csv_path = os.path.join(slimsearch_output_dir, "%s.summary.csv" % output_basename)

    command = [
        python_executable, slimsearch_script,
        "seqin=%s" % input_fasta_path,
        "motifs=%s" % known_motifs_file_path, 
        "goodmotif=%s" % motif_filter_pattern, 
        "resdir=%s/" % slimsearch_output_dir, # SLiMSuite expects trailing slash
        "resfile=%s" % occ_csv_path,
        "runid=%s_%s" % (group_name.replace(' ', '_'), sanitized_filter),
        "maxseq=%d" % max_seq,
        "slimchance=T",
        "v=1",               # Set verbosity to 1 for SLiMSearch
        "force=F", 
        "masking=F",         
        "efilter=F",         
    ]

    # Converted f-string to % formatting
    print("SLiMSearch command: %s" % ' '.join(command))

    try:
        # Replaced subprocess.run with subprocess.Popen for Python 2.7
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout_bytes, stderr_bytes = process.communicate()
        # Decode byte strings to regular strings
        stdout = stdout_bytes.decode('utf-8', errors='ignore') if stdout_bytes else ""
        stderr = stderr_bytes.decode('utf-8', errors='ignore') if stderr_bytes else ""

        if process.returncode != 0:
            # Custom error handling to get Python 3-like CalledProcessError details in Python 2.7
            print("SLiMSearch Stdout for %s with filter '%s' (Error):\n%s" % (group_name, motif_filter_pattern, stdout))
            print("SLiMSearch Stderr for %s with filter '%s' (Error):\n%s" % (group_name, motif_filter_pattern, stderr))
            raise subprocess.CalledProcessError(process.returncode, " ".join(command)) # Python 2.7 has less robust CalledProcessError

        # Converted f-string to % formatting
        print("SLiMSearch for %s with filter '%s' completed successfully." % (group_name, motif_filter_pattern))
        
        if stdout:
            # Converted f-string to % formatting
            print("SLiMSearch Stdout for %s:\n%s" % (group_name, stdout))
        if stderr:
            # Converted f-string to % formatting
            print("SLiMSearch Stderr for %s:\n%s" % (group_name, stderr))

        if os.path.exists(summary_csv_path):
            slim_summary_df = pd.read_csv(summary_csv_path)
            # Converted f-string to % formatting
            print("Found %d known motifs with summary results for %s." % (len(slim_summary_df), group_name))
            return slim_summary_df
        else:
            # Converted f-string to % formatting
            print("Warning: SLiMSearch summary file not found at %s" % summary_csv_path)
            return pd.DataFrame()

    except subprocess.CalledProcessError as e:
        # Converted f-string to % formatting
        print("Error running SLiMSearch for %s with filter '%s': %s" % (group_name, motif_filter_pattern, e))
        # The stdout/stderr were printed above before re-raising the error
        return pd.DataFrame()
    except Exception as e:
        # Converted f-string to % formatting
        print("An unexpected error occurred while running SLiMSearch for %s with filter '%s': %s" % (group_name, motif_filter_pattern, e))
        return pd.DataFrame()


# --- Main Script Logic to filter data and run SLiMFinder ---

print("\nFiltering Data ")
full_library_filtered = full_library_df[full_library_df['code'].isin(['VT', 'VP'])].copy()
RITA_exp_filtered = RITA_exp_df[RITA_exp_df['type'].isin(['VT', 'VP'])].copy()

# Removing leading M (from sequencing, exists in experimental peptides but not in true viral protein and would bias SLiM search)
RITA_exp_filtered['Aminoacids'] = RITA_exp_filtered['Aminoacids'].str[1:]

# Converted f-strings to % formatting
print("Number of peptides in full library (VT/VP): %d" % len(full_library_filtered))
print("Number of peptides used in RITA experiment (VT/VP): %d" % len(RITA_exp_filtered))

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
    # Uncomment if you want to include these groups
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
# Python 2.7 compatible os.makedirs
try:
    os.makedirs(slimfinder_output_base_dir)
except OSError as exc:
    if exc.errno == 17 and os.path.isdir(slimfinder_output_base_dir):
        pass
    else: raise

# tqdm.auto is Python 3. `tqdm` should be fine.
for group_name, group_df in tqdm(peptide_groups_dfs.items(), desc="Processing peptide groups for SLiMFinder"):
    if group_df.empty:
        # Converted f-string to % formatting
        print("Skipping SLiMFinder for empty group: %s" % group_name)
        slimfinder_results_summary[group_name] = pd.DataFrame()
        continue

    # Converted f-string in suffix
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix="_%s.fasta" % group_name.replace(' ', '_')) as tmp_fasta_file:
        fasta_input_path = tmp_fasta_file.name
    
    create_fasta_from_df(group_df, 'identifier', 'Aminoacids', fasta_input_path)

    group_slim_results = run_slimfinder(
        group_name=group_name,
        input_fasta_path=fasta_input_path,
        output_base_dir=slimfinder_output_base_dir,
        slimfinder_script=SLIMFINDER_SCRIPT,
        python_executable=PYTHON_EXECUTABLE, # This is now the Python 2.7 that runs the script
        slimlen=10, maxwild=2, minocc_perc=0.02, probcut=0.1, maxseq=0,
        nr_peptides = len(group_df), num_forks=NUM_FORKS_SLIMFINDER
    )
    slimfinder_results_summary[group_name] = group_slim_results

    if os.path.exists(fasta_input_path):
        os.remove(fasta_input_path)

print("\n--- SLiMFinder Analysis Complete ---")
print("Summary of significant motifs found per group:")
for group, results_df in slimfinder_results_summary.items():
    num_motifs = len(results_df)
    # Converted f-string to % formatting
    print("- %s: %d significant motifs" % (group, num_motifs))
    if num_motifs > 0:
        if 'Sig' in results_df.columns:
            print("  Top 5 Motifs (by Significance):")
            results_df['Sig'] = pd.to_numeric(results_df['Sig'], errors='coerce')
            print(results_df[['Pattern', 'Sig', 'Occ', 'Support']].dropna(subset=['Sig']).sort_values(by='Sig').head())
        else:
            print("  Top 5 Motifs:")
            print(results_df[['Pattern', 'Occ', 'Support']].head())
    print("-" * 30)

all_slimfinder_results_combined = pd.concat(
    [df.assign(Group=name) for name, df in slimfinder_results_summary.items() if not df.empty]
)
if not all_slimfinder_results_combined.empty:
    combined_slim_results_path = os.path.join(slimfinder_output_base_dir, "all_groups_slimfinder_combined_results.csv")
    all_slimfinder_results_combined.to_csv(combined_slim_results_path, index=False)
    # Converted f-string to % formatting
    print("\nAll SLiMFinder results combined and saved to: %s" % combined_slim_results_path)


# --- Run SLiMSearch for each peptide group ---
print("\n--- Starting SLiMSearch Analysis for Peptide Groups ---")
slimsearch_results_summary = {}
slimsearch_output_base_dir = os.path.join(RESULTS_DIR, "slimsearch_analyses")
# Python 2.7 compatible os.makedirs
try:
    os.makedirs(slimsearch_output_base_dir)
except OSError as exc:
    if exc.errno == 17 and os.path.isdir(slimsearch_output_base_dir):
        pass
    else: raise

MOTIF_FILTER_TO_USE = "ELM_*" # Adjust as needed

for group_name, group_df in tqdm(peptide_groups_dfs.items(), desc="Processing peptide groups for SLiMSearch"):
    if group_df.empty:
        # Converted f-string to % formatting
        print("Skipping SLiMSearch for empty group: %s" % group_name)
        slimsearch_results_summary[group_name] = pd.DataFrame()
        continue

    # Converted f-string in suffix
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix="_%s_slimsearch.fasta" % group_name.replace(' ', '_')) as tmp_fasta_file:
        fasta_input_path = tmp_fasta_file.name
    
    create_fasta_from_df(group_df, 'identifier', 'Aminoacids', fasta_input_path)

    group_slim_results = run_slimsearch(
        group_name=group_name,
        input_fasta_path=fasta_input_path,
        known_motifs_file_path=ELM_MOTIFS_FILE, 
        motif_filter_pattern=MOTIF_FILTER_TO_USE, 
        output_base_dir=slimsearch_output_base_dir,
        slimsearch_script=SLIMSEARCH_SCRIPT,
        python_executable=PYTHON_EXECUTABLE, # This is now the Python 2.7 that runs the script
        max_seq=0
    )
    slimsearch_results_summary[group_name] = group_slim_results

    if os.path.exists(fasta_input_path):
        os.remove(fasta_input_path)

print("\n--- SLiMSearch Analysis Complete ---")
print("Summary of known motifs found per group:")
for group, results_df in slimsearch_results_summary.items():
    num_motifs = len(results_df)
    # Converted f-string to % formatting
    print("- %s: %d known motifs (summary lines) with filter '%s'" % (group, num_motifs, MOTIF_FILTER_TO_USE))
    if num_motifs > 0:
        if 'p_UPC' in results_df.columns: 
            print("  Top 5 Motifs (by p_UPC - lower is more significant):")
            results_df['p_UPC'] = pd.to_numeric(results_df['p_UPC'], errors='coerce')
            print(results_df[['Pattern', 'N_UPC', 'E_UPC', 'p_UPC']].dropna(subset=['p_UPC']).sort_values(by='p_UPC').head())
        else:
            print("  Top 5 Motifs (No p_UPC column found):")
            print(results_df.head())
    print("-" * 30)

non_empty_slimsearch_dfs = [df.assign(Group=name, MotifFilter=MOTIF_FILTER_TO_USE) 
                            for name, df in slimsearch_results_summary.items() if not df.empty]

if non_empty_slimsearch_dfs: 
    all_slimsearch_results_combined = pd.concat(non_empty_slimsearch_dfs)
    sanitized_filter_for_filename = MOTIF_FILTER_TO_USE.replace('*', 'star').replace(',', '_')
    # Converted f-string to % formatting
    combined_slimsearch_results_path = os.path.join(slimsearch_output_base_dir, "all_groups_slimsearch_combined_summary_results_%s.csv" % sanitized_filter_for_filename)
    all_slimsearch_results_combined.to_csv(combined_slimsearch_results_path, index=False)
    # Converted f-string to % formatting
    print("\nAll SLiMSearch summary results combined and saved to: %s" % combined_slimsearch_results_path)
else:
    # Converted f-string to % formatting
    print("\nNo SLiMSearch results found for any group with filter '%s'. No combined file generated." % MOTIF_FILTER_TO_USE)

print("\nScript execution finished.")