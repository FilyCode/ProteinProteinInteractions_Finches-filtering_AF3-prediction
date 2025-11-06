import sys
import os
import subprocess # Needed to run external commands
import tempfile   # Needed to create temporary files
import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import metapredict as mpp # For disorder prediction
from tqdm.auto import tqdm
import multiprocessing  # For parallelization of metapredict calls
import math

# Define the absolute path to the cloned s4pred directory
s4pred_path = "/projectnb/cancergrp/Philipp/.conda/pkgs/s4pred"
run_model_script = os.path.join(s4pred_path, "run_model.py")

# CONSTANTS FOR STANDALONE NETSURFP-3 
NETSURFP3_STANDALONE_PATH = "/projectnb/cancergrp/Philipp/NetSurfP-3.0_standalone" 

# The name of the conda environment for the standalone NetSurfP-3 (as created from its environment.yml)
NETSURFP3_STANDALONE_ENV_NAME = "nsp3" 

# Full paths to the nsp3.py script and its model file
NSP3_SCRIPT_PATH = os.path.join(NETSURFP3_STANDALONE_PATH, "nsp3.py")
NSP3_MODEL_PATH = os.path.join(NETSURFP3_STANDALONE_PATH, "models", "nsp3.pth")

# Batching for parallel NetSurfP-3 calls
NETSURFP3_BATCH_SIZE = 50 # Number of proteins to process in each parallel NetSurfP-3.0 call

# Netsurfp3 parameter
RSA_BURIED_THRESHOLD = 0.25 # Relative Solvent Accessibility threshold: <=0.25 is buried, >0.25 is exposed


DATA_DIR = "/projectnb/cancergrp/Philipp/data/"
RESULTS_DIR = "/projectnb/cancergrp/Philipp/results/RITA_peptides"
os.makedirs(RESULTS_DIR, exist_ok=True)

full_library_df = pd.read_csv(f"{DATA_DIR}VP_library_all_sequences.csv")
RITA_exp_df = pd.read_excel(f"{DATA_DIR}RITA_and_ABT_pos_selection_screens.xlsx", sheet_name='RITA')

# Load the full proteins DataFrame
full_proteins_df = pd.read_csv(f"{DATA_DIR}full_library_virus_proteins.csv")
full_proteins_df['NCBI_id'] = full_proteins_df['NCBI_id'].str.split('|').str[0]
# Ensure the protein sequence column is named 'Sequence' for consistency
if 'Protein Sequence' in full_proteins_df.columns:
    full_proteins_df.rename(columns={'Protein Sequence': 'Sequence'}, inplace=True)
elif 'sequence' in full_proteins_df.columns: # Check for lowercase 'sequence' too
    full_proteins_df.rename(columns={'sequence': 'Sequence'}, inplace=True)


# Define the 20 standard amino acids for consistent ordering in plots
AMINO_ACIDS = sorted(list('ACDEFGHIKLMNPQRSTVWY'))

# Helper Function to Calculate Amino Acid Composition 
def get_amino_acid_composition(sequences):
    """
    Calculates the amino acid composition (percentage) for a list of peptide sequences.
    Handles empty sequences or non-string entries gracefully.
    """
    # CORRECTED: Use .empty to check if the pandas Series is empty
    if sequences.empty:
        return pd.Series({aa: 0.0 for aa in AMINO_ACIDS}, name="Composition")

    total_aa_counts = Counter()
    total_length = 0
    # Filter out non-strings or empty strings before processing
    valid_sequences = [s for s in sequences if isinstance(s, str) and s]

    # If after filtering, there are no valid sequences, return zeros
    if not valid_sequences:
        return pd.Series({aa: 0.0 for aa in AMINO_ACIDS}, name="Composition")

    for seq in valid_sequences:
        total_aa_counts.update(seq)
        total_length += len(seq)

    if total_length == 0: # This handles cases where valid_sequences might contain only empty strings
        return pd.Series({aa: 0.0 for aa in AMINO_ACIDS}, name="Composition")

    composition = {aa: (total_aa_counts.get(aa, 0) / total_length) * 100 for aa in AMINO_ACIDS}
    return pd.Series(composition, name="Composition")

# Helper Function to Plot Amino Acid Composition 
def plot_composition(composition_series_dict, title, filename_prefix, results_dir):
    """
    Plots amino acid composition for one or more groups using grouped bar plots.
    composition_series_dict: dict of {group_name: pandas.Series of composition}
    """
    if not composition_series_dict:
        print(f"Skipping plot '{title}': No data provided.")
        return

    # Convert dictionary of Series to a DataFrame for easier plotting
    plot_df_data = []
    for group_name, series in composition_series_dict.items():
        if series is not None and not series.empty: # Ensure series is not None or empty
            temp_df = series.reset_index()
            temp_df.columns = ['Amino Acid', 'Percentage']
            temp_df['Group'] = group_name
            plot_df_data.append(temp_df)
        else:
            print(f"Warning: No valid composition data for group '{group_name}' in '{title}'.")

    if not plot_df_data:
        print(f"Skipping plot '{title}': No valid dataframes to concatenate.")
        return

    plot_df = pd.concat(plot_df_data)

    plt.figure(figsize=(14, 7))
    sns.barplot(data=plot_df, x='Amino Acid', y='Percentage', hue='Group', palette='viridis', ci=None) # ci=None for no confidence intervals as it's aggregated data
    plt.title(f'Amino Acid Composition: {title}', fontsize=16)
    plt.xlabel('Amino Acid', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f"{filename_prefix}_amino_acid_composition.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved plot: {plot_path}")

# Filter DataFrames for VT or VP peptides 
print("\nFiltering Data ")
full_library_filtered = full_library_df[full_library_df['code'].isin(['VT', 'VP'])].copy()
RITA_exp_filtered = RITA_exp_df[RITA_exp_df['type'].isin(['VT', 'VP'])].copy()

print(f"Number of peptides in full library (VT/VP): {len(full_library_filtered)}")
print(f"Number of peptides used in RITA experiment (VT/VP): {len(RITA_exp_filtered)}")

print("\nGenerating comprehensive peptide amino acid composition and metadata table...")

# Start with the basic peptide info (identifier and sequence) from the filtered full library
comprehensive_peptide_table_aa = full_library_filtered[['identifier', 'Aminoacids']].copy()

# Calculate amino acid percentages for each individual peptide
aa_composition_per_peptide_df = comprehensive_peptide_table_aa['Aminoacids'].apply(lambda seq: pd.Series({
    aa: (Counter(seq).get(aa, 0) / len(seq)) * 100 if len(seq) > 0 else 0.0
    for aa in AMINO_ACIDS
}))

# Concatenate the calculated percentages with the initial identifier and Aminoacids columns
comprehensive_peptide_table_aa = pd.concat([comprehensive_peptide_table_aa, aa_composition_per_peptide_df], axis=1)

# Prepare RITA experiment data for merging
RITA_exp_filtered['identifier'] = RITA_exp_filtered['tileID']
rita_metadata_for_merge_aa = RITA_exp_filtered[['identifier', 'sig', 'log2FoldChange', 'padj']].copy()

# Ensure log2FoldChange and padj are numeric before potential calculations or display
rita_metadata_for_merge_aa['log2FoldChange'] = pd.to_numeric(rita_metadata_for_merge_aa['log2FoldChange'], errors='coerce')
rita_metadata_for_merge_aa['padj'] = pd.to_numeric(rita_metadata_for_merge_aa['padj'], errors='coerce')


# Merge RITA experiment metadata
# Use a left merge to ensure all peptides from comprehensive_peptide_table_aa are kept.
# Peptides not found in rita_metadata_for_merge_aa (i.e., not used in the experiment)
# will have NaN values in the newly merged 'significant', 'log_FC', and 'adj_p_val' columns.
comprehensive_peptide_table_aa = comprehensive_peptide_table_aa.merge(
    rita_metadata_for_merge_aa,
    on='identifier',
    how='left'
)

# Display the first few rows of the new table
print("\nFirst 5 rows of the comprehensive peptide amino acid composition table:")
print(comprehensive_peptide_table_aa.head())
print(f"\nShape of the comprehensive peptide amino acid composition table: {comprehensive_peptide_table_aa.shape}")
print(f"Columns in the comprehensive peptide amino acid composition table: {comprehensive_peptide_table_aa.columns.tolist()}")


# Save the comprehensive table to a CSV file
comprehensive_table_path_aa = os.path.join(RESULTS_DIR, "comprehensive_peptide_amino_acid_composition_and_metadata.csv")
comprehensive_peptide_table_aa.to_csv(comprehensive_table_path_aa, index=False)
print(f"\nSaved comprehensive peptide amino acid composition table to: {comprehensive_table_path_aa}")


# Calculate Amino Acid Compositions for Each Group 
print("\nCalculating Amino Acid Compositions ")

# Full Library (VT/VP only)
comp_full_library = get_amino_acid_composition(full_library_filtered['Aminoacids'])

# Experiment Used (VT/VP only)
comp_exp_used = get_amino_acid_composition(RITA_exp_filtered['Aminoacids'])

# Peptides Used vs. Not Used in Experiment (from the VT/VP filtered library)
used_sequences_set = set(RITA_exp_filtered['Aminoacids'].unique())
not_used_peptides_df = full_library_filtered[~full_library_filtered['Aminoacids'].isin(used_sequences_set)]

comp_exp_not_used = get_amino_acid_composition(not_used_peptides_df['Aminoacids'])

num_used = len(used_sequences_set)
num_not_used = len(not_used_peptides_df['Aminoacids'].unique()) # Unique counts for 'not used'
print(f"\nComparison of Used vs. Not Used peptides (from VT/VP library):")
print(f"  Total unique peptides in full library (VT/VP): {len(full_library_filtered['Aminoacids'].unique())}")
print(f"  Unique peptides USED in experiment: {num_used} ({num_used / len(full_library_filtered['Aminoacids'].unique()):.2%})")
print(f"  Unique peptides NOT USED in experiment: {num_not_used} ({num_not_used / len(full_library_filtered['Aminoacids'].unique()):.2%})")


# Experiment Significant vs. Non-Significant (from VT/VP used in experiment)
RITA_sig = RITA_exp_filtered[RITA_exp_filtered['sig'] == 'Yes']
RITA_non_sig = RITA_exp_filtered[RITA_exp_filtered['sig'] == 'No']

comp_exp_sig = get_amino_acid_composition(RITA_sig['Aminoacids'])
comp_exp_non_sig = get_amino_acid_composition(RITA_non_sig['Aminoacids'])

print(f"\nSignificant vs. Non-Significant peptides (from VT/VP used in experiment):")
print(f"  Number of significant peptides: {len(RITA_sig)}")
print(f"  Number of non-significant peptides: {len(RITA_non_sig)}")

# Experiment Upregulated vs. Downregulated Significant (from VT/VP used and significant)
# Ensure log2FoldChange is numeric before comparison
RITA_sig['log2FoldChange'] = pd.to_numeric(RITA_sig['log2FoldChange'], errors='coerce')
RITA_up = RITA_sig[RITA_sig['log2FoldChange'] > 0]
RITA_down = RITA_sig[RITA_sig['log2FoldChange'] < 0]

comp_exp_up = get_amino_acid_composition(RITA_up['Aminoacids'])
comp_exp_down = get_amino_acid_composition(RITA_down['Aminoacids'])

print(f"\nUpregulated vs. Downregulated Significant peptides:")
print(f"  Number of upregulated significant peptides: {len(RITA_up)}")
print(f"  Number of downregulated significant peptides: {len(RITA_down)}")
print(f"  Number of significant peptides with logFC = 0 (or NaN): {len(RITA_sig) - len(RITA_up) - len(RITA_down)}")


# Combine Compositions into a Summary DataFrame and Save 
print("\nGenerating Summary Table ")
all_compositions = pd.DataFrame({
    'Full_Library_VT_VP': comp_full_library,
    'Experiment_Used_VT_VP': comp_exp_used,
    'Experiment_Not_Used_VT_VP': comp_exp_not_used,
    'Experiment_Significant_VT_VP': comp_exp_sig,
    'Experiment_NonSignificant_VT_VP': comp_exp_non_sig,
    'Experiment_Upregulated_VT_VP': comp_exp_up,
    'Experiment_Downregulated_VT_VP': comp_exp_down
})

# Round to 2 decimal places for presentation
all_compositions = all_compositions.round(2)

print("\nAmino Acid Composition Summary (Percentages):")
print(all_compositions)

summary_table_path = os.path.join(RESULTS_DIR, "amino_acid_composition_summary.csv")
all_compositions.to_csv(summary_table_path)
print(f"\nSaved amino acid composition summary table: {summary_table_path}")

# Plotting the Compositions 
print("\nGenerating Plots ")

# Plot 1: Full Library vs. Experiment Used
plot_composition(
    {'Full Library (VT/VP)': comp_full_library, 'Experiment Used (VT/VP)': comp_exp_used},
    'Full Library vs. Experiment Used Peptides (VT/VP)',
    'full_vs_used',
    RESULTS_DIR
)

# Plot 2: Experiment Used vs. Not Used
plot_composition(
    {'Experiment Used (VT/VP)': comp_exp_used, 'Experiment Not Used (VT/VP)': comp_exp_not_used},
    'Experiment Used vs. Not Used Peptides (VT/VP)',
    'used_vs_not_used',
    RESULTS_DIR
)

# Plot 3: Experiment Significant vs. Non-Significant
plot_composition(
    {'Significant (VT/VP)': comp_exp_sig, 'Non-Significant (VT/VP)': comp_exp_non_sig},
    'Experiment Significant vs. Non-Significant Peptides (VT/VP)',
    'significant_vs_nonsignificant',
    RESULTS_DIR
)

# Plot 4: Experiment Upregulated vs. Downregulated Significant
plot_composition(
    {'Upregulated Significant (VT/VP)': comp_exp_up, 'Downregulated Significant (VT/VP)': comp_exp_down},
    'Upregulated vs. Downregulated Significant Peptides (VT/VP)',
    'upregulated_vs_downregulated',
    RESULTS_DIR
)

# Plot 5: Experiment Subset Comparisons
plot_composition(
    {'Full Library (VT/VP)': comp_full_library,
    'Significant (VT/VP)': comp_exp_sig, 
    'Non-Significant (VT/VP)': comp_exp_non_sig,
    'Upregulated Significant (VT/VP)': comp_exp_up, 
    'Downregulated Significant (VT/VP)': comp_exp_down},
    'Comparison of Peptides (VT/VP)',
    'comparison_peptide_amino_acid_comparison',
    RESULTS_DIR
)

print("\nAnalysis Complete! Check your results directory for the summary CSV and plots.")




def get_unique_full_protein_info(full_library_proteins_df):
    """
    Extracts unique full protein sequences and their NCBI_ids for NetSurfP-3 input.
    Returns a dictionary {NCBI_id: sequence} and a DataFrame suitable for NetsurfP-3 input.
    Assumes full_library_proteins_df has 'NCBI_id' and a 'Sequence' column.
    """
    # Ensure 'Sequence' column is string and not empty, and 'NCBI_id' is present
    unique_proteins_df = full_library_proteins_df[
        full_library_proteins_df['Sequence'].apply(lambda x: isinstance(x, str) and len(x) > 0)
    ].copy()
    
    # Drop duplicates based on NCBI_id and sequence to ensure unique proteins for prediction
    unique_proteins_df = unique_proteins_df.drop_duplicates(subset=['NCBI_id', 'Sequence'])

    protein_sequences_dict = unique_proteins_df.set_index('NCBI_id')['Sequence'].to_dict()
    # Create a DataFrame for input to NetSurfP-3
    protein_input_df = unique_proteins_df[['NCBI_id', 'Sequence']].rename(columns={'NCBI_id': 'identifier', 'Sequence': 'Aminoacids'})
    return protein_sequences_dict, protein_input_df

# Helper function for buried/exposed prediction with netsurfp3
def _run_single_netsurfp3_batch(batch_input_df_args):
    """
    Worker function to run standalone NetSurfP-3 for a single batch of proteins.
    Returns a dictionary of {protein_id: [RSA_scores]} for the batch.
    """
    batch_index, batch_df = batch_input_df_args
    batch_results = {}
    fasta_input_path = None
    output_dir = None

    KEEP_TEMP_FILES = False 

    try:
        # Check if batch_df is empty after potential subsetting or filtering
        if batch_df.empty:
            sys.stderr.write(f"Warning (Batch {batch_index}): Input DataFrame for NetSurfP-3 batch is empty. Skipping.\n")
            return batch_results

        # Create temporary FASTA input file for this batch
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=f"_batch{batch_index}.fasta") as fasta_file:
            fasta_input_path = fasta_file.name
            for _, row in batch_df.iterrows():
                fasta_file.write(f">{row['identifier']}\n{row['Aminoacids']}\n")

        # Create a temporary output directory for this batch's NetSurfP-3.0 results
        output_dir = tempfile.mkdtemp(prefix=f"nsp3_standalone_output_batch{batch_index}_")

        # Prepare and run the standalone NetSurfP-3.0 script using 'conda run'
        command = [
            "conda", "run", "-n", NETSURFP3_STANDALONE_ENV_NAME,
            "python", NSP3_SCRIPT_PATH,
            "-m", NSP3_MODEL_PATH,
            "-i", fasta_input_path,
            "-o", output_dir # Output directory, as per standalone README
        ]
        
        print(f" (Batch {batch_index}): Running NetSurfP-3 command: {' '.join(command)}")

        # Execute the command, capturing stdout and stderr.
        process = subprocess.run(command, check=True, capture_output=True, text=True)
               
        print(f"DEBUG (Batch {batch_index}): NetSurfP-3 standalone prediction command executed. Checking output directory...")

        # Find the subdirectory created by NetSurfP-3.0 (e.g., '01', '02', etc.)
        # This subdirectory contains the actual output files for the batch.
        subdirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
        
        if not subdirs:
            sys.stderr.write(f"Warning (Batch {batch_index}): No subdirectories found in {output_dir}. NetSurfP-3 output structure may have changed.\n")
            return batch_results # Return empty if no subdirs
        
        # Assume there's only one relevant subdirectory per batch for now (e.g., '01', '02', etc. per job run)
        job_output_subdir_name = subdirs[0] # e.g., '01' based on your debug output
        job_output_subdir_path = os.path.join(output_dir, job_output_subdir_name)

        # Construct the path to the main CSV output file within that subdirectory.
        # Based on your `head 01.csv` output, the file is named after the subdir (e.g., '01.csv').
        main_csv_filepath = os.path.join(job_output_subdir_path, f"{job_output_subdir_name}.csv") 

        if not os.path.exists(main_csv_filepath):
            sys.stderr.write(f"Warning (Batch {batch_index}): Expected CSV file '{main_csv_filepath}' not found.\n")
            sys.stderr.write(f"Files in {job_output_subdir_path}: {os.listdir(job_output_subdir_path)}\n")
            return batch_results

        protein_rsa_scores_accumulator = {} # Accumulate scores for each protein in this batch
        with open(main_csv_filepath, 'r') as csvfile:
            reader = csv.DictReader(csvfile)

            # --- CRITICAL FIX: Look for ' rsa' with a leading space and 'id' as key ---
            if ' rsa' not in reader.fieldnames: # (yes it needs to have a space, as the developer put spaces in the columns)
                sys.stderr.write(f"Warning (Batch {batch_index}): ' rsa' column not found in {main_csv_filepath}. Found: {reader.fieldnames}. Skipping this file.\n")
                return batch_results

            for row in reader:
                try:
                    # The 'id' column contains '>YP_009944365.1'. Strip the '>' prefix.
                    current_protein_id = row['id'].lstrip('>') 
                    rsa_value = float(row[' rsa']) # Get RSA value from lowercase ' rsa' column (yes it needs to have a space, as the developer put spaces in the columns)
                    
                    if current_protein_id not in protein_rsa_scores_accumulator:
                        protein_rsa_scores_accumulator[current_protein_id] = []
                    
                    protein_rsa_scores_accumulator[current_protein_id].append(rsa_value)
                except (ValueError, KeyError) as parse_e:
                    sys.stderr.write(f"Warning (Batch {batch_index}): Error parsing row in {main_csv_filepath}: {row}. Error: {parse_e}. Skipping row.\n")
                    continue
        
        # After processing all rows, populate batch_results
        batch_results.update(protein_rsa_scores_accumulator)
        
        if not batch_results:
            sys.stderr.write(f"Warning (Batch {batch_index}): No valid RSA scores were extracted from {main_csv_filepath}.\n")

    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"Error running standalone NetSurfP-3 for batch {batch_index} (command: {' '.join(command)}): {e}\n")
        sys.stderr.write(f"Stdout (first 500 chars):\n{e.stdout[:500]}...\n")
        sys.stderr.write(f"Stderr:\n{e.stderr}\n")
        raise # Re-raise the error as it's critical
    except (FileNotFoundError, ValueError) as e:
        sys.stderr.write(f"Error for batch {batch_index}: {e}\n")
        raise
    except Exception as e:
        sys.stderr.write(f"An unexpected error occurred during standalone NetSurfP-3 execution or output parsing for batch {batch_index}: {e}\n")
        raise
    finally:
        # Clean up temporary files and directories for this batch
        if not KEEP_TEMP_FILES: # Only delete if KEEP_TEMP_FILES is False
            if fasta_input_path and os.path.exists(fasta_input_path):
                os.remove(fasta_input_path)
            if output_dir and os.path.exists(output_dir):
                import shutil
                shutil.rmtree(output_dir)
            
    return batch_results


def run_netsurfp3_standalone_prediction(input_df, num_netsurfp3_processes=None):
    """
    Runs NetSurfP-3 prediction for a given DataFrame of proteins using the standalone package,
    parallelizing by splitting into batches.
    Args:
        input_df (pd.DataFrame): DataFrame with 'identifier' and 'Aminoacids' columns (for FASTA input).
        num_netsurfp3_processes (int, optional): Number of parallel processes to use for NetSurfP-3.
                                                If None, uses min(os.cpu_count() or 1, 4).
    Returns:
        dict: Parsed prediction results: {identifier: [RSA_scores_list]}.
              Returns an empty dict if input_df is empty.
    Raises:
        Exception: If any error occurs during standalone execution or parsing.
    """
    if input_df.empty:
        print("Warning: No proteins to predict for NetSurfP-3 standalone.")
        return {}

    if not os.path.exists(NSP3_SCRIPT_PATH):
        raise FileNotFoundError(f"NetSurfP-3 standalone script not found: {NSP3_SCRIPT_PATH}. "
                                f"Please verify NETSURFP3_STANDALONE_PATH.")
    if not os.path.exists(NSP3_MODEL_PATH):
        raise FileNotFoundError(f"NetSurfP-3 standalone model not found: {NSP3_MODEL_PATH}. "
                                f"Please verify NETSURFP3_STANDALONE_PATH and 'models/nsp3.pth'.")
    
    if num_netsurfp3_processes is None:
        num_netsurfp3_processes = min(os.cpu_count() or 1, 4)

    # Split the input DataFrame into batches
    num_proteins = len(input_df)
    num_batches = math.ceil(num_proteins / NETSURFP3_BATCH_SIZE)
    batches = []
    for i in range(num_batches):
        start_idx = i * NETSURFP3_BATCH_SIZE
        end_idx = min((i + 1) * NETSURFP3_BATCH_SIZE, num_proteins)
        batches.append((i, input_df.iloc[start_idx:end_idx])) # Pass index and batch_df

    print(f"\nRunning {num_proteins} proteins in {num_batches} batches on {num_netsurfp3_processes} processes for NetSurfP-3 (standalone)...")
    
    total_rsa_map = {}
    with multiprocessing.Pool(processes=num_netsurfp3_processes) as pool:
        # Use imap_unordered for tqdm progress bar and yield results as they complete
        for batch_results in tqdm(pool.imap_unordered(_run_single_netsurfp3_batch, batches),
                                  total=num_batches,
                                  desc=f"NetSurfP-3 batches ({num_netsurfp3_processes} cores)"):
            total_rsa_map.update(batch_results) # Aggregate results from each batch

    return total_rsa_map


# WORKER FUNCTION FOR PARALLEL PROCESSING
def _process_single_peptide_properties(args):
    """
    Worker function to calculate properties for a single peptide, including buried/exposed status.
    This function will be run in parallel processes.
    """
    peptide_id, seq, s4pred_pred_string, threshold_disorder, full_protein_seq, protein_rsa_scores, rsa_buried_threshold = args

    initial_props = {f'{prop}_perc': 0.0 for prop in PEPTIDE_PROPERTY_TYPES}
    return_dict = {'identifier': peptide_id, **initial_props}

    if len(seq) == 0:
        return return_dict

    total_residues = len(seq)

    # 1. Disorder Prediction (metapredict)
    disorder_scores = mpp.predict_disorder(seq)

    # Secondary structure prediction string is passed in
    if len(s4pred_pred_string) != len(seq):
        print(f"Warning (process {os.getpid()}): s4pred prediction length mismatch for peptide {peptide_id} ({len(s4pred_pred_string)} vs {len(seq)}). Assuming all coil for SS part.")
        s4pred_pred_string = 'C' * len(seq)

    # 2. Combine and make mutually exclusive: Disorder (D) overrides SS (H, E, C)
    combined_prediction = []
    for j in range(len(seq)):
        if disorder_scores[j] >= threshold_disorder:
            combined_prediction.append('D') # Disordered
        else:
            combined_prediction.append(s4pred_pred_string[j]) # H, E, or C
    combined_prediction_string = "".join(combined_prediction)

    # 3. Calculate SS and Disorder percentages
    return_dict['Disorder_perc'] = (combined_prediction_string.count('D') / total_residues) * 100
    return_dict['Helix_perc'] = (combined_prediction_string.count('H') / total_residues) * 100
    return_dict['Sheet_perc'] = (combined_prediction_string.count('E') / total_residues) * 100
    return_dict['Coil_perc'] = (combined_prediction_string.count('C') / total_residues) * 100

    # 4. Buried/Exposed calculation using full protein RSA scores
    if full_protein_seq and protein_rsa_scores and len(full_protein_seq) == len(protein_rsa_scores):
        # Remove leading M (artifact from sequencing) to match to original protein
        search_seq = seq
        stripped_m = False
        if seq.startswith('M') and len(seq) > 1:
            # If the peptide starts with 'M' and is long enough to strip,
            # use the M-stripped version for finding in the full protein.
            search_seq = seq[1:]
            stripped_m = True
            
        start_index = full_protein_seq.find(search_seq)        

        if start_index != -1:
            # Adjust slice length if 'M' was stripped, the RSA scores correspond to the full_protein_seq
            # so the slice should still be the length of the *original* peptide if we expect a 1:1 match.
            # However, if the M was spurious, we want the RSA for the *matched* part.
            # The RSA scores for the peptide should match the length of `search_seq`.
            # If the M was stripped, the peptide (seq) is 1 longer than search_seq.
            
            # This logic assumes protein_rsa_scores align with full_protein_seq, and we want
            # RSA for the *part of the protein that matches our peptide's core sequence*.
            peptide_rsa_slice = protein_rsa_scores[start_index : start_index + len(search_seq)]
            
            if len(peptide_rsa_slice) == len(search_seq):
                buried_count = sum(1 for rsa in peptide_rsa_slice if rsa < rsa_buried_threshold)
                exposed_count = len(peptide_rsa_slice) - buried_count
                
                return_dict['Buried_perc'] = (buried_count / len(search_seq)) * 100 # Calculate percentage over search_seq length
                return_dict['Exposed_perc'] = (exposed_count / len(search_seq)) * 100
            else:
                print(f"Warning: RSA slice length mismatch for peptide '{seq[:20]}...' (ID: {peptide_id}, stripped M: {stripped_m}). Expected {len(search_seq)}, got {len(peptide_rsa_slice)}. Buried/Exposed percentages will be 0.")
        else:
            reason = "(after stripping leading 'M')" if stripped_m else "(original sequence)"
            print(f"Warning: Peptide '{seq[:20]}...' (ID: {peptide_id}) {reason} not found in its full protein sequence. Buried/Exposed percentages will be 0.")
    else:
        print(f"Warning: Missing full protein sequence or RSA scores for peptide {peptide_id}'s protein, or length mismatch ({len(full_protein_seq) if full_protein_seq else 0} vs {len(protein_rsa_scores) if protein_rsa_scores else 0}). Buried/Exposed percentages will be 0.")

    return return_dict


# Helper Function to Get Peptide Properties (Disorder, Secondary Structure, Exposed/Buried)
def calculate_all_peptide_structural_properties(peptides_df_with_ncbi, full_proteins_df, threshold_disorder=0.5, rsa_buried_threshold=0.25, num_processes=None, num_netsurfp3_processes=None):
    """
    Calculates all peptide properties (Disorder, SS, Buried/Exposed).
    Orchestrates full protein RSA prediction and then individual peptide property calculation.
    
    Args:
        peptides_df_with_ncbi (pd.DataFrame): DataFrame with 'identifier', 'Aminoacids', and 'NCBI_id' columns.
        full_proteins_df (pd.DataFrame): DataFrame with 'NCBI_id' and 'Sequence' columns for full proteins.
        threshold_disorder (float): Disorder score threshold for metapredict.
        rsa_buried_threshold (float): RSA threshold for classifying residues as buried.
        num_processes (int, optional): Number of CPU cores to use for parallel processing for general peptide properties.
        num_netsurfp3_processes (int, optional): Number of CPU cores to use for parallel NetSurfP-3 calls.
    Returns:
        pd.DataFrame: A DataFrame indexed by 'identifier' with all calculated percentage columns.
    """
    if peptides_df_with_ncbi.empty:
        return pd.DataFrame(columns=[f'{prop}_perc' for prop in PEPTIDE_PROPERTY_TYPES])

    valid_peptides_df = peptides_df_with_ncbi[
        peptides_df_with_ncbi['Aminoacids'].apply(lambda x: isinstance(x, str) and len(x) > 0)
    ].copy()

    if valid_peptides_df.empty:
        return pd.DataFrame(columns=[f'{prop}_perc' for prop in PEPTIDE_PROPERTY_TYPES])

    # 1. Get unique full proteins and their sequences for NetSurfP-3 input
    full_protein_sequences_dict, full_protein_nsp3_input_df = get_unique_full_protein_info(full_proteins_df)

    # 2. Run NetSurfP-3 for RSA on all unique full proteins using the standalone version (PARALLELIZED)
    print(f"\nPredicting RSA for {len(full_protein_nsp3_input_df)} unique full proteins using NetSurfP-3 (standalone, parallelized)...")
    protein_rsa_map = run_netsurfp3_standalone_prediction(full_protein_nsp3_input_df, num_netsurfp3_processes=num_netsurfp3_processes)
    print("NetSurfP-3 RSA prediction complete.")
    
    # 3. Run S4PRED for secondary structure on all peptides (same logic as before)
    fasta_input_path_s4pred = None
    s4pred_output_path = None
    s4pred_map = {}

    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".fasta") as fasta_file:
            fasta_input_path_s4pred = fasta_file.name
            for _, row in valid_peptides_df.iterrows():
                fasta_file.write(f">{row['identifier']}\n{row['Aminoacids']}\n")

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".ss_fas") as output_file:
            s4pred_output_path = output_file.name
            command = [sys.executable, run_model_script, "--outfmt", "fas", fasta_input_path_s4pred]
            print(f"\nRunning s4pred via subprocess: {' '.join(command)}")
            try:
                subprocess.run(command, check=True, stdout=output_file, stderr=subprocess.PIPE, text=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running s4pred (command: {' '.join(command)}): {e}")
                print(f"Stderr: {e.stderr}")
                raise

        current_id = None
        current_seq_line = None
        with open(s4pred_output_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    current_id = line[1:]
                    current_seq_line = None
                elif current_id and current_seq_line is None:
                    current_seq_line = line
                elif current_id and current_seq_line is not None:
                    s4pred_map[current_id] = line
                    current_id = None
                    current_seq_line = None

    finally:
        if fasta_input_path_s4pred and os.path.exists(fasta_input_path_s4pred): os.remove(fasta_input_path_s4pred)
        if s4pred_output_path and os.path.exists(s4pred_output_path): os.remove(s4pred_output_path)
    # END S4PRED SUBPROCESS BLOCK

    # Prepare arguments for parallel processing for each peptide
    task_args = []
    for _, row in valid_peptides_df.iterrows():
        peptide_id = row['identifier']
        peptide_seq = row['Aminoacids']
        ncbi_id = row['NCBI_id']
        
        full_prot_seq = full_protein_sequences_dict.get(ncbi_id)
        prot_rsa_scores = protein_rsa_map.get(ncbi_id)
        s4pred_ss = s4pred_map.get(peptide_id, 'C' * len(peptide_seq))

        task_args.append(
            (peptide_id, peptide_seq, s4pred_ss, threshold_disorder, 
             full_prot_seq, prot_rsa_scores, rsa_buried_threshold)
        )

    print(f"Calculating peptide properties for {len(valid_peptides_df)} peptides in parallel (disorder overrides SS, plus buried/exposed)...")

    if num_processes is None:
        num_processes = min(os.cpu_count() or 1, 4)

    results = []
    with multiprocessing.Pool(processes=num_processes) as pool:
        for res in tqdm(pool.imap_unordered(_process_single_peptide_properties, task_args),
                        total=len(task_args),
                        desc=f"Processing peptides ({num_processes} cores)"):
            results.append(res)

    return pd.DataFrame(results).set_index('identifier')
    

# Helper Function to Plot Average Peptide Properties (no changes needed here)
def plot_average_properties(average_properties_dict, title, filename_prefix, results_dir):
    """
    Plots average peptide properties (Disorder, Helix, Sheet, Coil) for different groups.
    average_properties_dict: dict of {group_name: pandas.Series of average properties}
    """
    if not average_properties_dict:
        print(f"Skipping plot '{title}': No data provided.")
        return

    plot_data = []
    for group_name, series in average_properties_dict.items():
        if series is not None and not series.empty:
            # Reorder the series according to PEPTIDE_PROPERTY_TYPES for consistent plotting order
            ordered_series = series[[f'{prop}_perc' for prop in PEPTIDE_PROPERTY_TYPES]]
            temp_df = ordered_series.to_frame(name='Percentage').reset_index()
            temp_df.columns = ['Property', 'Percentage']
            # Clean up property names for plotting (remove '_perc')
            temp_df['Property'] = temp_df['Property'].str.replace('_perc', '')
            temp_df['Group'] = group_name
            plot_data.append(temp_df)
        else:
            print(f"Warning: No valid average property data for group '{group_name}' in '{title}'.")

    if not plot_data:
        print(f"Skipping plot '{title}': No valid dataframes to concatenate.")
        return

    plot_df = pd.concat(plot_data)

    plt.figure(figsize=(16, 8))
    sns.barplot(data=plot_df, x='Property', y='Percentage', hue='Group', palette='Spectral', ci=None)
    plt.title(f'Average Peptide Properties: {title}', fontsize=16)
    plt.xlabel('Peptide Property', fontsize=12)
    plt.ylabel('Average Percentage (%)', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f"{filename_prefix}_peptide_properties.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved plot: {plot_path}")



# Filter DataFrames for VT or VP peptides
print("\n Filtering Data")
full_library_filtered = full_library_df[full_library_df['code'].isin(['VT', 'VP'])].copy()
RITA_exp_filtered = RITA_exp_df[RITA_exp_df['type'].isin(['VT', 'VP'])].copy()
RITA_exp_filtered['identifier'] = RITA_exp_filtered['tileID']


print(f"Number of peptides in full library (VT/VP): {len(full_library_filtered)}")
print(f"Number of peptides used in RITA experiment (VT/VP): {len(RITA_exp_filtered)}")


# Peptides Used vs. Not Used in Experiment (from the VT/VP filtered library)
used_sequences_set = set(RITA_exp_filtered['Aminoacids'].unique())
not_used_peptides_df = full_library_filtered[~full_library_filtered['Aminoacids'].isin(used_sequences_set)].copy()

num_used = len(used_sequences_set)
num_not_used = len(not_used_peptides_df['Aminoacids'].unique()) # Unique counts for 'not used'
print(f"\nComparison of Used vs. Not Used peptides (from VT/VP library):")
print(f"  Total unique peptides in full library (VT/VP): {len(full_library_filtered['Aminoacids'].unique())}")
print(f"  Unique peptides USED in experiment: {num_used} ({num_used / len(full_library_filtered['Aminoacids'].unique()):.2%})")
print(f"  Unique peptides NOT USED in experiment: {num_not_used} ({num_not_used / len(full_library_filtered['Aminoacids'].unique()):.2%})")


# Experiment Significant vs. Non-Significant (from VT/VP used in experiment)
RITA_sig = RITA_exp_filtered[RITA_exp_filtered['sig'] == 'Yes'].copy()
RITA_non_sig = RITA_exp_filtered[RITA_exp_filtered['sig'] == 'No'].copy()


print(f"\nSignificant vs. Non-Significant peptides (from VT/VP used in experiment):")
print(f"  Number of significant peptides: {len(RITA_sig)}")
print(f"  Number of non-significant peptides: {len(RITA_non_sig)}")

# Experiment Upregulated vs. Downregulated Significant (from VT/VP used and significant)
# Ensure log2FoldChange is numeric before comparison
RITA_sig['log2FoldChange'] = pd.to_numeric(RITA_sig['log2FoldChange'], errors='coerce')
RITA_up = RITA_sig[RITA_sig['log2FoldChange'] > 0].copy()
RITA_down = RITA_sig[RITA_sig['log2FoldChange'] < 0].copy()


print(f"\nUpregulated vs. Downregulated Significant peptides:")
print(f"  Number of upregulated significant peptides: {len(RITA_up)}")
print(f"  Number of downregulated significant peptides: {len(RITA_down)}")
print(f"  Number of significant peptides with logFC = 0 (or NaN): {len(RITA_sig) - len(RITA_up) - len(RITA_down)}")


# Define the peptide property types for consistent ordering in plots.
# 'Disorder' will be mutually exclusive with Helix/Sheet/Coil.
PEPTIDE_PROPERTY_TYPES = ['Disorder', 'Helix', 'Sheet', 'Coil', 'Buried', 'Exposed']

# Calculate all peptide properties once 
print("\nCalculating all unique peptide properties (Disorder, Secondary Structure) ONCE")
all_peptides_properties_df = calculate_all_peptide_structural_properties(
    full_library_filtered[['identifier', 'Aminoacids', 'NCBI_id']],
    full_proteins_df,
    threshold_disorder=0.5,
    rsa_buried_threshold=RSA_BURIED_THRESHOLD,
    num_processes=None
)

print("\nGenerating comprehensive peptide properties and metadata table...")

# Start with the basic peptide info (identifier, sequence, NCBI_id) from the filtered full library
comprehensive_peptide_table = full_library_filtered[['identifier', 'Aminoacids', 'NCBI_id']].copy()

# Merge with the calculated structural properties from all_peptides_properties_df
comprehensive_peptide_table = comprehensive_peptide_table.merge(
    all_peptides_properties_df,
    left_on='identifier',
    right_index=True,
    how='left' # Use left merge to keep all peptides from base table
)

# Prepare RITA experiment data for merging
rita_metadata_for_merge = RITA_exp_filtered[['identifier', 'sig', 'log2FoldChange', 'padj']].copy()

# Merge RITA experiment metadata
comprehensive_peptide_table = comprehensive_peptide_table.merge(
    rita_metadata_for_merge,
    on='identifier',
    how='left'
)

# Display the first few rows of the new table
print("\nFirst 5 rows of the comprehensive peptide table:")
print(comprehensive_peptide_table.head())
print(f"\nShape of the comprehensive peptide table: {comprehensive_peptide_table.shape}")
print(f"Columns in the comprehensive peptide table: {comprehensive_peptide_table.columns.tolist()}")


# Save the comprehensive table to a CSV file
comprehensive_table_path = os.path.join(RESULTS_DIR, "comprehensive_peptide_properties_and_metadata_with_RSA.csv") # Updated filename
comprehensive_peptide_table.to_csv(comprehensive_table_path, index=False)
print(f"\nSaved comprehensive peptide table to: {comprehensive_table_path}")


#  Retrieve Peptide Properties for Each Group from the pre-calculated data 
print("\nRetrieving and Averaging Peptide Properties for Each Group")

average_properties_series = {}

def get_avg_props(identifiers_series, df_source):
    if identifiers_series.empty:
        return pd.Series({f'{prop}_perc': 0.0 for prop in PEPTIDE_PROPERTY_TYPES})
    unique_ids = identifiers_series.unique()
    existing_ids = df_source.index.intersection(unique_ids)
    if existing_ids.empty:
        print(f"Warning: No properties found for identifiers in group. Returning zeros for {len(unique_ids)} peptides.")
        return pd.Series({f'{prop}_perc': 0.0 for prop in PEPTIDE_PROPERTY_TYPES})
    
    props_df = df_source.loc[existing_ids]
    # Ensure only the percentage columns are selected for mean calculation
    return props_df[[f'{prop}_perc' for prop in PEPTIDE_PROPERTY_TYPES]].mean()


average_properties_series['Full_Library_VT_VP'] = get_avg_props(full_library_filtered['identifier'], all_peptides_properties_df)
average_properties_series['Experiment_Used_VT_VP'] = get_avg_props(RITA_exp_filtered['identifier'], all_peptides_properties_df)
average_properties_series['Experiment_Not_Used_VT_VP'] = get_avg_props(not_used_peptides_df['identifier'], all_peptides_properties_df)
average_properties_series['Experiment_Significant_VT_VP'] = get_avg_props(RITA_sig['identifier'], all_peptides_properties_df)
average_properties_series['Experiment_NonSignificant_VT_VP'] = get_avg_props(RITA_non_sig['identifier'], all_peptides_properties_df)
average_properties_series['Experiment_Upregulated_VT_VP'] = get_avg_props(RITA_up['identifier'], all_peptides_properties_df)
average_properties_series['Experiment_Downregulated_VT_VP'] = get_avg_props(RITA_down['identifier'], all_peptides_properties_df)


# Generate and Save Peptide Properties Summary Table
print("\nGenerating Peptide Properties Summary Table")
all_peptide_properties_avg = pd.DataFrame(average_properties_series).round(2)

print("\nAverage Peptide Properties Summary (Percentages):")
print(all_peptide_properties_avg)
properties_summary_table_path = os.path.join(RESULTS_DIR, "peptide_properties_summary_with_RSA.csv") # Updated filename
all_peptide_properties_avg.to_csv(properties_summary_table_path)
print(f"\nSaved average peptide properties summary table: {properties_summary_table_path}")


# Plotting the Peptide Properties
print("\nGenerating Peptide Properties Plots")

plot_average_properties(
    {'Full Library (VT/VP)': average_properties_series['Full_Library_VT_VP'],
     'Experiment Used (VT/VP)': average_properties_series['Experiment_Used_VT_VP']},
    'Full Library vs. Experiment Used Peptides (VT/VP)',
    'full_vs_used_properties_with_RSA', # Updated filename prefix
    RESULTS_DIR
)
plot_average_properties(
    {'Experiment Used (VT/VP)': average_properties_series['Experiment_Used_VT_VP'],
     'Experiment Not Used (VT/VP)': average_properties_series['Experiment_Not_Used_VT_VP']},
    'Experiment Used vs. Not Used Peptides (VT/VP)',
    'used_vs_not_used_properties_with_RSA', # Updated filename prefix
    RESULTS_DIR
)
plot_average_properties(
    {'Significant (VT/VP)': average_properties_series['Experiment_Significant_VT_VP'],
     'Non-Significant (VT/VP)': average_properties_series['Experiment_NonSignificant_VT_VP']},
    'Experiment Significant vs. Non-Significant Peptides (VT/VP)',
    'significant_vs_nonsignificant_properties_with_RSA', # Updated filename prefix
    RESULTS_DIR
)
plot_average_properties(
    {'Upregulated Significant (VT/VP)': average_properties_series['Experiment_Upregulated_VT_VP'],
     'Downregulated Significant (VT/VP)': average_properties_series['Experiment_Downregulated_VT_VP']},
    'Upregulated vs. Downregulated Significant Peptides (VT/VP)',
    'upregulated_vs_downregulated_properties_with_RSA', # Updated filename prefix
    RESULTS_DIR
)

plot_average_properties(
    {'Full Library (VT/VP)': average_properties_series['Full_Library_VT_VP'],
     'Significant (VT/VP)': average_properties_series['Experiment_Significant_VT_VP'],
     'Non-Significant (VT/VP)': average_properties_series['Experiment_NonSignificant_VT_VP'],
     'Upregulated Significant (VT/VP)': average_properties_series['Experiment_Upregulated_VT_VP'],
     'Downregulated Significant (VT/VP)': average_properties_series['Experiment_Downregulated_VT_VP']},
    'Comparison of Peptides (VT/VP)',
    'comparison_peptide_properties',
    RESULTS_DIR
)

print("\nAll Analysis Complete! Check your results directory for summary CSVs and plots.")