import pandas as pd
import requests
import time
import sys
from tqdm import tqdm

# --- Configuration ---
# Define input and output file paths
INPUT_FILE = "data/interacting-proteins_Lit-BM.tsv"
OUTPUT_FILE = "data/interacting-proteins-full-dataset_Lit-BM.csv"

# Define column names for the output CSV file
OUTPUT_ID_A_COLUMN = "ID_A_Interactor"
OUTPUT_SEQUENCE_A_COLUMN = "Sequence_A"
OUTPUT_ID_B_COLUMN = "ID_B_Interactor"
OUTPUT_SEQUENCE_B_COLUMN = "Sequence_B"

# Internal column names used when reading the input TSV, as it lacks a header
INPUT_ID_A_COLUMN_NAME = "Input_ID_A"
INPUT_ID_B_COLUMN_NAME = "Input_ID_B"

# UniProt API base URL for sequence search
UNIPROT_API_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"

# Rate limit for UniProt API requests (in seconds)
# This delay prevents hitting UniProt's rate limits and ensures polite API usage.
# UniProt recommends max 1 request/second for batch, or 0.1-0.5 for individual.
API_REQUEST_DELAY = 0.2  # 200 milliseconds

# --- Function to fetch sequence from UniProt ---
def fetch_sequence_from_ensembl_id(ensembl_id: str) -> str | None:
    """
    Fetches the protein sequence for a given Ensembl Gene ID from UniProtKB.
    It queries UniProt for reviewed entries linked to the Ensembl ID and
    extracts the primary protein sequence.

    Args:
        ensembl_id (str): The Ensembl Gene ID (e.g., ENSG00000001167).

    Returns:
        str | None: The protein sequence if found, otherwise None.
    """
    query_params = {
        "query": f"(reviewed:true) AND (xref:Ensembl-{ensembl_id})",
        "fields": "sequence",  # Request only the sequence field
        "format": "json"
    }

    try:
        response = requests.get(UNIPROT_API_SEARCH_URL, params=query_params)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        if data and 'results' in data and len(data['results']) > 0:
            # Return the sequence from the first result found (typically the canonical one)
            sequence = data['results'][0]['sequence']['value']
            return sequence
        else:
            # No reviewed UniProt entry found for the given Ensembl ID
            return None

    except requests.exceptions.RequestException as e:
        print(f"\nError fetching sequence for Ensembl ID {ensembl_id}: {e}", file=sys.stderr)
        return None
    except KeyError as e:
        print(f"\nError parsing UniProt response for Ensembl ID {ensembl_id}. Missing key: {e}", file=sys.stderr)
        return None

# --- Main Script Execution ---
def main():
    """
    Main function to orchestrate the script execution.
    It reads interacting protein pairs, fetches their sequences from UniProt,
    and saves the combined data to a new CSV file.
    """
    print(f"Starting to process {INPUT_FILE}...")

    try:
        # Read the input TSV file. 'header=None' is used as the file lacks a header row.
        # 'names' assigns temporary column names for easy access.
        df_input = pd.read_csv(INPUT_FILE, sep='\t', header=None,
                               names=[INPUT_ID_A_COLUMN_NAME, INPUT_ID_B_COLUMN_NAME])
        print(f"Read {len(df_input)} protein pairs from {INPUT_FILE}.")
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILE}' not found. Please check the path.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading input file {INPUT_FILE}: {e}", file=sys.stderr)
        sys.exit(1)

    # 1. Collect all unique Ensembl IDs from both columns
    all_ensembl_ids = set()
    all_ensembl_ids.update(df_input[INPUT_ID_A_COLUMN_NAME].unique())
    all_ensembl_ids.update(df_input[INPUT_ID_B_COLUMN_NAME].unique())
    print(f"Found {len(all_ensembl_ids)} unique Ensembl IDs to query for sequences.")

    # 2. Fetch sequences for all unique IDs and store them in a dictionary (acting as a cache)
    sequence_map = {}
    # tqdm provides a progress bar for the API calls, enhancing user experience.
    for ensembl_id in tqdm(list(all_ensembl_ids), desc="Fetching sequences from UniProt"):
        if ensembl_id not in sequence_map:
            sequence = fetch_sequence_from_ensembl_id(ensembl_id)
            sequence_map[ensembl_id] = sequence
            time.sleep(API_REQUEST_DELAY)  # Pause to respect UniProt API rate limits

    print("\nFinished fetching all unique sequences.")

    # 3. Create the output DataFrame by mapping fetched sequences to the original IDs
    df_output = pd.DataFrame()
    df_output[OUTPUT_ID_A_COLUMN] = df_input[INPUT_ID_A_COLUMN_NAME]
    df_output[OUTPUT_ID_B_COLUMN] = df_input[INPUT_ID_B_COLUMN_NAME]

    # Map Ensembl IDs to their fetched sequences. IDs for which no sequence was found
    # (i.e., `sequence_map` contains None) will be filled with "NOT_FOUND".
    df_output[OUTPUT_SEQUENCE_A_COLUMN] = df_output[OUTPUT_ID_A_COLUMN].map(sequence_map).fillna("NOT_FOUND")
    df_output[OUTPUT_SEQUENCE_B_COLUMN] = df_output[OUTPUT_ID_B_COLUMN].map(sequence_map).fillna("NOT_FOUND")

    # 4. Save the resulting DataFrame to a CSV file
    try:
        df_output.to_csv(OUTPUT_FILE, index=False)
        print(f"Successfully saved full dataset to {OUTPUT_FILE}.")
    except Exception as e:
        print(f"Error saving output file {OUTPUT_FILE}: {e}", file=sys.stderr)
        sys.exit(1)

    print("Script finished.")

if __name__ == "__main__":
    main()