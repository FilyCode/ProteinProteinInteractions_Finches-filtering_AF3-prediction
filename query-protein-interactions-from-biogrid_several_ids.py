import pandas as pd
import requests
import time
from collections import Counter, defaultdict
import json
import traceback
from UniProtMapper import ProtMapper
from typing import Set, Dict, List, Tuple
from tqdm import tqdm
import os

# --- Configuration ---
BIOGRID_ACCESS_KEY = "6a8be2d9ffdf1ad752599e23c0f5963b"  # IMPORTANT: Replace with your actual BioGRID access key!
SPECIES_ID = 9606  # Homo sapiens (human)
MIN_EVIDENCE_COUNT = 3  # Minimum number of evidence pieces required for an interaction
INPUT_CSV_FILE = 'data/Human-proteom_GCF_000001405.40.csv'
OUTPUT_DIR = 'data/interaction_results' # New directory for all output files

# Define your target proteins and families here.
# Each entry in 'individual_targets' is a tuple (UniProt_ID, Entrez_ID).
# Each entry in 'families' is a dictionary: {"family_name": [list of UniProt IDs for family members]}
FAMILIES_CONFIG = {
    "individual_targets": [
        ("P04637", "7157"), # Human P53
        ("P10415", "596"),  # Human BCL2
        ("Q07817", "598")   # Human BCL2L1 (BCLXL)
    ],
    "families": {
        "BCL2_family": ["P10415", "Q07817"] # Example: BCL2 and BCL2L1 form a family
    }
}

# UniProt API base URL (not directly used by ProtMapper, but kept for context if needed)
UNIPROT_ID_MAPPING_URL = "https://rest.uniprot.org/idmapping"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helper Function: map_ids_via_uniprot (uses uniprot-id-mapper) ---
def map_ids_via_uniprot(id_list: List[str], from_db_name: str, to_db_name: str) -> Dict[str, str]:
    """
    Maps a list of IDs from a specified source database to a target database
    using the uniprot-id-mapper package.
    Handles multiple target mappings for a single input ID by collecting all
    unique accessions into a comma-separated string.
    Implements client-side batching and **retry logic with exponential backoff**
    for large ID lists to prevent API errors.

    Args:
        id_list (list): A list of IDs to be mapped (e.g., RefSeq Protein IDs, GeneIDs, UniProtKB IDs).
        from_db_name (str): The name of the source database (e.g., "GeneID", "RefSeq_Protein", "UniProtKB_AC-ID").
        to_db_name (str): The name of the target database (e.g., "UniProtKB", "GeneID").

    Returns:
        dict: A dictionary mapping original ID to a comma-separated string of its mapped IDs.
              Returns an empty dictionary if no mappings are found or if an error occurs.
    """
    if not id_list:
        return {}

    unique_ids_sent = list(set(id_list))
    print(f"\nAttempting to map {len(unique_ids_sent)} unique IDs from '{from_db_name}' to '{to_db_name}' via uniprot-id-mapper...")

    mapper = ProtMapper()
    all_mappings_sets = {} # To store from_id -> set of target accessions
    cumulative_failed_ids = set() # To store failed IDs across all batches

    BATCH_SIZE = 1000
    MAX_RETRIES_PER_BATCH = 5 # Maximum attempts for a single batch
    INITIAL_RETRY_DELAY = 5 # Seconds for initial delay, will increase exponentially

    # Dynamically determine the target column name
    target_column_name = 'Entry' if to_db_name in ["UniProtKB", "UniProtKB_AC-ID"] else 'To'

    num_batches = (len(unique_ids_sent) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in tqdm(range(num_batches), desc=f"Mapping batches ({from_db_name} to {to_db_name})"):
        batch_ids = unique_ids_sent[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]

        attempt = 0
        success = False
        while attempt < MAX_RETRIES_PER_BATCH:
            try:
                mapped_df_batch, failed_ids_batch = mapper.get(
                    ids=batch_ids,
                    from_db=from_db_name,
                    to_db=to_db_name
                )

                if not mapped_df_batch.empty:
                    if target_column_name not in mapped_df_batch.columns:
                        print(f"\nError: Expected target column '{target_column_name}' not found in mapped DataFrame for batch {i+1}. Available columns: {mapped_df_batch.columns.tolist()}. Retrying batch...")
                        # Raise a KeyError to be caught by the outer try-except and trigger a retry for this batch
                        raise KeyError(f"Target column '{target_column_name}' missing in UniProtMapper response.")

                    for original_id, target_id in zip(mapped_df_batch['From'], mapped_df_batch[target_column_name]):
                        all_mappings_sets.setdefault(original_id, set()).add(target_id)

                if failed_ids_batch:
                    cumulative_failed_ids.update(failed_ids_batch)

                success = True # Batch successfully processed
                break # Exit retry loop if successful

            except (requests.exceptions.RequestException, requests.exceptions.HTTPError, json.JSONDecodeError, KeyError) as e:
                # Catch specific network/API errors and the KeyError from missing 'Entry'/'To' column
                attempt += 1
                delay = INITIAL_RETRY_DELAY * (2 ** (attempt - 1)) # Exponential backoff
                print(f"\nWarning: Error during mapping batch {i+1}/{num_batches} (attempt {attempt}/{MAX_RETRIES_PER_BATCH}): {e}")
                print(f"Full traceback for this error:\n{traceback.format_exc()}")
                if attempt < MAX_RETRIES_PER_BATCH:
                    print(f"Retrying batch in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"Max retries ({MAX_RETRIES_PER_BATCH}) reached for batch {i+1}. Marking as failed.")
            except Exception as e:
                # Catch any other unexpected errors (e.g., programming bugs)
                print(f"\nAn unexpected CRITICAL ERROR occurred during mapping batch {i+1}/{num_batches}: {e}")
                print(f"Full traceback for this error:\n{traceback.format_exc()}")
                break # Don't retry for unexpected errors, treat as a hard failure

        if not success:
            # If all retries failed for a batch, add all IDs from this batch to cumulative_failed_ids
            print(f"CRITICAL: Batch {i+1}/{num_batches} failed after {MAX_RETRIES_PER_BATCH} attempts. All {len(batch_ids)} IDs in this batch will be marked as unmapped.")
            cumulative_failed_ids.update(batch_ids)

        time.sleep(1) # Small delay between batches even after retries

    # --- END BATCHING & RETRY LOGIC ---

    if all_mappings_sets:
        print(f"\nMapping successful for {len(all_mappings_sets)} unique input IDs across all batches.")
        if cumulative_failed_ids:
            print(f"INFO: The uniprot-id-mapper reported {len(cumulative_failed_ids)} IDs that failed to map during API calls (including those from failed batches).")
    else:
        print(f"No mappings found by uniprot-id-mapper for any of the provided IDs from '{from_db_name}'.")
        if cumulative_failed_ids:
            print(f"All {len(cumulative_failed_ids)} IDs failed to map according to uniprot-id-mapper.")

    final_mappings_for_return = {}
    for from_id, accessions_set in all_mappings_sets.items():
        if accessions_set:
            final_mappings_for_return[from_id] = ", ".join(sorted(list(accessions_set)))

    # Identify any IDs that were sent but not found in mappings nor explicitly reported as failed (should be rare with batch retries)
    unmapped_from_ids_not_in_results = [uid for uid in unique_ids_sent if uid not in final_mappings_for_return and uid not in cumulative_failed_ids]
    if unmapped_from_ids_not_in_results:
        print(f"\nDEBUG: The following {len(unmapped_from_ids_not_in_results)} IDs from '{from_db_name}' were sent to UniProt but no mappings were returned and were not explicitly marked as failed:")
        debug_limit = 100
        if len(unmapped_from_ids_not_in_results) > debug_limit and from_db_name == "RefSeq_Protein":
            for uid in unmapped_from_ids_not_in_results[:debug_limit]:
                print(f"  - {uid}")
            print(f"  ... (first {debug_limit} of {len(unmapped_from_ids_not_in_results)} unmapped IDs shown)")
        else:
            for uid in unmapped_from_ids_not_in_results:
                print(f"  - {uid}")

    print(f"\nCompleted UniProt ID mapping from '{from_db_name}' to '{to_db_name}'. Total unique input IDs mapped: {len(final_mappings_for_return)}.")
    total_individual_uniprot_ids = sum(len(v.split(', ')) for v in final_mappings_for_return.values()) if final_mappings_for_return else 0
    print(f"Total individual IDs found: {total_individual_uniprot_ids}.")

    return final_mappings_for_return

    
# --- Function: get_biogrid_interactors ---
def get_biogrid_interactors(
    target_entrez_ids: List[str],
    species_id: int,
    min_evidence_count: int,
    biogrid_access_key: str,
    exclude_entrez_ids: Set[str] = None
) -> Tuple[Set[str], Dict[str, int]]:
    """
    Retrieves interactors for a given list of Entrez Gene IDs from BioGRID.
    Performs a single API call for the provided list.
    Filters interactions by species and minimum evidence count.
    Excludes specified Entrez IDs from the final set of filtered interactors.

    Args:
        target_entrez_ids (List[str]): A list of Entrez Gene IDs to query BioGRID for.
                                       BioGRID will return all interactions involving ANY of these IDs.
        species_id (int): The NCBI Taxonomy ID for the species (e.g., 9606 for Homo sapiens).
        min_evidence_count (int): Minimum number of evidence pieces required for an interaction.
        biogrid_access_key (str): Your BioGRID access key.
        exclude_entrez_ids (Set[str], optional): A set of Entrez IDs to explicitly exclude
                                                 from the final filtered set of interactors.
                                                 Defaults to None (no exclusions).

    Returns:
        Tuple[Set[str], Dict[str, int]]:
            - A set of Entrez IDs of interactors that meet the evidence criteria
              and are not in the exclude_entrez_ids set.
            - A dictionary mapping all found interactors (before exclusion) to their evidence counts.
              This is useful for debugging or if counts are needed elsewhere.
    """
    if not target_entrez_ids:
        return set(), {}

    if exclude_entrez_ids is None:
        exclude_entrez_ids = set()

    target_gene_list_str = ",".join(target_entrez_ids)
    
    biogrid_url = "https://webservice.thebiogrid.org/interactions/"
    biogrid_params = {
        "accesskey": biogrid_access_key,
        "geneList": target_gene_list_str,
        "speciesId": species_id,
        "includeInteractors": "true",
        "idType": "ENTREZ_GENE",
        "format": "json"
    }

    interactor_evidence_counts_entrez = Counter()
    all_found_interactor_entrez_ids = set() # To store all interactors before filtering

    try:
        response = requests.get(biogrid_url, params=biogrid_params, timeout=60)
        response.raise_for_status()
        interactions_data = response.json()

        if not isinstance(interactions_data, dict):
            print(f"Error: BioGRID API did not return a dictionary for target {target_gene_list_str}. Response: {response.text}")
            return set(), {}

        if not interactions_data or (len(interactions_data) == 1 and "error" in interactions_data):
            if "error" in interactions_data:
                print(f"  API Error Message for target {target_gene_list_str}: {interactions_data['error']}")
            return set(), {}
        else:
            for top_level_value in interactions_data.values():
                interaction = top_level_value

                if not isinstance(interaction, dict):
                    print(f"Skipping unexpected entry in BioGRID interaction list for target {target_gene_list_str}.")
                    continue

                entrez_a = str(interaction.get('ENTREZ_GENE_A'))
                entrez_b = str(interaction.get('ENTREZ_GENE_B'))
                organism_a = interaction.get('ORGANISM_A')
                organism_b = interaction.get('ORGANISM_B')

                # Ensure both interactors are human and are valid Entrez IDs
                if organism_a != species_id or organism_b != species_id:
                    continue
                if not entrez_a or entrez_a == '-' or not entrez_b or entrez_b == '-':
                    continue

                is_a_target = entrez_a in target_entrez_ids
                is_b_target = entrez_b in target_entrez_ids
                
                # We are interested in interactors that interact with *any* of the target_entrez_ids
                if is_a_target and entrez_b not in target_entrez_ids:
                    interactor_evidence_counts_entrez[entrez_b] += 1
                    all_found_interactor_entrez_ids.add(entrez_b)
                if is_b_target and entrez_a not in target_entrez_ids:
                    interactor_evidence_counts_entrez[entrez_a] += 1
                    all_found_interactor_entrez_ids.add(entrez_a)
                # Handle self-interactions if the target interacts with itself
                if is_a_target and is_b_target and entrez_a == entrez_b:
                    interactor_evidence_counts_entrez[entrez_a] += 1 # Count self-interaction evidence
                    all_found_interactor_entrez_ids.add(entrez_a) # Add self to the list of potential interactors (will be filtered by exclude later)


            interactors_filtered_by_evidence = set()
            for interactor, count in interactor_evidence_counts_entrez.items():
                if count >= min_evidence_count:
                    interactors_filtered_by_evidence.add(interactor)
            
            # Apply exclusion filter AFTER evidence filtering
            final_filtered_interactor_entrez_ids = interactors_filtered_by_evidence - exclude_entrez_ids

            return final_filtered_interactor_entrez_ids, interactor_evidence_counts_entrez

    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred during BioGRID API call for target {target_gene_list_str}: {e}")
        print(f"Raw response content: {response.text}")
        print("Please check your BioGRID access key. An invalid key often leads to HTTP 401 Unauthorized.")
        return set(), {}
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from BioGRID API for target {target_gene_list_str}: {e}")
        print(f"Raw response content: {response.text}")
        print("The BioGRID API likely returned non-JSON content or malformed JSON. This can happen with invalid access keys.")
        return set(), {}
    except Exception as e:
        print(f"An unexpected error occurred during BioGRID API call or processing for target {target_gene_list_str}: {e}")
        print(traceback.format_exc())
        return set(), {}

# --- New Function: run_interaction_analysis_for_target ---
def run_interaction_analysis_for_target(
    target_entrez_id: str,
    species_id: int,
    min_evidence_count: int,
    biogrid_access_key: str
) -> Tuple[Set[str], Set[str]]:
    """
    Retrieves and maps primary and secondary interactors for a single target Entrez ID.
    Returns sets of UniProt IDs for primary and secondary interactors.
    """
    print(f"\n--- Analyzing interactors for target Entrez ID: {target_entrez_id} ---")

    # 1. Get primary interactors
    primary_interactors_entrez, _ = get_biogrid_interactors(
        target_entrez_ids=[target_entrez_id],
        species_id=species_id,
        min_evidence_count=min_evidence_count,
        biogrid_access_key=biogrid_access_key,
        exclude_entrez_ids=set()
    )
    primary_interactors_entrez.discard(target_entrez_id) # Exclude self-interaction from the list of primary interactors

    primary_interactors_uniprot = set()
    if primary_interactors_entrez:
        print(f"Mapping {len(primary_interactors_entrez)} primary interactors (Entrez IDs) to UniProtKB IDs for {target_entrez_id}...")
        mapped_primary_interactors = map_ids_via_uniprot(list(primary_interactors_entrez), "GeneID", "UniProtKB")
        for uniprot_id_str in mapped_primary_interactors.values():
            primary_interactors_uniprot.update(uid.strip() for uid in uniprot_id_str.split(',') if uid.strip())
        print(f"  Found {len(primary_interactors_uniprot)} unique UniProtKB IDs for primary interactors of {target_entrez_id}.")
    else:
        print(f"No filtered primary interactors found for {target_entrez_id}.")

    # 2. Get secondary interactors
    secondary_interactors_entrez_cumulative = set()
    if primary_interactors_entrez:
        print(f"\nSearching for secondary interactors by querying each of the {len(primary_interactors_entrez)} primary interactors individually for {target_entrez_id}...")
        
        # Exclude target itself and its direct primary interactors from secondary results
        genes_to_exclude_from_secondary_results = primary_interactors_entrez.union({target_entrez_id})
        
        for primary_interactor_entrez in tqdm(list(primary_interactors_entrez), desc=f"Querying primary interactors for {target_entrez_id}'s secondary interactions"):
            current_secondary_interactors_entrez_for_this_primary, _ = get_biogrid_interactors(
                target_entrez_ids=[primary_interactor_entrez],
                species_id=species_id,
                min_evidence_count=min_evidence_count,
                biogrid_access_key=biogrid_access_key,
                exclude_entrez_ids=genes_to_exclude_from_secondary_results
            )
            secondary_interactors_entrez_cumulative.update(current_secondary_interactors_entrez_for_this_primary)
            time.sleep(1) # Be polite to the BioGRID API

    secondary_interactors_uniprot = set()
    if secondary_interactors_entrez_cumulative:
        print(f"\nMapping {len(secondary_interactors_entrez_cumulative)} secondary interactors (Entrez IDs) to UniProtKB IDs for {target_entrez_id}...")
        mapped_secondary_interactors = map_ids_via_uniprot(list(secondary_interactors_entrez_cumulative), "GeneID", "UniProtKB")
        for uniprot_id_str in mapped_secondary_interactors.values():
            secondary_interactors_uniprot.update(uid.strip() for uid in uniprot_id_str.split(',') if uid.strip())
        print(f"  Found {len(secondary_interactors_uniprot)} unique UniProtKB IDs for secondary interactors of {target_entrez_id}.")
    else:
        print(f"No filtered secondary interactors found for {target_entrez_id}.")

    return primary_interactors_uniprot, secondary_interactors_uniprot

# --- Function: check_for_interaction ---
def check_for_interaction(uniprot_ids_str: str, primary_set: Set[str], secondary_set: Set[str]) -> str:
    """
    Checks if a protein (identified by its UniProt IDs) is a primary or secondary interactor.
    Prioritizes 'primary' if it matches both.
    """
    if not isinstance(uniprot_ids_str, str) or not uniprot_ids_str.strip():
        return 'no interaction'

    current_protein_uniprot_ids = set(uid.strip() for uid in uniprot_ids_str.split(',') if uid.strip())

    if current_protein_uniprot_ids.intersection(primary_set):
        return 'primary'
    elif current_protein_uniprot_ids.intersection(secondary_set):
        return 'secondary'
    else:
        return 'no interaction'

# --- Interaction status to score mapping for family analysis ---
INTERACTION_SCORES = {
    'no interaction': 0,
    'secondary': 1,
    'primary': 2
}
SCORE_TO_STATUS = {v: k for k, v in INTERACTION_SCORES.items()}


# --- Main Script Logic ---
if __name__ == "__main__":
    print("--- Starting Interaction Analysis Script ---")

    # 1. Pre-process FAMILIES_CONFIG to get all unique target UniProt and Entrez IDs
    all_target_uniprot_ids_from_config = set()
    all_target_entrez_ids_from_config = set()
    uniprot_to_entrez_lookup = {} # For individual targets and mapping family members

    # Add individual targets
    for up_id, entrez_id in FAMILIES_CONFIG["individual_targets"]:
        all_target_uniprot_ids_from_config.add(up_id)
        all_target_entrez_ids_from_config.add(entrez_id)
        uniprot_to_entrez_lookup[up_id] = entrez_id # Assuming 1:1 for individual targets from config

    # Add family members (UniProt IDs)
    for family_name, uniprot_ids_list in FAMILIES_CONFIG["families"].items():
        all_target_uniprot_ids_from_config.update(uniprot_ids_list)

    # Map all unique UniProt IDs (from families and individual targets) to their Entrez IDs if not already known
    print("\nMapping all necessary UniProt IDs to Entrez Gene IDs for BioGRID querying...")
    uniprot_to_entrez_map_for_query = map_ids_via_uniprot(list(all_target_uniprot_ids_from_config), "UniProtKB_AC-ID", "GeneID")
    
    # Update master uniprot_to_entrez_lookup, handling multiple Entrez for one UniProt by taking the first one (BioGRID queries often need specific IDs)
    # A more robust solution might handle this ambiguity or require clearer input if multiple Entrez IDs are possible per UniProt.
    # For now, we'll try to use the most common mapping or just the first if multiple are returned for a UniProt.
    for up_id, entrez_id_str in uniprot_to_entrez_map_for_query.items():
        if up_id not in uniprot_to_entrez_lookup: # Only add if not already explicitly defined in individual_targets
            first_entrez = entrez_id_str.split(', ')[0]
            uniprot_to_entrez_lookup[up_id] = first_entrez
            all_target_entrez_ids_from_config.add(first_entrez) # Add to the master list of Entrez IDs to query

    print(f"Consolidated {len(all_target_uniprot_ids_from_config)} unique UniProt IDs, and {len(all_target_entrez_ids_from_config)} unique Entrez IDs for analysis.")

    # Store interactors for each target Entrez ID
    all_primary_interactors_uniprot_per_target: Dict[str, Set[str]] = {}
    all_secondary_interactors_uniprot_per_target: Dict[str, Set[str]] = {}

    # 2. Iterate through all unique target Entrez IDs to fetch their interactors
    # This loop will ensure interactors for all proteins listed in config (individual targets & family members) are found.
    print("\n--- Fetching primary and secondary interactors from BioGRID for all unique target proteins ---")
    for target_entrez_id in tqdm(list(all_target_entrez_ids_from_config), desc="Overall BioGRID Interaction Fetch"):
        if target_entrez_id:
            primary_uniprot, secondary_uniprot = run_interaction_analysis_for_target(
                target_entrez_id, SPECIES_ID, MIN_EVIDENCE_COUNT, BIOGRID_ACCESS_KEY
            )
            all_primary_interactors_uniprot_per_target[target_entrez_id] = primary_uniprot
            all_secondary_interactors_uniprot_per_target[target_entrez_id] = secondary_uniprot
        time.sleep(1) # Small delay between full target analyses

    # 3. Load and process your protein list (mapping RefSeq to UniProtKB IDs)
    print(f"\nProcessing {INPUT_CSV_FILE}...")
    try:
        df = pd.read_csv(INPUT_CSV_FILE)
    except FileNotFoundError:
        print(f"Error: The file '{INPUT_CSV_FILE}' was not found.")
        exit()

    if 'ID' not in df.columns:
        print(f"Error: The input CSV '{INPUT_CSV_FILE}' must contain an 'ID' column.")
        exit()

    refseq_ids_from_file = df['ID'].unique().tolist()
    # This is the call that was failing due to large size, now it will be batched
    refseq_to_uniprot_map = map_ids_via_uniprot(refseq_ids_from_file, "RefSeq_Protein", "UniProtKB")
    df['UniProt_ID'] = df['ID'].map(refseq_to_uniprot_map).fillna('')

    df_for_interaction_check = df[df['UniProt_ID'].astype(str).str.strip() != ''].copy()
    unmapped_count = len(df) - len(df_for_interaction_check)
    if unmapped_count > 0:
        print(f"Warning: {unmapped_count} protein IDs from your input file could not be mapped to valid UniProt IDs and will be marked as 'no interaction'.")

    # 4. Annotate individual target interactions
    print("\n--- Annotating individual target interactions ---")
    for target_up_id, target_entrez_id in FAMILIES_CONFIG["individual_targets"]:
        column_name = f'Interaction_with_{target_up_id}'
        
        # Ensure interactors were found for this target
        primary_set = all_primary_interactors_uniprot_per_target.get(target_entrez_id, set())
        secondary_set = all_secondary_interactors_uniprot_per_target.get(target_entrez_id, set())

        df[column_name] = df['UniProt_ID'].apply(lambda x: check_for_interaction(x, primary_set, secondary_set))
        print(f"Added column: '{column_name}'")

    # 5. Annotate family-based interactions
    print("\n--- Annotating family-based interactions ---")
    for family_name, family_uniprot_members in FAMILIES_CONFIG["families"].items():
        print(f"\nProcessing family: {family_name} with members: {family_uniprot_members}")
        
        # Map family UniProt members to their Entrez IDs for lookup
        family_entrez_members = []
        for up_member in family_uniprot_members:
            entrez_member = uniprot_to_entrez_lookup.get(up_member)
            if entrez_member:
                family_entrez_members.append(entrez_member)
            else:
                print(f"Warning: UniProt ID {up_member} from family {family_name} could not be mapped to an Entrez ID. Skipping this member for family analysis.")
        
        if not family_entrez_members:
            print(f"No valid Entrez IDs for family {family_name}. Skipping family interaction analysis.")
            continue

        # Initialize temporary columns for scores
        for up_member, entrez_member in zip(family_uniprot_members, family_entrez_members):
            temp_col_name = f'temp_score_{up_member}'
            primary_set = all_primary_interactors_uniprot_per_target.get(entrez_member, set())
            secondary_set = all_secondary_interactors_uniprot_per_target.get(entrez_member, set())
            
            df[temp_col_name] = df['UniProt_ID'].apply(
                lambda x: INTERACTION_SCORES[check_for_interaction(x, primary_set, secondary_set)]
            )

        # Calculate family-id_intersection
        def _calculate_intersection_status(row, family_uniprot_members, family_entrez_members, temp_prefix='temp_score_'):
            scores = []
            for up_member in family_uniprot_members:
                temp_col_name = f'{temp_prefix}{up_member}'
                if temp_col_name in row:
                    scores.append(row[temp_col_name])
            
            if not scores: # Should not happen if temp columns are created, but defensive
                return 'no interaction'

            min_score = min(scores)
            return SCORE_TO_STATUS[min_score]

        df[f'{family_name}_intersection'] = df.apply(
            lambda row: _calculate_intersection_status(row, family_uniprot_members, family_entrez_members), axis=1
        )
        print(f"Added column: '{family_name}_intersection'")

        # Calculate family-id_combined
        def _calculate_combined_status(row, family_uniprot_members, family_entrez_members, temp_prefix='temp_score_'):
            scores = []
            for up_member in family_uniprot_members:
                temp_col_name = f'{temp_prefix}{up_member}'
                if temp_col_name in row:
                    scores.append(row[temp_col_name])
            
            if not scores: # Should not happen
                return 'no interaction'

            max_score = max(scores)
            return SCORE_TO_STATUS[max_score]

        df[f'{family_name}_combined'] = df.apply(
            lambda row: _calculate_combined_status(row, family_uniprot_members, family_entrez_members), axis=1
        )
        print(f"Added column: '{family_name}_combined'")

        # Drop temporary score columns
        for up_member in family_uniprot_members:
            temp_col_name = f'temp_score_{up_member}'
            if temp_col_name in df.columns:
                df = df.drop(columns=[temp_col_name])

    # 6. Save the final CSV file
    final_output_csv_file = os.path.join(OUTPUT_DIR, 'human-proteom-interactions_multi_target_and_families.csv')
    df.to_csv(final_output_csv_file, index=False)
    print(f"\nProcessing complete! Results saved to '{final_output_csv_file}'.")
    print("\nFirst few rows of the output file:")
    print(df.head())