import pandas as pd
import requests
import time
from collections import Counter
import json
import traceback
from UniProtMapper import ProtMapper
from typing import Set, Dict, List, Tuple
from tqdm import tqdm

# --- Configuration ---
BIOGRID_ACCESS_KEY = "6a8be2d9ffdf1ad752599e23c0f5963b"  # IMPORTANT: Replace with your actual BioGRID access key!
P53_UNIPROT_ID = "P04637"  # Human P53 (TP53) UniProt ID (used for reference, not direct query in this flow)
P53_ENTREZ_ID = "7157"  # Human P53 (TP53) Entrez Gene ID (used for BioGRID query)
SPECIES_ID = 9606  # Homo sapiens (human)
MIN_EVIDENCE_COUNT = 3  # Minimum number of evidence pieces required for an interaction
INPUT_CSV_FILE = 'data/Human-proteom_GCF_000001405.40.csv'
OUTPUT_CSV_FILE = 'data/human-proteom-interactions.csv'
BIOGRID_PRIMARY_MAPPED_OUTPUT_CSV_FILE = 'data/biogrid-p53-primary-interactors-uniprot-mapped.csv' # Output file for BioGRID primary interactors
BIOGRID_SECONDARY_MAPPED_OUTPUT_CSV_FILE = 'data/biogrid-p53-secondary-interactors-uniprot-mapped.csv' # New output file for BioGRID secondary interactors

# UniProt API base URL (not directly used by ProtMapper, but kept for context if needed)
UNIPROT_ID_MAPPING_URL = "https://rest.uniprot.org/idmapping"

# --- Function: map_ids_via_uniprot (uses uniprot-id-mapper) ---
def map_ids_via_uniprot(id_list: List[str], from_db_name: str, to_db_name: str) -> Dict[str, str]:
    """
    Maps a list of IDs from a specified source database to UniProtKB IDs
    using the uniprot-id-mapper package.
    Handles multiple UniProtKB mappings for a single input ID by collecting all
    unique accessions into a comma-separated string.

    Args:
        id_list (list): A list of IDs to be mapped (e.g., RefSeq Protein IDs, GeneIDs).
        from_db_name (str): The name of the source database (e.g., "GeneID", "RefSeq_Protein").
        to_db_name (str): The name of the target database (e.g., "UniProtKB").

    Returns:
        dict: A dictionary mapping original ID to a comma-separated string of its mapped UniProtKB IDs.
              Returns an empty dictionary if no mappings are found or if an error occurs.
    """
    if not id_list:
        return {}

    unique_ids_sent = list(set(id_list))
    print(f"\nAttempting to map {len(unique_ids_sent)} unique IDs from '{from_db_name}' to '{to_db_name}' via uniprot-id-mapper...")

    mapper = ProtMapper()

    all_mappings_sets = {} # To store from_id -> set of UniProtKB accessions

    try:
        mapped_df, failed_ids = mapper.get(
            ids=unique_ids_sent,
            from_db=from_db_name,
            to_db=to_db_name
        )
        
        if not mapped_df.empty:
            for original_id, uniprot_id in zip(mapped_df['From'], mapped_df['Entry']):
                all_mappings_sets.setdefault(original_id, set()).add(uniprot_id)

            print(f"Mapping successful for {len(all_mappings_sets)} unique input IDs.")
            if failed_ids:
                print(f"INFO: The uniprot-id-mapper reported {len(failed_ids)} IDs that failed to map during the API call.")
        else:
            print(f"No mappings found by uniprot-id-mapper for any of the provided IDs from '{from_db_name}'.")
            if failed_ids:
                print(f"All {len(failed_ids)} IDs failed to map according to uniprot-id-mapper.")

    except Exception as e:
        print(f"An **CRITICAL ERROR** occurred during mapping with uniprot-id-mapper for '{from_db_name}':")
        print(traceback.format_exc())
        return {}

    final_mappings_for_return = {}
    for from_id, accessions_set in all_mappings_sets.items():
        if accessions_set:
            final_mappings_for_return[from_id] = ", ".join(sorted(list(accessions_set)))

    unmapped_from_ids_not_in_results = [uid for uid in unique_ids_sent if uid not in final_mappings_for_return]
    if unmapped_from_ids_not_in_results:
        print(f"\nDEBUG: The following {len(unmapped_from_ids_not_in_results)} IDs from '{from_db_name}' were sent to UniProt but no mappings were returned by uniprot-id-mapper:")
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
    print(f"Total individual UniProt IDs found: {total_individual_uniprot_ids}.")

    return final_mappings_for_return

# --- NEW Function: get_biogrid_interactors (single API call per target_entrez_ids list) ---
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
                                       This list should be appropriately sized for a single API call.
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
        # print("Warning: No target Entrez IDs provided for BioGRID query.") # Too chatty for individual calls in loop
        return set(), {}

    if exclude_entrez_ids is None:
        exclude_entrez_ids = set()

    # print(f"Querying BioGRID for interactors of target Entrez ID(s): {target_entrez_ids}...") # Too chatty

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
        response = requests.get(biogrid_url, params=biogrid_params, timeout=60) # Increased timeout to 60s
        response.raise_for_status()
        interactions_data = response.json()

        if not isinstance(interactions_data, dict):
            # print("Error: BioGRID API did not return a dictionary. This usually means an invalid access key or API error.") # Too chatty
            # print("Raw response content (if available):", response.text) # Too chatty
            return set(), {}

        if not interactions_data or (len(interactions_data) == 1 and "error" in interactions_data):
            # print("No interactions found or BioGRID API returned an error:") # Too chatty
            if "error" in interactions_data:
                print(f"  API Error Message for target {target_gene_list_str}: {interactions_data['error']}")
            return set(), {}
        else:
            for top_level_value in interactions_data.values():
                interaction = top_level_value

                if not isinstance(interaction, dict):
                    print(f"Skipping unexpected entry in BioGRID interaction list for target {target_gene_list_str}.")
                    continue

                entrez_a = interaction.get('ENTREZ_GENE_A')
                entrez_b = interaction.get('ENTREZ_GENE_B')
                organism_a = interaction.get('ORGANISM_A')
                organism_b = interaction.get('ORGANISM_B')

                # Ensure both interactors are human and are valid Entrez IDs
                if organism_a != species_id or organism_b != species_id:
                    continue
                if not entrez_a or entrez_a == '-' or not entrez_b or entrez_b == '-':
                    continue

                is_a_target = entrez_a in target_entrez_ids
                is_b_target = entrez_b in target_entrez_ids
                
                # We are interested in interactors (entrez_b or entrez_a) that are NOT in the target_entrez_ids list for "secondary" search
                # but for evidence counting, we count any interaction involving one of our target genes.
                if is_a_target: 
                    interactor_evidence_counts_entrez[entrez_b] += 1
                    all_found_interactor_entrez_ids.add(entrez_b)
                if is_b_target and entrez_a != entrez_b: # Avoid double counting if already handled by 'is_a_target'
                    interactor_evidence_counts_entrez[entrez_a] += 1
                    all_found_interactor_entrez_ids.add(entrez_a)
                elif is_a_target and is_b_target and entrez_a == entrez_b: # self-interaction of a target gene
                     interactor_evidence_counts_entrez[entrez_a] += 1
                     all_found_interactor_entrez_ids.add(entrez_a)

            # print(f"  Found {len(all_found_interactor_entrez_ids)} unique potential interactors (Entrez IDs) before evidence and exclusion filtering for {target_gene_list_str}.") # Too chatty

            interactors_filtered_by_evidence = set()
            for interactor, count in interactor_evidence_counts_entrez.items():
                if count >= min_evidence_count:
                    interactors_filtered_by_evidence.add(interactor)
            
            # Apply exclusion filter AFTER evidence filtering
            final_filtered_interactor_entrez_ids = interactors_filtered_by_evidence - exclude_entrez_ids

            # print(f"  After filtering for >= {min_evidence_count} pieces of evidence and excluding {len(exclude_entrez_ids)} specified IDs, {len(final_filtered_interactor_entrez_ids)} unique interactors (Entrez IDs) remain for {target_gene_list_str}.") # Too chatty
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
        print(traceback.format_exc()) # Added for better debugging
        return set(), {}

# --- 1. Get all known human P53 primary interactors from BioGRID and count evidence ---
print(f"Retrieving P53 primary interactors from BioGRID (Entrez ID: {P53_ENTREZ_ID})...")

# Get primary interactors of P53. No exclusions for interactors (P53 self-interaction is handled by discard).
# The target_entrez_ids list contains only P53 for this query.
p53_primary_interactors_entrez_ids, _ = get_biogrid_interactors(
    target_entrez_ids=[P53_ENTREZ_ID], # Query for P53 only
    species_id=SPECIES_ID,
    min_evidence_count=MIN_EVIDENCE_COUNT,
    biogrid_access_key=BIOGRID_ACCESS_KEY,
    exclude_entrez_ids=set() # No exclusions for initial query interactors
)
# We want a list of *distinct* proteins that interact with P53. P53 interacting with itself isn't a "different" interactor.
p53_primary_interactors_entrez_ids.discard(P53_ENTREZ_ID) 

# --- Map filtered BioGRID primary interactors (Entrez IDs) to UniProtKB IDs ---
p53_primary_interactors_all_uniprot_ids = set()
entrez_to_uniprot_primary_interactor_map_full = {}

if p53_primary_interactors_entrez_ids:
    print(f"\nMapping {len(p53_primary_interactors_entrez_ids)} filtered BioGRID primary interactors (Entrez IDs) to UniProtKB IDs...")
    entrez_to_uniprot_primary_interactor_map = map_ids_via_uniprot(list(p53_primary_interactors_entrez_ids), "GeneID", "UniProtKB")

    entrez_to_uniprot_primary_interactor_map_full = entrez_to_uniprot_primary_interactor_map

    for uniprot_id_string in entrez_to_uniprot_primary_interactor_map.values():
        for uniprot_id in uniprot_id_string.split(', '):
            if uniprot_id:
                p53_primary_interactors_all_uniprot_ids.add(uniprot_id)

    print(f"Successfully mapped BioGRID primary interactors to {len(p53_primary_interactors_all_uniprot_ids)} unique UniProtKB IDs.")

    # --- Save the BioGRID mapped primary IDs to a CSV file ---
    if entrez_to_uniprot_primary_interactor_map_full:
        biogrid_primary_mapped_df = pd.DataFrame(entrez_to_uniprot_primary_interactor_map_full.items(), columns=['GeneID', 'UniProt_IDs'])
        biogrid_primary_mapped_df.to_csv(BIOGRID_PRIMARY_MAPPED_OUTPUT_CSV_FILE, index=False)
        print(f"Mapped BioGRID primary interactors saved to '{BIOGRID_PRIMARY_MAPPED_OUTPUT_CSV_FILE}'.")
    else:
        print("No BioGRID primary interactors were successfully mapped to UniProt, so no file will be saved.")
else:
    print("No filtered P53 primary interactors from BioGRID, so skipping mapping primary interactors to UniProtKB IDs.")


# --- 2. Get all known human P53 secondary interactors from BioGRID and count evidence ---
secondary_interactors_all_uniprot_ids = set()
entrez_to_uniprot_secondary_interactor_map_full = {}
secondary_interactors_filtered_entrez_ids_cumulative = set() # Use a cumulative set for all secondary interactors

if p53_primary_interactors_entrez_ids:
    print(f"\nSearching for secondary interactors by querying each of the {len(p53_primary_interactors_entrez_ids)} primary interactors individually...")
    
    # Define the set of genes to exclude from the secondary interaction results.
    # This includes P53 itself and all identified primary interactors,
    # to ensure that secondary interactors are distinct from P53 and primary interactors.
    genes_to_exclude_from_secondary_results = p53_primary_interactors_entrez_ids.union({P53_ENTREZ_ID})
    
    # Loop through each primary interactor Entrez ID
    for primary_interactor_entrez_id in tqdm(list(p53_primary_interactors_entrez_ids), desc="Querying primary interactors for secondary interactions"):
        # Query BioGRID for interactors of the CURRENT primary interactor
        current_secondary_interactors_entrez_for_this_primary, _ = get_biogrid_interactors(
            target_entrez_ids=[primary_interactor_entrez_id], # Query for only ONE primary interactor at a time
            species_id=SPECIES_ID,
            min_evidence_count=MIN_EVIDENCE_COUNT,
            biogrid_access_key=BIOGRID_ACCESS_KEY,
            exclude_entrez_ids=genes_to_exclude_from_secondary_results
        )
        # Add found secondary interactors to the cumulative set
        secondary_interactors_filtered_entrez_ids_cumulative.update(current_secondary_interactors_entrez_for_this_primary)
        
        time.sleep(1) # Be polite to the BioGRID API between individual queries

    if secondary_interactors_filtered_entrez_ids_cumulative:
        print(f"\nFound a total of {len(secondary_interactors_filtered_entrez_ids_cumulative)} unique filtered BioGRID secondary interactors (Entrez IDs).")
        print(f"\nMapping {len(secondary_interactors_filtered_entrez_ids_cumulative)} filtered BioGRID secondary interactors (Entrez IDs) to UniProtKB IDs...")
        
        secondary_entrez_to_uniprot_interactor_map = map_ids_via_uniprot(list(secondary_interactors_filtered_entrez_ids_cumulative), "GeneID", "UniProtKB")

        entrez_to_uniprot_secondary_interactor_map_full = secondary_entrez_to_uniprot_interactor_map

        for uniprot_id_string in secondary_entrez_to_uniprot_interactor_map.values():
            for uniprot_id in uniprot_id_string.split(', '):
                if uniprot_id:
                    secondary_interactors_all_uniprot_ids.add(uniprot_id)

        print(f"Successfully mapped BioGRID secondary interactors to {len(secondary_interactors_all_uniprot_ids)} unique UniProtKB IDs.")

        # --- Save the BioGRID mapped secondary IDs to a CSV file ---
        if entrez_to_uniprot_secondary_interactor_map_full:
            biogrid_secondary_mapped_df = pd.DataFrame(entrez_to_uniprot_secondary_interactor_map_full.items(), columns=['GeneID', 'UniProt_IDs'])
            biogrid_secondary_mapped_df.to_csv(BIOGRID_SECONDARY_MAPPED_OUTPUT_CSV_FILE, index=False)
            print(f"Mapped BioGRID secondary interactors saved to '{BIOGRID_SECONDARY_MAPPED_OUTPUT_CSV_FILE}'.")
        else:
            print("No BioGRID secondary interactors were successfully mapped to UniProt, so no file will be saved.")
    else:
        print("No filtered P53 secondary interactors found from BioGRID.")
else:
    print("No primary P53 interactors found, so skipping search for secondary interactors.")


# --- 3. Process your protein list (MAPPING REFSEQ TO UNIPROTKB IDS) ---
print(f"\nProcessing {INPUT_CSV_FILE}...")
try:
    df = pd.read_csv(INPUT_CSV_FILE)
    # Uncomment the line below for faster testing on a subset of your data
    # df = df.head(100) 
except FileNotFoundError:
    print(f"Error: The file '{INPUT_CSV_FILE}' was not found.")
    exit()

if 'ID' not in df.columns:
    print(f"Error: The input CSV '{INPUT_CSV_FILE}' must contain an 'ID' column.")
    exit()

refseq_ids_from_file = df['ID'].unique().tolist()

refseq_to_uniprot_map = map_ids_via_uniprot(refseq_ids_from_file, "RefSeq_Protein", "UniProtKB")

df['UniProt_ID'] = df['ID'].map(refseq_to_uniprot_map)

df['UniProt_ID'] = df['UniProt_ID'].fillna('')

# We check for interaction only on rows that have a UniProt_ID
df_for_interaction_check = df[df['UniProt_ID'].astype(str).str.strip() != ''].copy()

unmapped_count = len(df) - len(df_for_interaction_check)
if unmapped_count > 0:
    print(f"Warning: {unmapped_count} protein IDs from your input file could not be mapped to valid UniProt IDs and will be marked as 'no interaction'.")

df['P53_Interaction'] = 'no interaction' # Initialize column

# --- Update interaction check logic to include secondary interactors ---
def check_for_interaction(uniprot_ids_str: str, primary_set: Set[str], secondary_set: Set[str]) -> str:
    """
    Checks if a protein (identified by its UniProt IDs) is a primary or secondary interactor.
    Prioritizes 'primary' if it matches both.
    """
    if not uniprot_ids_str.strip():
        return 'no interaction'

    current_protein_uniprot_ids = set(uid.strip() for uid in uniprot_ids_str.split(',') if uid.strip())

    if current_protein_uniprot_ids.intersection(primary_set):
        return 'primary'
    elif current_protein_uniprot_ids.intersection(secondary_set):
        return 'secondary'
    else:
        return 'no interaction'

# Apply the updated interaction check
if not df_for_interaction_check.empty:
    df['P53_Interaction'] = df['UniProt_ID'].apply(lambda x: check_for_interaction(x, p53_primary_interactors_all_uniprot_ids, secondary_interactors_all_uniprot_ids))
else:
    print("No RefSeq IDs from your input file could be successfully mapped to UniProt IDs, so all proteins will be marked as 'no interaction'.")


# --- 4. Write the new CSV file ---
final_df = df

final_df.to_csv(OUTPUT_CSV_FILE, index=False)
print(f"\nProcessing complete! Results saved to '{OUTPUT_CSV_FILE}'.")
print("First few rows of the output file:")
print(final_df.head())