import pandas as pd
import requests
import time
import io
import re
import json
import os
from collections import defaultdict
from tqdm.notebook import tqdm as default_tqdm

# --- Configuration ---
BIOGRID_FILE = 'data/BIOGRID-ORGANISM-Homo_sapiens-4.4.249.tsv' # Path to your BioGRID file
OUTPUT_CSV_FILE = 'data/biogrid_interacting_protein_sequences.csv'

# Checkpoint directory and file paths
CHECKPOINT_DIR = 'checkpoints'
BIOGRID_XREFS_CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, 'biogrid_xrefs_checkpoint.json')
FINAL_UNIPROT_MAP_CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, 'final_uniprot_map_checkpoint.json')

# Your email address (recommended for UniProt API requests)
YOUR_EMAIL = "phitro@bu.edu" # Change this to your actual email
# BioGRID Access Key
BIOGRID_ACCESS_KEY = "6a8be2d9ffdf1ad752599e23c0f5963b"

# Define a User-Agent header for all requests
HEADERS = {
    "User-Agent": f"PythonScriptForBioGRIDParsing/1.0 ({YOUR_EMAIL})"
}

# --- UniProt API Endpoints ---
UNIPROT_INFO_API_URL = "https://rest.uniprot.org/uniprotkb/{}.json"
UNIPROT_SEQUENCE_API_URL = "https://rest.uniprot.org/uniprotkb/{}.fasta"
UNIPROT_SEARCH_API_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_TAXONOMY_URL = "https://rest.uniprot.org/taxonomy/{taxId}.json" # Correct for keyword formatting


# BioGRID API Endpoint for Interactions (THIS IS THE WORKING ENDPOINT)
BIOGRID_INTERACTIONS_API_URL = "https://webservice.thebiogrid.org/interactions/?"


# --- List of definitive physical protein-protein interaction types ---
PHYSICAL_INTERACTION_TYPES = [
    'Two-hybrid',
    'Affinity Capture-Luminescence',
    'Affinity Capture-Western',
    'Affinity Capture-MS',
    'Reconstituted Complex',
    'FRET',
    'Co-purification',
    'Protein-peptide',
    'Co-crystal Structure',
    'Far Western',
    'Co-fractionation'
]

# --- Caches for API responses to avoid redundant calls ---
sequence_cache = {}
uniprot_search_results_cache = {} # Cache for raw search results
uniprot_partner_select_cache = {} # Cache for (identifier_to_search, taxon_id, id_type_hint, original_input_id) -> uniprot_acc
organism_info_cache = {} # Cache for UniProt entry info including organism name and full protein name


# --- Checkpoint Helper Functions ---

def _serialize_complex_data(data):
    if isinstance(data, defaultdict):
        return {
            (f"{k[0]}::{k[1]}" if isinstance(k, tuple) else k): _serialize_complex_data(v)
            for k, v in dict(data).items()
        }
    elif isinstance(data, dict):
        return {
            (f"{k[0]}::{k[1]}" if isinstance(k, tuple) else k): _serialize_complex_data(v)
            for k, v in data.items()
        }
    elif isinstance(data, set):
        return sorted(list(data))
    elif isinstance(data, list):
        return [_serialize_complex_data(item) for item in data]
    else:
        return data

def _deserialize_complex_data(data):
    if isinstance(data, dict):
        new_dict = {}
        for k, v in data.items():
            deserialized_v = _deserialize_complex_data(v)
            if isinstance(k, str) and '::' in k:
                k = tuple(k.split('::'))
            new_dict[k] = deserialized_v
        
        if all(isinstance(val, dict) for val in new_dict.values()):
            if all(isinstance(inner_val, set) for val_dict in new_dict.values() for inner_val in val_dict.values()):
                outer_default = defaultdict(lambda: defaultdict(set))
                for k_outer, v_inner_dict in new_dict.items():
                    outer_default[k_outer] = defaultdict(set, v_inner_dict)
                return outer_default
            else:
                return {k: defaultdict(set, v) if all(isinstance(val, set) for val in v.values()) else v for k,v in new_dict.items()}
        elif all(isinstance(val, set) for val in new_dict.values()):
            return defaultdict(set, new_dict)
        elif all(isinstance(val, str) for val in new_dict.values()):
            return defaultdict(str, new_dict)
        return new_dict
    elif isinstance(data, list):
        if all(isinstance(item, str) for item in data):
            return set(data)
        return [_deserialize_complex_data(item) for item in data]
    else:
        return data

def save_checkpoint(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(_serialize_complex_data(data), f, indent=4)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        print(f"Checkpoint loaded from {filename}")
        return _deserialize_complex_data(data)
    print(f"No checkpoint found at {filename}")
    return None


# --- UniProt API Helper Functions (from your working script, adapted) ---

def clean_uniprot_id(uniprot_id):
    """Ensures UniProt ID is canonical (e.g., P05221-1 -> P05221)."""
    if not uniprot_id or pd.isna(uniprot_id) or not isinstance(uniprot_id, str):
        return ""
    return uniprot_id.split('-')[0].strip()

def get_uniprot_info(uniprot_id):
    """Fetches UniProt entry details including organism name and full protein name."""
    uniprot_id_clean = clean_uniprot_id(uniprot_id)
    if not uniprot_id_clean:
        return None

    if uniprot_id_clean in organism_info_cache:
        return organism_info_cache[uniprot_id_clean]

    try:
        response = requests.get(UNIPROT_INFO_API_URL.format(uniprot_id_clean), timeout=10)
        response.raise_for_status()
        data = response.json()
        organism_name = data.get('organism', {}).get('scientificName', 'Unknown organism')
        protein_full_name = data.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', '')
        
        info = {'organism_name': organism_name, 'full_protein_name': protein_full_name}
        organism_info_cache[uniprot_id_clean] = info
        time.sleep(0.1) # Be polite to the UniProt API
        return info
    except requests.exceptions.RequestException as e:
        print(f"  Error fetching UniProt info for {uniprot_id_clean}: {e}")
        return None

def get_protein_sequence(uniprot_id_or_acc):
    """Fetches protein sequence from UniProt API."""
    uniprot_id_clean = clean_uniprot_id(uniprot_id_or_acc)
    if not uniprot_id_clean:
        return ""
    
    if uniprot_id_clean in sequence_cache:
        return sequence_cache[uniprot_id_clean]

    try:
        response = requests.get(UNIPROT_SEQUENCE_API_URL.format(uniprot_id_clean), timeout=10)
        response.raise_for_status()
        fasta_content = response.text
        sequence = "".join(fasta_content.splitlines()[1:]).strip()
        if not sequence:
            print(f"  Warning: No sequence content for {uniprot_id_clean} from UniProt.")
        sequence_cache[uniprot_id_clean] = sequence
        time.sleep(0.1) # Be polite to the UniProt API
        return sequence
    except requests.exceptions.RequestException as e:
        print(f"  Error fetching sequence for {uniprot_id_clean} from UniProt: {e}")
        return ""

def get_organism_names(tax_ids_list):
    """
    Retrieves scientific names for a list of NCBI Taxonomy IDs using the UniProt Taxonomy API.
    Returns a dictionary: {tax_id: organism_name}
    """
    if not tax_ids_list:
        print("Skipping organism name retrieval as no Taxonomy IDs provided.")
        return {}

    print(f"Retrieving organism names for {len(tax_ids_list)} unique taxonomy IDs from UniProt Taxonomy API...")
    
    organism_name_map = {}
    _tqdm = default_tqdm 

    for tax_id in _tqdm(tax_ids_list, desc="Fetching organism names"):
        tax_id_str = str(int(tax_id)) if pd.notna(tax_id) else None
        if not tax_id_str:
            continue

        try:
            # CORRECTED UniProt Taxonomy API endpoint: https://rest.uniprot.org/taxonomy/{taxId}.json
            response = requests.get(UNIPROT_TAXONOMY_URL.format(taxId=tax_id_str), headers={"Accept": "application/json"})
            response.raise_for_status()
            data = response.json()
            organism_name_map[tax_id_str] = data.get('scientificName', f"Unknown (Tax ID: {tax_id_str})")
        except requests.exceptions.RequestException as e:
            print(f"WARNING: Error fetching organism name for Tax ID {tax_id_str} from UniProt Taxonomy API: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"   Response status code: {e.response.status_code}")
                print(f"   Response text: {e.response.text}")
            organism_name_map[tax_id_str] = f"Error (Tax ID: {tax_id_str})"
        time.sleep(0.5) 
        
    print(f"Successfully retrieved {len(organism_name_map)} organism names.")
    return organism_name_map


def search_uniprot_for_accession(identifier_to_search, taxon_id, id_type_hint=None, original_input_id=None):
    """
    Searches UniProtKB for a given identifier, filtered by taxon, using various search query strategies.
    Returns the selected UniProt primary accession (e.g., P12345), or None if not found.
    """
    if not identifier_to_search or not taxon_id:
        return None
    
    # Cache key: (identifier_to_search, taxon_id, id_type_hint, original_input_id)
    cache_key = (identifier_to_search, taxon_id, id_type_hint, original_input_id)
    if cache_key in uniprot_partner_select_cache:
        return uniprot_partner_select_cache[cache_key]

    # --- Construct UniProt search queries (prioritized) ---
    queries = []

    # 1. Prioritize direct accession search
    if id_type_hint == 'accession' or re.match(r'[OPQ]\d[A-Z0-9]{5}|\w\d[A-Z0-9]{4}', identifier_to_search):
        queries.append(f'accession:{identifier_to_search}')
    
    # 2. Search by Entrez Gene ID
    if id_type_hint == 'geneid' or (re.match(r'^\d+$', identifier_to_search) and len(identifier_to_search) > 3):
        queries.append(f'gene_id:{identifier_to_search}')

    # 3. Search by Gene Name/Symbol
    if id_type_hint == 'gene' or id_type_hint == 'gene_symbol' or re.match(r'^[A-Z0-9_-]+$', identifier_to_search):
        queries.append(f'gene:{identifier_to_search}')
    
    # 4. Search by RefSeq Protein/Nucleotide ID
    if id_type_hint == 'refseq_prot' or re.match(r'NP_|XP_|WP_', identifier_to_search):
        queries.append(f'xref:refseq-{identifier_to_search}')
    if id_type_hint == 'refseq_nucl' or re.match(r'NM_|NR_', identifier_to_search):
        queries.append(f'xref:refseq-{identifier_to_search}')

    # 5. Search by Ensembl Gene/Protein ID
    if id_type_hint == 'ensembl_gene' or identifier_to_search.startswith('ENSG'):
        queries.append(f'xref:ensembl-{identifier_to_search}')
    if id_type_hint == 'ensembl_prot' or identifier_to_search.startswith('ENSP'):
        queries.append(f'xref:ensembl-{identifier_to_search}')

    # Fallback: generic text search if no specific queries formed, or to broaden the search
    if not queries:
        queries.append(f'text:{identifier_to_search}')
    
    # Combine queries with OR, and add taxonomy filter
    full_query = f'({" OR ".join(queries)}) AND taxonomy_id:{taxon_id}'
    
    # Cache for raw search results
    raw_search_cache_key = (full_query)
    if raw_search_cache_key in uniprot_search_results_cache:
        all_results = uniprot_search_results_cache[raw_search_cache_key]
    else:
        try:
            params = {
                'query': full_query,
                'fields': 'accession,protein_name,organism_name,gene_names', # Request fields needed for filtering
                'format': 'json',
                'size': 100 # Fetch up to 100 relevant results
            }
            response = requests.get(UNIPROT_SEARCH_API_URL, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
            all_results = data.get('results', [])
            uniprot_search_results_cache[raw_search_cache_key] = all_results
            time.sleep(0.1)
        except requests.exceptions.RequestException as e:
            uniprot_partner_select_cache[cache_key] = None
            return None

    if not all_results:
        uniprot_partner_select_cache[cache_key] = None
        return None

    # --- Filter and select best candidate ---
    organism_candidates = []
    
    for res in all_results:
        accession = res.get('primaryAccession')
        uniprot_full_protein_name = res.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', '')
        uniprot_organism_id = str(res.get('organism', {}).get('taxId', ''))
        gene_names_list = [g.get('value', '') for g in res.get('geneNames', [])]

        if uniprot_organism_id == taxon_id: # Direct taxId match
            is_relevant = False
            id_lower = identifier_to_search.lower()
            
            if id_lower in uniprot_full_protein_name.lower():
                is_relevant = True
            elif any(id_lower in gn.lower() for gn in gene_names_list):
                is_relevant = True
            elif original_input_id and original_input_id.lower() in uniprot_full_protein_name.lower():
                is_relevant = True
            elif original_input_id and any(original_input_id.lower() in gn.lower() for gn in gene_names_list):
                is_relevant = True
            
            if re.match(r'^\d+$', identifier_to_search) and any(identifier_to_search == str(xref.get('id')) for xref in res.get('uniProtKBCrossReferences', []) if xref.get('database') == 'GeneID'):
                is_relevant = True
            
            if is_relevant:
                organism_candidates.append(accession)
    
    if organism_candidates:
        selected_acc = clean_uniprot_id(organism_candidates[0])
        uniprot_partner_select_cache[cache_key] = selected_acc
        return selected_acc
    
    # Fallback: if no highly relevant candidate, but some results exist, take the first from the specified taxon
    for res in all_results:
        accession = res.get('primaryAccession')
        uniprot_organism_id = str(res.get('organism', {}).get('taxId', ''))

        if uniprot_organism_id == taxon_id:
            selected_acc = clean_uniprot_id(accession)
            uniprot_partner_select_cache[cache_key] = selected_acc
            return selected_acc
            
    uniprot_partner_select_cache[cache_key] = None
    return None


# --- BioGRID API Helper Function (Modified for correct endpoint and better parsing) ---

def biogrid_fetch_gene_xrefs_from_interactions(identifiers_list, organism_id, biogrid_access_key, max_results_per_call=10000):
    """
    Queries the BioGRID /interactions/ endpoint to find interactions involving the given identifiers.
    Extracts cross-references (UniProt, Entrez, RefSeq, Ensembl) for these identifiers
    from the returned interaction objects.
    Returns a dictionary: {original_query_id: {uniprot: set(), entrez_gene: set(), ...}}
    """
    if not identifiers_list:
        return {}

    all_interactions_data = {} 
    biogrid_results_xrefs = defaultdict(lambda: defaultdict(set)) 
    
    _tqdm = default_tqdm

    valid_identifiers = [str(ident).strip() for ident in identifiers_list if pd.notna(ident) and str(ident).strip() != '-']
    if not valid_identifiers:
        return {}
    
    original_query_ids_set = set(valid_identifiers) 

    print(f"  Fetching interactions for {len(valid_identifiers)} IDs for organism {organism_id} (max {max_results_per_call} per call)...")
    
    gene_list_batch_size = 50 

    for i in _tqdm(range(0, len(valid_identifiers), gene_list_batch_size), desc=f"BioGRID Interaction Lookup for {organism_id}"):
        gene_batch = valid_identifiers[i:i+gene_list_batch_size]
        gene_list_str = "|".join(gene_batch)

        current_start_index = 0
        current_page = 0 
        
        while True: # Loop for pagination for the current gene_batch
            params = {
                "accesskey": biogrid_access_key,
                "taxId": organism_id,
                "geneList": gene_list_str,
                "format": "jsonExtended", 
                "searchIds": "true",           
                "searchNames": "true",         
                "searchSynonyms": "true",      
                "additionalIdentifierTypes": "UNIPROT|REFSEQ|ENSEMBL|ENTREZ_GENE|BIOGRID", # Comprehensive search types
                "includeInteractors": "true", # Important to get all interactions involving the gene
                "includeInteractorInteractions": "false", 
                "start": current_start_index,
                "max": max_results_per_call
            }

            try:
                response = requests.get(BIOGRID_INTERACTIONS_API_URL, params=params, headers=HEADERS)
                response.raise_for_status()
                interactions_data_page = response.json()

                if not interactions_data_page:
                    break 

                for int_id, details in interactions_data_page.items():
                    all_interactions_data[str(int_id)] = details

                if len(interactions_data_page) < max_results_per_call:
                    break 
                else:
                    current_start_index += max_results_per_call 
                    current_page += 1
                    time.sleep(0.5) 
            
            except requests.exceptions.RequestException as e:
                full_url = requests.Request('GET', BIOGRID_INTERACTIONS_API_URL, params=params).prepare().url
                print(f"WARNING: BioGRID API /interactions/ GET query failed for organism {organism_id}, gene batch starting with '{gene_batch[0]}' (page {current_page}, start {current_start_index}): {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"   Response status code: {e.response.status_code}")
                    print(f"   Response text: {e.response.text}")
                print(f"   Failing URL (truncated to 500 chars): {full_url[:500]}...")
                break 
            except Exception as e:
                print(f"WARNING: Unexpected error processing BioGRID API /interactions/ GET response for organism {organism_id}, gene batch starting with '{gene_batch[0]}' (page {current_page}, start {current_start_index}): {e}")
                break 
        
        time.sleep(0.2) # General sleep between geneList batches

    print(f"  Successfully fetched {len(all_interactions_data)} total interactions from BioGRID for organism {organism_id}.")

    # Now, parse the collected interactions to build cross-reference mappings
    # This loop is crucial for correctly linking BioGRID's returned data back to your original input IDs.
    for interaction_id, interaction_details in _tqdm(all_interactions_data.items(), desc=f"Parsing Interactions for {organism_id}"):
        for suffix in ['A', 'B']:
            # Extract all relevant identifiers for this interactor (A or B) from BioGRID's response
            biogrid_id = str(interaction_details.get(f'BIOGRID_ID_{suffix}')) if interaction_details.get(f'BIOGRID_ID_{suffix}') else None
            official_symbol = interaction_details.get(f'OFFICIAL_SYMBOL_{suffix}')
            entrez_gene_id = str(interaction_details.get(f'ENTREZ_GENE_{suffix}')) if interaction_details.get(f'ENTREZ_GENE_{suffix}') else None
            uniprot_id_direct_field = interaction_details.get(f'UNIPROT_ID_{suffix}') # Top-level UniProt ID if available
            additional_ids_str = interaction_details.get(f'ADDITIONAL_IDENTIFIERS_{suffix}', '') # Pipe-separated string

            # Set of all identifiers BioGRID has for THIS interactor from the interaction data
            biogrid_reported_ids_for_this_interactor = set()
            if biogrid_id: biogrid_reported_ids_for_this_interactor.add(biogrid_id)
            if official_symbol: biogrid_reported_ids_for_this_interactor.add(official_symbol)
            if entrez_gene_id: biogrid_reported_ids_for_this_interactor.add(entrez_gene_id)
            if uniprot_id_direct_field: biogrid_reported_ids_for_this_interactor.add(uniprot_id_direct_field)
            
            # Parse from ADDITIONAL_IDENTIFIERS_ string
            current_uniprot_ids_parsed = set()
            if uniprot_id_direct_field: current_uniprot_ids_parsed.add(uniprot_id_direct_field)
            current_uniprot_ids_parsed.update([m.group(1) for m in re.finditer(r'UNIPROTKB:(.*?)(?:\||$)', additional_ids_str)])
            current_uniprot_ids_parsed.update([m.group(1) for m in re.finditer(r'UNIPROT-ACCESSION:(.*?)(?:\||$)', additional_ids_str)])
            # Add these to the pool of all reported IDs for matching
            biogrid_reported_ids_for_this_interactor.update(current_uniprot_ids_parsed)

            current_entrez_gene_ids_parsed = set()
            if entrez_gene_id: current_entrez_gene_ids_parsed.add(entrez_gene_id) 
            current_entrez_gene_ids_parsed.update([m.group(1) for m in re.finditer(r'ENTREZ_GENE:(.*?)(?:\||$)', additional_ids_str)])
            biogrid_reported_ids_for_this_interactor.update(current_entrez_gene_ids_parsed)

            current_refseq_ids_parsed = set([m.group(1) for m in re.finditer(r'REFSEQ:(.*?)(?:\||$)', additional_ids_str)])
            biogrid_reported_ids_for_this_interactor.update(current_refseq_ids_parsed)

            current_ensembl_ids_parsed = set([m.group(1) for m in re.finditer(r'ENSEMBL:(.*?)(?:\||$)', additional_ids_str)])
            biogrid_reported_ids_for_this_interactor.update(current_ensembl_ids_parsed)
            
            current_gene_symbols_parsed = set()
            if official_symbol: current_gene_symbols_parsed.add(official_symbol)


            # --- Crucial Matching Step: Link BioGRID's interactor data back to YOUR original input IDs ---
            # For each original ID you asked BioGRID about, check if this interactor's identifiers match it.
            for original_id_from_input in original_query_ids_set: 
                is_match = False
                
                # Check for direct string match of original ID in any of BioGRID's reported IDs for this interactor
                if original_id_from_input in biogrid_reported_ids_for_this_interactor:
                    is_match = True
                
                # Special handling for ETG<num> input IDs: check if numeric part matches any reported Entrez ID
                elif original_id_from_input.startswith('ETG'):
                    match = re.match(r'ETG(\d+)', original_id_from_input)
                    if match:
                        entrez_num_from_etg = match.group(1)
                        if entrez_num_from_etg in biogrid_reported_ids_for_this_interactor: 
                            is_match = True
                
                if is_match:
                    # If a match is found, attribute ALL relevant discovered xrefs to this original_id_from_input
                    # This ensures maximum coverage of cross-references for that specific input ID
                    if current_uniprot_ids_parsed:
                        biogrid_results_xrefs[original_id_from_input]['uniprot'].update(current_uniprot_ids_parsed)
                    if current_entrez_gene_ids_parsed:
                        biogrid_results_xrefs[original_id_from_input]['entrez_gene'].update(current_entrez_gene_ids_parsed)
                    if current_refseq_ids_parsed:
                        biogrid_results_xrefs[original_id_from_input]['refseq'].update(current_refseq_ids_parsed)
                    if current_ensembl_ids_parsed:
                        biogrid_results_xrefs[original_id_from_input]['ensembl'].update(current_ensembl_ids_parsed)
                    if current_gene_symbols_parsed: 
                        biogrid_results_xrefs[original_id_from_input]['gene_symbol'].update(current_gene_symbols_parsed)
                    
                    # Also, explicitly add common derivations of IDs if they exist and are useful for UniProt mapping.
                    # This includes adding the numeric part of ETG IDs to the 'entrez_gene' set.
                    if original_id_from_input.startswith('ETG'):
                        match = re.match(r'ETG(\d+)', original_id_from_input)
                        if match: biogrid_results_xrefs[original_id_from_input]['entrez_gene'].add(match.group(1))
                    
                    # Ensure the original_id_from_input itself is available as a gene_symbol if appropriate
                    # This helps UniProt if it recognizes the raw string as a gene name.
                    if original_id_from_input not in biogrid_results_xrefs[original_id_from_input]['gene_symbol'] and \
                       not re.match(r'ETG\d+|R[Pp]\d+-|DADB-|[OPQ]\d[A-Z0-9]{5}|\w\d[A-Z0-9]{4}|\w+-\d+\.?\d*$', original_id_from_input): 
                        biogrid_results_xrefs[original_id_from_input]['gene_symbol'].add(original_id_from_input)

    return biogrid_results_xrefs


# --- Main Script ---
def main():
    if not BIOGRID_ACCESS_KEY or BIOGRID_ACCESS_KEY == "YOUR_BIOGRID_ACCESS_KEY":
        print("ERROR: Please update BIOGRID_ACCESS_KEY in the script with your actual BioGRID API key.")
        return

    print(f"Loading BioGRID file: {BIOGRID_FILE}")
    try:
        df = pd.read_csv(BIOGRID_FILE, sep='\t', low_memory=False)
    except FileNotFoundError:
        print(f"ERROR: BioGRID file not found at {BIOGRID_FILE}. Please check the path.")
        return
    except Exception as e:
        print(f"ERROR: Failed to read BioGRID file {BIOGRID_FILE}: {e}")
        return
    
    
    initial_rows = len(df)
    print(f"Initial BioGRID dataset size (after subsetting): {initial_rows} rows.")


    # --- FILTERING STEP ---
    df_filtered = df[df['EXPERIMENTAL_SYSTEM'].isin(PHYSICAL_INTERACTION_TYPES)].copy()
    
    filtered_rows = len(df_filtered)
    print(f"Filtered to {filtered_rows} physical interactions based on EXPERIMENTAL_SYSTEM.")
    if filtered_rows == 0:
        print("No interactions found after filtering. Exiting.")
        return

    # Collect ALL unique identifiers (from INTERACTOR_A/B, OFFICIAL_SYMBOL_A/B, ALIASES_FOR_A/B)
    # These are the identifiers we want to map.
    all_unique_ids_by_organism = defaultdict(set)
    all_tax_ids = set()

    print("Collecting ALL unique identifiers (from INTERACTOR_A/B, OFFICIAL_SYMBOL, ALIASES) grouped by organism...")
    for index, row in default_tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Collecting IDs"):
        for suffix in ['A', 'B']:
            organism_id = str(int(row[f'ORGANISM_{suffix}_ID'])) if pd.notna(row[f'ORGANISM_{suffix}_ID']) else None
            if organism_id:
                all_tax_ids.add(organism_id)
                
                # Collect INTERACTOR_A/B values
                interactor_id = str(row[f'INTERACTOR_{suffix}']).strip() if pd.notna(row[f'INTERACTOR_{suffix}']) and str(row[f'INTERACTOR_{suffix}']).strip() != "-" else None
                if interactor_id:
                    all_unique_ids_by_organism[organism_id].add(interactor_id)

                # Collect OFFICIAL_SYMBOL_A/B
                symbol = str(row[f'OFFICIAL_SYMBOL_{suffix}']).strip() if pd.notna(row[f'OFFICIAL_SYMBOL_{suffix}']) and str(row[f'OFFICIAL_SYMBOL_{suffix}']).strip() != "-" else None
                if symbol: 
                    all_unique_ids_by_organism[organism_id].add(symbol)

                # Collect ALIASES_FOR_A/B
                aliases_str = str(row[f'ALIASES_FOR_{suffix}']) if pd.notna(row[f'ALIASES_FOR_{suffix}']) and str(row[f'ALIASES_FOR_{suffix}']).strip() != "-" else ""
                for alias in aliases_str.split('|'):
                    cleaned_alias = alias.strip()
                    if cleaned_alias:
                        all_unique_ids_by_organism[organism_id].add(cleaned_alias)

    print(f"Found {len(all_tax_ids)} unique organism taxonomy IDs in the filtered data.")
    tax_id_to_name = get_organism_names(list(all_tax_ids))


    # --- Step 1: Query BioGRID API for all collected unique identifiers ---
    biogrid_xrefs_for_input_ids = load_checkpoint(BIOGRID_XREFS_CHECKPOINT_FILE)
    if biogrid_xrefs_for_input_ids is None:
        biogrid_xrefs_for_input_ids = defaultdict(lambda: defaultdict(set))
    
    print("\n--- Starting BioGRID API queries for ALL unique input identifiers ---")
    
    missing_identifiers_overall_for_biogrid = set()

    for current_tax_id in sorted(list(all_tax_ids)):
        identifiers_for_org = list(all_unique_ids_by_organism[current_tax_id])
        missing_identifiers_for_org = [
            ident for ident in identifiers_for_org
            if (ident, current_tax_id) not in biogrid_xrefs_for_input_ids
        ]
        
        if missing_identifiers_for_org: 
            missing_identifiers_overall_for_biogrid.update({ (ident, current_tax_id) for ident in missing_identifiers_for_org })

    if missing_identifiers_overall_for_biogrid:
        print(f"  {len(missing_identifiers_overall_for_biogrid)} total identifiers still need BioGRID API lookup.")
        for current_tax_id in default_tqdm(sorted(list(all_tax_ids)), desc="Querying BioGRID per Organism (missing IDs)"):
            organism_display_name = tax_id_to_name.get(current_tax_id, current_tax_id) 
            
            current_org_missing_ids = [
                ident for (ident, tax_id) in missing_identifiers_overall_for_biogrid
                if tax_id == current_tax_id
            ]

            if not current_org_missing_ids:
                continue

            print(f"\n  Querying BioGRID interactions for {len(current_org_missing_ids)} missing IDs for {organism_display_name} ({current_tax_id})...")
            
            # REVERTED: Use biogrid_fetch_gene_xrefs_from_interactions for /interactions/ endpoint
            org_biogrid_results = biogrid_fetch_gene_xrefs_from_interactions(current_org_missing_ids, current_tax_id, BIOGRID_ACCESS_KEY)
            
            for original_query_id, xrefs in org_biogrid_results.items():
                for xref_type, ids in xrefs.items():
                    biogrid_xrefs_for_input_ids[(original_query_id, current_tax_id)][xref_type].update(ids)
            print(f"  Collected BioGRID cross-references for {len(org_biogrid_results)} unique original input IDs for {organism_display_name}.")
        
        save_checkpoint(biogrid_xrefs_for_input_ids, BIOGRID_XREFS_CHECKPOINT_FILE)
    else:
        print("  All BioGRID cross-references already collected or no new identifiers to process.")


    # --- Step 2: Consolidate IDs to UniProt Accessions using BioGRID xrefs and UniProt Search API ---
    final_id_to_uniprot_map = load_checkpoint(FINAL_UNIPROT_MAP_CHECKPOINT_FILE)
    if final_id_to_uniprot_map is None:
        final_id_to_uniprot_map = {}
    
    print("\n--- Consolidating IDs to UniProt Accessions using BioGRID xrefs and UniProt Search API ---")

    all_input_id_taxon_pairs = set()
    for taxon_id, ids_set in all_unique_ids_by_organism.items():
        for ident in ids_set:
            all_input_id_taxon_pairs.add((ident, taxon_id))

    unmapped_original_id_pairs = [
        (original_id, taxon_id) for (original_id, taxon_id) in all_input_id_taxon_pairs
        if (original_id, taxon_id) not in final_id_to_uniprot_map
    ]

    print(f"  {len(unmapped_original_id_pairs)} original input IDs still need UniProt accession mapping.")

    for (original_id, taxon_id) in default_tqdm(unmapped_original_id_pairs, desc="Mapping to UniProt Accessions"):
        found_uniprot_acc = None
        
        xrefs = biogrid_xrefs_for_input_ids.get(original_id, defaultdict(set)) 

        # Priority 1: Direct UniProt Accessions from BioGRID xrefs
        if 'uniprot' in xrefs and xrefs['uniprot']:
            for uniprot_acc_candidate in xrefs['uniprot']:
                if uniprot_acc_candidate and clean_uniprot_id(uniprot_acc_candidate):
                    found_uniprot_acc = clean_uniprot_id(uniprot_acc_candidate)
                    break
        if found_uniprot_acc:
            final_id_to_uniprot_map[(original_id, taxon_id)] = found_uniprot_acc
            continue

        # Priority 2: Try UniProt Search API with BioGRID-provided Entrez Gene IDs
        if 'entrez_gene' in xrefs and xrefs['entrez_gene']:
            for entrez_id_candidate in xrefs['entrez_gene']:
                cache_check_key = (entrez_id_candidate, taxon_id, 'geneid', original_id)
                if cache_check_key in uniprot_partner_select_cache:
                    found_uniprot_acc = uniprot_partner_select_cache[cache_check_key]
                else:
                    found_uniprot_acc = search_uniprot_for_accession(entrez_id_candidate, taxon_id, id_type_hint="geneid", original_input_id=original_id)
                if found_uniprot_acc: break
        if found_uniprot_acc:
            final_id_to_uniprot_map[(original_id, taxon_id)] = found_uniprot_acc
            continue
        
        # Priority 3: Try UniProt Search API with BioGRID-provided Gene Symbols
        if 'gene_symbol' in xrefs and xrefs['gene_symbol']:
            for symbol_candidate in xrefs['gene_symbol']:
                cache_check_key = (symbol_candidate, taxon_id, 'gene', original_id)
                if cache_check_key in uniprot_partner_select_cache:
                    found_uniprot_acc = uniprot_partner_select_cache[cache_check_key]
                else:
                    found_uniprot_acc = search_uniprot_for_accession(symbol_candidate, taxon_id, id_type_hint="gene", original_input_id=original_id)
                if found_uniprot_acc: break
        if found_uniprot_acc:
            final_id_to_uniprot_map[(original_id, taxon_id)] = found_uniprot_acc
            continue

        # Priority 4: Try the original_id itself via UniProt Search API (pattern matching for best type inference)
        inferred_type_for_original_id = None
        if original_id.startswith('ETG'): 
            inferred_type_for_original_id = "geneid" 
        elif re.match(r'[OPQ]\d[A-Z0-9]{5}|\w\d[A-Z0-9]{4}', original_id): inferred_type_for_original_id = "accession"
        elif re.match(r'NP_|XP_|WP_', original_id): inferred_type_for_original_id = "refseq_prot"
        elif re.match(r'NM_|NR_', original_id): inferred_type_for_original_id = "refseq_nucl"
        elif original_id.startswith('ENSG'): inferred_type_for_original_id = "ensembl_gene"
        elif original_id.startswith('ENSP'): inferred_type_for_original_id = "ensembl_prot"
        
        if inferred_type_for_original_id:
            cache_check_key = (original_id, taxon_id, inferred_type_for_original_id, original_id)
            if cache_check_key in uniprot_partner_select_cache:
                found_uniprot_acc = uniprot_partner_select_cache[cache_check_key]
            else:
                found_uniprot_acc = search_uniprot_for_accession(original_id, taxon_id, id_type_hint=inferred_type_for_original_id, original_input_id=original_id)
        
        # Fallback to general gene/accession search for original_id if not found yet
        if not found_uniprot_acc:
            cache_check_key = (original_id, taxon_id, 'gene', original_id)
            if cache_check_key in uniprot_partner_select_cache:
                found_uniprot_acc = uniprot_partner_select_cache[cache_check_key]
            else:
                found_uniprot_acc = search_uniprot_for_accession(original_id, taxon_id, id_type_hint="gene", original_input_id=original_id)
        if not found_uniprot_acc:
            cache_check_key = (original_id, taxon_id, 'accession', original_id)
            if cache_check_key in uniprot_partner_select_cache:
                found_uniprot_acc = uniprot_partner_select_cache[cache_check_key]
            else:
                found_uniprot_acc = search_uniprot_for_accession(original_id, taxon_id, id_type_hint="accession", original_input_id=original_id)

        if found_uniprot_acc:
            final_id_to_uniprot_map[(original_id, taxon_id)] = found_uniprot_acc
            continue
        
        # Priority 5: Fallback to other BioGRID xrefs (RefSeq, Ensembl) via UniProt Search API
        for xref_type_key in ['refseq', 'ensembl']: 
            if xref_type_key in xrefs and xrefs[xref_type_key]:
                for candidate_id in xrefs[xref_type_key]:
                    inferred_type = None
                    if re.match(r'NP_|XP_|WP_', candidate_id): inferred_type = "refseq_prot"
                    elif re.match(r'NM_|NR_', candidate_id): inferred_type = "refseq_nucl"
                    elif candidate_id.startswith('ENSG'): inferred_type = "ensembl_gene"
                    elif candidate_id.startswith('ENSP'): inferred_type = "ensembl_prot"
                    
                    if inferred_type:
                        cache_check_key = (candidate_id, taxon_id, inferred_type, original_id)
                        if cache_check_key in uniprot_partner_select_cache:
                            found_uniprot_acc = uniprot_partner_select_cache[cache_check_key]
                        else:
                            found_uniprot_acc = search_uniprot_for_accession(candidate_id, taxon_id, id_type_hint=inferred_type, original_input_id=original_id)
                    
                    if found_uniprot_acc:
                        break
                if found_uniprot_acc:
                    break
        
        if found_uniprot_acc:
            final_id_to_uniprot_map[(original_id, taxon_id)] = found_uniprot_acc
            continue
        
    save_checkpoint(final_id_to_uniprot_map, FINAL_UNIPROT_MAP_CHECKPOINT_FILE)
    print(f"\nTotal final master ID to UniProt map contains {len(final_id_to_uniprot_map)} entries across all organisms.")


    # 5. Get sequences for all unique UniProt accessions found in our master map
    uniprot_acc_to_sequence = {}
    
    accessions_to_fetch_seq_for = list(set(acc for acc in final_id_to_uniprot_map.values() if acc))
    if accessions_to_fetch_seq_for:
        print(f"Retrieving sequences for {len(accessions_to_fetch_seq_for)} unique UniProt accessions...")
        _tqdm = default_tqdm 
        for accession in _tqdm(accessions_to_fetch_seq_for, desc="Fetching sequences"):
            if accession not in sequence_cache:
                sequence = get_protein_sequence(accession)
                if sequence:
                    uniprot_acc_to_sequence[accession] = sequence
            else:
                uniprot_acc_to_sequence[accession] = sequence_cache[accession]
        print(f"Successfully retrieved {len(uniprot_acc_to_sequence)} sequences.")
    else:
        print("No UniProt accessions found in the final map to fetch sequences for.")


    # Prepare output data
    output_rows = []
    
    print("\nProcessing filtered BioGRID interactions and compiling results...")
    for index, row in default_tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Compiling results"):
        def get_final_uniprot_id_for_interactor(interactor_id_val, symbol_val, aliases_str_val, organism_id_val):
            org_id_str = str(int(organism_id_val)) if pd.notna(organism_id_val) else None
            if not org_id_str:
                return None

            if interactor_id_val and interactor_id_val != "-":
                key = (interactor_id_val, org_id_str)
                if key in final_id_to_uniprot_map:
                    return final_id_to_uniprot_map[key]
            
            if symbol_val and symbol_val != "-":
                key = (symbol_val, org_id_str)
                if key in final_id_to_uniprot_map:
                    return final_id_to_uniprot_map[key]

            if aliases_str_val:
                for alias in aliases_str_val.split('|'):
                    cleaned_alias = alias.strip()
                    if cleaned_alias and cleaned_alias != "-":
                        key = (cleaned_alias, org_id_str)
                        if key in final_id_to_uniprot_map:
                            return final_id_to_uniprot_map[key]
            
            return None 

        interactor_a_val = str(row['INTERACTOR_A']).strip() if pd.notna(row['INTERACTOR_A']) and str(row['INTERACTOR_A']).strip() != "-" else None
        symbol_a_val = str(row['OFFICIAL_SYMBOL_A']).strip() if pd.notna(row['OFFICIAL_SYMBOL_A']) and str(row['OFFICIAL_SYMBOL_A']).strip() != "-" else None
        aliases_a_str_val = str(row['ALIASES_FOR_A']) if pd.notna(row['ALIASES_FOR_A']) and str(row['ALIASES_FOR_A']).strip() != "-" else ""
        organism_a_id = str(int(row['ORGANISM_A_ID'])) if pd.notna(row['ORGANISM_A_ID']) else None
        
        interactor_b_val = str(row['INTERACTOR_B']).strip() if pd.notna(row['INTERACTOR_B']) and str(row['INTERACTOR_B']).strip() != "-" else None
        symbol_b_val = str(row['OFFICIAL_SYMBOL_B']).strip() if pd.notna(row['OFFICIAL_SYMBOL_B']) and str(row['OFFICIAL_SYMBOL_B']).strip() != "-" else None
        aliases_b_str_val = str(row['ALIASES_FOR_B']) if pd.notna(row['ALIASES_FOR_B']) and str(row['ALIASES_FOR_B']).strip() != "-" else ""
        organism_b_id = str(int(row['ORGANISM_B_ID'])) if pd.notna(row['ORGANISM_B_ID']) else None

        uniprot_a = get_final_uniprot_id_for_interactor(interactor_a_val, symbol_a_val, aliases_a_str_val, organism_a_id)
        uniprot_b = get_final_uniprot_id_for_interactor(interactor_b_val, symbol_b_val, aliases_b_str_val, organism_b_id)

        sequence_a = uniprot_acc_to_sequence.get(uniprot_a, "")
        sequence_b = uniprot_acc_to_sequence.get(uniprot_b, "")
        
        organism_name_a = tax_id_to_name.get(organism_a_id, f"TaxID:{organism_a_id}") if organism_a_id else ""
        organism_name_b = tax_id_to_name.get(organism_b_id, f"TaxID:{organism_b_id}") if organism_b_id else ""

        output_rows.append({
            'ID_A_Interactor': interactor_a_val,
            'ID_A_Official_Symbol': symbol_a_val,
            'ID_A_Aliases': aliases_a_str_val,
            'UniProt_ID_A': uniprot_a, 
            'Sequence_A': sequence_a,
            'Organism_A': organism_name_a,
            'ID_B_Interactor': interactor_b_val,
            'ID_B_Official_Symbol': symbol_b_val,
            'ID_B_Aliases': aliases_b_str_val,
            'UniProt_ID_B': uniprot_b,
            'Sequence_B': sequence_b,
            'Organism_B': organism_name_b,
            'Experimental_System': row['EXPERIMENTAL_SYSTEM']
        })

    output_df = pd.DataFrame(output_rows)
    output_df = output_df[(output_df['Sequence_A'] != "") & (output_df['Sequence_B'] != "")].copy()
    
    print(f"\nSaving final dataset with {len(output_df)} interactions (where both proteins had sequences) to {OUTPUT_CSV_FILE}")
    output_df.to_csv(OUTPUT_CSV_FILE, index=False)
    print("Process complete!")

if __name__ == "__main__":
    main()