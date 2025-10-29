import argparse
import pandas as pd
from Bio import SeqIO
import metapredict as mpred
import concurrent.futures
import os
import sys

def parse_sequences(input_file_path, file_format):
    """
    Parses protein sequences from a given file.
    Args:
        input_file_path (str): Path to the input file.
        file_format (str): 'fasta' or 'csv'.
    Returns:
        list: A list of tuples, where each tuple is (protein_id, sequence).
    """
    sequences = []
    print(f"Attempting to parse sequences from {input_file_path} as {file_format} format...")
    if file_format == 'fasta':
        try:
            # SeqIO.parse returns an iterator, converting to list to allow multiple iterations if needed
            # and to pass to ProcessPoolExecutor which expects iterable inputs.
            for record in SeqIO.parse(input_file_path, "fasta"):
                if record.seq: # Ensure sequence is not empty
                    sequences.append((record.id, str(record.seq)))
                else:
                    print(f"Warning: Skipping empty sequence for protein ID: {record.id}", file=sys.stderr)
        except FileNotFoundError:
            print(f"Error: Input FASTA file not found at {input_file_path}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error parsing FASTA file {input_file_path}: {e}", file=sys.stderr)
            sys.exit(1)
    elif file_format == 'csv':
        try:
            df = pd.read_csv(input_file_path)
            # Assuming CSV has 'protein_id' and 'sequence' columns
            if 'ID' not in df.columns or 'Sequence' not in df.columns:
                raise ValueError("CSV file must contain 'ID' and 'Sequence' columns.")
            
            # Filter out rows with empty sequences
            df_filtered = df.dropna(subset=['Sequence'])
            empty_seq_count = len(df) - len(df_filtered)
            if empty_seq_count > 0:
                print(f"Warning: Skipping {empty_seq_count} proteins with empty sequences from CSV.", file=sys.stderr)

            sequences = list(df_filtered[['ID', 'Sequence']].itertuples(index=False, name=None))

        except FileNotFoundError:
            print(f"Error: Input CSV file not found at {input_file_path}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error parsing CSV file {input_file_path}: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Error: Unsupported file format: {file_format}. Choose 'fasta' or 'csv'.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Successfully parsed {len(sequences)} sequences from {input_file_path}.")
    return sequences

def predict_idr_for_protein(protein_id, sequence):
    """
    Predicts IDR regions for a single protein sequence using metapredict.
    Args:
        protein_id (str): The identifier of the protein.
        sequence (str): The protein sequence.
    Returns:
        tuple: A tuple containing:
            - dict: Data for the overview CSV (Protein_ID, Has_IDR, IDR_Start_Positions, IDR_End_Positions, Number_of_IDRs).
            - list: A list of dicts, each for an IDR sequence (Protein_ID, IDR_Sequence, IDR_Start, IDR_End).
    """
    if not sequence or len(sequence) == 0:
        print(f"Warning: Empty sequence for protein ID: {protein_id}. Skipping IDR prediction.", file=sys.stderr)
        return {
            'Protein_ID': protein_id,
            'Has_IDR': False,
            'IDR_Start_Positions': '',
            'IDR_End_Positions': '',
            'Number_of_IDRs': 0
        }, []

    try:
        # This will make mpred.predict_disorder return a DisorderObject, which has the .disordered_domain_boundaries attribute
        prediction_object = mpred.predict_disorder(sequence, version='V3', return_domains=True)
        
        idr_labels = prediction_object.disordered_domain_boundaries
        
        has_idr = bool(idr_labels)
        num_idrs = len(idr_labels)
        
        # Ensure start/end positions are 1-indexed for human readability in CSV.
        # metapredict's DisorderObject returns 0-indexed boundaries.
        start_positions = [str(start + 1) for start, _ in idr_labels]
        end_positions = [str(end) for _, end in idr_labels]
        
        overview_row = {
            'Protein_ID': protein_id,
            'Has_IDR': has_idr,
            'IDR_Start_Positions': ','.join(start_positions),
            'IDR_End_Positions': ','.join(end_positions),
            'Number_of_IDRs': num_idrs
        }
        
        idr_sequence_rows = []
        for i, (start, end) in enumerate(idr_labels):
            idr_counter = i + 1 
            idr_unique_id = f"{protein_id}_{idr_counter}"
            idr_seq = sequence[start:end] 
            
            idr_sequence_rows.append({
                'ID': protein_id,
                'IDR_Unique_ID': idr_unique_id, 
                'IDR_Counter': idr_counter,     
                'Sequence': idr_seq,
                'IDR_Length': end - start + 1,
                'IDR_Start': start + 1,         # Store 1-indexed start position
                'IDR_End': end                  # Store 0-indexed end position
            })
            
        return overview_row, idr_sequence_rows        

    except Exception as e:
        print(f"Error predicting IDR for protein {protein_id}: {e}", file=sys.stderr)
        # Return empty/default values on error to ensure job completion
        return {
            'Protein_ID': protein_id,
            'Has_IDR': False,
            'IDR_Start_Positions': '',
            'IDR_End_Positions': '',
            'Number_of_IDRs': 0
        }, []

def main():
    parser = argparse.ArgumentParser(description="Predict IDR regions in protein sequences using metapredict.")
    parser.add_argument("--input_file", required=True, 
                        help="Path to the input FASTA or CSV file containing protein sequences.")
    parser.add_argument("--input_format", default="fasta", choices=["fasta", "csv"],
                        help="Format of the input file (fasta or csv). Default is 'fasta'.")
    parser.add_argument("--output_overview_file", default="data/human-idr-region-overview.csv",
                        help="Path to the output CSV file for IDR overview. Default is 'data/human-idr-region-overview.csv'.")
    parser.add_argument("--output_idr_sequences_file", default="data/human-idr-regions.csv",
                        help="Path to the output CSV file for IDR sequences. Default is 'data/human-idr-regions.csv'.")
    parser.add_argument("--num_processes", type=int, default=1,
                        help="Number of CPU processes to use for prediction. Default is 1 (no multiprocessing).")
    
    args = parser.parse_args()

    # Ensure output directories exist
    os.makedirs(os.path.dirname(args.output_overview_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_idr_sequences_file), exist_ok=True)

    print(f"Starting IDR prediction for input: {args.input_file} (format: {args.input_format})...")
    
    # Parse sequences
    all_sequences = parse_sequences(args.input_file, args.input_format)

    if not all_sequences:
        print("No sequences found or parsed. Exiting.", file=sys.stderr)
        sys.exit(0)

    overview_data = []
    idr_sequences_data = []

    # Prepare inputs for multiprocessing: split protein_ids and sequences into separate lists
    protein_ids = [seq_id for seq_id, _ in all_sequences]
    sequences = [seq for _, seq in all_sequences]

    if args.num_processes > 1:
        print(f"Using {args.num_processes} processes for prediction...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_processes) as executor:
            # executor.map ensures order of results corresponds to order of inputs
            results = list(executor.map(predict_idr_for_protein, protein_ids, sequences))

            for overview_row, idr_seq_rows in results:
                overview_data.append(overview_row)
                idr_sequences_data.extend(idr_seq_rows) # extend with list of dictionaries

    else:
        print("Running in single-process mode...")
        for i, (protein_id, sequence) in enumerate(all_sequences):
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(all_sequences)} proteins...", end='\r')
            overview_row, idr_seq_rows = predict_idr_for_protein(protein_id, sequence)
            overview_data.append(overview_row)
            idr_sequences_data.extend(idr_seq_rows)
        print(f"Processed {len(all_sequences)}/{len(all_sequences)} proteins.           ") # Clear the line

    # Convert results to pandas DataFrames and save to CSV
    print(f"Generated {len(overview_data)} overview entries and {len(idr_sequences_data)} IDR sequence entries.")

    overview_df = pd.DataFrame(overview_data)
    idr_sequences_df = pd.DataFrame(idr_sequences_data)

    print(f"Saving IDR overview to {args.output_overview_file}...")
    overview_df.to_csv(args.output_overview_file, index=False)
    
    print(f"Saving IDR sequences to {args.output_idr_sequences_file}...")
    idr_sequences_df.to_csv(args.output_idr_sequences_file, index=False)

    print("IDR prediction complete!")

if __name__ == "__main__":
    main()