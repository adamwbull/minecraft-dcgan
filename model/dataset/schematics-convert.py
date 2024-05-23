import h5py
import numpy as np
import pickle
from vector import string_to_vector, vector_to_string
import argparse
import time
import torch 

def load_embeddings(embeddings_file):
    embeddings = None
    # First, check if we can open the file without using torch.load
    try:
        with open(embeddings_file, 'rb') as file:
            embeddings = pickle.load(file)
    except Exception as e:
        print(f"Error loading embeddings with pickle: {e}")
        # If error occurs, it's likely due to the file containing PyTorch tensors
        with open(embeddings_file, 'rb') as file:
            # Use torch.load with explicit map_location to ensure tensors are loaded onto the CPU
            embeddings = torch.load(file, map_location=torch.device('cpu'))
    return embeddings

# Logging function
def log(text="", console_only=False):
    current_date = time.strftime("%Y%m%d")
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")

    print(timestamp + " " + text)
    
    if not console_only:
        # Write to the latest.log file
        with open('./schematics-convert.log', 'a') as log_file:
            log_file.write(f"{timestamp} {text}\n")

        # Write to the daily log file
        daily_log_filename = f'./{current_date}.log'
        with open(daily_log_filename, 'a') as daily_log_file:
            daily_log_file.write(f"{timestamp} {text}\n")

# Function to manage log files
def initialize_logs():
    open('./schematics-convert.log', 'w').close()

# Initialize log files
initialize_logs()

# Main function to convert schematics to embeddings
def convert_schematics_to_embeddings(schematics_file, embeddings_file, output_file):
    # Load embeddings
    embeddings = load_embeddings(embeddings_file)
    
    # Open schematics HDF5
    with h5py.File(schematics_file, 'r') as schematics_db, h5py.File(output_file, 'w') as output_db:
        for structure_name in schematics_db.keys():
            structure = schematics_db[structure_name][:]
            embedded_structure = np.zeros(shape=structure.shape[:-1] + (len(next(iter(embeddings.values()))[0]),), dtype=np.float32)
            
            for i in range(structure.shape[0]):
                for j in range(structure.shape[1]):
                    for k in range(structure.shape[2]):
                        vector_key = vector_to_string(structure[i, j, k])
                        if vector_key in embeddings:
                            # Get the first embedding for simplicity; modify if the number of entries are increased
                            # later and more advanced picking is needed.
                            embedded_structure[i, j, k] = embeddings[vector_key][0]
                        else:
                            log(f"No embedding found for block {str(structure[i, j, k])} -> {vector_to_string(structure[i, j, k])} at {i}, {j}, {k}, using zeros")
            
            output_db.create_dataset(structure_name, data=embedded_structure)
            log(f"Processed and saved structure: {structure_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Minecraft schematics to embeddings.")
    parser.add_argument("--schematics_file", type=str, required=True, help="Path to the schematics HDF5 database.")
    parser.add_argument("--embeddings_file", type=str, required=True, help="Path to the embeddings pickle file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path for the output HDF5 database.")
    
    args = parser.parse_args()
    
    convert_schematics_to_embeddings(args.schematics_file, args.embeddings_file, args.output_file)
