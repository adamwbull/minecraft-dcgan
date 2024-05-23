import h5py
import numpy as np
import csv
import os

def compute_hash(tensor):
    # Simple hash function: Converts tensor to bytes and hashes
    return hash(tensor.tobytes())

def read_hash_db(csv_filename):
    hash_db = {}
    if os.path.exists(csv_filename):
        with open(csv_filename, mode='r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                hash_db[row[0]] = row[1]
    return hash_db

def write_hash_db(csv_filename, hash_db):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for key, value in hash_db.items():
            writer.writerow([key, value])

def find_duplicates(hdf5_filename, csv_filename):
    hash_db = read_hash_db(csv_filename)
    duplicates = set()
    updated = False

    with h5py.File(hdf5_filename, 'r') as f:
        for dataset_name in f:
            if dataset_name not in hash_db:
                tensor = f[dataset_name][...]
                tensor_hash = compute_hash(tensor)
                hash_db[dataset_name] = tensor_hash
                updated = True
            else:
                tensor_hash = hash_db[dataset_name]

            # Check for duplicates
            if list(hash_db.values()).count(tensor_hash) > 1:
                duplicates.add(dataset_name)

    if updated:
        write_hash_db(csv_filename, hash_db)

    return duplicates

if __name__ == "__main__":
    hdf5_file = "../dataset/schematics.hdf5"
    csv_file = "hashes.csv"
    duplicate_entries = find_duplicates(hdf5_file, csv_file)
    print(f"Duplicate entries: {duplicate_entries}")
