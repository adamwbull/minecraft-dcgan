# dataset_blocks_build.py

import h5py
import numpy as np
from collections import defaultdict
import random
from vector import string_to_vector, vector_to_string
from dataset import MinecraftDataset
import os
import argparse
import time
import shutil 

threshold = 100000  # Threshold to start saving checkpoint HDF5s
interval = 50000  # Interval at which to save checkpoint HDF5s

# Logging function
def log(text="", console_only=False):
    current_date = time.strftime("%Y%m%d")
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")

    print(timestamp + " " + text)
    
    if not console_only:
        # Write to the latest file
        with open('./embeddings/logs/dataset-build-latest.log', 'a') as log_file:
            log_file.write(f"{timestamp} {text}\n")


# Function to manage log files
def initialize_files():
    if not os.path.exists('./embeddings/dataset/logs/'):
        os.makedirs('./embeddings/dataset/logs/')
    open('./embeddings/dataset/logs/dataset-build-latest.log', 'w').close()

# Initialize log files
initialize_files()

def calculate_sampling_probabilities(minecraft_dataset):
    
    sampling_probabilities_path = f"sampling-probabilities-{len(minecraft_dataset)}.npy"
    sampling_probabilities = None
    if os.path.exists(sampling_probabilities_path):
        log("Loading established probabilities...")
        sampling_probabilities = np.load(sampling_probabilities_path, allow_pickle=True).item()
    else:
        block_counts = defaultdict(int)
        log("Calculating new sampling probabilities...")
        total_length = len(minecraft_dataset)
        for i, structure in enumerate(minecraft_dataset):
            reshaped_structure = structure.reshape(-1, structure.shape[-1])
            unique_vectors, counts = np.unique(reshaped_structure, axis=0, return_counts=True)
            for vector, count in zip(unique_vectors, counts):
                block_counts[vector_to_string(vector)] += count
            if (i + 1) % 500 == 0:
                log(f"{i+1}/{total_length} structures counted")

        total_blocks = sum(block_counts.values())
        block_probabilities = {block: count / total_blocks for block, count in block_counts.items()}
        sampling_probabilities = {block: (np.sqrt(probability / 0.001) + 1) * (0.001 / probability) for block, probability in block_probabilities.items()}
        
        # Normalize the probabilities
        sum_probabilities = sum(sampling_probabilities.values())
        sampling_probabilities = {block: prob / sum_probabilities for block, prob in sampling_probabilities.items()}
        
        np.save(sampling_probabilities_path, sampling_probabilities)
        log(f"Sampling probabilities saved to {sampling_probabilities_path}.")

        # Pretty print each entry in sampling_probabilities
        log("Final Sampling Probabilities:")
        for block, prob in sorted(sampling_probabilities.items(), key=lambda item: item[1], reverse=True):
            log(f"{block}: {prob}")
        
    return sampling_probabilities

def is_far_enough(x, y, z, nx, ny, nz, min_neg_distance):
    return abs(x - nx) >= min_neg_distance or abs(y - ny) >= min_neg_distance or abs(z - nz) >= min_neg_distance

def sample_blocks(structure_idx, structure, sampling_probabilities, min_neg_distance, added_entries, sample_size, existing_samples, target_occurrences, current_occurrences, 
                  attempts_per_structure=1024):

    if existing_samples is None:
        existing_samples = set()

    valid_dim = 32
    pad_width = 1
    
    dirt_vector = string_to_vector("minecraft:dirt")
    air_vector = string_to_vector("minecraft:air")
    
    padded_structure = np.pad(structure, 
                                pad_width=((pad_width, pad_width), (pad_width, pad_width), (pad_width, pad_width), (0, 0)),
                                mode='constant', constant_values=0)
        
    # Apply air padding to all sides except the bottom, which gets a dirt layer
    padded_structure[:pad_width, :, :] = air_vector  # -X
    padded_structure[:, :pad_width, :] = dirt_vector  # -Y
    padded_structure[:, :, :pad_width] = air_vector  # -Z
    padded_structure[:, :, -pad_width:] = air_vector # +Z
    padded_structure[:, -pad_width:, :] = air_vector  # +Y
    padded_structure[-pad_width:, :, :] = air_vector  # +X

    entries = []
    attempts = 0

    structure_indices = [(x, y, z) for x in range(pad_width, valid_dim + pad_width)
                                  for y in range(pad_width, valid_dim + pad_width)
                                  for z in range(pad_width, valid_dim + pad_width)]
    random.shuffle(structure_indices)  # Randomize the order of indices

    while attempts < attempts_per_structure and added_entries < sample_size:
        for x, y, z in structure_indices:
            if attempts >= attempts_per_structure or added_entries >= sample_size:
                break  # Exit if max attempts or desired sample size is reached

            # Check for existence in DB already.
            sample_id = f"{structure_idx}-{x}-{y}-{z}" # Here we identify this x y z position in the given structure.
            if sample_id not in existing_samples:
                target = padded_structure[x, y, z]
                target_string = vector_to_string(target)
                # Check if this block type has reached its target occurrence
                if current_occurrences.get(target_string, 0) >= target_occurrences.get(target_string, float('inf')):
                    continue 
                if target_string in sampling_probabilities and random.random() < sampling_probabilities[target_string]:
                    #log(f"{target_string} passed sampling check")
                    context = [padded_structure[x + dx, y + dy, z + dz] for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]]
                    negative = []
                    negative_breakout = 0
                    while len(negative) < 6:
                        if negative_breakout >= 32768:
                            break
                        #log(f"Attempting to find valid negative for {target_string}")
                        nx, ny, nz = (random.randint(0, valid_dim - 1) + pad_width for _ in range(3))
                        if is_far_enough(x, y, z, nx, ny, nz, min_neg_distance):
                            new_negative = padded_structure[nx, ny, nz]
                            #log(f"{target_string} negative found: {vector_to_string(new_negative)}")
                            negative.append(new_negative)
                        else:
                            negative_breakout += 1
                    if len(negative) == 6:
                        entries.append((target, context, negative))
                        added_entries += 1
                        existing_samples.add(sample_id)
                        current_occurrences[target_string] = current_occurrences.get(target_string, 0) + 1
                        #log(f"{target_string} added to dataset")
                    else:
                        log(f"{target_string} skipped due to failed negative loop")
            
            attempts += 1

    return entries, added_entries

def log_block_occurrences(blocks_hdf5_path, sampling_probabilities):
    block_counts = defaultdict(int)
    
    with h5py.File(blocks_hdf5_path, 'r') as f:
        for entry_name in f.keys():
            entry = f[entry_name]
            target = entry['target'][:]
            target_string = vector_to_string(target)
            block_counts[target_string] += 1

    total_blocks = sum(block_counts.values())
    log("Block Occurrences in Dataset vs Sampling Targets:")
    log("")
    for block, count in block_counts.items():
        log(f"Block: {block}")
        log(f"Dataset: {(count / total_blocks):.4f}")
        log(f"Average: {sampling_probabilities[block]:.4f}")
        log("")

def calculate_target_occurrences(sampling_probabilities, sample_size):
    target_occurrences = {}
    for block, probability in sampling_probabilities.items():
        # Calculate the target number of occurrences for each block
        target_occurrences[block] = int(round(probability * sample_size))
    return target_occurrences

def load_existing(blocks_hdf5_path):
    current_occurrences = defaultdict(int)
    existing_samples = set()
    added_entries = 0
    
    with h5py.File(blocks_hdf5_path, 'r') as f:
        for entry_name in f.keys():
            entry = f[entry_name]
            target = entry['target'][:]
            target_string = vector_to_string(target)
            current_occurrences[target_string] += 1
            added_entries += 1
            
            # Assuming the structure index, x, y, z can be uniquely identified for each entry
            # If the structure index is not stored, you might need a different method to ensure uniqueness
            sample_id = entry_name  # Or construct this based on other data if available
            existing_samples.add(sample_id)
            
    # Estimate the next checkpoint based on added_entries
    checkpoint_next = ((added_entries // interval) + 1) * interval
    if checkpoint_next < threshold:
        checkpoint_next = threshold

    return current_occurrences, existing_samples, added_entries, checkpoint_next


def main(minecraft_dataset_path, blocks_hdf5_basepath, min_neg_distance, sample_size):
    minecraft_dataset = MinecraftDataset(minecraft_dataset_path)
    sampling_probabilities = calculate_sampling_probabilities(minecraft_dataset)
    target_occurrences = calculate_target_occurrences(sampling_probabilities, sample_size)

    current_occurrences = defaultdict(int)
    existing_samples = set()
    added_entries = 0
    checkpoint_next = threshold
    cycles_through_dataset = 0

    # Initialize the HDF5 file path
    blocks_hdf5_path = f"{blocks_hdf5_basepath}-{min_neg_distance}-target.hdf5"
    
    # Check if this file exists. If it does, update relevant variables to resume the collection up to sample size.
    if os.path.exists(blocks_hdf5_path):
        current_occurrences, existing_samples, added_entries, checkpoint_next = load_existing(blocks_hdf5_path)
        log(f"Loaded existing dataset info:")
        log(f"current_occurrences: {current_occurrences}, len(existing_samples): {len(existing_samples)}, added_entries: {added_entries}, checkpoint_next: {checkpoint_next}")

    while added_entries < sample_size:
        with h5py.File(blocks_hdf5_path, 'a') as f:
            for k, structure in enumerate(minecraft_dataset):
                if added_entries >= sample_size:
                    break  # Exit if the desired sample size is reached
                
                entries, added_entries = sample_blocks(k, structure, sampling_probabilities, min_neg_distance, added_entries, sample_size, existing_samples, target_occurrences, current_occurrences)
                for i, (target, context, negative) in enumerate(entries):
                    entry_num = added_entries - len(entries) + i + 1
                    grp = f.create_group(f"entry_{entry_num}")
                    grp.create_dataset("target", data=target)
                    grp.create_dataset("context", data=np.array(context))
                    grp.create_dataset("negative", data=np.array(negative))
                    existing_samples.add(f"{k}-{target}-{context}-{negative}")
                if len(entries) > 0:
                    log(f"Added {len(entries)} entries from structure {k}. Total: {added_entries}/{sample_size}")
                else:
                    log(f"Having some trouble finding entries. I'll keep looking!")

                # Check if it's time to create a checkpoint
                if added_entries >= checkpoint_next:
                    # Ensure data is written to disk before copying
                    f.flush()
                    checkpoint_path = f"{blocks_hdf5_basepath}-{min_neg_distance}-{checkpoint_next}.hdf5"
                    shutil.copy2(blocks_hdf5_path, checkpoint_path)
                    log(f"Checkpoint created at {checkpoint_next} entries: {checkpoint_path}")
                    checkpoint_next += interval
            cycles_through_dataset += 1

            # No explicit save or close needed here; the `with` block handles that automatically

    # Log final block occurrences and any other final information needed
    log_block_occurrences(blocks_hdf5_path, sampling_probabilities)
    log(f"Total cycles through the dataset to reach the sample size: {cycles_through_dataset}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build embeddings dataset.")
    parser.add_argument("--minecraft_dataset_path", type=str, default="../dataset/schematics.hdf5", help="Path to the Minecraft dataset.")
    parser.add_argument("--blocks_hdf5_path", type=str, default="../dataset/blocks", help="Path for the blocks HDF5 file.")
    parser.add_argument("--min_neg_distance", type=int, default=16, help="Minimum distance for negative samples.")
    parser.add_argument("--sample_size", type=int, default=200000, help="Total sample size.")
    args = parser.parse_args()

    main(args.minecraft_dataset_path, args.blocks_hdf5_path, args.min_neg_distance, args.sample_size)
