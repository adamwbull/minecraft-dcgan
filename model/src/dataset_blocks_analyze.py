# dataset_blocks_analyze
import h5py
import numpy as np
from collections import defaultdict
import random
from vector import string_to_vector, vector_to_string
from dataset import MinecraftDataset
import os
import argparse


def calculate_sampling_probabilities(minecraft_dataset):
    sampling_probabilities_path = f"sampling-probabilities-{len(minecraft_dataset)}.npy"
    sampling_probabilities = None
    if os.path.exists(sampling_probabilities_path):
        print("Loading established probabilities...")
        sampling_probabilities = np.load(sampling_probabilities_path, allow_pickle=True).item()
    else:
        block_counts = defaultdict(int)
        print("Calculating new sampling probabilities...")
        total_length = len(minecraft_dataset)
        for i, structure in enumerate(minecraft_dataset):
            reshaped_structure = structure.reshape(-1, structure.shape[-1])
            unique_vectors, counts = np.unique(reshaped_structure, axis=0, return_counts=True)
            for vector, count in zip(unique_vectors, counts):
                block_counts[vector_to_string(vector)] += count
            if (i + 1) % 500 == 0:
                print(f"{i+1}/{total_length} structures counted")

        total_blocks = sum(block_counts.values())
        block_probabilities = {block: count / total_blocks for block, count in block_counts.items()}
        sampling_probabilities = {block: (np.sqrt(probability / 0.001) + 1) * (0.001 / probability) for block, probability in block_probabilities.items()}
        
        # Normalize the probabilities
        sum_probabilities = sum(sampling_probabilities.values())
        sampling_probabilities = {block: prob / sum_probabilities for block, prob in sampling_probabilities.items()}
        
        np.save(sampling_probabilities_path, sampling_probabilities)
        print(f"Sampling probabilities saved to {sampling_probabilities_path}.")
        
    return sampling_probabilities

def print_block_occurrences(blocks_hdf5_path, sampling_probabilities=None):
    block_counts = defaultdict(int)
    
    with h5py.File(blocks_hdf5_path, 'r') as f:
        for entry_name in f.keys():
            entry = f[entry_name]
            target = entry['target'][:]
            target_string = vector_to_string(target)
            block_counts[target_string] += 1
        print(f"entry count: {len(f.keys())}")

    total_blocks = sum(block_counts.values())
    print("Block Occurrences in Dataset vs Sampling Targets (if provided):")
    print("")
    for block, count in block_counts.items():
        dataset_percentage = count / total_blocks
        print(f"Block: {block}")
        print(f"Dataset: {dataset_percentage:.4f}")
        if sampling_probabilities and block in sampling_probabilities:
            print(f"Target: {sampling_probabilities[block]:.4f}")
        print("")

def analyze_existing_blocks_dataset(blocks_hdf5_path, minecraft_dataset_path, min_neg_distance):
    minecraft_dataset = MinecraftDataset(minecraft_dataset_path)
    sampling_probabilities = calculate_sampling_probabilities(minecraft_dataset)
    print_block_occurrences(blocks_hdf5_path, sampling_probabilities)

analyze_existing_blocks_dataset("../dataset/blocks-16-target.hdf5", "../dataset/schematics.hdf5", 3)