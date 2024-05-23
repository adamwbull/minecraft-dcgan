# dataset.py

import torch
from torch.utils.data import Dataset
import h5py
import os
import time

testing = False

# Logging function
def log(text="", console_only=False, should_log=False):

    if should_log:
        current_date = time.strftime("%Y%m%d")
        timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")

        print(timestamp + " " + text)
        
        if not console_only:
            # Write to the latest.log file
            with open('./embeddings/dataset/logs/latest.log', 'a') as log_file:
                log_file.write(f"{timestamp} {text}\n")

            # Write to the daily log file
            daily_log_filename = f'./embeddings/dataset/logs/{current_date}.log'
            with open(daily_log_filename, 'a') as daily_log_file:
                daily_log_file.write(f"{timestamp} {text}\n")

# Function to manage log files
def initialize_files():
    if not os.path.exists('./embeddings/dataset/logs'):
        os.makedirs('./embeddings/dataset/logs')
    open('./embeddings/dataset/logs/latest.log', 'w').close()

class BlockDataset(Dataset):
    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path
        with h5py.File(hdf5_path, 'r') as f:
            self.entries = list(f.keys())

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_path, 'r') as f:
            entry = f[self.entries[idx]]
            target = entry['target'][:]
            context = entry['context'][:]
            negative = entry['negative'][:]
        return target, context, negative

class MinecraftDataset(Dataset):
    
    def __init__(self, hdf5_path='../dataset/schematics_embeddings_32.hdf5'):
        with h5py.File(hdf5_path, 'r') as f:
            self.keys = list(f.keys())
        self.hdf5_path = hdf5_path

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        with h5py.File(self.hdf5_path, 'r') as f:
            key = self.keys[index]
            matrix = f[key][:]
            return torch.tensor(matrix, dtype=torch.float32)
