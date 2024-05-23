# dataset.py

import torch
from torch.utils.data import Dataset
import h5py

class MinecraftDataset(Dataset):
    
    def __init__(self, hdf5_path='../dataset/schematics.hdf5'):
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
