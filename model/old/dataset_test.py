# dataset_test.py

import dataset
import random

def main():
    # Initialize the Minecraft dataset
    hdf5_path = '../dataset/schematics.hdf5'  # Adjust the path as needed
    minecraft_dataset = dataset.MinecraftDataset(hdf5_path=hdf5_path)
    
    # Initialize the BlockDataset with the Minecraft dataset
    block_dataset = dataset.BlockDataset(minecraft_dataset=minecraft_dataset)
    
    # Select a random index to fetch an item
    random_idx = random.randint(0, len(block_dataset) - 1)
    
    # Fetch a random entry from the BlockDataset
    target, context_blocks, negative_blocks = block_dataset[random_idx]
    
    # For this test, we'll just print the shapes of the returned arrays
    print("Target Block:", target.shape)
    print("Context Blocks:", [cb.shape for cb in context_blocks])
    print("Negative Blocks:", [nb.shape for nb in negative_blocks])

if __name__ == "__main__":
    main()
