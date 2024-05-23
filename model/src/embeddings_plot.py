# embeddings_plot.

import os
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances 
import numpy as np
import argparse

def wrap_labels(labels, wrap_length):
    """Wrap labels to a specified length."""
    return ['\n'.join(textwrap.wrap(label, wrap_length)) for label in labels]

color_map = {
    "minecraft:air": "#58B4ED",
    "minecraft:dirt": "#8d6545",
    "minecraft:stone": "#8a8a8a",
    "minecraft:cobblestone": "#a8a8a8",
    "minecraft:stone_bricks": "#696666",
    "minecraft:oak_planks": "#f1a62f",
    "minecraft:oak_log[axis=x]": "#f1c988",
    "minecraft:oak_log[axis=y]": "#f1c988",
    "minecraft:oak_log[axis=z]": "#f1c988",
    "minecraft:glass": "#dfd5d5",
    "minecraft:white_wool": "#ffffff",
    "minecraft:oak_stairs[facing=north,half=bottom,shape=straight,waterlogged=false]": "#dcb06a",
    "minecraft:oak_stairs[facing=east,half=bottom,shape=straight,waterlogged=false]": "#dcb06a",
    "minecraft:oak_stairs[facing=south,half=bottom,shape=straight,waterlogged=false]": "#dcb06a",
    "minecraft:oak_stairs[facing=west,half=bottom,shape=straight,waterlogged=false]": "#dcb06a",
    "minecraft:oak_slab[type=bottom,waterlogged=false]": "#b7945c",
    "minecraft:oak_slab[type=top,waterlogged=false]": "#b7945c"
}

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #

# Ensure seaborn is set up for the heatmaps
sns.set_theme()

# Plotting functions for end-of-epoch.

# Note that the embeddings is a dict where:
# embeddings[block_id string] = [list of embeddings for that block id]
# block_id is a string representation of the block, and each embedding entry is a 32 length vector with ranging float values
# in contextual setting, this list is long, where with the normal it will be a list with 1 embedding.
# Use PCA to plot embeddings.

def plot_embeddings_3d(embeddings, epoch, name, save_path='embeddings/plots/3d'):
    try:
        labels = []  # To store block IDs
        all_embeddings = []  # To store embeddings
        for block_id, embedding in embeddings.items():
            # Move each embedding to CPU and then extend the list
            all_embeddings.extend([e.cpu() for e in embedding])  # Adjusted to move tensors to CPU
            labels.extend([block_id] * len(embedding))  # Repeat block_id for each embedding
        
        # Convert embeddings to numpy array for PCA
        all_embeddings = np.vstack([e.numpy() for e in all_embeddings])  # Adjusted for CPU tensors
        
        # Apply PCA to reduce to 3 dimensions
        pca = PCA(n_components=3)
        embeddings_reduced = pca.fit_transform(all_embeddings)

        # Plotting in 3D
        fig = plt.figure(figsize=(18, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create a color list for the labels
        unique_labels = list(set(labels))
        colors = [color_map.get(label, '#000000') for label in labels]
        
        for i, label in enumerate(unique_labels):
            idxs = [idx for idx, lbl in enumerate(labels) if lbl == label]
            ax.scatter(embeddings_reduced[idxs, 0], embeddings_reduced[idxs, 1], embeddings_reduced[idxs, 2], color=color_map.get(label, '#000000'), label=label)

        # Adding legend and labels with a check to avoid too many labels
        if len(unique_labels) <= 30:  # Arbitrary limit to avoid overcrowding
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title='Blocks')
        
        ax.set_title(f'3D Block Embeddings Epoch {epoch}')
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        ax.set_zlabel('PCA 3')
        
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f'{save_path}{name}_{epoch}.png')
        plt.close()
    except Exception as e:
        print(f"plot_embeddings_3d failed: {e}")

# Plot the non contextual embeddings.
def plot_embeddings(embeddings, epoch, name):
    #print("plot_embeddings")
    try:
        labels = []  # To store block IDs
        all_embeddings = []  # To store embeddings
        for block_id, embedding in embeddings.items():
            all_embeddings.extend(embedding)  # Assuming each block_id has a list of embeddings
            labels.extend([block_id] * len(embedding))  # Repeat block_id for each embedding

        # Apply PCA to reduce to 2 dimensions
        pca = PCA(n_components=2)
        all_embeddings = np.vstack([embedding.cpu().numpy() for embedding in all_embeddings])
        embeddings_reduced = pca.fit_transform(all_embeddings)

        # Plotting with labels
        plt.figure(figsize=(14, 7))
        for i, label in enumerate(labels):
            plt.scatter(embeddings_reduced[i, 0], embeddings_reduced[i, 1], color=color_map.get(labels[i], '#000000'))
            plt.text(embeddings_reduced[i, 0], embeddings_reduced[i, 1], label, fontsize=9)

        plt.title(f'Block Embeddings Epoch {epoch}')
        os.makedirs('embeddings/plots/2d', exist_ok=True)
        plt.savefig(f'embeddings/plots/2d/{name}_{epoch}.png')
        plt.close()
    except Exception as e:
        print(f"plot_embeddings failed: {e}")

def create_confusion_matrix(embeddings, epoch, name):
    #print("create_confusion_matrix")
    try:
        # Assuming `embeddings` is a dictionary with block_ids as keys and embeddings as values
        labels = list(embeddings.keys())
        # Flatten the list of embeddings and keep track of labels for each embedding
        embeddings_list = []
        for embedding_group in embeddings.values():
            for embedding in list(embedding_group):  # Assuming each group is already a numpy array
                embeddings_list.append(embedding)

        if not embeddings_list:  # Check if the list is empty
            print("No embeddings to create a confusion matrix.")
            return

        embeddings_list = [embedding.cpu() for embedding in embeddings_list]  # Move each tensor to CPU
        embeddings_matrix = np.vstack([embedding.numpy() for embedding in embeddings_list])

        # Calculate pairwise distances
        distances = pairwise_distances(embeddings_matrix, metric='euclidean')
        
        plt.figure(figsize=(20, 20))
        sns.heatmap(distances, xticklabels=labels, yticklabels=labels, cmap='viridis')
        plt.title(f'Embeddings Distance Matrix Epoch {epoch}')
        os.makedirs('embeddings/confusion', exist_ok=True)
        plt.savefig(f'embeddings/confusion/{name}_{epoch}.png')
        plt.close()
    except Exception as e:
        print(f"create_confusion_matrix failed: {e}")

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #

if __name__ == '__main__':
    main()

def main():

    parser = argparse.ArgumentParser(description="create a 3d plot for a given embedding table")
    parser.add_argument("dict_file", type=str, help="KPL file to target")

    args = parser.parse_args()

    kpl = {}
    if args.dict_file:
        with open(args.dict_file, 'rb') as f:
            kpl = pickle.load(f)
        log(f"Loaded dictionary from {args.dict_file}")

    plot_embeddings_3d(kpl, '100', 'Final Embeddings', './')

