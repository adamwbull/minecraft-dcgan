import torch
import argparse
import numpy as np
import pickle
from collections import Counter
from embeddings_model import AdvancedBlockEmbeddings  # Ensure this is correctly imported
from vector import get_unique_block_vectors, vector_to_string

def main(checkpoint_path, output_pickle_file):
    # Ensure the model is compatible with this setup
    model = AdvancedBlockEmbeddings().to(device)
    model.eval()  # Set the model to evaluation mode

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    # Process unique blocks
    embeddings = {}
    unique_blocks = get_unique_block_vectors()  # This function needs to be defined appropriately
    with torch.no_grad():
        for block in unique_blocks:
            block_vector = torch.tensor(block, dtype=torch.float).to(device)
            embedding = model.block_processor(block_vector.unsqueeze(0))  # Add batch dimension if necessary
            embedding = embedding.squeeze(0).cpu().numpy()  # Remove batch dimension and move to CPU

            block_id = vector_to_string(block)  # Ensure this function correctly converts blocks to their ID representation

            if block_id not in embeddings:
                embeddings[block_id] = []
            # Ensure embedding uniqueness before adding
            if not any(np.array_equal(embedding, np.array(x)) for x in embeddings[block_id]):
                embeddings[block_id].append(embedding.tolist())

    # Save the embeddings to a pickle file
    with open(output_pickle_file, 'wb') as f:
        pickle.dump(embeddings, f)

    print(f"Embeddings saved to {output_pickle_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for unique blocks and save them to a pickle file.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the .pth.tar model checkpoint.")
    parser.add_argument("--output_pickle_file", type=str, required=True, help="Output path for the embeddings pickle file.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    main(args.checkpoint_path, args.output_pickle_file)
