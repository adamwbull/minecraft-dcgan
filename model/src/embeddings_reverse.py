import numpy as np
import pickle
from scipy.spatial import distance

# TODO: Explore multiple methods of mapping embeddings back into vector space.

# Use euclidean distance.
def find_closest_embedding_context(new_embedding, path_to_dictionary):
    # Load the dictionary of embeddings
    with open(path_to_dictionary, 'rb') as file:
        embeddings_dict = pickle.load(file)

    min_distance = np.inf
    closest_block_id = None

    # Iterate through the dictionary to find the closest embedding
    for block_id, embeddings in embeddings_dict.items():
        for embedding in embeddings:
            # Calculate the Euclidean distance between the new and stored embeddings
            dist = distance.euclidean(new_embedding, embedding)
            if dist < min_distance:
                min_distance = dist
                closest_block_id = block_id

    return closest_block_id
