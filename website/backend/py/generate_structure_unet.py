# generate_structure.py
import base64
import torch
import numpy as np
#import mcschematic
from hyperparameters import feature_map_size, block_vector_dim, noise_dim, embedding_vector_dim, ldm_latent_dim
from generator import RGDCGANGenerator, EDCGANGenerator, UNetWithMultiLevelCrossAttention, UNetDecoder
import os
import random
import string
import argparse
import sqlite3
from vector import get_block_id, get_properties
from datetime import datetime
import pickle

#from scipy.spatial.distance import cdist

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

# For development printing
testing = False

# Parse command line arguments for generator name
parser = argparse.ArgumentParser(description='Generate structures using a specified generator model.')
parser.add_argument('generator_name', type=str, help='Name of the generator model to use')
parser.add_argument('generator_type', type=int, help='Choose between model architectures with an int.')
parser.add_argument('rga', type=str, help='level of RGA to use')
args = parser.parse_args()

# Assuming Generator is similar to WGPGenerator and softmaxgens.py is the relevant module
# and using the same device setting as in multigenerate.py
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_random_vector(size):
    return torch.randn(1, size, 1, 1, 1)

# Generator Initialization (Now including a choice of generators)
def initialize_generator(checkpoint_name, checkpoint_type, rga):

    generator = None
    decoder = None

    if checkpoint_type == 0:
        # Vectors
        generator = RGDCGANGenerator(noise_dim, block_vector_dim, feature_map_size, rga).to(device)
    elif checkpoint_type == 1:
        # Embeddings
        generator = EDCGANGenerator(noise_dim, embedding_vector_dim, feature_map_size).to(device)
    elif checkpoint_type == 2:
        # Embeddings LDM v1
        generator = UNetWithMultiLevelCrossAttention(in_channels=ldm_latent_dim, out_channels=embedding_vector_dim)
        decoder = UNetDecoder(latent_dim=ldm_latent_dim, embedding_dim=embedding_vector_dim)
    
    # Acquire states from checkpoint.
    generator_state_dict = None
    decoder_state_dict = None
    
    checkpoint = torch.load(f'./py/models/{checkpoint_name}.pth.tar', map_location=device)

    if checkpoint_type <= 1:
        generator_state_dict = checkpoint['generator_state']
    elif checkpoint_type == 2:
        generator_state_dict = checkpoint['unet_state_dict']
        decoder_state_dict = checkpoint['decoder_state_dict']
        
    # Load in states.
        
    # Generator.
    if list(generator_state_dict.keys())[0].startswith("module."):
        generator_state_dict = {key[len("module."):]: value for key, value in generator_state_dict.items()}
    generator.load_state_dict(generator_state_dict)
    
    # Decoder.
    if checkpoint_type == 2:
        decoder.load_state_dict(decoder_state_dict)

    return generator, decoder

# Generate Structure
def generate_structure(generator, decoder, checkpoint_type):
    if checkpoint_type <= 1:
        generator.eval()
        noise = torch.randn(1, noise_dim, device=device)
        with torch.no_grad():
            generated_sample = generator(noise).squeeze(0)
        reshaped_sample = generated_sample.permute(1, 2, 3, 0)
        return reshaped_sample.cpu().numpy() 
    elif checkpoint_type == 2:
        generator.eval()
        decoder.eval()
        with torch.no_grad():
            # Grab a random vector from the latent space.
            random_latent_vector = generate_random_vector(ldm_latent_dim)
            
            # Pass through model to generate a unique structure.
            generated_latent = generator(random_latent_vector)
            generated_structure = decoder(generated_latent)
            return generated_structure
    else:
        return None

# Postprocess Structure (Adapting from postprocessing.py)
def normalize_vector(vector):
    block_type = vector[:11]
    max_index_block = np.argmax(np.abs(block_type))
    normalized_block = np.zeros_like(block_type)
    normalized_block[max_index_block] = 1

    direction = vector[11:15]
    max_index_direction = np.argmax(np.abs(direction))
    normalized_direction = np.zeros_like(direction)
    normalized_direction[max_index_direction] = 1

    last_bit = [1 if vector[15] > 0.5 else 0]
    normalized_vector = np.concatenate([normalized_block, normalized_direction, last_bit])
    return normalized_vector.tolist()

def init_postprocess_structure(structure, checkpoint_type):

    if checkpoint_type == 0:
        # Normalize each vector into the one-hot categories.
        return np.array([[[normalize_vector(block) for block in zArray] for zArray in yArray] for yArray in structure])
    else:
        # So far, all other models are embeddings.
        # For embeddings, we will be mapping them back to block ids in 
        # our final convert_to_schem_and_encode postprocessing step.
        return structure

def generate_random_filename(length=10):
    """Generate a random filename of given length."""
    letters_and_digits = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters_and_digits) for i in range(length))

def get_or_create_model_id(model_name, db_path='py/structures.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Check if the model exists and get its ID
    cursor.execute('SELECT id FROM models WHERE name = ?', (model_name,))
    model_id = cursor.fetchone()
    if model_id:
        model_id = model_id[0]
    else:
        # If not exists, insert the new model and get its ID
        current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        cursor.execute('INSERT INTO models (name, created_at) VALUES (?, ?)', (model_name, current_datetime))
        conn.commit()
        model_id = cursor.lastrowid
    conn.close()
    return model_id

def record_schematic_in_db(name, filepath, generator_name, checkpoint_type, db_path='py/structures.db'):
    model_id = get_or_create_model_id(generator_name, db_path)
    current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO model_schematics (name, filepath, model_id, created_at, model_type) VALUES (?, ?, ?, ?, ?)
    ''', (name, filepath, model_id, current_datetime, checkpoint_type))
    conn.commit()
    conn.close()

def find_nearest_embedding(embedding, embeddings_dict):
    # Flatten the embedding to 1D if it's not already
    embedding_flattened = embedding.flatten()
    # Calculate distances (using cosine similarity or Euclidean distance)
    min_distance = float('inf')
    nearest_block_id = None
    for block_id, block_embeddings in embeddings_dict.items():
        for block_embedding in block_embeddings:
            distance = np.linalg.norm(embedding_flattened - np.array(block_embedding))
            if distance < min_distance:
                min_distance = distance
                nearest_block_id = block_id
    return nearest_block_id

def convert_to_schem_and_encode(structure, generator_name, checkpoint_type, db_path='py/structures.db'):
    schem = mcschematic.MCSchematic()

    # For use if we are an embeddings generator output.
    embeddings = None
    if checkpoint_type == 1:
        embeddings = load_embeddings('./embeddings.pkl')

    for x in range(structure.shape[0]):
        for y in range(structure.shape[1]):
            for z in range(structure.shape[2]):

                if checkpoint_type == 0:
                    # Vectors.
                    vector = structure[x, y, z]
                    block_id, block_type_index = get_block_id(vector)
                    properties = get_properties(vector, block_type_index)
                    block_string = block_id + (properties if properties else "")
                    schem.setBlock((x, y, z), block_string)
                elif checkpoint_type >= 1:
                    # Embeddings.
                    generated_embedding = structure[x, y, z]
                    nearest_block_id = find_nearest_embedding(generated_embedding, embeddings)
                    # Use the nearest_block_id to setBlock
                    schem.setBlock((x, y, z), nearest_block_id)
    
    # Adjust the directory structure to include the model name
    model_directory = os.path.join('py/schematics', generator_name)
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    
    filename = generate_random_filename(10)
    filename_with_schem = filename + '.schem'
    filepath = os.path.join(model_directory, filename_with_schem)
    schem.save(outputFolderPath=model_directory, schemName=filename, version=mcschematic.Version.JE_1_19)
    
    # Encode the schematic content
    with open(filepath, 'rb') as file:
        schematic_content = file.read()
    encoded_schem_content = base64.b64encode(schematic_content).decode('utf-8')
    
    # Record schematic in the database
    record_schematic_in_db(filename, filepath, generator_name, checkpoint_type, db_path)

    return filename, encoded_schem_content

# Ensure the 'schematics' directory exists
if not os.path.exists('schematics'):
    os.makedirs('schematics')

if __name__ == "__main__":

    # Create structure.
    generator, decoder = initialize_generator(args.generator_name, args.generator_type, int(args.rga))
    structure = generate_structure(generator, decoder, args.generator_type)
    postprocessed_structure = init_postprocess_structure(structure, args.generator_type)

    print('Structure generated.')

    print(structure)

    # Save structure.
    #filename, base64_schem_content = convert_to_schem_and_encode(postprocessed_structure, args.generator_name, args.generator_type)

    # Return structure data to any callers.
    #print(filename)
    #print(base64_schem_content)