# gen_area.py

import argparse
import os
import torch

from generators.proceduralgenerator import ProceduralGenerator
from generators.cdcgan import TextEncoder, Generator
from variables.hyperparameters import vocab, feature_map_size, block_vector_dim, text_embed_dim, noise_dim, embedding_dim

parser = argparse.ArgumentParser(description="Generate a larger composite structure using multiple cDCGAN outputs.")

# Argument for the checkpoint name (e.g. "checkpoint_100")
parser.add_argument("checkpoint", type=str, help="Name of the checkpoint including epoch (e.g. checkpoint_100).")
args = parser.parse_args()

# Load the checkpoint based on the provided name
checkpoint_path = f"{args.checkpoint_name}.pth.tar"
if not os.path.exists(checkpoint_path):
    print(f"Error: {checkpoint_path} does not exist!")
    exit(1)

checkpoint = torch.load(checkpoint_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the generator and text encoder from the checkpoint
generator = Generator(text_embed_dim, noise_dim, block_vector_dim, feature_map_size).to(device)
text_encoder = TextEncoder(len(vocab), embedding_dim).to(device)

generator.load_state_dict(checkpoint['generator_state'])
text_encoder.load_state_dict(checkpoint['text_encoder_state']) 

# Initialize the ProceduralGenerator and generate composite structures as needed
proc_gen = ProceduralGenerator(generator, text_encoder, vocab)

# Further commands for generating structures or any other procedures can be added here...
# For demonstration purposes:
descriptions = proc_gen.random_descriptions(10)
composite_structure = proc_gen.generate_composite((256, 100, 256), descriptions)
print("Generated composite structure!")