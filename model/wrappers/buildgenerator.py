# buildgenerator.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from generators.cdcgan import Generator
from variables.hyperparameters import text_embed_dim, noise_dim, block_vector_dim, feature_map_size

class BuildGenerator:

    def __init__(self, checkpoint_path):
        self.generator = self.load_generator(checkpoint_path)
            
    # Load generator from checkpoint.
    @staticmethod
    def load_generator(checkpoint_path):
        generator = Generator(text_embed_dim, noise_dim, block_vector_dim, feature_map_size)
        checkpoint = torch.load(checkpoint_path)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        generator.eval()  # Set the model to evaluation mode
        return generator

    # Process a text prompt to fit a specific length.
    # If longer than the desired length, truncate it. If shorter, pad it with whitespace.
    @staticmethod
    def process_text_prompt(prompt, max_length):

        if len(prompt) > max_length:
            # Convert the prompt into a comma-separated format to save space
            prompt = ','.join(prompt.split())

            # If still too long, truncate it
            if len(prompt) > max_length:
                prompt = prompt[:max_length]
        else:
            # If shorter than desired, pad it with whitespace
            prompt = prompt.ljust(max_length)

        return prompt

    # Generate a single image using cDCGAN.
    def generate_build(self, height, width, depth, prompt):
        # Process the text prompt
        processed_prompt = self.process_text_prompt(prompt, text_embed_dim)

        # Sample noise. For the text embedding, transform the processed prompt into a tensor.
        # Note: This is a naive way to convert the prompt into a tensor. Depending on your model, you might want to have a more sophisticated method.
        noise = torch.randn((1, noise_dim))
        text_embed = torch.FloatTensor([ord(char) for char in processed_prompt]).unsqueeze(0)

        # Generate the build
        build = self.generator(noise, text_embed)
        
        return build
