# proceduralgenerator.py

import argparse
import torch

# Just some procedural generation tech I whipped together.
# Perhaps someday, this can be refined more to cover larger generated areas requested by an application.

class ProceduralGenerator:
    def __init__(self, generator, text_encoder, vocab, noise_dim):
        self.generator = generator
        self.text_encoder = text_encoder
        self.vocab = vocab
        self.noise_dim = noise_dim

    def _generate_building(self, desc):
        """Generate a building using the description."""
        # Convert text_data to suitable tensor
        text_data = [self.vocab.get(word, self.vocab['<UNK>']) for word in desc]
        text_data = torch.nn.utils.rnn.pad_sequence([torch.tensor(text_data)], batch_first=True, padding_value=self.vocab['<PAD>'])
        text_embed = self.text_encoder(text_data)

        noise = torch.randn(1, self.noise_dim)
        fake_data = self.generator(noise, text_embed)
        return fake_data

    def generate_composite(self, target_size, descriptions):
        """Generate a composite structure of target size using a list of descriptions."""
        composite_structure = torch.zeros(target_size)  # initialize with zeros

        # This is a simplistic approach where we fill the target area sequentially with generated buildings.
        # More sophisticated algorithms can be designed to optimally fill space.
        z_idx, y_idx, x_idx = 0, 0, 0
        for desc in descriptions:
            building = self._generate_building(desc)
            b_z, b_y, b_x = building.shape[2:]
            
            # Assuming 3D structures and filling in a 3D space.
            if x_idx + b_x > target_size[2]:
                x_idx = 0
                y_idx += b_y
            if y_idx + b_y > target_size[1]:
                y_idx = 0
                z_idx += b_z
            if z_idx + b_z > target_size[0]:
                break  # we've filled the target area

            composite_structure[z_idx:z_idx+b_z, y_idx:y_idx+b_y, x_idx:x_idx+b_x] = building
            x_idx += b_x

        return composite_structure

    def random_descriptions(self, n):
        """Generate a list of n random descriptions."""
        descriptions = []
        for _ in range(n):
            desc = [random.choice(self.vocab[key]) for key in self.vocab.keys() if key != '<PAD>' and key != '<UNK>']
            descriptions.append(desc)
        return descriptions


# Execution.