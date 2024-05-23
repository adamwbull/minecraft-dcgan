import torch
import argparse
from softmaxgens import WGPGenerator
from variables.hyperparameters import feature_map_size, block_vector_dim, noise_dim
import os
import json
import random
import string
import time

def generate_random_string(length=5):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def generate_sample_from_checkpoint(checkpoint_name, noise_dim, device):

    # Initialize generator
    generator = WGPGenerator(noise_dim, block_vector_dim, feature_map_size).to(device)
    
    # Load checkpoint
    checkpoint = torch.load('./models/'+checkpoint_name+'.pth.tar')
    
    # Handle the DDP 'module.' prefix
    state_dict = checkpoint['generator_state']
    if list(state_dict.keys())[0].startswith("module."):
        # Remove 'module.' prefix
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    
    generator.load_state_dict(state_dict)
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    print(timestamp + f" Generating sample from {checkpoint_name}.")
    
    # Generate sample
    noise = torch.randn(1, noise_dim).to(device)
    with torch.no_grad():
        generated_sample = generator(noise)

    # Reshape and permute the tensor to the desired format
    reshaped_sample = generated_sample.squeeze(0)  # Remove the batch dimension
    reshaped_sample = reshaped_sample.permute(1, 2, 3, 0)
    
    return reshaped_sample

# Convert the 4D tensor into a JSON serializable format.
def tensor_to_json_format(tensor):
    result = []
    for i in range(tensor.shape[0]): 
        x_array = []
        for j in range(tensor.shape[1]): 
            y_array = []
            for k in range(tensor.shape[2]):
                vector = tensor[i, j, k].tolist() 
                y_array.append(vector)
            x_array.append(y_array)
        result.append(x_array)
    
    return result

def save_sample_as_json(sample, model_name, output_folder='generations'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    unique_string = generate_random_string()
    filename = f"{model_name}_{unique_string}.json"
    filepath = os.path.join(output_folder, filename)

    while os.path.exists(filepath):
        unique_string = generate_random_string()
        filename = f"{model_name}_{unique_string}.json"
        filepath = os.path.join(output_folder, filename)

    formatted_sample = tensor_to_json_format(sample)
    with open(filepath, 'w') as f:
        json.dump(formatted_sample, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate multiple samples from GAN models.")
    parser.add_argument('--samples', type=int, default=1, help="Number of samples to generate per model.")
    parser.add_argument('--ignore', nargs='*', default=[], help="List of models to ignore.")
    parser.add_argument('--only', nargs='*', default=[], help="List of specific models to generate samples for.")  # New argument for --only
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = [f for f in os.listdir('./models') if f.endswith('.pth.tar')]

    if args.only:
        models_to_use = [m for m in args.only if m + '.pth.tar' in models and m not in args.ignore]
    else:
        models_to_use = [m for m in models if m.replace('.pth.tar', '') not in args.ignore]

    for model in models_to_use:
        model_name = model.replace('.pth.tar', '')
        for _ in range(args.samples):
            generated_sample = generate_sample_from_checkpoint(model_name, noise_dim, device)
            save_sample_as_json(generated_sample, model_name)
