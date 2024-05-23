# train.py

import argparse
import os
import random
import string
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ldm import Encoder, UNetSimple, UNetWithMultiLevelCrossAttention, Decoder
from dataset import MinecraftDataset
import time

# Logging function
def log(text="", console_only=False):
    current_date = time.strftime("%Y%m%d")
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")

    print(timestamp + " " + text)
    
    if not console_only:
        # Write to the latest.log file
        with open('./logs/latest.log', 'a') as log_file:
            log_file.write(f"{timestamp} {text}\n")

        # Write to the daily log file
        daily_log_filename = f'./logs/{current_date}.log'
        with open(daily_log_filename, 'a') as daily_log_file:
            daily_log_file.write(f"{timestamp} {text}\n")

# Function to manage log files
def initialize_logs():
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    open('./logs/latest.log', 'w').close()

# Initialize log files
initialize_logs()

def generate_random_string(length=5):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

def save_checkpoint(state, filename="checkpoint.pth.tar", args_dict=None):

    # Create or append to the list of training sessions
    if 'saves' not in state:
        state['saves'] = []
    
    # Add a timestamp and the args_dict to the current save state
    current_save_info = {}
    if args_dict is not None:
        current_save_info = args_dict.copy()
        current_save_info['saved'] = time.strftime("%Y-%m-%d %H:%M:%S")
    state['saves'].append(current_save_info)

    torch.save(state, filename)

def load_checkpoint(checkpoint_path, encoder, unet, decoder, optimizer):
    checkpoint = torch.load(checkpoint_path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    unet.load_state_dict(checkpoint['unet_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']

def apply_noise(data, step, max_steps, device):
    """Applies progressive noise to the data based on the current step."""
    noise_level = step / max_steps
    return data + noise_level * torch.randn_like(data).to(device)

def train_loop(encoder, unet, decoder, dataloader, optimizer, criterion, device, current_step, max_steps):
    """A single training loop over the dataset, applying noising and learning to denoise."""
    for batch_idx, data in enumerate(dataloader):
        data = data.to(device)
        noised_data = apply_noise(data, current_step, max_steps, device)
        optimizer.zero_grad()
        latent = encoder(noised_data)
        denoised = unet(latent)
        outputs = decoder(denoised)
        loss = criterion(outputs, data)  # Target is the original, clean data
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            log(f"Step: {current_step}, Batch: {batch_idx}, Loss: {loss.item()}")

def train(args):
    
    cuda_available = torch.cuda.is_available()
    log(f"torch.cuda.is_available(): {cuda_available}")
    device = torch.device("cuda" if cuda_available else "cpu")
    dataset = MinecraftDataset()  # Assumes this returns a torch Dataset
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    encoder = Encoder(args.embedding_dim, args.latent_dim).to(device)

    # Determine which architecture to use.
    unet = None
    if args.attention:
        log(f"Selected UNetWithMultiLevelCrossAttention as denoiser.")
        unet = UNetWithMultiLevelCrossAttention(args.latent_dim, args.latent_dim).to(device)
    else:
        log(f"Selected UNetSimple as denoiser.")
        unet = UNetSimple(args.latent_dim, args.latent_dim).to(device)

    decoder = Decoder(args.latent_dim, args.embedding_dim).to(device)
    optimizer = optim.Adam(list(encoder.parameters()) + list(unet.parameters()) + list(decoder.parameters()), lr=args.learning_rate)
    criterion = nn.MSELoss()  # Consider a loss function that can handle the progressive noising
    start_epoch = 0
    model_directory = f"./models"

    if os.path.exists(model_directory):
        for file in os.listdir(model_directory):
            if file.startswith(args.name) and file.endswith("_target.pth.tar"):
                checkpoint_path = './models/' + file 
                start_epoch = load_checkpoint(checkpoint_path, encoder, unet, decoder, optimizer)
                log(f"Resuming training from epoch {start_epoch}")

    # Convert args to a dictionary and remove non-serializable entries
    args_dict = vars(args)
    args_dict.pop('attention', None)  # Remove non-serializable entries if necessary

    for epoch in range(1, args.num_epochs + 1):
        
        log(f"Starting epoch {epoch}/{args.num_epochs}")

        for current_step in range(1, args.num_steps + 1):
            train_loop(encoder, unet, decoder, dataloader, optimizer, criterion, device, current_step, args.num_steps)
        
        log(f"Completed epoch {epoch}/{args.num_epochs}")

        if epoch % args.save_every_epoch == 0:
            checkpoint_name = f"{args.name}_{args.batch_size}batch_{epoch}epochs_{generate_random_string()}.pth.tar"
                
            save_checkpoint({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'unet_state_dict': unet.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join("./models", checkpoint_name), args_dict)
            log(f"Checkpoint saved.")

    # Save final checkpoint
    random_str = generate_random_string()
    save_checkpoint({
        'epoch': args.num_epochs,
        'encoder_state_dict': encoder.state_dict(),
        'unet_state_dict': unet.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, f"./models/{args.name}_{args.batch_size}batch_{args.num_epochs}epochs_{random_str}_terminated.pth.tar", args_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Latent Diffusion Model for Minecraft Structure Generation")
    parser.add_argument("--name", type=str, required=True, help="Model name")
    parser.add_argument("--attention", type=bool, default=True, help="Model name")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--num_steps", type=int, default=100, help="Number of de/noising steps")
    parser.add_argument("--save_every_epoch", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--embedding_dim", type=int, default=32, help="Dimension of embedding vectors")
    parser.add_argument("--latent_dim", type=int, default=64, help="Dimension of latent space vectors")

    args = parser.parse_args()
    
    if not os.path.exists("./models"):
        os.makedirs("./models")
    
    train(args)