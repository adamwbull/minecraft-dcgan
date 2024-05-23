# trainfidcnn.py

import torch
import torch.optim as optim
import torch.nn as nn
from fidcnn import FID3DCNN
from torch.utils.data import DataLoader
from dataset import MinecraftDataset
import os
import time
import argparse
from variables.fidhyperparameters import default_batch_size, default_epochs, default_lr, default_save_every_epoch

# Argument parsing
parser = argparse.ArgumentParser(description="Train a GAN model on Minecraft dataset.")
parser.add_argument("name", type=str, help="Name for the trial. Specify an existing trial to continue it.")
parser.add_argument("--epochs", type=int, default=default_epochs, help="Number of training epochs.")
parser.add_argument("--lr", type=float, default=default_lr, help="Learning rate for training.")
parser.add_argument("--batch_size", type=int, default=default_batch_size, help="Training batch size.")
parser.add_argument("--save_every_epoch", type=int, default=default_save_every_epoch, help="Save checkpoint every X epochs.")
args = parser.parse_args()

# Logging function
def log(text="", console_only=False):
    current_date = time.strftime("%Y%m%d")
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")

    print(timestamp + " " + text)
    
    if not console_only:
        # Write to the latest.log file
        with open('./logs/fid-latest.log', 'a') as log_file:
            log_file.write(f"{timestamp} {text}\n")

        # Write to the daily log file
        daily_log_filename = f'./logs/fid-{current_date}.log'
        with open(daily_log_filename, 'a') as daily_log_file:
            daily_log_file.write(f"{timestamp} {text}\n")

# Function to manage log files
def initialize_logs():
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    open('./logs/fid-latest.log', 'w').close()

initialize_logs()

# Function to save checkpoints
def save_checkpoint(state, filename):
    torch.save(state, filename)

# Function to load checkpoint
def load_checkpoint(model, optimizer, filename):
    if os.path.isfile(filename):
        log(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        log(f"Loaded checkpoint '{filename}' (epoch {checkpoint['epoch']})")
    else:
        log(f"No checkpoint found at '{filename}'")
        start_epoch = 0
    return start_epoch

def train(model, dataloader, epochs, learning_rate, checkpoint_name):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    start_epoch = 0
    checkpoint_path = f'models/{checkpoint_name}.pth'
    if checkpoint_path is not None:
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path)

    for epoch in range(start_epoch, epochs):
        for data in dataloader:

            # Load in and re-arrange tensor dimensions to match generator/discriminator.
            data = data.permute(0, 4, 1, 2, 3)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()
        log(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

        # Save checkpoint periodically
        if epoch % 10 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
            }, f'fid_{checkpoint_name}_epoch_{epoch+1}.pth')

def main():
    data = MinecraftDataset()
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True)

    model = FID3DCNN()
    train(model, dataloader, args.epochs, args.lr, args.name)

if __name__ == "__main__":
    main()
