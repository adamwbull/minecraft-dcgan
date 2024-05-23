# train_distributed.py

import argparse
import os
import random
import string
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.multiprocessing import spawn
from ldm import Encoder, UNetSimple, UNetWithMultiLevelCrossAttention, Decoder
from dataset import MinecraftDataset
import time
from sklearn.model_selection import train_test_split
import numpy as np
import torch.distributed as dist
from ssim import SSIM3DLossMultiChannel

# Early stopping for validation loop.
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, rank=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.rank = rank

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
    
    def check_stop(self):
        """
        Checks if early stopping is triggered and broadcasts the decision to all processes
        """
        if self.rank == 0:
            if self.early_stop:
                dist.broadcast(torch.tensor([1]), 0)
            else:
                dist.broadcast(torch.tensor([0]), 0)
        else:
            stop_signal = torch.tensor([0])
            dist.broadcast(stop_signal, 0)
            self.early_stop = True if stop_signal.item() == 1 else False

# For reducing losses across the distributed setup.
def reduce_mean(tensor, nprocs):
    """
    Reduces the tensor data across all processes
    :param tensor: Tensor to be averaged
    :param nprocs: Number of processes participating in the job
    """
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= nprocs
    return rt

# Split the dataset into training and validation.
def split_dataset(dataset, train_size=0.7, seed=42):
    # Ensure reproducibility of the split
    random_state = np.random.RandomState(seed)
    train_indices, val_indices = train_test_split(range(len(dataset)),
                                                  train_size=train_size,
                                                  random_state=random_state)
    return torch.utils.data.Subset(dataset, train_indices), torch.utils.data.Subset(dataset, val_indices)

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

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_loop(encoder, unet, decoder, dataloader, optimizer, criterion, device, current_step, max_epochs, rank, world_size):
    local_loss = 0.0
    num_batches = len(dataloader)  # Number of batches processed by this GPU

    for data in dataloader:
        data = data.to(device)
        data = data.permute(0, 4, 1, 2, 3) 
        noised_data = apply_noise(data, current_step, max_epochs, device)
        optimizer.zero_grad()
        latent = encoder(noised_data)
        denoised = unet(latent)
        outputs = decoder(denoised)
        loss = criterion(outputs, data)
        loss.backward()
        optimizer.step()

        # Accumulate loss
        local_loss += loss.item()

    # Average the loss over all batches for this GPU
    local_loss /= num_batches

    # Reduce across all processes to get the average loss across all GPUs
    reduced_loss = reduce_mean(torch.tensor(local_loss, device=device), world_size).item()

    if rank == 0:
        log(f"Step: {current_step}, Avg Training Loss: {reduced_loss}")
 
def train(rank, world_size, args):
    
    setup(rank, world_size)
    
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    dataset = MinecraftDataset()
    train_dataset, val_dataset = split_dataset(dataset, train_size=0.7, seed=42)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, sampler=val_sampler)

    encoder = Encoder(args.embedding_dim, args.latent_dim).to(device)

    if args.attention:
        if rank == 0:
            log(f"Selected UNetWithMultiLevelCrossAttention as denoiser.")
        unet = UNetWithMultiLevelCrossAttention(args.latent_dim, args.latent_dim).to(device)
    else:
        if rank == 0:
            log(f"Selected UNetSimple as denoiser.")
        unet = UNetSimple(args.latent_dim, args.latent_dim).to(device)

    encoder = DDP(encoder, device_ids=[rank])
    unet = DDP(unet, device_ids=[rank])
    decoder = Decoder(args.latent_dim, args.embedding_dim).to(device)
    decoder = DDP(decoder, device_ids=[rank])

    optimizer = optim.Adam(list(encoder.parameters()) + list(unet.parameters()) + list(decoder.parameters()), lr=args.learning_rate)
    criterion = None
    if args.lossoption == 0:
        if rank == 0:
            log(f"Using MSELoss as criterion.")
        criterion = nn.MSELoss()
    elif args.lossoption == 1:
        if rank == 0:
            log(f"Using SSIM3DLossMultiChannel as criterion.")
        criterion = SSIM3DLossMultiChannel(window_size=11, window_sigma=1.5, data_range=1, size_average=True, channels=args.embedding_dim)

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

    # Early stopper for validation
    #early_stopper = EarlyStopping(patience=10, min_delta=0.0001, rank=rank)

    # Train on the dataset num_epochs times.
    for epoch in range(1, args.num_epochs + 1):
        
        if rank == 0:
            log(f"Starting epoch {epoch}/{args.num_epochs}")

        # Training loop
        encoder.train()
        unet.train()
        decoder.train()
        for current_step in range(1, args.num_steps + 1):
            train_loop(encoder, unet, decoder, train_dataloader, optimizer, criterion, device, current_step, args.num_steps, rank, world_size)

        if (epoch % args.save_every_epoch == 0 or epoch <= 4) and rank == 0:
            checkpoint_name = f"{args.name}_{args.batch_size}batch_{epoch}epochs_{generate_random_string()}.pth.tar"
            save_checkpoint({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'unet_state_dict': unet.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join("./models", checkpoint_name), args_dict)
            log(f"Checkpoint saved.")
        
        # Validation loop
        encoder.eval()
        unet.eval()
        decoder.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_dataloader:
                data = data.to(device)
                data = data.permute(0, 4, 1, 2, 3) 
                latent = encoder(data)
                denoised = unet(latent)
                outputs = decoder(denoised)
                loss = criterion(outputs, data)
                val_loss += loss.item()
        
        # Calculate validation loss across ranks.
        val_loss = torch.tensor(val_loss, device=device)
        val_loss = reduce_mean(val_loss, world_size).item()
        val_loss /= len(val_dataloader)

        if rank == 0:
            log(f"Avg Validation Loss: {val_loss}")
            log(f"Completed epoch {epoch}/{args.num_epochs}")

        # Early Stopping
        #early_stopper(val_loss)
        #early_stopper.check_stop()
        #if early_stopper.early_stop:
        #    if rank == 0:
        #        checkpoint_name = f"{args.name}_{args.batch_size}batch_{epoch}epochs_{generate_random_string()}_earlystop.pth.tar"
        #        save_checkpoint({
        #            'epoch': epoch,
        #            'encoder_state_dict': encoder.state_dict(),
        #            'unet_state_dict': unet.state_dict(),
        #            'decoder_state_dict': decoder.state_dict(),
        #            'optimizer_state_dict': optimizer.state_dict(),
        #        }, os.path.join("./models", checkpoint_name), args_dict)
        #        log("Early stopping triggered; checkpoint saved.")
        #    break

    if rank == 0:
        # Save final checkpoint
        random_str = generate_random_string()
        save_checkpoint({
            'epoch': args.num_epochs,
            'encoder_state_dict': encoder.state_dict(),
            'unet_state_dict': unet.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, f"./models/{args.name}_{args.batch_size}batch_{args.num_epochs}epochs_{random_str}_terminated.pth.tar", args_dict)

# Start the model with DDP.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Latent Diffusion Model for Minecraft Structure Generation")
    parser.add_argument("--name", type=str, required=True, help="Model name")
    parser.add_argument("--attention", type=bool, default=True, help="Use UNet with multi-level cross attention")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--num_steps", type=int, default=100, help="Number of de/noising steps")
    parser.add_argument("--save_every_epoch", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--embedding_dim", type=int, default=16, help="Dimension of embedding vectors")
    parser.add_argument("--lossoption", type=int, default=0, help="0 - MSELoss, 1 - SSIM3D")
    parser.add_argument("--latent_dim", type=int, default=64, help="Dimension of latent space vectors")

    args = parser.parse_args()
    
    if not os.path.exists("./models"):
        os.makedirs("./models")

    world_size = torch.cuda.device_count()
    log("World size: " + str(world_size))

    for i in range(world_size):
        log("Found GPU " + str(i) + ": " + str(torch.cuda.get_device_properties(i)))


    spawn(train, args=(world_size, args), nprocs=world_size, join=True)