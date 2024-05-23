# train.py

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCELoss, SyncBatchNorm
import torch.nn.functional as F

import argparse
import os
import time
import random
import string
import pickle

from dataset import MinecraftDataset
from eradcgan import Generator, Discriminator
from variables.hyperparameters import feature_map_size, block_embedding_dim, noise_dim, betas, default_epochs, default_batch_size, default_lr, default_save_every_epoch
from variables.globals import detailed_printing

# WGAN-GP Hyperparameters
lambda_gp = 10
discriminator_updates = 5
save_max_difference = 10
save_streak_needed = 5

import torch.distributed as dist
import torch.utils.data.distributed
from torch.nn.parallel import DistributedDataParallel as DDP

def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

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

# Initialize the distributed environment.
def setup(rank, world_size):
    log("Setting up " + str(rank) + " of " + str(world_size))
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# Clean distributed processes.
def cleanup():
    dist.destroy_process_group()

# Argument parsing
parser = argparse.ArgumentParser(description="Train a GAN model on Minecraft dataset.")
parser.add_argument("name", type=str, help="Name for the trial. Specify an existing trial to continue it.")
parser.add_argument("--epochs", type=int, default=default_epochs, help="Number of training epochs.")
parser.add_argument("--lr", type=float, default=default_lr, help="Learning rate for training.")
parser.add_argument("--batch_size", type=int, default=default_batch_size, help="Training batch size.")
parser.add_argument("--save_every_epoch", type=int, default=default_save_every_epoch, help="Save checkpoint every X epochs.")
args = parser.parse_args()

def custom_collate(batch):
    # Find the maximum size along each dimension
    max_d = max([item.size(0) for item in batch])
    max_h = max([item.size(1) for item in batch])
    max_w = max([item.size(2) for item in batch])
    max_c = max([item.size(3) for item in batch])

    # Pad the items in the batch to the maximum size
    padded_batch = [F.pad(item, (0, max_c - item.size(3), 0, max_w - item.size(2), 0, max_h - item.size(1), 0, max_d - item.size(0))) for item in batch]
    
    # Stack the padded tensors
    return torch.stack(padded_batch)

if detailed_printing:
    log("Dataset initialized successfully.")

# Checkpoint handling functions
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, './models/'+filename)

def load_checkpoint(name, generator, discriminator):
    fileFound = False
    checkpoint_path = ""
    model_directory = "./models" 

    # Check if the directory exists
    if not os.path.exists(model_directory):
        print(f"Directory {model_directory} does not exist.")
        return 0, float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), None

    # Check for a terminated pth tar.
    #log(f"model_directory: {model_directory}")
    #log(f"name: {name}")

    for file in os.listdir(model_directory): 
        #log(f"file: {file}") 
        starts_with = file.startswith(name)
        #log(f"starts_with: {starts_with}")
        ends_with = file.endswith("terminated.pth.tar")
        #log(f"ends_with: {ends_with}")
        if starts_with and ends_with:
            #log(f"terminated pth.tar found!")
            fileFound = True
            checkpoint_path = os.path.join(model_directory, file)

    # Also check for a latest pth tar if the terminated one doesnt exist.
    if fileFound == False:
        for file in os.listdir(model_directory):
            if file.startswith(name) and file.endswith("latest.pth.tar"):
                fileFound = True
                checkpoint_path = os.path.join(model_directory, file)
    
    if fileFound == False:
        return 0, float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), None

    checkpoint = torch.load(checkpoint_path)

    new_gen_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['generator_state'].items()}
    generator.load_state_dict(new_gen_state_dict)

    new_disc_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['discriminator_state'].items()}
    discriminator.load_state_dict(new_disc_state_dict)

    best_loss_d_tmp = checkpoint.get('best_loss_d', float('inf')) 
    best_loss_g_tmp = checkpoint.get('best_loss_g', float('inf'))
    cur_loss_d = checkpoint.get('loss_d', float('inf')) 
    cur_loss_g = checkpoint.get('loss_g', float('inf'))
    best_loss_d_closest_to_half_tmp = checkpoint.get('best_loss_d_closest_to_half', float('inf'))
    
    latest_epoch = checkpoint['epoch']
    
    return latest_epoch, cur_loss_d, cur_loss_g, best_loss_d_tmp, best_loss_g_tmp, best_loss_d_closest_to_half_tmp, checkpoint

def rotate_checkpoints(name):
    base_path = './models/'
    if os.path.exists(f'{base_path}{name}_latest_1.pth.tar'):
        os.rename(f'{base_path}{name}_latest_1.pth.tar', f'{base_path}{name}_latest_2.pth.tar')
    if os.path.exists(f'{base_path}{name}_latest.pth.tar'):
        os.rename(f'{base_path}{name}_latest.pth.tar', f'{base_path}{name}_latest_1.pth.tar')

def clear_and_report():

    torch.cuda.memory_summary()

    torch.cuda.empty_cache()
    log("CUDA cache cleared. Exiting train.py...")

    torch.cuda.memory_summary()

    return

def main(rank, world_size, batch_size, epochs, lr, name, save_every_epoch):
    
    setup(rank, world_size)

    # Define the device for the current process
    device = torch.device("cuda:{}".format(rank))

    log("Initialized main on device: " + str(device) + " with rank " + str(rank))

    # Dataset
    dataset = MinecraftDataset()

    # Use DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler, collate_fn=custom_collate)

    # Initialize models.
    generator = Generator(noise_dim, block_embedding_dim, feature_map_size)
    discriminator = Discriminator(block_embedding_dim)

    # Load checkpoint.
    start_epoch, loss_d, loss_g, best_loss_d, best_loss_g, best_loss_d_closest_to_half, checkpoint = load_checkpoint(name, generator, discriminator)

    # Initialize optimizers.
    optimizer_g = Adam(generator.parameters(), lr=lr, betas=betas)
    optimizer_d = Adam(discriminator.parameters(), lr=lr, betas=betas)
    #criterion = BCELoss()

    # Move models to device.
    generator.to(device)
    discriminator.to(device)

    # Load optimizer state (AFTER) models are on device.
    if checkpoint is not None:
        optimizer_g.load_state_dict(checkpoint['optimizer_g_state'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d_state'])

    # Convert discriminator BatchNorm to SyncBatchNorm and wrap with DDP
    discriminator = SyncBatchNorm.convert_sync_batchnorm(discriminator)
    generator = DDP(generator, device_ids=[rank])
    discriminator = DDP(discriminator, device_ids=[rank])

    log(f"Loaded model {name} on rank {rank} in local epoch {start_epoch} | loss_d: {loss_d} | loss_g: {loss_g} ")
    
    # Function to compute gradient penalty
    def compute_gradient_penalty(D, real_samples, fake_samples):
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand((real_samples.size(0), 1, 1, 1, 1), device=real_samples.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = torch.ones(d_interpolates.shape, requires_grad=False, device=device)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    # Initialize a list to track the streak for each GPU
    save_streaks = [0] * world_size

    # Initialize a list to keep track of the previous difference for each GPU
    prev_differences = [None] * world_size

    # Loop through our goal number of epochs.
    for epoch in range(start_epoch, epochs):

        # Loop through whole dataset.
        for i, real_data in enumerate(dataloader): 

            # Load in and re-arrange tensor dimensions to match generator/discriminator.
            real_data = real_data.to(device)
            real_data = real_data.permute(0, 4, 1, 2, 3)
            batch_size = real_data.size(0)
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            # Where the gradient error is caught.
            with torch.autograd.set_detect_anomaly(True):

                # Per recommended WGAN-GP training, update the discriminator more often than generator.
                for _ in range(discriminator_updates):
                    optimizer_d.zero_grad()

                    # Real data
                    output_real = discriminator(real_data)
                    loss_real = -torch.mean(output_real)

                    # Fake data
                    noise = torch.randn(batch_size, noise_dim).to(device)
                    fake_data = generator(noise)
                    output_fake = discriminator(fake_data.detach())
                    loss_fake = torch.mean(output_fake)

                    # Compute the gradient penalty
                    gradient_penalty = compute_gradient_penalty(discriminator, real_data, fake_data)

                    # Combine losses
                    loss_d = loss_real + loss_fake + lambda_gp * gradient_penalty
                    loss_d.backward()
                    optimizer_d.step()
            
            # ---------------------
            #  Train Generator
            # ---------------------

            optimizer_g.zero_grad()

            # Generate fake data
            noise = torch.randn(batch_size, noise_dim, device=device)
            fake_data = generator(noise)

            # Get discriminator's judgement of the fake data
            output = discriminator(fake_data)

            # For WGAN-GP, we maximize the discriminator's output for the fake data
            loss_g = -torch.mean(output)

            # Backpropagate the generator loss
            loss_g.backward()
            optimizer_g.step()
        
        # Calculate the difference between generator and discriminator losses
        difference = loss_g - loss_d

        # Check if the difference is within the defined threshold (save_max_difference)
        for i in range(world_size):
            if prev_differences[i] is not None and abs(difference - prev_differences[i]) <= save_max_difference:
                save_streaks[i] += 1
            else:
                save_streaks[i] = 0

            # Update the previous difference for each GPU
            prev_differences[i] = difference

        # Check if any of the GPUs reached the save_streak_needed
        if any(streak == save_streak_needed for streak in save_streaks) and rank == 0:
            randstring = generate_random_string(5)
            checkpoint_filename = f"{args.name}_streak_{epoch}_{randstring}.pth.tar"
            rotate_checkpoints(args.name)
            save_checkpoint({
                'best_loss_d_closest_to_half': best_loss_d_closest_to_half,
                'best_loss_g': best_loss_g,
                'best_loss_d': best_loss_d,
                'loss_d': loss_d,
                'loss_g': loss_g,
                'epoch': epoch,
                'generator_state': generator.state_dict(),
                'discriminator_state': discriminator.state_dict(),
                'optimizer_g_state': optimizer_g.state_dict(),
                'optimizer_d_state': optimizer_d.state_dict(),
            }, checkpoint_filename)
            log(f"Saved checkpoint {checkpoint_filename} with streak {save_streaks}")
            # Reset all streaks for each GPU
            save_streaks = [0] * world_size
            
        # Save intermediate checkpoints.
        if epoch % 100 in [x for x in range(10, 100, 10)] and rank == 0:
            checkpoint_filename = f"{args.name}_latest.pth.tar"
            rotate_checkpoints(args.name)
            save_checkpoint({
                'best_loss_d_closest_to_half': best_loss_d_closest_to_half,
                'best_loss_g': best_loss_g,
                'best_loss_d': best_loss_d,
                'loss_d': loss_d,
                'loss_g': loss_g,
                'epoch': epoch,
                'generator_state': generator.state_dict(),
                'discriminator_state': discriminator.state_dict(),
                'optimizer_g_state': optimizer_g.state_dict(),
                'optimizer_d_state': optimizer_d.state_dict(),
            }, checkpoint_filename)
            log(f"Saved latest checkpoint {checkpoint_filename}")
            log(f"Saved Epoch {epoch}/{epochs} | Discriminator Loss: {loss_d:.6f} | Generator Loss: {loss_g:.6f}")
        else:
            log(f"GPU {rank}: Epoch {epoch}/{epochs} | Discriminator Loss: {loss_d:.6f} | Generator Loss: {loss_g:.6f}")
            
    # A final save if we reach this, indicating we have done 100 epochs since our for loop will exit early.
    if rank == 0:
        # Last Epoch info
        log(f"Epoch {epochs}/{epochs} | Discriminator Loss: {loss_d:.6f} | Generator Loss: {loss_g:.6f}")

        # Save.
        checkpoint_filename = f"{args.name}_terminated.pth.tar"
        save_checkpoint({
            'best_loss_d_closest_to_half': best_loss_d_closest_to_half,
            'best_loss_g': best_loss_g,
            'best_loss_d': best_loss_d,
            'loss_d': loss_d,
            'loss_g': loss_g,
            'epoch': 0,
            'generator_state': generator.state_dict(),
            'discriminator_state': discriminator.state_dict(),
            'optimizer_g_state': optimizer_g.state_dict(),
            'optimizer_d_state': optimizer_d.state_dict(),
        }, checkpoint_filename)
        log(f"Saved checkpoint {checkpoint_filename}.")

    cleanup()

    #clear_and_report()

if __name__ == "__main__":

    world_size = torch.cuda.device_count()  # Number of GPUs

    log("World size: " + str(world_size))

    for i in range(torch.cuda.device_count()):
        log("Found GPU " + str(i) + ": " + str(torch.cuda.get_device_properties(i)))

    # Include other parsed arguments as needed, divide batch size evenly across GPUs.
    spawn_args = (world_size, args.batch_size // world_size, args.epochs, args.lr, args.name, args.save_every_epoch)

    torch.multiprocessing.spawn(main, args=spawn_args, nprocs=world_size, join=True)
