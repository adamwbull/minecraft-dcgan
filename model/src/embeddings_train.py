# embeddings_train.py

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import BlockDataset
import string
import random
import argparse
from embeddings_model import AdvancedBlockEmbeddings
from embeddings_loss import ContrastiveLoss
import numpy as np
from collections import Counter
from vector import vector_to_string
import time
from vector import get_unique_block_vectors
import pickle

# If training on my PC, plot
from embeddings_plot import plot_embeddings, plot_embeddings_3d, create_confusion_matrix

plotting = True

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #

# Parallelization

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #

# Logging and file functions.

# Logging function
def log(text="", console_only=False):
    current_date = time.strftime("%Y%m%d")
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")

    print(timestamp + " " + text)
    
    if not console_only:
        # Write to the latest.log file
        with open('./embeddings/logs/latest.log', 'a') as log_file:
            log_file.write(f"{timestamp} {text}\n")

        # Write to the daily log file
        daily_log_filename = f'./embeddings/logs/{current_date}.log'
        with open(daily_log_filename, 'a') as daily_log_file:
            daily_log_file.write(f"{timestamp} {text}\n")

# Function to manage log files
def initialize_files():
    if not os.path.exists('./embeddings'):
        os.makedirs('./embeddings')
    if not os.path.exists('./embeddings/logs'):
        os.makedirs('./embeddings/logs')
    if not os.path.exists('./embeddings/checkpoints'):
        os.makedirs('./embeddings/checkpoints')
    if not os.path.exists('./embeddings/dicts'):
        os.makedirs('./embeddings/dicts')
    open('./embeddings/logs/latest.log', 'w').close()

# Initialize log files
initialize_files()

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #

# CUDA info.
log(f"CUDA Available: {torch.cuda.is_available()}")
log(f"CUDA Version: {torch.version.cuda}")
log(f"PyTorch Version: {torch.__version__}")
if torch.cuda.is_available():
    log(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

# Decide on target device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Using device: {device}")

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #

# Training parameters, hyperparams can be overridden by argparse.

# Maintain train/validation dataset split across training sessions, assuming the seed and dataset size don't change.
seed = 1202000
torch.manual_seed(seed)

# Hyperparams.
num_epochs = 1
learning_rate = 0.001
batch_size = 10

# Parse command line arguments
parser = argparse.ArgumentParser(description="Load checkpoint for Block2AttentionVec model")
parser.add_argument("name", type=str, help="Name for the trial. Specify an existing trial to continue it.")
parser.add_argument("--epochs", type=int, default=num_epochs, help="Number of training epochs.")
parser.add_argument("--lr", type=float, default=learning_rate, help="Learning rate for training.")
parser.add_argument("--batch_size", type=int, default=batch_size, help="Training batch size.")

args = parser.parse_args()

name = args.name
num_epochs = args.epochs
learning_rate = args.lr
batch_size = args.batch_size

# Function to generate random string of characters
def generate_random_string(length=6):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

def save_checkpoint(
    epoch, 
    model, 
    optimizer, 
    total_train_loss, 
    total_val_loss, 
    avg_train_loss, 
    avg_val_loss, 
    best_train_loss, 
    best_val_loss, 
    sampled_blocks_counter,
    filename="block2vec-models/block2vec_checkpoint", 
    is_best=False
):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'total_train_loss': total_train_loss,
        'total_val_loss': total_val_loss,
        'avg_train_loss': avg_train_loss,
        'avg_val_loss': avg_val_loss,
        'best_train_loss': best_train_loss,
        'best_val_loss': best_val_loss,
        'sampled_blocks_counter': dict(sampled_blocks_counter)
    }
    file_extension = ".pth.tar"
    if is_best:
        # Append random characters to filename for uniqueness
        unique_filename = f"{filename}_{generate_random_string()}{file_extension}"
        torch.save(state, unique_filename)
    else:
        torch.save(state, filename + file_extension)

def get_context_blocks(structure, x, y, z):
    neighbors = []
    # Consider including the position as part of the context
    for dx, dy, dz in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
        nx, ny, nz = x + dx, y + dy, z + dz
        if 0 <= nx < 32 and 0 <= ny < 32 and 0 <= nz < 32:
            neighbors.append(structure[nx, ny, nz])
    return torch.stack(neighbors) if neighbors else None

# Initialize model, loss function, and optimizer.
unique_blocks = get_unique_block_vectors()
model = AdvancedBlockEmbeddings()
model = model.to(device)
criterion =ContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

log("Initializing dataset...")
dataset = BlockDataset("../dataset/blocks-16-100000.hdf5")
log("Dataset initialized.")

# Split into training and validation.
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

best_train_loss = float('inf')
best_val_loss = float('inf')

start_epoch = 0
sampled_blocks_counter = Counter() 

# Load checkpoint if specified
if name:
    filepath = "embeddings/checkpoints/" + name + "_target.pth.tar"
    if os.path.isfile(filepath):
        log(f"Loading checkpoint '{filepath}'")
        checkpoint = torch.load(filepath, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        best_train_loss = checkpoint['best_train_loss']
        best_val_loss = checkpoint['best_val_loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        log(f"Loaded checkpoint '{filepath}' (epoch {checkpoint['epoch']})")
    else:
        log(f"No checkpoint found at '{filepath}', starting from scratch.")

total_val_loss = 0
total_train_loss = 0
avg_train_loss = 0
avg_val_loss = 0
accuracy = -1

for epoch in range(start_epoch, num_epochs):

    model.train()  # Set model to training mode

    total_train_loss = 0

    for target, context_blocks, negative_blocks in train_dataloader:

        target = target.to(device)

        # Stack the context and negative blocks if they are not empty
        context_blocks = context_blocks.to(device)
        negative_blocks = negative_blocks.to(device)

        target_strings = vector_to_string(target, should_print=False) 

        # Ensure 'target_strings' is always a list for consistency, even if it's just one item.
        # Will never happen, unless torch changes how batch=1 in the future tbh.
        if isinstance(target_strings, str):
            target_strings = [target_strings]

        # Loop through each string in the batch and update the counter.
        for target_string in target_strings:
            sampled_blocks_counter[target_string] += 1

        # Reset the gradient.  
        model.zero_grad()

        # Train.
        target_embedding, context_embeddings, negative_embeddings = model(target, context_blocks, negative_blocks)
        loss = criterion(target_embedding, context_embeddings, negative_embeddings)

        # Backward pass.
        loss.backward()
        optimizer.step()

        # Record performance.
        total_train_loss += loss.item()

    # Validation phase.

    # Create embedding  mappings out of our validation set.
    # Should be block_id (target) -> list of embeddings
    validation_embeddings = dict()

    model.eval()  
    total_val_loss = 0
    with torch.no_grad(): # No gradients needed for validation
        for target, context_blocks, negative_blocks in val_dataloader:

            target = target.to(device)
            context_blocks = context_blocks.to(device)
            negative_blocks = negative_blocks.to(device)

            target_embedding, context_embeddings, negative_embeddings = model(target, context_blocks, negative_blocks)
            target_strings = vector_to_string(target, should_print=False) 

            # Ensure 'target_strings' is always a list for consistency, even if it's just one item.
            # Will never happen, unless torch changes how batch=1 in the future tbh.
            if isinstance(target_strings, str):
                target_strings = [target_strings]

            for i, target_id in enumerate(target_strings):
                if target_id not in validation_embeddings:
                    validation_embeddings[target_id] = []

                if not any(np.array_equal(target_embedding[i].cpu().numpy(), np.array(x)) for x in validation_embeddings[target_id]):
                    validation_embeddings[target_id].append(target_embedding[i].cpu().numpy())

            # Compute loss.
            loss = criterion(target_embedding, context_embeddings, negative_embeddings)

            # Accumulate validation loss
            total_val_loss += loss.item()

    # Calculate average losses
    avg_train_loss = total_train_loss / len(train_dataloader)
    avg_val_loss = total_val_loss / len(val_dataloader)

    # Save new bests if applicable.
    best_saved = False
    if total_train_loss < best_train_loss:
        best_saved = True
        best_train_loss = total_train_loss
        save_checkpoint(
            epoch, 
            model, 
            optimizer, 
            total_train_loss, 
            total_val_loss, 
            avg_train_loss, 
            avg_val_loss, 
            best_train_loss, 
            best_val_loss, 
            sampled_blocks_counter, 
            filename=f"embeddings/checkpoints/best_train_{epoch}epochs", 
            is_best=True
        )

    if total_val_loss < best_val_loss:
        best_saved = True
        best_val_loss = total_val_loss
        save_checkpoint(
            epoch, 
            model, 
            optimizer, 
            total_train_loss, 
            total_val_loss, 
            avg_train_loss, 
            avg_val_loss, 
            best_train_loss, 
            best_val_loss, 
            sampled_blocks_counter, 
            filename=f"embeddings/checkpoints/best_val_{epoch}epochs", 
            is_best=True
        )

    # Save at the end of every epoch, since our dataset is massive an epoch is tough to complete.
    if not best_saved:
        save_checkpoint(
            epoch, 
            model, 
            optimizer,  
            total_train_loss, 
            total_val_loss,
            avg_train_loss, 
            avg_val_loss, 
            best_train_loss, 
            best_val_loss,
            sampled_blocks_counter, 
            filename=f"embeddings/checkpoints/checkpoint_terminated_{start_epoch+num_epochs}epochs",
            is_best=False
        )
    
    log(f'Epoch {epoch}, total_train_loss: {total_train_loss}, total_val_loss: {total_val_loss}')

    # Save our contextual embeddings.
    validation_embeddings_filename = f'embeddings/dicts/{name}_epoch{epoch}_validation.pkl'
    with open(validation_embeddings_filename, 'wb') as file:
        pickle.dump(validation_embeddings, file)

    # Generate embeddings for all possible unique 
    # one-hot encoded vectors 
    embeddings = dict()
    with torch.no_grad():
        for block in unique_blocks:
            block_vector = torch.tensor(block, dtype=torch.float)
            block_vector = block_vector.to(device)  # 'device' should be defined elsewhere, e.g., device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            embedding = model.block_processor(block_vector) 

            # Move embedding to CPU and convert to numpy array for serialization compatibility
            embedding = embedding.cpu().numpy()

            block_id = vector_to_string(block_vector.cpu())  # Ensure block_id generation does not cause device mismatch

            if block_id not in embeddings:
                embeddings[block_id] = []

            # Ensure embedding uniqueness before adding
            if not any(np.array_equal(embedding, np.array(x)) for x in embeddings[block_id]):
                embeddings[block_id].append(embedding.tolist())

    # Save and plot our noncontextual embeddings.
    embeddings_filename = f'embeddings/dicts/{name}_epoch{epoch}_unique.pkl'
    with open(embeddings_filename, 'wb') as file:
        pickle.dump(embeddings, file)

    if plotting:
        plot_embeddings(embeddings, epoch, name)
        plot_embeddings_3d(embeddings, epoch, name)
        create_confusion_matrix(embeddings, epoch, name)

# Final save before quitting.
save_checkpoint(
    start_epoch+num_epochs, 
    model, 
    optimizer, 
    total_train_loss, 
    total_val_loss, 
    avg_train_loss, 
    avg_val_loss, 
    best_train_loss, 
    best_val_loss, 
    sampled_blocks_counter, 
    filename=f"embeddings/checkpoints/checkpoint_terminated_{start_epoch+num_epochs}epochs",
    is_best=True
)

