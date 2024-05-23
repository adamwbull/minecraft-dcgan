import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from ..src.dataset import MinecraftDataset
from model_loader import preload_models_from_standard_weights
from ddpm import DDPMSampler  # Assuming ddpm.py contains DDPMSampler

# Configure device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset setup (modify according to your data)
dataset = MinecraftDataset()
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Model loading
ckpt_path = 'path/to/your/model.ckpt'
models = preload_models_from_standard_weights(ckpt_path, device)
encoder = models['encoder']
decoder = models['decoder']
diffusion = models['diffusion']

# DDPMSampler initialization
generator = torch.Generator(device=device)
ddpm_sampler = DDPMSampler(generator=generator, num_training_steps=1000)

# Loss and Optimizer
mse_loss = nn.MSELoss()
params = list(encoder.parameters()) + list(decoder.parameters()) + list(diffusion.parameters())
optimizer = optim.Adam(params, lr=1e-4)

# Training loop
epochs = 10  # Set appropriate number of epochs
for epoch in range(epochs):
    for images, _ in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
        images = images.to(device)
        
        # Forward pass through encoder and add noise using DDPMSampler
        latent_images = encoder(images)
        noisy_images = ddpm_sampler.add_noise(latent_images, torch.tensor([0] * len(images)).to(device))

        # Diffusion model predicts noise
        predicted_noise = diffusion(noisy_images) 

        # Calculate loss 
        loss = mse_loss(predicted_noise, images)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1} completed with loss {loss.item()}")

print("Training completed.")
