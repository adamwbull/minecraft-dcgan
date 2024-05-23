# eradcgan.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from variables.globals import detailed_printing
from attention import RegionLocalizedAttention

# Generator.
class Generator(nn.Module):

    def __init__(self, noise_dim, output_channels, feature_map_size):
        super(Generator, self).__init__()

        self.fc = nn.Linear(noise_dim, 256 * feature_map_size**3) 
        self.feature_map_size = feature_map_size

        self.deconv1 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)
        self.attn_final = RegionLocalizedAttention(64, region_factor=2)
        self.deconv3 = nn.ConvTranspose3d(64, output_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, noise):

        x = self.fc(noise).view(-1, 256, self.feature_map_size, self.feature_map_size, self.feature_map_size) 

        if detailed_printing:
            print(f"After fc: {x.shape}")

        x = F.relu(self.deconv1(x))

        if detailed_printing:
            print(f"After deconv1: {x.shape}")

        x = F.relu(self.deconv2(x))

        if detailed_printing:
            print(f"After deconv2: {x.shape}")

        x = F.relu(self.attn_final(x))

        if detailed_printing:
            print(f"After attn_final: {x.shape}")

        x = torch.tanh(self.deconv3(x))

        if detailed_printing:
            print(f"After deconv3: {x.shape}")

        return x
    
class Discriminator(nn.Module):
    def __init__(self, input_channels):

        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=4, stride=2, padding=1)
        
        self.attn_d = RegionLocalizedAttention(64, region_factor=2)
        
        self.conv2 = nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1)

        self.conv3 = nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1)

        self.fc = nn.Linear(256*4*4*4, 1)

    def forward(self, x):

        if detailed_printing:
            print(f"Discriminator input shape: {x.shape}")

        x = F.leaky_relu(self.conv1(x), 0.2)

        if detailed_printing:
            print(f"After conv1: {x.shape}")

        x = F.leaky_relu(self.attn_d(x), 0.2)

        if detailed_printing:
            print(f"After attn_d: {x.shape}")

        x = F.leaky_relu(self.conv2(x), 0.2)
        
        if detailed_printing:
            print(f"After conv2: {x.shape}")

        x = F.leaky_relu(self.conv3(x), 0.2)

        if detailed_printing:
            print(f"After conv3: {x.shape}")

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if detailed_printing:
            print(f"After fc: {x.shape} {x}")

        return x
