# ladcgan.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from variables.hyperparameters import global_dimension
from variables.globals import detailed_printing
from attention import LocalizedAttention, GlobalAttention, RegionLocalizedAttention

class WGPGenerator(nn.Module):

    def __init__(self, noise_dim, output_channels, feature_map_size):
        super(WGPGenerator, self).__init__()

        self.fc = nn.Linear(noise_dim, 256 * feature_map_size**3) 
        self.feature_map_size = feature_map_size

        self.deconv1 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)
        self.global_attn_final = RegionLocalizedAttention(64, region_factor=2)
        self.deconv3 = nn.ConvTranspose3d(64, output_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, noise):

        x = self.fc(noise).view(-1, 256, self.feature_map_size, self.feature_map_size, self.feature_map_size) 

        if detailed_printing:
            print(f"After fc: {x.shape}")

        x = F.relu(self.deconv1(x))

        if detailed_printing:
            print(f"After deconv1: {x.shape}")

        x = self.deconv2(x)

        if detailed_printing:
            print(f"After deconv2: {x.shape}")

        x = F.relu(self.global_attn_final(x))

        if detailed_printing:
            print(f"After local_attn_final: {x.shape}")

        #x = torch.tanh(self.deconv3(x))
        x = self.deconv3(x)

        # Apply softmax separately to the block type and directionality sections
        block_type_probs = F.softmax(x[:, 0:11, :, :, :], dim=1)
        directionality_probs = F.softmax(x[:, 11:15, :, :, :], dim=1)

        # Apply sigmoid to the vertical directionality section
        vertical_directionality_probs = torch.sigmoid(x[:, 15, :, :, :].unsqueeze(1))

        # Concatenate the processed sections back together
        x = torch.cat([block_type_probs, directionality_probs, vertical_directionality_probs], dim=1)

        if detailed_printing:
            print(f"After deconv3: {x.shape}")

        return x
    
class RegionGenerator(nn.Module):

    def __init__(self, noise_dim, output_channels, feature_map_size):
        super(RegionGenerator, self).__init__()

        self.fc = nn.Linear(noise_dim, 256 * feature_map_size**3)
        self.feature_map_size = feature_map_size

        self.deconv1 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)
        self.global_attn_final = RegionLocalizedAttention(64, region_factor=2)
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

        x = F.relu(self.global_attn_final(x))

        if detailed_printing:
            print(f"After global_attn_final: {x.shape}")

        #x = torch.tanh(self.deconv3(x))
        x = self.deconv3(x)

        # Apply softmax separately to the block type and directionality sections
        block_type_probs = F.softmax(x[:, 0:11, :, :, :], dim=1)
        directionality_probs = F.softmax(x[:, 11:15, :, :, :], dim=1)

        # Apply sigmoid to the vertical directionality section
        vertical_directionality_probs = torch.sigmoid(x[:, 15, :, :, :].unsqueeze(1))

        # Concatenate the processed sections back together
        x = torch.cat([block_type_probs, directionality_probs, vertical_directionality_probs], dim=1)

        if detailed_printing:
            print(f"After deconv3: {x.shape}")

        return x
    
class GlobalGenerator(nn.Module):

    def __init__(self, noise_dim, output_channels, feature_map_size):
        super(GlobalGenerator, self).__init__()

        self.fc = nn.Linear(noise_dim, 256 * feature_map_size**3)
        self.feature_map_size = feature_map_size

        self.deconv1 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)
        self.global_attn_final = GlobalAttention(64)
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

        x = F.relu(self.global_attn_final(x))

        if detailed_printing:
            print(f"After global_attn_final: {x.shape}")

        #x = torch.tanh(self.deconv3(x))
        x = self.deconv3(x)

        # Apply softmax separately to the block type and directionality sections
        block_type_probs = F.softmax(x[:, 0:11, :, :, :], dim=1)
        directionality_probs = F.softmax(x[:, 11:15, :, :, :], dim=1)

        # Apply sigmoid to the vertical directionality section
        vertical_directionality_probs = torch.sigmoid(x[:, 15, :, :, :].unsqueeze(1))

        # Concatenate the processed sections back together
        x = torch.cat([block_type_probs, directionality_probs, vertical_directionality_probs], dim=1)

        if detailed_printing:
            print(f"After deconv3: {x.shape}")

        return x

# Generator.
class Generator(nn.Module):

    def __init__(self, noise_dim, output_channels, feature_map_size):
        super(Generator, self).__init__()

        self.fc = nn.Linear(noise_dim, 256 * feature_map_size**3)
        self.feature_map_size = feature_map_size

        self.deconv1 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)
        #self.local_attn_final = LocalizedAttention(64, local_window_size=3)
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

        #x = F.relu(self.local_attn_final(x))

        if detailed_printing:
            print(f"After local_attn_final: {x.shape}")

        #x = torch.tanh(self.deconv3(x))
        x = self.deconv3(x)

        # Apply softmax separately to the block type and directionality sections
        block_type_probs = F.softmax(x[:, 0:11, :, :, :], dim=1)
        directionality_probs = F.softmax(x[:, 11:15, :, :, :], dim=1)

        # Apply sigmoid to the vertical directionality section
        vertical_directionality_probs = torch.sigmoid(x[:, 15, :, :, :].unsqueeze(1))

        # Concatenate the processed sections back together
        x = torch.cat([block_type_probs, directionality_probs, vertical_directionality_probs], dim=1)

        if detailed_printing:
            print(f"After deconv3: {x.shape}")

        return x

# AttentionGenerator.
class AttentionGenerator(nn.Module):

    def __init__(self, noise_dim, output_channels, feature_map_size):
        super(AttentionGenerator, self).__init__()

        self.fc = nn.Linear(noise_dim, 256 * feature_map_size**3) 
        self.feature_map_size = feature_map_size

        self.deconv1 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)
        self.local_attn_final = LocalizedAttention(64, local_window_size=3)
        self.deconv3 = nn.ConvTranspose3d(64, output_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, noise):

        x = self.fc(noise).view(-1, 256, self.feature_map_size, self.feature_map_size, self.feature_map_size) 

        if detailed_printing:
            print(f"After fc: {x.shape}")

        x = F.relu(self.deconv1(x))

        if detailed_printing:
            print(f"After deconv1: {x.shape}")

        x = self.deconv2(x)

        if detailed_printing:
            print(f"After deconv2: {x.shape}")

        x = F.relu(self.local_attn_final(x))

        if detailed_printing:
            print(f"After local_attn_final: {x.shape}")

        x = self.deconv3(x)

        # Apply softmax separately to the block type and directionality sections
        block_type_probs = F.softmax(x[:, 0:11, :, :, :], dim=1)
        directionality_probs = F.softmax(x[:, 11:15, :, :, :], dim=1)

        # Apply sigmoid to the vertical directionality section
        vertical_directionality_probs = torch.sigmoid(x[:, 15, :, :, :].unsqueeze(1))

        # Concatenate the processed sections back together
        x = torch.cat([block_type_probs, directionality_probs, vertical_directionality_probs], dim=1)

        if detailed_printing:
            print(f"After deconv3: {x.shape}")

        return x