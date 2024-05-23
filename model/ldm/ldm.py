# ldm.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# Compress the high-dimensional embedding input data into a lower-dimensional latent representation.
class Encoder(nn.Module):
    def __init__(self, embedding_dim, latent_dim):
        super(Encoder, self).__init__()
        self.initial_layer = nn.Conv3d(embedding_dim, 64, kernel_size=3, stride=1, padding=1)
        
        # Adjusted down-sampling blocks
        # Removed one down-sampling block to reduce the extent of size reduction
        self.down1 = self._down_block(64, 128, 1)
        self.down2 = self._down_block(128, 256, 2)
        self.down3 = self._down_block(256, 512, 1)
        
        self.final_layer = nn.Conv3d(512, latent_dim, kernel_size=3, stride=1, padding=1)  # Adjusted kernel_size and padding
        
    def _down_block(self, in_channels, out_channels, stride):
        layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        return layers

    def forward(self, x):
        #print(f"decoder input shape: {x.shape}")
        x = F.relu(self.initial_layer(x))
        #print(f"relu shape: {x.shape}")
        x = self.down1(x)
        #print(f"down1 shape: {x.shape}")
        x = self.down2(x)
        #print(f"down2 shape: {x.shape}")
        # No further downsampling here to preserve the spatial dimensions for the desired output
        x = self.down3(x)  # This layer now maintains the spatial dimension
        #print(f"down3 shape: {x.shape}")
        x = self.final_layer(x)  # This layer also maintains the spatial dimension
        #print(f"final_layer shape: {x.shape}")
        return x

# A wonderful implementation of crossattention similar to Rombach et al.
class CrossAttention(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CrossAttention, self).__init__()
        self.query_conv = nn.Conv3d(in_channels=channel, out_channels=channel // reduction, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=channel, out_channels=channel // reduction, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=channel, out_channels=channel, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, depth * height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, depth * height * width)
        value = self.value_conv(x).view(batch_size, -1, depth * height * width)

        attention = torch.bmm(query, key)
        attention = self.softmax(attention)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, depth, height, width)

        return out + x

# An expanded version of original UNet application that closer mimics
# High-Resolution Image Synthesis with Latent Diffusion Models by Rombach et al
# with cross attention layers to capture contexts better at each level.
class UNetWithMultiLevelCrossAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetWithMultiLevelCrossAttention, self).__init__()
        self.encoder1 = self.contracting_block(in_channels, 64)
        self.cross_attention1 = CrossAttention(64)
        self.encoder2 = self.contracting_block(64, 128)
        self.cross_attention2 = CrossAttention(128)
        self.encoder3 = self.contracting_block(128, 256)
        self.cross_attention3 = CrossAttention(256)
        
        self.bottleneck = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.cross_attention_bottleneck = CrossAttention(512)
        
        self.decoder1 = self.expansive_block(512, 256, 256)
        self.cross_attention4 = CrossAttention(256)
        self.decoder2 = self.expansive_block(256, 128, 128)
        self.cross_attention5 = CrossAttention(128)
        self.decoder3 = self.expansive_block(128, 64, out_channels)
        self.cross_attention6 = CrossAttention(64)

    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        return block
    
    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv3d(in_channels, mid_channel, kernel_size, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channel, mid_channel, kernel_size, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(mid_channel, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )
        return block

    def forward(self, x):
        #print(f"u-net input shape: {x.shape}")

        # Encoder with cross-attention after each block
        enc1 = self.encoder1(x)
        #print(f"enc1 shape: {enc1.shape}")
        ca1 = self.cross_attention1(enc1)
        #print(f"ca1 (after enc1 + CA) shape: {ca1.shape}")
        enc2 = self.encoder2(ca1)
        #print(f"enc2 shape: {enc2.shape}")
        ca2 = self.cross_attention2(enc2)
        #print(f"ca2 (after enc2 + CA) shape: {ca2.shape}")
        enc3 = self.encoder3(ca2)
        #print(f"enc3 shape: {enc3.shape}")
        ca3 = self.cross_attention3(enc3)
        #print(f"ca3 (after enc3 + CA) shape: {ca3.shape}")

        # Bottleneck with cross-attention
        bottleneck = self.bottleneck(ca3)
        #print(f"bottleneck shape: {bottleneck.shape}")
        ca_bottleneck = self.cross_attention_bottleneck(bottleneck)
        #print(f"ca_bottleneck (after bottleneck + CA) shape: {ca_bottleneck.shape}")

        # Decoder with cross-attention before each block
        # and incorporating skip connections immediately after encoder blocks
        dec1 = self.decoder1(ca_bottleneck)
        #print(f"dec1 shape: {dec1.shape}")
        ca4 = self.cross_attention4(dec1)
        #print(f"ca4 (after dec1 + CA) shape: {ca4.shape}")
        dec1_ca4 = ca4 + enc3  # Skip connection
        #print(f"dec1_ca4 (after skip connection with enc3) shape: {dec1_ca4.shape}")

        dec2 = self.decoder2(dec1_ca4)
        #print(f"dec2 shape: {dec2.shape}")
        ca5 = self.cross_attention5(dec2)
        #print(f"ca5 (after dec2 + CA) shape: {ca5.shape}")
        dec3_ca5 = ca5 + enc2  # Skip connection
        #print(f"dec2_ca5 (after skip connection with enc2) shape: {dec3_ca5.shape}")

        dec3 = self.decoder3(dec3_ca5)
        #print(f"dec3 shape: {dec3.shape}")
        ca6 = self.cross_attention6(dec3)
        #print(f"ca6 (after dec4 + CA) shape: {ca6.shape}")
        final_output = ca6 + enc1  # Skip connection
        #print(f"final_output (after skip connection with enc1) shape: {final_output.shape}")

        return final_output

# Reconstruct the outputs from the latent space back to the original data space.
class Decoder(nn.Module):
    def __init__(self, latent_dim, embedding_dim):
        super(Decoder, self).__init__()

        self.initial_layer = nn.ConvTranspose3d(latent_dim, 256, kernel_size=3, stride=1, padding=1)
        
        # Maintain the same upsampling blocks but add one more to reach the desired dimension.
        self.up1 = self._up_block(256, 128, 2, 1)
        self.up2 = self._up_block(128, 64, 2, 1)
        self.up3 = self._up_block(64, 32, 1, 0)

        # Adjust the final layer to output the correct embedding dimension.
        self.final_layer = nn.Sequential(
            nn.Conv3d(32, embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU()  # Adding ReLU activation to match output space with embeddings
        )
        
    def _up_block(self, in_channels, out_channels, stride, output_padding):
        layers = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=output_padding),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        return layers

    def forward(self, x):
        #print(f"decoder input shape: {x.shape}")
        x = F.relu(self.initial_layer(x))
        #print(f"relu shape: {x.shape}")
        x = self.up1(x)
        #print(f"up1 shape: {x.shape}")
        x = self.up2(x)
        #print(f"up2 shape: {x.shape}")
        x = self.up3(x)
        #print(f"up3 shape: {x.shape}")
        x = self.final_layer(x)
        #print(f"final_layer shape: {x.shape}")
        return x

# A simple core of the LDM with only contracting and expanding blocks.
# It follows the U-Net architecture with symmetric downscaling and upscaling paths and skip connections.
class UNetSimple(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetSimple, self).__init__()
        self.encoder1 = self.contracting_block(in_channels, 64)
        self.encoder2 = self.contracting_block(64, 128)
        self.encoder3 = self.contracting_block(128, 256)
        self.encoder4 = self.contracting_block(256, 512)
        
        self.bottleneck = nn.Sequential(
            nn.Conv3d(512, 1024, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(1024, 1024, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        
        self.decoder1 = self.expansive_block(1024, 512, 512)
        self.decoder2 = self.expansive_block(512, 256, 256)
        self.decoder3 = self.expansive_block(256, 128, 128)
        self.decoder4 = self.expansive_block(128, 64, out_channels)
    
    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        return block
    
    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv3d(in_channels, mid_channel, kernel_size, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channel, mid_channel, kernel_size, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(mid_channel, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )
        return block

    def forward(self, x):

        # Encoder.
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        
        # Bottleneck.
        bottleneck = self.bottleneck(enc4)
        
        # Decoder with skip connections.
        dec1 = self.decoder1(bottleneck)
        dec1 = dec1 + enc4
        dec2 = self.decoder2(dec1)
        dec2 = dec2 + enc3
        dec3 = self.decoder3(dec2)
        dec3 = dec3 + enc2
        dec4 = self.decoder4(dec3)
        dec4 = dec4 + enc1
        return dec4
