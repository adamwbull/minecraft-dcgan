import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import RegionLocalizedAttention

# ERA-DCGAN
class ERADCGAN(nn.Module):

    def __init__(self, noise_dim, output_channels, feature_map_size, rga):
        super(ERADCGAN, self).__init__()

        self.fc = nn.Linear(noise_dim, 256 * feature_map_size**3) 
        self.feature_map_size = feature_map_size

        self.deconv1 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)
        self.attn_final = RegionLocalizedAttention(64, region_factor=rga)
        self.deconv3 = nn.ConvTranspose3d(64, output_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, noise):

        x = self.fc(noise).view(-1, 256, self.feature_map_size, self.feature_map_size, self.feature_map_size) 

        x = F.relu(self.deconv1(x))

        x = F.relu(self.deconv2(x))

        x = F.relu(self.attn_final(x))
        
        x = torch.tanh(self.deconv3(x))
        return x
    
# RA-DCGAN
class RGDCGANGenerator(nn.Module):

    def __init__(self, noise_dim, output_channels, feature_map_size, rga):
        super(RGDCGANGenerator, self).__init__()

        self.fc = nn.Linear(noise_dim, 256 * feature_map_size**3) 
        self.feature_map_size = feature_map_size

        self.deconv1 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)
        self.global_attn_final = RegionLocalizedAttention(64, region_factor=rga)
        self.deconv3 = nn.ConvTranspose3d(64, output_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, noise):

        x = self.fc(noise).view(-1, 256, self.feature_map_size, self.feature_map_size, self.feature_map_size) 

        x = F.relu(self.deconv1(x))

        x = self.deconv2(x)

        x = F.relu(self.global_attn_final(x))

        x = self.deconv3(x)
        block_type_probs = F.softmax(x[:, 0:11, :, :, :], dim=1)
        directionality_probs = F.softmax(x[:, 11:15, :, :, :], dim=1)
        vertical_directionality_probs = torch.sigmoid(x[:, 15, :, :, :].unsqueeze(1))
        x = torch.cat([block_type_probs, directionality_probs, vertical_directionality_probs], dim=1)

        return x
    
# E-DCGAN
class EDCGANGenerator(nn.Module):

    def __init__(self, noise_dim, output_channels, feature_map_size):
        super(EDCGANGenerator, self).__init__()

        self.fc = nn.Linear(noise_dim, 256 * feature_map_size**3) 
        self.feature_map_size = feature_map_size

        self.deconv1 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose3d(64, output_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, noise):

        x = self.fc(noise).view(-1, 256, self.feature_map_size, self.feature_map_size, self.feature_map_size) 

        x = F.relu(self.deconv1(x))

        x = F.relu(self.deconv2(x))

        x = torch.tanh(self.deconv3(x))
        
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
class UNetDecoder(nn.Module):
    def __init__(self, latent_dim, embedding_dim):
        super(UNetDecoder, self).__init__()

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
