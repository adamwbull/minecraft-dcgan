# attention.py

from variables.globals import detailed_printing
import torch
import torch.nn as nn
import torch.nn.functional as F

class RegionLocalizedAttention(nn.Module):
    def __init__(self, in_channels, region_factor=1):
        super(RegionLocalizedAttention, self).__init__()
        self.in_channels = in_channels
        self.region_factor = region_factor
        self.query_conv = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, D, H, W = x.size()
        RD, RH, RW = D // self.region_factor, H // self.region_factor, W // self.region_factor

        out = torch.zeros_like(x)
        for d in range(self.region_factor):
            for h in range(self.region_factor):
                for w in range(self.region_factor):
                    # Extract region
                    region = x[:, :, d*RD:(d+1)*RD, h*RH:(h+1)*RH, w*RW:(w+1)*RW]

                    # Apply attention to region
                    proj_query = self.query_conv(region).view(batch_size, -1, RD*RH*RW).permute(0, 2, 1)
                    proj_key = self.key_conv(region).view(batch_size, -1, RD*RH*RW)
                    proj_value = self.value_conv(region).view(batch_size, -1, RD*RH*RW)

                    energy = torch.bmm(proj_query, proj_key)
                    attention = F.softmax(energy, dim=-1)
                    region_out = torch.bmm(proj_value, attention.permute(0, 2, 1))
                    region_out = region_out.view(batch_size, C, RD, RH, RW)

                    # Add original input
                    region_out = self.gamma * region_out + region

                    # Place computed region in output tensor
                    out[:, :, d*RD:(d+1)*RD, h*RH:(h+1)*RH, w*RW:(w+1)*RW] = region_out

        return out

class GlobalAttention(nn.Module):
    def __init__(self, in_channels):
        super(GlobalAttention, self).__init__()
        self.query_conv = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, D, H, W = x.size()

        # Generate queries, keys, values
        proj_query = self.query_conv(x).view(batch_size, -1, D*H*W).permute(0, 2, 1)  # [B, N, C]
        proj_key = self.key_conv(x).view(batch_size, -1, D*H*W)  # [B, C, N]
        proj_value = self.value_conv(x).view(batch_size, -1, D*H*W)  # [B, C, N]

        # Attention map
        energy = torch.bmm(proj_query, proj_key)  # [B, N, N]
        attention = F.softmax(energy, dim=-1)  # [B, N, N]

        # Apply attention
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, D, H, W)

        # Add original input
        out = self.gamma * out + x

        return out

class LocalizedAttention(nn.Module):
    
    def __init__(self, in_channels, local_window_size):

        if detailed_printing:
            print('in_channels:', in_channels)
            print('local_window_size:', local_window_size)

        super(LocalizedAttention, self).__init__()
        self.in_channels = in_channels
        self.local_window_size = local_window_size
        self.query_conv = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.padding = local_window_size // 2

    def forward(self, x):
        
        batch_size, C, D, H, W = x.size()

        if detailed_printing:
            print('batch_size, C, D, H, W: ', batch_size, C, D, H, W)
            print('x.shape: ', x.shape)

        # Semantically (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)
        x_padded = F.pad(x, (self.padding, self.padding, self.padding, self.padding, self.padding, self.padding))

        # I set up the possibility of averaging based on more than just the center block, but for now only the center of the local cube will be added here.
        # It may be beneficial to consider a sub-cube of the local space and average these together for each local_attention value, but not going to investigate now.
        local_attention = torch.zeros_like(x)
        #update_counts = torch.zeros_like(x)

        # Process each local window
        for i in range(D):
            for j in range(H):
                for k in range(W):

                    # Our local cube to consider for attention.
                    local_cube = x_padded[:, :, i:i+self.local_window_size, j:j+self.local_window_size, k:k+self.local_window_size]

                    if detailed_printing:
                        print('local_cube.shape:', local_cube.shape)

                    # Key and Value for the whole cube
                    key = self.key_conv(local_cube).view(batch_size, -1, self.local_window_size**3)
                    value = self.value_conv(local_cube).view(batch_size, -1, self.local_window_size**3)

                    if detailed_printing:
                        print('key.shape:', key.shape)
                        print('value.shape:', value.shape)

                    # Extract the central block
                    central_block = local_cube[:, :, self.padding:self.padding+1, self.padding:self.padding+1, self.padding:self.padding+1]

                    if detailed_printing:
                        print('central_block.shape:', central_block.shape)

                    # Compute the query for the central block, ensuring the output channels match self.local_window_size**3
                    query = self.query_conv(central_block)

                    if detailed_printing:
                        print('query.shape:', query.shape)

                    # Ensure the dimensions are [batch_size, self.local_window_size**3, 1]
                    #query = query.view(batch_size, self.local_window_size**3, 1) # This crashes

                    # Ensure the dimensions of query are compatible for batch matrix multiplication with key
                    # The number of output channels for query_conv should match the flattened dimensions of key (which is in_channels // 8 * local_window_size**3)
                    query = query.view(batch_size, 1, -1)  # No need to match local_window_size**3 here, as we are only interested in the central block

                    if detailed_printing:
                        print('query.shape post view:', query.shape)

                    # Calculate energy scores
                    energy = torch.bmm(query, key)
                    attention = F.softmax(energy, dim=-1)

                    if detailed_printing:
                        print('energy.shape:', key.shape)
                        print('attention.shape:', value.shape)

                    # Apply attention to the value tensor for central block
                    attention_applied = torch.bmm(value, attention.permute(0, 2, 1))

                    # Update the central voxel of local_attention
                    local_attention[:, :, i, j, k] = attention_applied.squeeze(-1)
                    #update_counts[:, :, i, j, k] += 1
 
        # Averages the contributions in local_attention.
        # Not needed currently.
        #local_attention = local_attention / update_counts.clamp(min=1)

        # Perform our gamma operation.
        out = self.gamma * local_attention + x

        return out
