# attention.py

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
