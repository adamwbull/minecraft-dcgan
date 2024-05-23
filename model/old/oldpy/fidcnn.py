# fidcnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import LocalizedAttention  # Import LocalizedAttention from ladcgan.py

class FID3DCNN(nn.Module):

    def __init__(self):
        super(FID3DCNN, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(32)

        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(64)

        self.local_attn = LocalizedAttention(in_channels=64, local_window_size=3)
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(128)

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc = nn.Linear(128, 128) 

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))

        x = F.relu(self.bn2(self.conv2(x)))

        x = self.local_attn(x)
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.avg_pool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x