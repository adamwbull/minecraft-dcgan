import torch
import torch.nn.functional as F

def gaussian_window(size, sigma):
    """Generates a 3D Gaussian window."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    grid = torch.stack(torch.meshgrid([coords, coords, coords], indexing='ij'), dim=0)
    std2 = 2 * sigma * sigma
    window = torch.exp(-torch.sum(grid ** 2, dim=0) / std2)
    return window / window.sum()

def ssim_3d(img1, img2, window_size, window_sigma, data_range, size_average=True):
    """Computes the mean Structural Similarity Index (SSIM) over a 3D volume."""
    window = gaussian_window(window_size, window_sigma).to(img1.device)

    mu1 = F.conv3d(img1, window.unsqueeze(0).unsqueeze(0), padding=window_size//2, groups=1)
    mu2 = F.conv3d(img2, window.unsqueeze(0).unsqueeze(0), padding=window_size//2, groups=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, window.unsqueeze(0).unsqueeze(0), padding=window_size//2, groups=1) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window.unsqueeze(0).unsqueeze(0), padding=window_size//2, groups=1) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window.unsqueeze(0).unsqueeze(0), padding=window_size//2, groups=1) - mu1_mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM3DLossSingleChannel(torch.nn.Module):
    def __init__(self, window_size=11, window_sigma=1.5, data_range=1, size_average=True):
        super(SSIM3DLossSingleChannel, self).__init__()
        self.window_size = window_size
        self.window_sigma = window_sigma
        self.data_range = data_range
        self.size_average = size_average

    def forward(self, img1, img2):
        return 1 - ssim_3d(img1, img2, self.window_size, self.window_sigma, self.data_range, self.size_average)

class SSIM3DLossMultiChannel(torch.nn.Module):
    def __init__(self, window_size=11, window_sigma=1.5, data_range=1, size_average=True, channels=32):
        super(SSIM3DLossMultiChannel, self).__init__()
        self.window_size = window_size
        self.window_sigma = window_sigma
        self.data_range = data_range
        self.size_average = size_average
        self.channels = channels

    def forward(self, img1, img2):
        ssim_total = 0
        for i in range(self.channels):
            ssim_total += ssim_3d(img1[:, i:i+1, ...], img2[:, i:i+1, ...],
                                  self.window_size, self.window_sigma,
                                  self.data_range, self.size_average)
        return 1 - ssim_total / self.channels
