import torch.nn as nn
"""
Feature Pyramid Network helper functions.
"""
# -----------------------------------------------------------

class UpSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 256, kernel_size=1, stride=1, padding=0)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.conv(x2)
        return x1 + x2

# -------------------------------------------------------------------


class Smooth(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        return self.smooth(x)
