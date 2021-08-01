import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv_block = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, kernel_size=(3,3), padding='same'),
                nn.BatchNorm2d(channels_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels_out, channels_out, kernel_size=(3,3), padding='same'),
                nn.BatchNorm2d(channels_out),
                nn.ReLU(inplace=True)
            )
    def forward(self, x):
        return self.conv_block(x)
      
      
      
      
# ------------------------------------------------------------------------------------





class DownBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.down = nn.Sequential(
            DoubleConv(channels_in, channels_out),
            nn.MaxPool2d(2)
        )

    def forward(self,x):
        return self.down(x)
      
      
      
      
      
# ----------------------------------------------------------------------------------








class UpBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.up = nn.ConvTranspose2d(channels_in, channels_out, kernel_size=(2,2), stride=2)
        self.conv = DoubleConv(channels_in, channels_out)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w //2, diff_h // 2, diff_h - diff_h // 2])

        x = torch.cat([x2,x1], dim=1)
        return self.conv(x)
      
      
      
      
      
      
      
# ------------------------------------------------------------------------------------------------
      
      
      
      
 class Unet(nn.Module):
    def __init__(
        self,
        n_channels = 1, 
        n_classes = 1,
        lr = 1e-3
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.lr = lr

        self.input = DoubleConv(self.n_channels,32)
        self.down1 = DownBlock(32, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.up1 = UpBlock(256, 128)
        self.up2 = UpBlock(128, 64)
        self.up3 = UpBlock(64, 32)
        self.out = DoubleConv(32,self.n_classes)
  
    def forward(self, x):
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        return self.out(x)
      
      
  
      
      
      
      
      
      
