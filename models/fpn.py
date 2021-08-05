import torch
import torch.nn as nn
import torchvision
from models.backbones.resnet18 import ResNet18


## ----------------------------------------------------------------
## FEATURE PYRAMID NETWORK WITH RESNET OR DENSENET BACKBONE
## -----------------------------------------------------------------



class FeaturePyramidNetwork(nn.Module):
  def __init__(self, backbone: nn.Module, num_classes: int):
    super().__init__()
    conv_layers, lateral_layers, dealiasing_layers, num_features_out = backbone.get_features()
    self.input, self.conv2, self.conv3, self.conv4, self.conv5 = conv_layers
    self.lateral_c2, self.lateral_c3, self.lateral_c4, self.lateral_c5 = lateral_layers
    self.dealiasing_p2, self.dealiasing_p3, self.dealiasing_p4 = dealiasing_layers
    self.num_classes = num_classes

  def forward(self, x):

    # bottom up
    c1 = self.input(x)
    c2 = self.conv2(c1)
    c3 = self.conv3(c2)
    c4 = self.conv4(c3)
    c5 = self.conv5(c4)

    # top down and lateral connection
    p5 = self.lateral_c5(c5)
    p4 = self.lateral_c4(c4) + F.interploate(input=p5, size = (c4.shape[2], c4.shape[3]), mode='nearest')
    p3 = self.lateral_c3(c3) + F.interpolate(input=p4, size = (c3.shape[2], c3.shape[3]), mode='nearest')
    p2 = self.lateral_c2(c2) + F.interpolate(input=p3, size = (c2.shape[2], c2.shape[3]), mode='nearest')

    # reduce aliasing
    p4 = self.dealiasing_p4(p4)
    p3 = self.dealiasing_p3(p3)
    p2 = self.dealiasing_p2(p2)

    p6 = F.max_pool2d(input=p5, kernel_size=1, stride=2)

    return p2, p3, p4, p6
      
      
