import torch.nn as nn
import torchvision
import torch
from typing import NewType

class ResNet50(nn.Module):
  def __init__(self, pretrained: bool = True, num_features_out: int = 256):
    super().__init__()
    self.pretrained = pretrained
    self.num_features_out = num_features_out
    self.resent50 = torchvision.models.resnet50(pretrained=self.pretrained)

  def get_features(self):
    features = list(self.resent50.children())

    input = nn.Sequential(*features[:4])
    conv2 = features[4]
    conv3 = features[5]
    conv4 = features[6]
    conv5 = features[7]

    lateral_c2 = nn.Conv2d(in_channels=256, out_channels=self.num_features_out, kernel_size=1)
    lateral_c3 = nn.Conv2d(in_channels=512, out_channels=self.num_features_out, kernel_size=1)
    lateral_c4 = nn.Conv2d(in_channels=1024, out_channels=self.num_features_out, kernel_size=1)
    lateral_c5 = nn.Conv2d(in_channels=2048, out_channels=self.num_features_out, kernel_size=1)

    dealiasing_p2 = nn.Conv2d(in_channels=self.num_features_out, out_channels=self.num_features_out, kernel_size=3, padding=1)
    dealiasing_p3 = nn.Conv2d(in_channels=self.num_features_out, out_channels=self.num_features_out, kernel_size=3, padding=1)
    dealiasing_p4 = nn.Conv2d(in_channels=self.num_features_out, out_channels=self.num_features_out, kernel_size=3, padding=1)

    # Freeze backbone
    for parameters in [module.parameters() for module in [input, conv2]]:
      for parameter in parameters:
        parameter.requires_grad = False

    module = NewType('layers',nn.Module)
    conv_layers = (module(x) for x in (input, conv2, conv3, conv4, conv5))
    lateral_layers = (module(x) for x in (lateral_c2, lateral_c3, lateral_c4, lateral_c5))
    dealiasing_layers = (module(x) for x in (dealiasing_p2, dealiasing_p3, dealiasing_p4))

    return conv_layers, lateral_layers, dealiasing_layers, self.num_features_out

  def forward(self,x):
    return self.resent50(x)
  
