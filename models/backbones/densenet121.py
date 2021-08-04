import torch
from torch import nn
from typing import NewType

class DenseNet121(nn.Module):
  def __init__(self. pretrained: bool = True, num_features_out: int = 256):
    super().__init__()
    self.pretrained = pretrained
    self.num_features_out = num_features_out
    self.densenet121 = torchvision.models.densenet121(pretrained=self.pretrained)
    
   def get_features(self):
    features = list(self.densenet121.children())
    
    input = nn.Sequential(features[0][:3]) 
    conv2 = nn.Sequential(features[0].denseblock1, features[0].transition1) # in channels = 64
    conv3 = nn.Sequential(features[0].denseblock2, features[0].transition2) # in channels = 128
    conv4 = nn.Sequential(features[0].denseblock3, features[0].transition3) # in channels = 256
    conv5 = nn.Sequential(features[0].denseblock4, features[0].norm5) # in channels = 512
    
    lateral_c2 = nn.Conv2d(in_channels = 64, out_channels = self.num_features_out, kernel_size = 1)
    lateral_c3 = nn.Conv2d(in_channels = 128, out_channels = self.num_features_out, kernel_size = 1)
    lateral_c4 = nn.Conv2d(in_channels = 256, out_channels = self.num_features_out, kernel_size = 1)
    lateral_c5 = nn.Conv2d(in_channels = 512, out_channels = self.num_features_out, kernel_size = 1)
    
    dealiasing_p2 = nn.Conv2d(in_channels = self.num_features_out, out_channels = self.num_features_out, kernel_size = 3, padding = 1)
    dealiasing_p3 = nn.Conv2d(in_channels = self.num_features_out, out_channels = self.num_features_out, kernel_size = 3, padding = 1)
    dealiasing_p4 = nn.Conv2d(in_channels = self.num_features_out, out_channels = self.num_features_out, kernel_size = 3, padding = 1)
    
    # Freeze backbone
    for parameters in [module.parameters() for module in [input, conv2]]:
      for parameter in parameters:
        parameter.require_grad = False
        
    module = NewType('layers', nn.Module)
    conv_layers = (module(x) for x in (input, conv2, conv3, conv4, conv5))
    lateral_layers = (module(x) for x in (lateral_c2, lateral_c3, lateral_c4, lateral_c5))
    dealiasing_layers = (module(x) for x in (dealiasing_p2, dealiasing_p3, dealiasing_p4))
    
    return conv_layers, lateral_layers, dealiasing_layers, self.num_features_out
  
 def forward(self, x):
  return self.densenet121(x)
    
