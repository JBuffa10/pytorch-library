import torch.nn as nn
import torchvision
import torch
from collections import OrderedDict
from utils.fpn_blocks import UpSample, Smooth

# --------------------------- Feature Pyramid Network with ResNet or DenseNet backbone ---------------------------



class FeaturePyramidNetwork(nn.Module):
    def __init__(self, backbone:str, n_layers, pretrain:bool=True, memory_efficient:bool=True):
        super().__init__()
        """
        args:
            backbone (string): Backbone model. Must be in ['densenet', 'resnet']. Default is 'resnet'
            n_layers (string or integer): Number of layers in backbone. 
                For 'resnet' must be in [18, 34, 50, 101, 152].
                For 'densenet' must be in [121, 161, 169, 201].
            pretrain (boolean): Whether to utilize pretrained backbone. Default is True.
            memory_efficient (boolean): Whether to utilize memory efficient densenet.
        returns:
            fpn (OrderedDict): OrderedDictionary of output channels from Feature Pyramid Network.
        """
        assert backbone in ['resnet', 'densenet'], 'Must be one of ["resnet", "densenet"]'

        n_layers = str(n_layers)

        if backbone == 'densenet':
            assert n_layers in ['121', '161', '169', '201'], 'Must one of [121, 161, 169, 201]'
        elif backbone == 'resnet':
            assert n_layers in ['18', '34', '50', '101', '152'], 'Must be one of [18, 34, 50, 101, 152]'
        
        # Get ResNet model
        if backbone == 'resnet':
            if n_layers == '18':
                self.backbone = torchvision.models.resnet18(pretrained=pretrain)
            elif n_layers == '34':
                self.backbone = torchvision.models.resnet34(pretrained=pretrain)
            elif n_layers == '50':
                self.backbone = torchvision.models.resnet50(pretrained=pretrain)
            elif n_layers == '101':
                self.backbone = torchvision.models.resnet101(pretrained=pretrain)
            elif n_layers == '152':
                self.backbone = torchvision.models.resnet152(pretrained=pretrain)
            
            # Freeze backbone
            for param in self.backbone.parameters():
                param.requires_grad = False
        
            # Break down layers
            self.input = nn.Sequential(
                self.backbone.conv1,
                self.backbone.bn1,
                self.backbone.relu,
                self.backbone.maxpool
            )
            self.c1 = self.backbone.layer1
            self.c2 = self.backbone.layer2
            self.c3 = self.backbone.layer3
            self.c4 = self.backbone.layer4
            
            # Get num out channels for ResNet layers 1 - 4
            if n_layers in ['18', '34']:
                self.resnet_out_channels = [c[-1].conv2.out_channels for c in (self.c1, self.c2, self.c3, self.c4)]
            if n_layers in ['50', '101', '152']:
                self.resnet_out_channels = [c[-1].conv3.out_channels for c in (self.c1, self.c2, self.c3, self.c4)]

            # Create Feature Pyramid Network with ResNet backbone
            self.fpn = torchvision.ops.FeaturePyramidNetwork(self.resnet_out_channels, 256)
        
        # Get DenseNet model
        elif backbone == 'densenet':
            if n_layers == '121':
                self.backbone = torchvision.models.densenet121(pretrained=pretrain, memory_efficient=memory_efficient)
            elif n_layers == '161':
                self.backbone = torchvision.models.densenet161(pretrained=pretrain, memory_efficient=memory_efficient)
            elif n_layers == '169':
                self.backbone = torchvision.models.densenet169(pretrained=pretrain, memory_efficient=memory_efficient)
            elif n_layers == '201':
                self.backbone = torchvision.models.densenet201(pretrained=pretrain, memory_efficient=memory_efficient)
            
            # Freeze backbone
            for param in self.backbone.parameters():
                param.requires_grad = False
                
            # Break down layers
            self.input = nn.Sequential(
                self.backbone.features.conv0,
                self.backbone.features.norm0,
                self.backbone.features.relu0,
                self.backbone.features.relu0, 
            )

            self.c1 = nn.Sequential(
                self.backbone.features.denseblock1,
                self.backbone.features.transition1
            )

            self.c2 = nn.Sequential(
                self.backbone.features.denseblock2,
                self.backbone.features.transition2            
            )

            self.c3 = nn.Sequential(
                self.backbone.features.denseblock3,
                self.backbone.features.transition3            
            )

            self.c4 = nn.Sequential(
                self.backbone.features.denseblock4,
                self.backbone.features.norm5            
            )
            
            # Get num out channels for DenseNet blocks
            self.densenet_out_channels = [self.backbone.features.transition1.conv.out_channels, 
                                        self.backbone.features.transition2.conv.out_channels, 
                                        self.backbone.features.transition3.conv.out_channels, 
                                        self.backbone.features.norm5.num_features]

            # Create Feature Pyramid Network with DenseNet backbone
            self.fpn = torchvision.ops.FeaturePyramidNetwork(self.densenet_out_channels, 256)
            
    def forward(self, x):
        # backbone 
        x = self.input(x)
        c1 = self.c1(x)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        
        # Create Ordered Dictionary of output channels
        c = OrderedDict()
        c['p2'] = c1
        c['p3'] = c2
        c['p4'] = c3
        c['p5'] = c4
        
        # Pass backbone outputs to Feature Pyramid Network and return results
        return self.fpn(c)
      
      
      
      
      
      
      
# --------------------------- Feature Pyramid Network with VGG16 backbone from scratch ---------------------------






class VGG16FPN(nn.Module):
    """
    Feature Pyramid Network with VGG16 backbone.
    """
    def __init__(self, num_channels=3):
        super().__init__()
        self.num_channels = num_channels
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_channels, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.bottleneck = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)

        self.up1 = UpSample(512)
        self.up2 = UpSample(256)
        self.up3 = UpSample(128)
        self.up4 = UpSample(64)

        self.smooth = Smooth()
    
    def forward(self, x):
        c1 = self.conv_block_1(x)
        c2 = self.conv_block_2(c1)
        c3 = self.conv_block_3(c2)
        c4 = self.conv_block_4(c3)
        c5 = self.conv_block_5(c4)

        neck = self.bottleneck(c5)

        p5 = self.up1(neck, c4)
        p4 = self.up2(p5, c3)
        p3 = self.up3(p4, c2)
        p2 = self.up4(p3, c1)

        p5 = self.smooth(p5)
        p4 = self.smooth(p4)
        p3 = self.smooth(p3)
        p2 = self.smooth(p2)

        return p2, p3, p4, p5
      
      
      
