"""
modules.py - This file stores the rathering boring network blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from model.propagation.efficientnet import mod_efficientnet
from model.propagation.efficientnet import cbam


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class FeatureFusionBlock(nn.Module):
    def __init__(self, indim, outdim):
        super().__init__()

        self.block1 = ResBlock(indim, outdim)
        self.attention = cbam.CBAM(outdim)
        self.block2 = ResBlock(outdim, outdim)

    def forward(self, x, f16):
        x = torch.cat([x, f16], 1)
        x = self.block1(x)
        r = self.attention(x)
        x = self.block2(x + r)

        return x


# Single object version, used only in static image pretraining
# This will be loaded and modified into the multiple objects version later (in stage 1/2/3)
# See model.py (load_network) for the modification procedure
class ValueEncoderSO(nn.Module):
    def __init__(self):
        super().__init__()

        efficientnet = mod_efficientnet.efficientnet_b0(pretrained=True,
                                                        extra_channels=1)
        self.block1 = efficientnet.features[0]
        self.block2 = efficientnet.features[1]

        self.block3 = efficientnet.features[2]  # 1/4, 24
        self.block4 = efficientnet.features[3]  # 1/8, 40
        self.block5 = efficientnet.features[4]
        self.block6 = efficientnet.features[5]  # 1/16, 112

        self.fuser = FeatureFusionBlock(120 + 112, 64)

    def forward(self, image, key_f16, mask):
        # key_f16 is the feature from the key encoder

        f = torch.cat([image, mask], 1)

        x = self.block1(f)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        x = self.fuser(x, key_f16)

        return x


# Multiple objects version, used in other times
# Multiple objects version, used in other times
class ValueEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        efficientnet = mod_efficientnet.efficientnet_b0(pretrained=True,
                                                        extra_channels=2)
        self.block1 = efficientnet.features[0]
        self.block2 = efficientnet.features[1]

        self.block3 = efficientnet.features[2]  # 1/4, 24
        self.block4 = efficientnet.features[3]  # 1/8, 40
        self.block5 = efficientnet.features[4]
        self.block6 = efficientnet.features[5]  # 1/16, 112

        self.fuser = FeatureFusionBlock(120 + 112, 64)

    def forward(self, image, key_f16, mask, other_masks):
        # key_f16 is the feature from the key encoder

        f = torch.cat([image, mask, other_masks], 1)

        x = self.block1(f)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        x = self.fuser(x, key_f16)

        return x


class KeyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        efficientnet = models.efficientnet_b2(pretrained=True)
        self.block1 = efficientnet.features[0]
        self.block2 = efficientnet.features[1]

        self.block3 = efficientnet.features[2]  # 1/4, 24
        self.block4 = efficientnet.features[3]  # 1/8, 48
        self.block5 = efficientnet.features[4]
        self.block6 = efficientnet.features[5]  # 1/16, 120

    def forward(self, f):
        x = self.block1(f)
        x = self.block2(x)
        f4 = self.block3(x)   # 1/4, 24
        f8 = self.block4(f4)  # 1/8, 48
        f16 = self.block6(self.block5(f8))  # 1/16, 120

        return f16, f8, f4


class UpsampleBlock(nn.Module):
    def __init__(self, skip_c, up_c, out_c, scale_factor=2):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_c, up_c, kernel_size=3, padding=1)
        self.out_conv = ResBlock(up_c, out_c)
        self.scale_factor = scale_factor

    def forward(self, skip_f, up_f):
        x = self.skip_conv(skip_f)
        x = x + F.interpolate(up_f, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x = self.out_conv(x)
        return x


class KeyProjection(nn.Module):
    def __init__(self, indim, keydim):
        super().__init__()
        self.key_proj = nn.Conv2d(indim, keydim, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)

    def forward(self, x):
        return self.key_proj(x)
