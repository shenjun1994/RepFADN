import torch
import torch.nn as nn
import numpy as np
from basicsr.utils.registry import ARCH_REGISTRY


def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


def bilinear_upsample_weights(filter_size, weights):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """
    f_out = weights.size(0)
    f_in = weights.size(1)
    weights = np.zeros((f_out,
                        f_in,
                        4,
                        4), dtype=np.float32)

    upsample_kernel = upsample_filt(filter_size)

    for i in range(f_out):
        for j in range(f_in):
            weights[i, j, :, :] = upsample_kernel
    return torch.Tensor(weights)


class FeatureExtraction(nn.Module):
    def __init__(self, level, in_channels):
        super(FeatureExtraction, self).__init__()
        if level == 1:
            self.conv0 = nn.Conv2d(in_channels, 64, (3, 3), (1, 1), (1, 1))
        else:
            self.conv0 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.convt_F = nn.ConvTranspose2d(64, 64, (4, 4), (2, 2), (1, 1))
        self.LReLus = nn.LeakyReLU(negative_slope=0.2)
        self.convt_F.weight.data.copy_(bilinear_upsample_weights(4, self.convt_F.weight))

    def forward(self, x):
        out = self.LReLus(self.conv0(x))
        out = self.LReLus(self.conv1(out))
        out = self.LReLus(self.conv2(out))
        out = self.LReLus(self.conv3(out))
        out = self.LReLus(self.conv4(out))
        out = self.LReLus(self.conv5(out))
        out = self.LReLus(self.convt_F(out))
        return out


class ImageReconstruction(nn.Module):
    def __init__(self, out_channels):
        super(ImageReconstruction, self).__init__()
        self.conv_R = nn.Conv2d(64, out_channels, (3, 3), (1, 1), (1, 1))
        self.convt_I = nn.ConvTranspose2d(out_channels, out_channels, (4, 4), (2, 2), (1, 1))
        self.convt_I.weight.data.copy_(bilinear_upsample_weights(4, self.convt_I.weight))

    def forward(self, LR, convt_F):
        convt_I = self.convt_I(LR)
        conv_R = self.conv_R(convt_F)

        HR = convt_I + conv_R
        return HR

@ARCH_REGISTRY.register()
class LapSRN_x2(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(LapSRN_x2, self).__init__()
        self.FeatureExtraction1 = FeatureExtraction(level=1, in_channels=in_chans)
        self.ImageReconstruction1 = ImageReconstruction(out_chans)

    def forward(self, LR):
        convt_F1 = self.FeatureExtraction1(LR)
        HR_2 = self.ImageReconstruction1(LR, convt_F1)


        return HR_2

@ARCH_REGISTRY.register()
class LapSRN_x4(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(LapSRN_x4, self).__init__()
        self.FeatureExtraction1 = FeatureExtraction(level=1, in_channels=in_chans)
        self.FeatureExtraction2 = FeatureExtraction(level=2, in_channels=in_chans)
        self.ImageReconstruction1 = ImageReconstruction(out_chans)
        self.ImageReconstruction2 = ImageReconstruction(out_chans)

    def forward(self, LR):
        convt_F1 = self.FeatureExtraction1(LR)
        HR_2 = self.ImageReconstruction1(LR, convt_F1)

        convt_F2 = self.FeatureExtraction2(convt_F1)
        HR_4 = self.ImageReconstruction2(HR_2, convt_F2)
        return HR_4

@ARCH_REGISTRY.register()
class LapSRN_x8(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(LapSRN_x8, self).__init__()
        self.FeatureExtraction1 = FeatureExtraction(level=1, in_channels=in_chans)
        self.FeatureExtraction2 = FeatureExtraction(level=2, in_channels=in_chans)
        self.FeatureExtraction3 = FeatureExtraction(level=3, in_channels=in_chans)
        self.ImageReconstruction1 = ImageReconstruction(out_chans)
        self.ImageReconstruction2 = ImageReconstruction(out_chans)
        self.ImageReconstruction3 = ImageReconstruction(out_chans)

    def forward(self, LR):
        convt_F1 = self.FeatureExtraction1(LR)
        HR_2 = self.ImageReconstruction1(LR, convt_F1)

        convt_F2 = self.FeatureExtraction2(convt_F1)
        HR_4 = self.ImageReconstruction2(HR_2, convt_F2)

        convt_F3 = self.FeatureExtraction3(convt_F2)
        HR_8 = self.ImageReconstruction3(HR_4, convt_F3)

        return HR_8