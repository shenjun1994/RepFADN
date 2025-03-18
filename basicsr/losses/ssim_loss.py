import torch
import torch.nn as nn
import cv2
import numpy as np
from torch.nn import functional as F
from basicsr.utils.color_util import rgb2ycbcr_pt
from basicsr.utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class SSIMLoss(nn.Module):
    def __init__(self, crop_border=4, use_y_channel=False, loss_weight=0.1, **kwargs):
        super(SSIMLoss, self).__init__()
        self.crop_border = crop_border
        self.use_y_channel = use_y_channel
        self.loss_weight = loss_weight

    def _ssim_pth(self, img, img2):
        """Calculate SSIM (structural similarity) (PyTorch version).

        It is called by func:`calculate_ssim_pt`.

        Args:
            img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
            img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).

        Returns:
            float: SSIM result.
        """
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())
        window = torch.from_numpy(window).view(1, 1, 11, 11).expand(img.size(1), 1, 11, 11).to(img.dtype).to(img.device)

        mu1 = F.conv2d(img, window, stride=1, padding=0, groups=img.shape[1])  # valid mode
        mu2 = F.conv2d(img2, window, stride=1, padding=0, groups=img2.shape[1])  # valid mode
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img * img, window, stride=1, padding=0, groups=img.shape[1]) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu2_sq
        sigma12 = F.conv2d(img * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu1_mu2

        cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
        ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
        return ssim_map.mean([1, 2, 3])

    def calculate_ssim_pt(self, img, img2, **kwargs):
        """Calculate SSIM (structural similarity) (PyTorch version).

        ``Paper: Image quality assessment: From error visibility to structural similarity``

        The results are the same as that of the official released MATLAB code in
        https://ece.uwaterloo.ca/~z70wang/research/ssim/.

        For three-channel images, SSIM is calculated for each channel and then
        averaged.

        Args:
            img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
            img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
            crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
            test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

        Returns:
            float: SSIM result.
        """

        assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

        if self.crop_border != 0:
            img = img[:, :, self.crop_border:-self.crop_border, self.crop_border:-self.crop_border]
            img2 = img2[:, :, self.crop_border:-self.crop_border, self.crop_border:-self.crop_border]

        if self.use_y_channel:
            img = rgb2ycbcr_pt(img, y_only=True)
            img2 = rgb2ycbcr_pt(img2, y_only=True)

        img = img.to(torch.float64)
        img2 = img2.to(torch.float64)

        ssim = self._ssim_pth(img * 255., img2 * 255.)
        return ssim.mean()

    def forward(self, pred, gt, **kwargs):
        ssim = self.calculate_ssim_pt(pred, gt)
        red_ssim = 1 - ssim
        return self.loss_weight * red_ssim