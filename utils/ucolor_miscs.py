""" code adopted from Ucolor pytorch implementation
https://github.com/59Kkk/pytorch_Ucolor_lcy
"""

import numpy as np
import cv2
import torch
from torch import nn
# from __future__ import division #needed


def HSVLoss(im):
        eps = 1e-7
        img = im * 0.5 + 0.5
        hue = torch.Tensor(im.shape[0], im.shape[2], im.shape[3]).to(im.device)

        hue[img[:, 2] == img.max(1)[0]] = 4.0 + ((img[:, 0] - img[:, 1]) / (img.max(1)[0] - img.min(1)[0] + eps))[
            img[:, 2] == img.max(1)[0]]
        hue[img[:, 1] == img.max(1)[0]] = 2.0 + ((img[:, 2] - img[:, 0]) / (img.max(1)[0] - img.min(1)[0] + eps))[
            img[:, 1] == img.max(1)[0]]
        hue[img[:, 0] == img.max(1)[0]] = (0.0 + ((img[:, 1] - img[:, 2]) / (img.max(1)[0] - img.min(1)[0] + eps))[
            img[:, 0] == img.max(1)[0]]) % 6

        hue[img.min(1)[0] == img.max(1)[0]] = 0.0
        hue = hue / 6

        saturation = (img.max(1)[0] - img.min(1)[0]) / (img.max(1)[0] + eps)
        saturation[img.max(1)[0] == 0] = 0

        value = img.max(1)[0]

        cc = torch.stack([hue, saturation, value], dim=1)
        return cc

""" code from rgb_lab_formulation.py """


def check_image(image):
    # Ensure the last dimension has 3 channels
    if image.shape[-1] != 3:
        raise ValueError("Image must have 3 color channels")

    # Check the number of dimensions
    if image.ndim not in (3, 4):
        raise ValueError("Image must be either 3 or 4 dimensions")

    # Explicitly set the last dimension to 3 (this is for documentation or future shape assumptions)
    # PyTorch doesn't have a direct `set_shape` method, but we can verify shape consistency
    if len(image.shape) == 3:
        expected_shape = (image.shape[0], image.shape[1], 3)
    elif len(image.shape) == 4:
        expected_shape = (image.shape[0], image.shape[1], image.shape[2], 3)

    assert image.shape == expected_shape, f"Expected shape {expected_shape}, but got {image.shape}"

    return image


def rgb_to_lab(srgb):
    """
    Converts an RGB image to LAB color space.
    :param srgb: Input image in RGB format with shape (H, W, 3) or (N, H, W, 3)
    :return: Image in LAB color space with the same shape as the input
    """
    # Ensure the image has 3 color channels
    srgb = check_image(srgb)

    # Flatten the image to process pixels
    srgb_pixels = srgb.view(-1, 3)

    # Step 1: Convert sRGB to XYZ
    linear_mask = (srgb_pixels <= 0.04045).float()
    exponential_mask = (srgb_pixels > 0.04045).float()
    rgb_pixels = (srgb_pixels / 12.92) * linear_mask + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask

    rgb_to_xyz = torch.tensor([
        [0.412453, 0.212671, 0.019334],  # R
        [0.357580, 0.715160, 0.119193],  # G
        [0.180423, 0.072169, 0.950227],  # B
    ], dtype=torch.float32, device=srgb.device)

    xyz_pixels = torch.matmul(rgb_pixels, rgb_to_xyz)

    # Step 2: Convert XYZ to LAB
    # Normalize for D65 white point
    xyz_normalized_pixels = xyz_pixels * torch.tensor(
        [1/0.950456, 1.0, 1/1.088754], dtype=torch.float32, device=srgb.device
    )

    epsilon = 6 / 29
    linear_mask = (xyz_normalized_pixels <= (epsilon ** 3)).float()
    exponential_mask = (xyz_normalized_pixels > (epsilon ** 3)).float()

    fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + \
                    (xyz_normalized_pixels ** (1/3)) * exponential_mask

    fxfyfz_to_lab = torch.tensor([
        [0.0, 500.0, 0.0],    # fx
        [116.0, -500.0, 200.0],  # fy
        [0.0, 0.0, -200.0],   # fz
    ], dtype=torch.float32, device=srgb.device)

    lab_pixels = torch.matmul(fxfyfz_pixels, fxfyfz_to_lab) + torch.tensor(
        [-16.0, 0.0, 0.0], dtype=torch.float32, device=srgb.device
    )

    # Reshape back to the original shape
    return lab_pixels.view_as(srgb)