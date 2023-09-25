import numpy as np
import cv2
import torch
import math


def keep_largest_blob(mask: np.ndarray):
    img = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Find largest contour in intermediate image
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return mask
    largest_contour = max(contours, key=cv2.contourArea)

    # Output
    out_mask = np.zeros(img.shape, np.uint8)
    cv2.drawContours(out_mask, [largest_contour], -1, 1, cv2.FILLED)
    out_mask = cv2.bitwise_and(mask.astype(np.uint8), out_mask).astype(np.float32)
    return out_mask


def gaussian_kernel_2d(kernel_size: int, sigma: float, device: str = "cuda"):
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1) / 2.0
    var = sigma**2.0

    kernel = (1 / (2 * math.pi * var)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2.0, dim=-1) / (2 * var)
    )
    return (kernel / torch.sum(kernel)).to(device).repeat(1, 1, 1, 1)
