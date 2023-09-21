"""Segmentation related plotting functions."""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from geda.utils.colors import color_map, color_map_viz
from typing import Callable
from src.metrics.results import SegmentationResult

COLORMAP = color_map(256)


def plot_segmentation_results(
    results: list[SegmentationResult],
    labels: list[str],
    inverse_preprocessing: Callable,
    filepath: str | None,
) -> None:
    """Plot image, y_true and y_pred (masks) for each result."""
    nrows = len(results)
    fig, axes = plt.subplots(nrows + 1, 3, figsize=(20, 7 * nrows))

    color_map_viz(labels, background=0, void=255, ax=axes[0][0])

    for i, result in enumerate(results):
        ax = axes[i + 1]
        image, y_pred, y_true = result.image, result.y_pred, result.y_true
        image = inverse_preprocessing(image)

        y_pred = y_pred.argmax(0)
        y_true = colorize_2d_segmentation_mask(y_true)
        y_pred = colorize_2d_segmentation_mask(y_pred)
        ax[0].imshow(image)
        ax[1].imshow(y_true)
        ax[2].imshow(y_pred)
        for _ax in ax:
            _ax.set_axis_off()

    if filepath is not None:
        fig.savefig(filepath, bbox_inches="tight")
    plt.close()


def plot_mask_on_image(image: np.ndarray, mask: np.ndarray, txt: str | None = None) -> np.ndarray:
    """Put segmentation mask on image."""
    fontscale = 0.5  # line width
    fw = 1
    alpha = 0.9
    BLACK, WHITE = (0, 0, 0), (255, 255, 255)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    cv2.addWeighted(mask, alpha, image, 1.0, 0, image)
    if txt is not None:
        w, h = cv2.getTextSize(txt, 0, fontScale=fontscale, thickness=fw)[0]
        cv2.rectangle(image, (0, 0), (w + h // 2, h + h // 2), BLACK, -1, cv2.LINE_AA)
        cv2.putText(image, txt, (h // 4, h + h // 4), 0, fontscale, WHITE, fw, cv2.LINE_AA)
    return image


def colorize_2d_segmentation_mask(mask: np.ndarray, colormap: np.ndarray = COLORMAP):
    """parse 2d segmentation mask (grayscale) to RGB mask."""
    mask = np.expand_dims(mask, axis=-1)
    cmap = colormap[:, np.newaxis, :]
    new_mask = np.dot(mask == 0, cmap[0])
    unique_labels = np.unique(mask)
    for label_id in unique_labels:
        new_mask += np.dot(mask == label_id, cmap[label_id])
    return new_mask
