"""Evaluator class for segmentation task."""

import albumentations as A
import torch
from src.model.model import BaseModel


class BaseEvaluator:
    def __init__(
        self,
        model: BaseModel,
        input_shape: tuple[int, int, int, int] = (1, 3, 256, 256),
        device: str = "cuda:0",
    ):
        self.model = model.to(device)
        self.device = device
        self.input_shape = input_shape

    def _dummy_input(self) -> torch.Tensor:
        """Return dummy input (according to input_shape)."""
        return torch.randn(*self.input_shape).to(self.device)

    def _warmup(self, n_iter: int = 5) -> None:
        """Warmup the model for correct inference time measurement."""
        for _ in range(n_iter):
            self.model(self._dummy_input())
