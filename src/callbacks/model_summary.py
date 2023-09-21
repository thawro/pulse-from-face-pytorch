from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.model.module.trainer import Trainer

from torchinfo import summary

from .base import BaseCallback
from src.logging import get_pylogger
from src.utils.files import save_txt_to_file

log = get_pylogger(__name__)


class ModelSummary(BaseCallback):
    def __init__(
        self,
        input_size: tuple[int, ...] | list[tuple[int, ...]],
        depth: int = 4,
        filepath: str | None = None,
    ):
        self.depth = depth
        self.input_size = input_size
        self.filepath = filepath

    def on_fit_start(self, trainer: Trainer):
        model_summary = str(
            summary(
                trainer.module.model,
                input_size=self.input_size,
                depth=self.depth,
                col_names=[
                    "input_size",
                    "output_size",
                    "num_params",
                    "params_percent",
                    "mult_adds",
                    # "trainable",
                ],
            )
        )
        if self.filepath is not None:
            save_txt_to_file(model_summary, self.filepath)
