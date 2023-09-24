"""Training callbacks."""

from src.logging import get_pylogger
from src.visualization import plot_segmentation_results
from src.callbacks.base import BaseCallback
from src.model.module.trainer import Trainer

from typing import Callable

log = get_pylogger(__name__)


class SegmentationExamplesPlotterCallback(BaseCallback):
    """Plot prediction examples"""

    def __init__(
        self,
        inverse_preprocessing: Callable,
        cmap: list[tuple[int, int, int]],
        stage: str,
        dirpath: str | None = None,
    ):
        self.dirpath = dirpath
        self.stage = stage
        self.inverse_preprocessing = inverse_preprocessing
        self.cmap = cmap

    def log(self, trainer: Trainer) -> None:
        results = trainer.module.results[self.stage]
        filepath = f"{self.dirpath}/{trainer.current_step}"
        plot_segmentation_results(results, self.cmap, self.inverse_preprocessing, filepath)
