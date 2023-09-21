from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.model.module.trainer import Trainer

from .base import BaseCallback
from src.logging import get_pylogger
from src.utils.files import save_yaml
from src.visualization import plot_metrics

log = get_pylogger(__name__)


class MetricsPlotterCallback(BaseCallback):
    """Plot per epoch metrics"""

    def __init__(self, filepath: str | None = None):
        if filepath is None:
            filepath = "metrics.jpg"
        self.filepath = filepath

    def on_epoch_end(self, trainer: Trainer) -> None:
        plot_metrics(trainer.module.epochs_metrics_storage, filepath=self.filepath)


class MetricsSaverCallback(BaseCallback):
    """Plot per epoch metrics"""

    def __init__(self, filepath: str | None = None):
        if filepath is None:
            filepath = "results.yaml"
        self.filepath = filepath

    def on_epoch_end(self, trainer: Trainer) -> None:
        metrics_storage = trainer.module.epochs_metrics_storage
        metrics = metrics_storage.to_dict()
        save_yaml(metrics, self.filepath)
