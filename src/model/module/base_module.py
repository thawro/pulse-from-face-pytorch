from abc import abstractmethod

import torch
from torch import nn, Tensor

from src.data.datamodule import DataModule
from src.logging import get_pylogger
from src.logging.loggers import BaseLogger
from src.metrics import MetricsStorage, Result
from src.model.model import BaseModel
from src.model.loss import BaseLoss
from src.model.metrics import BaseMetrics

from src.callbacks import Callbacks

log = get_pylogger(__name__)

SPLITS = ["train", "val", "test"]


class BaseModule:
    logger: BaseLogger
    device: torch.device
    datamodule: DataModule
    callbacks: "Callbacks"
    current_epoch: int

    def __init__(
        self,
        model: BaseModel,
        loss_fn: BaseLoss,
        metrics: BaseMetrics,
        optimizers: dict[str, torch.optim.Optimizer],
        schedulers: dict[str, torch.optim.lr_scheduler.LRScheduler] = {},
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.metrics = metrics
        self.steps_metrics_storage = MetricsStorage()
        self.epochs_metrics_storage = MetricsStorage()
        self.results: dict[str, list[Result]] = {split: [] for split in SPLITS}

    def pass_attributes(
        self,
        device: torch.device,
        logger: BaseLogger,
        callbacks: "Callbacks",
        datamodule: DataModule,
    ):
        self.logger = logger
        self.callbacks = callbacks
        self.datamodule = datamodule
        self.device = device

    def set_attributes(self, **attributes):
        for name, attr in attributes.items():
            setattr(self, name, attr)

    def load_state_dict(self, state_dict: dict):
        self.model.load_state_dict(state_dict["model"])
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state_dict["optimizers"][name])
        for name, scheduler in self.schedulers.items():
            scheduler.load_state_dict(state_dict["schedulers"][name])
        self.steps_metrics_storage.load_state_dict(state_dict["metrics"]["steps"])
        self.epochs_metrics_storage.load_state_dict(state_dict["metrics"]["epochs"])

    def state_dict(self) -> dict:
        optimizers_state = {
            name: optimizer.state_dict() for name, optimizer in self.optimizers.items()
        }
        schedulers_state = {
            name: scheduler.state_dict() for name, scheduler in self.schedulers.items()
        }
        metrics_state = {
            "steps": self.steps_metrics_storage.state_dict(),
            "epochs": self.epochs_metrics_storage.state_dict(),
        }
        model_state = {"model": self.model.state_dict()}
        model_state.update(
            {
                "optimizers": optimizers_state,
                "schedulers": schedulers_state,
                "metrics": metrics_state,
            }
        )
        return model_state

    def _common_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int, stage: str):
        data, y_true = batch
        y_pred = self.model(data)
        loss = self.loss_fn(y_pred, y_true)
        if stage == "train":
            self.optimizers["optim_0"].zero_grad()
            loss.backward()
            self.optimizers["optim_0"].step()
        losses = {
            "loss": loss,
        }
        metrics = self.metrics.calculate_metrics(y_pred=y_pred, y_true=y_true)
        self.steps_metrics_storage.append(losses, stage)
        self.steps_metrics_storage.append(metrics, stage)
        results = {"data": data, "y_true": y_true, "y_pred": y_pred}
        if batch_idx == 0:
            num_examples = min(10, len(results["image"]))
            for name, tensor in results.items():
                results[name] = tensor[:num_examples].cpu().detach().numpy()

            for i in range(num_examples):
                result = Result(
                    data=results["data"][i],
                    y_true=results["y_true"][i],
                    y_pred=results["y_pred"][i],
                )
                self.results[stage].append(result)

    def training_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int):
        self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int):
        with torch.no_grad():
            self._common_step(batch, batch_idx, "val")

    @abstractmethod
    def on_train_epoch_start(self):
        pass

    @abstractmethod
    def on_validation_epoch_start(self):
        pass

    def _common_epoch_end(self, stage: str) -> None:
        batch_metrics = self.steps_metrics_storage.inverse_nest()[stage]
        mean_metrics = {name: sum(values) / len(values) for name, values in batch_metrics.items()}
        msg = [f"Epoch: {self.current_epoch}"]
        for name, value in mean_metrics.items():
            msg.append(f"{stage}/{name}: {round(value, 3)}")
        log.info("  ".join(msg))
        self.epochs_metrics_storage.append(mean_metrics, split=stage)

    def on_train_epoch_end(self) -> None:
        self._common_epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._common_epoch_end("val")

    def on_epoch_end(self):
        optizers_lr = {
            f"{name}_LR": optimizer.param_groups[0]["lr"]
            for name, optimizer in self.optimizers.items()
        }
        self.epochs_metrics_storage.append(optizers_lr, split="train")
