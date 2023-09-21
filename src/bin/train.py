"""Train the model"""

import torch
from torch import nn, optim
from src.data import DataModule, BaseDataset
from src.data.transforms import train_transform, inference_transform
from src.logging import TerminalLogger, get_pylogger
from src.callbacks import (
    LoadModelCheckpoint,
    MetricsPlotterCallback,
    MetricsSaverCallback,
    ModelSummary,
    SaveModelCheckpoint,
)

from src.model.model.base import BaseModel
from src.model.loss import BaseLoss, WeightedLoss
from src.model.module import Trainer, BaseModule
from src.model.metrics import BaseMetrics, BaseMetric
from src.model.utils import seed_everything

from src.utils import DS_ROOT, NOW, ROOT

log = get_pylogger(__name__)

EXPERIMENT_NAME = "watermak_bbox_localizer"

CONFIG = {
    "seed": 42,
    "dataset": "<ds_name>",
    "input_size": (32, 3, 256, 256),
    "max_epochs": 500,
    "batch_size": 48,
    "device": "cuda",
    "loss_weights": {"loss": 1.0},
    # "ckpt_path": "/home/tomhaw/nn-watermarks/results/watermark_test/13-09-2023_07:39:52/checkpoints/last.pt",
    "limit_batches": -1,
}

if CONFIG["limit_batches"] != -1:
    EXPERIMENT_NAME = "debug"

RUN_NAME = f"{NOW}"
CONFIG["logs_path"] = str(ROOT / "results" / EXPERIMENT_NAME / RUN_NAME)


def create_datamodule() -> DataModule:
    tr_transform = train_transform(CONFIG["image_size"], CONFIG["image_size"])
    val_transform = inference_transform(CONFIG["image_size"])

    ds_path = str(DS_ROOT / CONFIG["dataset"])
    train_ds = BaseDataset(ds_path, "trainaug", tr_transform)
    val_ds = BaseDataset(ds_path, "val", val_transform)

    return DataModule(
        train_ds=train_ds, val_ds=val_ds, test_ds=None, batch_size=CONFIG["batch_size"]
    )


def create_model() -> BaseModel:
    net = nn.Identity()
    return BaseModel(
        net=net, input_size=CONFIG["input_size"], input_names=["images"], output_names=["output"]
    )


def create_loss_fn() -> BaseLoss:
    loss_fn = WeightedLoss(nn.MSELoss(), weight=CONFIG["loss_weights"]["loss"])
    return BaseLoss(loss_fn)


def create_callbacks(logger) -> list:
    ckpt_saver_params = dict(ckpt_dir=logger.ckpt_dir, stage="val", mode="min")
    summary_filepath = str(logger.model_dir / "model_summary.txt")
    callbacks = [
        MetricsPlotterCallback(str(logger.log_path / "metrics.jpg")),
        MetricsSaverCallback(str(logger.log_path / "metrics.yaml")),
        ModelSummary(input_size=CONFIG["input_size"], depth=4, filepath=summary_filepath),
        SaveModelCheckpoint(name="best_G", metric="enc_dec_loss", **ckpt_saver_params),
        SaveModelCheckpoint(name="best_D", metric="disc_loss", **ckpt_saver_params),
        SaveModelCheckpoint(name="last", last=True, top_k=0, **ckpt_saver_params),
    ]
    if "ckpt_path" in CONFIG and CONFIG["ckpt_path"] is not None:
        callbacks.append(LoadModelCheckpoint(CONFIG["ckpt_path"]))
    return callbacks


def main() -> None:
    seed_everything(CONFIG["seed"])
    torch.set_float32_matmul_precision("medium")

    datamodule = create_datamodule()
    model = create_model()
    loss_fn = create_loss_fn()
    logger = TerminalLogger(CONFIG["logs_path"], config=CONFIG)
    callbacks = create_callbacks(logger)
    optimizers = {
        "optim_0": optim.SGD(model.parameters(), lr=CONFIG["lr"], momentum=1e-4, nesterov=True)
    }
    module = BaseModule(
        model=model,
        loss_fn=loss_fn,
        metrics=BaseMetrics([BaseMetric()]),
        optimizers=optimizers,
        schedulers={},
    )
    logger.log_config()

    trainer = Trainer(
        logger=logger,
        device=CONFIG["device"],
        callbacks=callbacks,
        max_epochs=CONFIG["max_epochs"],
        limit_batches=CONFIG["limit_batches"],
    )
    trainer.fit(module, datamodule)


if __name__ == "__main__":
    main()
