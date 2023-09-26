"""Train the model"""

import torch
from torch import nn, optim
from src.data import DataModule, CelebAMaskDataset
from src.data.transforms import CelebATransform
from src.logging import TerminalLogger, get_pylogger
from src.callbacks import (
    LoadModelCheckpoint,
    MetricsPlotterCallback,
    MetricsSaverCallback,
    ModelSummary,
    SaveModelCheckpoint,
    SegmentationExamplesPlotterCallback,
)

from src.model.architectures.psp_net import PSPNet
from src.model.model.segmentation import SegmentationModel
from src.model.loss import WeightedLoss, AuxiliarySegmentationLoss
from src.model.module import Trainer, SegmentationModule
from src.model.metrics.segmentation import SegmentationMetrics
from src.model.utils import seed_everything

from src.bin.config import (
    IMGSZ,
    MEAN,
    STD,
    SEED,
    DS_PATH,
    N_CLASSES,
    BATCH_SIZE,
    MODEL_INPUT_SIZE,
    CKPT_PATH,
    LOGS_PATH,
    LOG_EVERY_N_STEPS,
    CONFIG,
    DEVICE,
    MAX_EPOCHS,
    LIMIT_BATCHES,
    PALETTE,
)

log = get_pylogger(__name__)


transform = CelebATransform(IMGSZ, MEAN, STD)


def create_datamodule() -> DataModule:
    train_ds = CelebAMaskDataset(str(DS_PATH), N_CLASSES, "train", transform.train)
    val_ds = CelebAMaskDataset(str(DS_PATH), N_CLASSES, "val", transform.inference)
    test_ds = CelebAMaskDataset(str(DS_PATH), N_CLASSES, "test", transform.inference)
    return DataModule(train_ds=train_ds, val_ds=val_ds, test_ds=test_ds, batch_size=BATCH_SIZE)


def create_module() -> SegmentationModule:
    loss_fn = AuxiliarySegmentationLoss(
        seg_loss=WeightedLoss(nn.CrossEntropyLoss(), weight=1),
        cls_loss=WeightedLoss(nn.BCEWithLogitsLoss(), weight=0.4),
    )

    model = SegmentationModel(
        net=PSPNet(num_classes=N_CLASSES, cls_dropout=0.5, backbone="resnet101"),
        input_size=MODEL_INPUT_SIZE,
        input_names=["images"],
        output_names=["masks", "class_probs"],
    )

    optimizer = optim.SGD(
        model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4, nesterov=True
    )
    scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=100_000, power=0.9)

    module = SegmentationModule(
        model=model,
        loss_fn=loss_fn,
        metrics=SegmentationMetrics(N_CLASSES),
        optimizers={"optim": optimizer},
        schedulers={"optim": scheduler},
    )
    return module


def create_callbacks(logger: TerminalLogger) -> list:
    ckpt_saver_params = dict(ckpt_dir=logger.ckpt_dir, stage="val", mode="min")
    summary_filepath = str(logger.model_dir / "model_summary.txt")
    examples_dirpath = logger.log_path / "steps_examples"
    examples_dirpath.mkdir()
    callbacks = [
        MetricsPlotterCallback(str(logger.log_path / "epoch_metrics.jpg")),
        MetricsSaverCallback(str(logger.log_path / "epoch_metrics.yaml")),
        ModelSummary(input_size=MODEL_INPUT_SIZE, depth=4, filepath=summary_filepath),
        SaveModelCheckpoint(name="best", metric="mean_IoU", **ckpt_saver_params),
        SaveModelCheckpoint(name="last", last=True, top_k=0, **ckpt_saver_params),
        SegmentationExamplesPlotterCallback(
            inverse_preprocessing=transform.inverse_preprocessing,
            cmap=PALETTE,
            stage="val",
            dirpath=str(examples_dirpath),
        ),
    ]
    if CKPT_PATH is not None:
        callbacks.append(LoadModelCheckpoint(CKPT_PATH))
    return callbacks


def main() -> None:
    seed_everything(SEED)
    torch.set_float32_matmul_precision("medium")

    datamodule = create_datamodule()
    module = create_module()

    logger = TerminalLogger(LOGS_PATH, config=CONFIG)
    callbacks = create_callbacks(logger)

    trainer = Trainer(
        logger=logger,
        device=DEVICE,
        callbacks=callbacks,
        max_epochs=MAX_EPOCHS,
        limit_batches=LIMIT_BATCHES,
        log_every_n_steps=LOG_EVERY_N_STEPS,
    )
    trainer.fit(module, datamodule)


if __name__ == "__main__":
    main()
