import torch
from tqdm.auto import tqdm

from src.data.datamodule import DataModule
from src.logging import get_pylogger
from src.logging.loggers import BaseLogger
from src.model.utils import save_checkpoint

log = get_pylogger(__name__)

from src.model.module.base_module import BaseModule
from src.callbacks import BaseCallback, Callbacks


class Trainer:
    module: BaseModule
    datamodule: DataModule

    def __init__(
        self,
        logger: BaseLogger,
        device: torch.device,
        callbacks: list[BaseCallback],
        max_epochs: int = 100,
        limit_batches: int = -1,
    ):
        self.logger = logger
        self.device = device
        self.callbacks = Callbacks(callbacks)
        self.max_epochs = max_epochs
        self._limit_batches = limit_batches
        self.current_epoch = 0
        self.best_metrics = {}

    def train_epoch(self):
        self.module.on_train_epoch_start()
        self.callbacks.on_train_epoch_start(self)
        loop = tqdm(self.datamodule.train_dataloader(), leave=False, desc="Train")
        limit_batches = int(self._limit_batches)
        for i, batch in enumerate(loop):
            for j in range(len(batch)):
                batch[j] = batch[j].to(self.device)
            self.module.training_step(batch, i)
            limit_batches -= 1
            if limit_batches == 0:
                break
        self.module.on_train_epoch_end()
        self.callbacks.on_train_epoch_end(self)

    def val_epoch(self):
        with torch.no_grad():
            self.module.on_validation_epoch_start()
            self.callbacks.on_validation_epoch_start(self)
            loop = tqdm(self.datamodule.val_dataloader(), leave=False, desc="Val")
            limit_batches = int(self._limit_batches)
            for i, batch in enumerate(loop):
                for j in range(len(batch)):
                    batch[j] = batch[j].to(self.device)
                self.module.validation_step(batch, i)
                limit_batches -= 1
                if limit_batches == 0:
                    break
            self.module.on_validation_epoch_end()
            self.callbacks.on_validation_epoch_end(self)

    def fit(self, module: BaseModule, datamodule: DataModule):
        if self._limit_batches > 0:
            datamodule._set_shuffle(False)
        self.datamodule = datamodule
        module.model = module.model.to(self.device)
        self.module = module
        self.module.pass_attributes(self.device, self.logger, self.callbacks, datamodule)
        self.callbacks.on_fit_start(self)
        for epoch in range(self.current_epoch, self.current_epoch + self.max_epochs):
            self.current_epoch = epoch
            module.set_attributes(current_epoch=epoch)
            self.train_epoch()
            self.val_epoch()
            module.on_epoch_end()
            self.callbacks.on_epoch_end(self)
            print()

    def load_checkpoint(self, ckpt_path: str):
        log.info(f"Loading checkpoint from {ckpt_path}")
        ckpt_state = torch.load(ckpt_path)
        self.module.load_state_dict(ckpt_state["module"])
        self.datamodule.load_state_dict(ckpt_state["datamodule"])
        self.callbacks.load_state_dict(ckpt_state["callbacks"])
        self.current_epoch = ckpt_state["epoch"] + 1

    def save_checkpoint(self, ckpt_path: str):
        module_state = self.module.state_dict()
        datamodule_state = self.datamodule.state_dict()
        callbacks_state = self.callbacks.state_dict()
        ckpt_state = {
            "module": module_state,
            "datamodule": datamodule_state,
            "callbacks": callbacks_state,
            "epoch": self.current_epoch,
        }
        save_checkpoint(ckpt_state, ckpt_path)
