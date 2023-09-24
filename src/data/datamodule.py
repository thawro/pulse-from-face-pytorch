"""DataModule used to load DataLoaders"""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.logging import get_pylogger

from .dataset import BaseDataset

log = get_pylogger(__name__)


class DataModule:
    def __init__(
        self,
        train_ds: BaseDataset,
        val_ds: BaseDataset,
        test_ds: BaseDataset | None,
        batch_size: int = 16,
        num_workers: int = 8,
        pin_memory: bool = True,
        drop_last: bool = True,
    ):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self._shuffle = None
        self.total_batches = {
            "train": len(self.train_dataloader()),
            "val": len(self.val_dataloader()),
        }
        if test_ds is not None:
            self.total_batches["test"] = len(self.test_dataloader())

    def _set_shuffle(self, shuffle: bool):
        self._shuffle = shuffle

    def train_dataloader(self) -> DataLoader:
        shuffle = self._shuffle if self._shuffle is not None else True
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def val_dataloader(self) -> DataLoader:
        shuffle = self._shuffle if self._shuffle is not None else False
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def test_dataloader(self) -> DataLoader:
        shuffle = self._shuffle if self._shuffle is not None else False
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def state_dict(self) -> dict:
        return {
            "random_state": random.getstate(),
            "torch_random_state": torch.random.get_rng_state(),
            "numpy_random_state": np.random.get_state(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        random.setstate(state_dict["random_state"])
        torch.random.set_rng_state(state_dict["torch_random_state"])
        np.random.set_state(state_dict["numpy_random_state"])
