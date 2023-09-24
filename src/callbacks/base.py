from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.model.module.trainer import Trainer

from abc import abstractmethod
from src.logging import get_pylogger

log = get_pylogger(__name__)


class BaseCallback:
    @abstractmethod
    def on_fit_start(self, trainer: Trainer):
        pass

    @abstractmethod
    def on_train_epoch_start(self, trainer: Trainer):
        pass

    @abstractmethod
    def on_train_epoch_end(self, trainer: Trainer):
        pass

    @abstractmethod
    def on_validation_epoch_start(self, trainer: Trainer):
        pass

    @abstractmethod
    def on_validation_epoch_end(self, trainer: Trainer):
        pass

    @abstractmethod
    def on_epoch_end(self, trainer: Trainer):
        pass

    @abstractmethod
    def log(self, trainer: Trainer):
        pass

    def state_dict(self) -> dict:
        return {}

    @abstractmethod
    def load_state_dict(self, state_dict: dict):
        pass


class Callbacks:
    def __init__(self, callbacks: list[BaseCallback]):
        self.callbacks = callbacks

    def on_fit_start(self, trainer: Trainer):
        for callback in self.callbacks:
            callback.on_fit_start(trainer)

    def on_train_epoch_start(self, trainer: Trainer):
        for callback in self.callbacks:
            callback.on_train_epoch_start(trainer)

    def on_train_epoch_end(self, trainer: Trainer):
        for callback in self.callbacks:
            callback.on_train_epoch_end(trainer)

    def on_validation_epoch_start(self, trainer: Trainer):
        for callback in self.callbacks:
            callback.on_validation_epoch_start(trainer)

    def on_validation_epoch_end(self, trainer: Trainer):
        for callback in self.callbacks:
            callback.on_validation_epoch_end(trainer)

    def on_epoch_end(self, trainer: Trainer):
        for callback in self.callbacks:
            callback.on_epoch_end(trainer)

    def log(self, trainer: Trainer):
        for callback in self.callbacks:
            callback.log(trainer)

    @abstractmethod
    def state_dict(self):
        state_dict = {}
        for callback in self.callbacks:
            state_dict.update(callback.state_dict())
        return state_dict

    @abstractmethod
    def load_state_dict(self, state_dict: dict):
        for callback in self.callbacks:
            callback.load_state_dict(state_dict)
