"""Implementation of loss Base and weighted loss classes"""

from torch import Tensor
from torch.nn.modules.loss import _Loss


class WeightedLoss(_Loss):
    def __init__(self, criterion: _Loss, weight: float):
        super().__init__()
        self.criterion = criterion
        self.weight = weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        loss = self.criterion(pred, target)
        return self.weight * loss
