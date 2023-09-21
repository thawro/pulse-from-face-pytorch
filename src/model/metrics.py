from torch import Tensor


class BaseMetric:
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> dict[str, float]:
        raise NotImplementedError()


class BaseMetrics:
    def __init__(self, metrics: list[BaseMetric]):
        self.metrics = metrics

    def calculate_metrics(self, y_pred: Tensor, y_true: Tensor) -> dict[str, float]:
        metrics = {}
        for metric in self.metrics:
            metrics.update(metric(y_pred, y_true))
        return metrics
