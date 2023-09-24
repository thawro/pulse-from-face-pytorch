from torch import Tensor
from .base import BaseModel


class SegmentationModel(BaseModel):
    def segment(self, images: Tensor) -> tuple[Tensor, Tensor]:
        seg_out, cls_out = self.net(images)
        return seg_out, cls_out
