from torch import Tensor
from .base import BaseModel
from src.model.architectures.segmentation.base import SegmentationNet


class SegmentationModel(BaseModel):
    net: SegmentationNet

    def segment(self, images: Tensor) -> Tensor:
        seg_out, cls_out = self.net(images)
        return seg_out
