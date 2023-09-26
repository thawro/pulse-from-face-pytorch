"""Functions used to load model with trained weights"""
from src.model.model.segmentation import SegmentationModel
from src.model.architectures.segmentation.psp_net import PSPNet
import torch


def load_model(num_classes: int, input_size: tuple[int, ...], ckpt_path: str, device: str):
    model = SegmentationModel(
        net=PSPNet(num_classes=num_classes, cls_dropout=0.5, backbone="resnet101"),
        input_size=input_size,
        input_names=["images"],
        output_names=["masks", "class_probs"],
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt = ckpt["module"]["model"]
    model.load_state_dict(ckpt)
    return model.to(device)
