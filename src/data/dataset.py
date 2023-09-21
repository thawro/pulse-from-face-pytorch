"""Dataset classes"""

import torchvision.datasets
import albumentations as A
from pathlib import Path


class BaseDataset(torchvision.datasets.VisionDataset):
    root: Path

    def __init__(
        self,
        root: str,
        split: str = "test",
        transform: A.Compose | None = None,
        target_transform: A.Compose | None = None,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = split
        self.root = Path(root)
