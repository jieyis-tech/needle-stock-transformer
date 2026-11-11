from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
import struct

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        with gzip.open(image_filename, "rb") as image_file:
            magic, count, rows, cols = struct.unpack(">IIII", image_file.read(16))
            if magic != 2051:
                raise ValueError("Invalid MNIST image magic number")
            images = np.frombuffer(image_file.read(), dtype=np.uint8)
        with gzip.open(label_filename, "rb") as label_file:
            label_magic, label_count = struct.unpack(">II", label_file.read(8))
            if label_magic != 2049 or count != label_count:
                raise ValueError("Invalid MNIST labels")
            labels = np.frombuffer(label_file.read(), dtype=np.uint8)
        if images.size != count * rows * cols or labels.size != count:
            raise ValueError("Invalid MNIST data length")
        self.images = images.reshape(count, rows, cols, 1).astype(np.float32) / np.float32(255.0)
        self.labels = labels.copy()
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        images = self.images[index]
        labels = self.labels[index]
        if self.transforms is not None:
            if images.ndim == 3:
                images = self.apply_transforms(images)
            else:
                images = np.stack([self.apply_transforms(image) for image in images])
        return images, labels
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.labels.shape[0]
        ### END YOUR SOLUTION
