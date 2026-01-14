from torchvision.transforms.v2 import functional as TF
import random
from torch.utils.data import Dataset
import numpy as np
import torch
from src.datasets.utils import Utils
import h5py
import cv2
from typing import List, Optional, Union, Sequence

class iScatDataset(Dataset):
    def __init__(
        self,
        hdf5_path: str,
        classes: Sequence[int] = (0, 1, 2),
        apply_augmentation: bool = False,
        normalize: str = "minmax",
        indices: Optional[Sequence[int]] = None,
        multi_class: bool = False,
        chunk_size: int = 32,
        num_z_slices: Optional[Union[int, str]] = "all",
    ):
        """
        PyTorch Dataset for microscopy data stored in an HDF5 file.

        Args:
            hdf5_path (str): Path to the HDF5 file.
            classes (list): Classes to include in the mask.
            apply_augmentation (bool): Whether to apply random flips/rotations.
            normalize (str): Normalization method ('minmax', 'zscore', or None).
            indices (list): Optional list of indices to subset the dataset.
            multi_class (bool): If True, output multiclass masks instead of binary.
            chunk_size (int): Number of frames to average for each image.
            num_z_slices (int | str | None): Limit number of z-slices, or "all"/None for all.
        """
        self.hdf5_path = hdf5_path
        self.classes = classes
        self.apply_augmentation = apply_augmentation
        self.normalize = normalize
        self.multi_class = multi_class
        self.chunk_size = chunk_size
        self.num_z_slices = num_z_slices

        # Validate num_z_slices
        if isinstance(self.num_z_slices, str) and self.num_z_slices.lower() == "all":
            self.num_z_slices = None
        elif isinstance(self.num_z_slices, int):
            if not (0 < self.num_z_slices < 202):
                raise ValueError("num_z_slices must be a positive integer < 202.")
        elif self.num_z_slices is not None:
            raise ValueError("num_z_slices must be 'all', None, or int.")

        # Get dataset size (open once just for metadata)
        with h5py.File(hdf5_path, "r") as f:
            self.image_dataset_size = f["image_patches"].shape[0]

        self.indices = indices if indices is not None else range(self.image_dataset_size)

        # Worker-specific HDF5 file handle (lazy init)
        self._file: Optional[h5py.File] = None

    def _get_file(self) -> h5py.File:
        """Open HDF5 file lazily per worker."""
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r")
        return self._file

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        idx = self.indices[idx]
        f = self._get_file()

        # Load data
        image = f["image_patches"][idx].astype(np.float32)  # (Z, H, W)
        masks = f["mask_patches"][idx].astype(np.uint8)     # (C, H, W)

        # Z-slice filtering
        if self.num_z_slices is not None:
            image = image[: self.num_z_slices]

        # Average frames
        image = Utils.extract_averaged_frames(image, num_frames=self.chunk_size) # (chunk_size, H, W)

        # Convert to tensors
        image = torch.from_numpy(image) # (chunk_size, H, W)
        masks = torch.from_numpy(masks) # (C, H, W)

        # Normalize
        if self.normalize == "minmax":
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        elif self.normalize == "zscore":
            mean, std = image.mean(), image.std() + 1e-8
            image = (image - mean) / std
        elif self.normalize is None:
            pass
        else:
            raise ValueError("normalize must be 'minmax', 'zscore', or None.")

        # Process masks
        if len(self.classes) == 1:
            mask = masks[self.classes[0]]
        else:
            if self.multi_class:
                mask = torch.zeros_like(masks[0], dtype=torch.uint8)
                for i, cls in enumerate(self.classes, start=1):
                    mask[masks[cls] > 0] = i
            else:
                mask = torch.zeros_like(masks[0], dtype=torch.uint8)
                for cls in self.classes:
                    mask |= masks[cls]
                mask[mask > 1] = 1

        # Augmentation
        if self.apply_augmentation:
            if random.random() > 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            if random.random() > 0.5:
                image, mask = TF.vflip(image), TF.vflip(mask)
            if random.random() > 0.5:
                angle = random.choice([90, -90])
                image, mask = TF.rotate(image, angle), TF.rotate(mask, angle)

        return image, mask #(chunk_size, H, W) mask shape is  (H,W) where 0 is background 1 is particle if multiclass is True (1 is 80nm 2 is 300nm ...)