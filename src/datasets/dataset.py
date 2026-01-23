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
        one_hot_encode: bool = False,
    ):
        """
        PyTorch Dataset for microscopy data stored in an HDF5 file.

        Args:
            hdf5_path (str): Path to the HDF5 file.
            classes (list): Classes to include in the mask. 0: 80nm, 1: 300nm, 2: 600nm.
            apply_augmentation (bool): Whether to apply random flips/rotations.
            normalize (str): Normalization method ('minmax', 'zscore', 'global_zscore', or None).
            indices (list): Optional list of indices to subset the dataset.
            multi_class (bool): If True, output multiclass masks instead of binary.
            chunk_size (int): Number of frames to average for each image.
            num_z_slices (int | str | None): Limit number of z-slices, or "all"/None for all.
            one_hot_encode (bool): If True, return one-hot encoded masks of shape (C, H, W).
        """
        self.hdf5_path = hdf5_path
        self.classes = classes
        self.apply_augmentation = apply_augmentation
        self.normalize = normalize
        self.multi_class = multi_class
        self.chunk_size = chunk_size
        self.num_z_slices = num_z_slices
        self.one_hot_encode = one_hot_encode

        # Global Statistics provided by user
        self.GLOBAL_MEAN = 7846.924091
        self.GLOBAL_STD = 1514.261195

        # Validate num_z_slices
        if isinstance(self.num_z_slices, str) and self.num_z_slices.lower() == "all":
            self.num_z_slices = None
        elif isinstance(self.num_z_slices, int):
            if not (0 < self.num_z_slices < 202):
                raise ValueError("num_z_slices must be a positive integer < 202.")
        elif self.num_z_slices is not None:
            raise ValueError("num_z_slices must be 'all', None, or int.")

        with h5py.File(hdf5_path, "r") as f:
            self.image_dataset_size = f["image_patches"].shape[0]

        self.indices = indices if indices is not None else range(self.image_dataset_size)
        self._file: Optional[h5py.File] = None

    def _get_file(self) -> h5py.File:
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
        # --- NORMALIZATION (Moved before averaging) ---
        if self.normalize == "global_zscore":
            image = (image - self.GLOBAL_MEAN) / (self.GLOBAL_STD + 1e-8)
            # image = torch.clamp(torch.from_numpy(image), -3.0, 3.0).numpy()
        elif self.normalize == "minmax":
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        elif self.normalize == "zscore":
            image = (image - image.mean()) / (image.std() + 1e-8)
        elif self.normalize is not None:
            raise ValueError("normalize must be 'minmax', 'zscore', 'global_zscore', or None.")

        # Average frames
        image = Utils.extract_averaged_frames(image, num_frames=self.chunk_size) # (chunk_size, H, W)

        # Convert to tensors
        image = torch.from_numpy(image)
        masks = torch.from_numpy(masks)

        # Process masks
        if len(self.classes) == 1:
            mask = masks[self.classes[0]]
        else:
            mask = torch.zeros_like(masks[0], dtype=torch.uint8)
            if self.multi_class:
                for i, cls in enumerate(self.classes, start=1):
                    mask[masks[cls] > 0] = i
            else:
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
        
        mask = mask.float()
        
        if self.one_hot_encode:
            num_classes = len(self.classes) + 1 if self.multi_class else 2
            mask_one_hot = torch.nn.functional.one_hot(mask.long(), num_classes=num_classes)
            mask = mask_one_hot.permute(2, 0, 1).float()
        
        return image, mask