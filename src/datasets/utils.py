import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
from typing import Union, List, Tuple
import torch.nn.functional as F
import h5py

class Utils:

    @staticmethod
    def compute_mean_slices(z_stack, num_chunks):
        """
        Divides the z_stack into `num_chunks` parts along the z-axis and computes the mean
        image for each part.

        Parameters:
            z_stack (numpy.ndarray): The input 3D numpy array with shape (200, X, X).
            num_chunks (int): Number of chunks to divide the z-axis into.

        Returns:
            list: A list of numpy arrays, each representing the mean image of a chunk.
        """
        # Ensure the z_stack is a 3D array
        if z_stack.ndim != 3:
            raise ValueError("Input z_stack must be a 3D numpy array.")

        # Get the number of slices along the z-axis
        num_slices = z_stack.shape[0]

        # Compute the size of each chunk
        chunk_size = num_slices // num_chunks

        # Initialize a list to store the mean images
        mean_images = []

        for i in range(num_chunks):
            start_idx = i * chunk_size
            # For the last chunk, include all remaining slices
            end_idx = (
                (i + 1) * chunk_size if i != num_chunks - 1 else num_slices
            )

            # Compute the mean along the z-axis for the current chunk
            mean_image = np.mean(z_stack[start_idx:end_idx], axis=0)
            mean_images.append(mean_image.astype(np.uint16))

        return mean_images

    @staticmethod
    def load_np_masks(
        target_paths: List[Tuple[str, str, str]],
        fluo_masks_indices,
        seg_method: str = "comdet",
    ):
        """
        Load the masks corresponding to the fluorescence images.
        """
        all_masks = []
        all_masks_paths = []
        if seg_method == "comdet":
            mask_suffix = "_mask.npy"
        elif seg_method == "kmeans":
            mask_suffix = "_mask_kmeans.npy"
        else:
            raise ValueError(f"Invalid segmentation method: {seg_method}")
        for target_path in target_paths:
            masks_path = []
            for fluo_mask_idx in fluo_masks_indices:
                mask_path = target_path[fluo_mask_idx].replace(
                    ".tif", mask_suffix
                )
                if os.path.exists(mask_path):
                    masks_path.append(mask_path)
                else:
                    raise FileNotFoundError(f"Mask file not found: {mask_path}")
            all_masks_paths.append(masks_path)
        for masks_path in all_masks_paths:
            masks = []
            for mask_path in masks_path:
                mask = np.load(mask_path)
                masks.append(mask)
            combined_mask = np.zeros_like(masks[0])
            for class_index, mask in enumerate(masks, start=1):
                combined_mask[mask == 1] = class_index
            # mask= sum(masks)
            # mask[mask>1]=1
            # all_masks.append(mask)
            all_masks.append(combined_mask)
        all_masks = np.concatenate([all_masks], axis=0)
        return all_masks

    @staticmethod
    def visualize_labeled_canvas(mask: np.ndarray):
        # Plot the labeled canvas
        plt.figure(figsize=(10, 10))
        plt.title("Labeled Canvas")
        plt.imshow(mask, cmap="viridis")
        plt.colorbar(label="Labels (0: Background, 1: Particle)")
        plt.axis("off")
        plt.show()

    @staticmethod
    def calculate_class_weights_from_masks(masks: torch.Tensor) -> torch.Tensor:
        """
        Calculate class weights for a segmentation task with any number of classes.

        Args:
            masks (torch.Tensor): Tensor of shape (N, H, W), where N is the number of samples,
                                and each value in the masks represents a class (0 to C-1).

        Returns:
            torch.Tensor: Class weights of shape (C,), where C is the number of classes.
        """
        masks = torch.Tensor(masks)
        num_classes = int(
            masks.max().item() + 1
        )  # Assume classes are from 0 to C-1
        class_counts = torch.zeros(
            num_classes, dtype=torch.float, device=masks.device
        )
        flattened_masks = masks.reshape(-1)
        unique, counts = torch.unique(flattened_masks, return_counts=True)
        for label, count in zip(unique, counts):
            class_counts[
                int(label)
            ] += count  # Accumulate pixel counts for each class
        total_pixels = class_counts.sum()
        class_weights = total_pixels / (num_classes * class_counts)
        class_weights = class_weights / class_weights.sum()
        return class_weights

    @staticmethod
    def z_score_normalize(
        images: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        eps: float = 1e-8,
    ):
        normalized_images = (images - mean) / (std + eps)
        return normalized_images

    @staticmethod
    def shift_segmentation_masks(masks, shift_x=1, shift_y=1):
        """
        Shift segmentation masks in a batch to the bottom-right by specified pixels.

        Args:
            masks (torch.Tensor): A tensor of shape (B, H, W) or (B, C, H, W) representing a batch of segmentation masks.
            shift_x (int): Number of pixels to shift to the right.
            shift_y (int): Number of pixels to shift down.

        Returns:
            torch.Tensor: The shifted masks tensor with the same shape as input.
        """
        # Ensure shift_x and shift_y are non-negative
        if shift_x < 0 or shift_y < 0:
            raise ValueError("shift_x and shift_y must be non-negative.")

        # Determine padding amounts
        pad_left = shift_x
        pad_right = 0
        pad_top = shift_y
        pad_bottom = 0

        # Check if the input has a channel dimension (B, C, H, W)
        has_channels = masks.ndim == 4

        if has_channels:
            # Apply padding and slicing while keeping the batch and channel dimensions
            padded_masks = F.pad(
                masks,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=0,
            )
            shifted_masks = padded_masks[
                :,
                :,
                : -shift_y if shift_y > 0 else None,
                : -shift_x if shift_x > 0 else None,
            ]
        else:
            # Apply padding and slicing for (B, H, W) shape
            padded_masks = F.pad(
                masks,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=0,
            )
            shifted_masks = padded_masks[
                :,
                : -shift_y if shift_y > 0 else None,
                : -shift_x if shift_x > 0 else None,
            ]

        return shifted_masks

    @staticmethod
    def load_masks_from_hdf5(hdf5_path, indices=None):
        """
        Load mask patches from an HDF5 file.

        Args:
            hdf5_path (str): Path to the HDF5 file.
            indices (list or None): Optional list of indices to load specific masks.
                                    If None, all masks are loaded.

        Returns:
            np.ndarray: Loaded mask patches (subset if indices are specified).
        """
        with h5py.File(hdf5_path, "r") as f:
            mask_patches = f["mask_patches"]

            # If indices are provided, select only those masks
            if indices is not None:
                masks = mask_patches[indices]
            else:
                masks = mask_patches[:]

        return masks

    @staticmethod
    def extract_averaged_frames(image, num_frames=12):
        """
        Extracts the average of evenly spaced frame chunks from a 3D image.

        Parameters:
        image (numpy.ndarray): 3D image with shape (Z, H, W)
        num_frames (int): Number of frames to extract

        Returns:
        numpy.ndarray: 3D array of extracted averaged frames with shape (num_frames, H, W)
        """
        z_len = image.shape[0]
        chunk_size = z_len // num_frames

        averaged_frames = []
        for i in range(num_frames):
            start = i * chunk_size
            end = start + chunk_size
            chunk = image[start:end]
            averaged_frames.append(np.mean(chunk, axis=0))

        return np.array(averaged_frames)
