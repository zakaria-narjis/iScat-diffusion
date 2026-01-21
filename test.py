"""
DDPM Conditional Diffusion Testing Script / Function

Purpose:
- Load trained DDPM model weights.
- Run inference on test dataset using noisy input + mask.
- Save quantitative metrics and example visualizations.

Main Steps:

1. Load Model
- Args:
    - model_class / type (U-Net, AttU_Net, etc.)
    - model_path (trained weights)
    - device (GPU / CPU)
- Load state_dict into model and move to device.

2. Load Test Dataset
- Args:
    - hdf5_path or dataset folder
    - indices for test set
    - normalization, mask info, multi-class flags
- Wrap in DataLoader for batch inference.
- Optional: DistributedSampler if running DDP.

3. Run Inference
- For each batch:
    - If DDPM, sample random noise x_T
    - Iteratively denoise using model conditioned on mask
    - Store predicted images

4. Compute Metrics
- Compute L2 loss / PSNR / SSIM per sample or batch.
- Aggregate metrics over test set.

5. Visualization
- For a few selected indices:
    - Save original image, segmentation mask, predicted/generated image
    - Use matplotlib or batch plotting functions
    - Save figures in experiment output folder

6. Save Results
- Save metrics dict as JSON for reproducibility
- Optionally save numpy arrays of predicted images

7. DDP Handling
- Aggregate metrics across ranks if using multiple GPUs
- Only rank 0 saves plots and JSON files

8. Return
- Return metrics dictionary and optionally generated images
"""

"""
DPM Conditional Diffusion Testing Script

Runs inference with a trained DDPM-like model conditioned on a mask, computes
L2 / PSNR / SSIM metrics, and saves visualizations and JSON results.

This script is intentionally generic; plug in your own model class,
beta schedule, and dataset structure as needed.
"""

import argparse
import json
import math
import os
from typing import Dict, Tuple, Optional, List

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# ---------------------------
# 1. Model loading utilities
# ---------------------------


def get_model_class(name: str):
    """
    Map a model name string to an actual class.

    Replace this stub with your own imports / registry:
        if name == "UNet": from my_models import UNet; return UNet
    """
    raise NotImplementedError(
        f"Model class resolver not implemented. "
        f"Implement get_model_class('{name}') to return your model class."
    )


def load_model(
    model_type: str,
    model_path: str,
    device: torch.device,
    model_kwargs: Optional[Dict] = None,
) -> torch.nn.Module:
    """
    Load model weights and move to device.
    """
    if model_kwargs is None:
        model_kwargs = {}

    model_cls = get_model_class(model_type)
    model = model_cls(**model_kwargs)
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ---------------------------
# 2. Dataset & DataLoader
# ---------------------------


class HDF5SegDataset(Dataset):
    """
    Example HDF5 dataset for image+mask (and optional target).

    Expected layout in the HDF5 file (adjust as needed):
        /images      : float32 or uint8, shape (N, C, H, W)
        /masks       : float32 or uint8, shape (N, 1, H, W) or (N, H, W)
        /targets     : (optional) same shape as images or desired label maps

    Normalization is applied to images and targets (if provided).
    """

    def __init__(
        self,
        hdf5_path: str,
        indices: Optional[List[int]] = None,
        normalize: bool = True,
        multi_class: bool = False,
        has_target: bool = True,
    ):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.normalize = normalize
        self.multi_class = multi_class
        self.has_target = has_target

        self._h5: Optional[h5py.File] = None

        with h5py.File(self.hdf5_path, "r") as f:
            n_samples = f["images"].shape[0]
        if indices is None:
            self.indices = list(range(n_samples))
        else:
            self.indices = indices

    def _require_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        self._require_h5()
        real_idx = self.indices[idx]

        img = self._h5["images"][real_idx]  # (C,H,W) or (H,W)
        mask = self._h5["masks"][real_idx]

        if self.has_target:
            target = self._h5["targets"][real_idx]
        else:
            target = img  # fall back to image if no dedicated target

        img = torch.from_numpy(np.asarray(img)).float()
        mask = torch.from_numpy(np.asarray(mask)).float()
        target = torch.from_numpy(np.asarray(target)).float()

        # Ensure channel dimension
        if img.ndim == 2:
            img = img.unsqueeze(0)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if target.ndim == 2:
            target = target.unsqueeze(0)

        if self.normalize:
            # Simple [0,1] normalization for uint8 images; customize as needed.
            img = img / 255.0 if img.max() > 1.0 else img
            target = target / 255.0 if target.max() > 1.0 else target

        return {
            "image": img,
            "mask": mask,
            "target": target,
            "index": real_idx,
        }

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None


def build_dataloader(
    hdf5_path: str,
    indices: Optional[List[int]],
    batch_size: int,
    num_workers: int,
    normalize: bool,
    multi_class: bool,
    has_target: bool,
    distributed: bool,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    dataset = HDF5SegDataset(
        hdf5_path=hdf5_path,
        indices=indices,
        normalize=normalize,
        multi_class=multi_class,
        has_target=has_target,
    )

    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=False)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader, sampler


# ---------------------------
# 3. DDPM sampling (reverse)
# ---------------------------


class DDPMDiffusion:
    """
    Minimal DDPM reverse process helper.

    Assumes a known beta schedule; plug in your training schedule.
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: torch.device = torch.device("cpu"),
    ):
        self.timesteps = timesteps
        self.device = device

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alpha_bars = alpha_bars.to(device)

    def p_sample(
        self,
        model: torch.nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single reverse diffusion step x_{t-1} <- x_t.

        model is assumed to predict noise epsilon given (x_t, t, cond).
        Adapt this if your model has a different signature.
        """
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1, 1)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)

        # Predict noise
        eps_theta = model(x_t, t, cond)

        # Standard DDPM sampling formula
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - beta_t / torch.sqrt(1.0 - alpha_bar_t) * eps_theta
        )

        if (t == 0).all():
            return mean

        noise = torch.randn_like(x_t)
        sigma_t = torch.sqrt(beta_t)
        return mean + sigma_t * noise

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        shape: Tuple[int, int, int, int],
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full reverse process starting from x_T ~ N(0, I).
        """
        batch_size = shape[0]
        x_t = torch.randn(shape, device=self.device)

        for step in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), step, device=self.device, dtype=torch.long)
            x_t = self.p_sample(model, x_t, t, cond)

        return x_t.clamp(0.0, 1.0)


# ---------------------------
# 4. Metrics: L2 / PSNR / SSIM
# ---------------------------


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target, reduction="mean")


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio in dB for tensors in [0, max_val].
    """
    mse = F.mse_loss(pred, target, reduction="mean").item()
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0,
    window_size: int = 11,
    channel: int = 1,
) -> float:
    """
    Simple SSIM implementation on torch tensors (N,C,H,W) in [0,max_val].
    Not optimized; use a library like piq or pytorch-msssim for production. [web:12][web:9]
    """
    # Conversion to single image if batch > 1
    if pred.ndim != 4 or target.ndim != 4:
        raise ValueError("SSIM expects 4D tensors (N,C,H,W).")

    # Compute per-image SSIM and average
    ssim_vals: List[float] = []
    for i in range(pred.size(0)):
        x = pred[i : i + 1]
        y = target[i : i + 1]

        # Gaussian window (separable)
        def gaussian_window(ws: int, sigma: float) -> torch.Tensor:
            coords = torch.arange(ws).float() - ws // 2
            g = torch.exp(-(coords**2) / (2 * sigma**2))
            g = g / g.sum()
            window_1d = g.unsqueeze(0)
            window_2d = (window_1d.t() @ window_1d).unsqueeze(0).unsqueeze(0)
            return window_2d

        window = gaussian_window(window_size, 1.5).to(x.device)
        window = window.expand(channel, 1, window_size, window_size)

        mu_x = F.conv2d(x, window, padding=window_size // 2, groups=channel)
        mu_y = F.conv2d(y, window, padding=window_size // 2, groups=channel)

        mu_x2 = mu_x.pow(2)
        mu_y2 = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_x2 = (
            F.conv2d(x * x, window, padding=window_size // 2, groups=channel) - mu_x2
        )
        sigma_y2 = (
            F.conv2d(y * y, window, padding=window_size // 2, groups=channel) - mu_y2
        )
        sigma_xy = (
            F.conv2d(x * y, window, padding=window_size // 2, groups=channel) - mu_xy
        )

        c1 = (0.01 * max_val) ** 2
        c2 = (0.03 * max_val) ** 2

        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
            (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
        )
        ssim_vals.append(ssim_map.mean().item())

    return float(np.mean(ssim_vals))


# ---------------------------
# 5. Visualization utilities
# ---------------------------


def save_example_figure(
    output_dir: str,
    index: int,
    image: torch.Tensor,
    mask: torch.Tensor,
    pred: torch.Tensor,
):
    """
    Save original, mask, and prediction for a single sample.

    image, mask, pred: (C,H,W) or (1,H,W) tensors in [0,1].
    """
    os.makedirs(output_dir, exist_ok=True)

    img = image.detach().cpu().numpy()
    msk = mask.detach().cpu().numpy()
    prd = pred.detach().cpu().numpy()

    # Squeeze channels for visualization
    if img.ndim == 3 and img.shape[0] in (1, 3):
        if img.shape[0] == 1:
            img_vis = img[0]
            cmap_img = "gray"
        else:
            img_vis = np.transpose(img, (1, 2, 0))
            cmap_img = None
    else:
        img_vis = img.squeeze()
        cmap_img = "gray"

    msk_vis = msk.squeeze()
    prd_vis = prd.squeeze()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_vis, cmap=cmap_img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(msk_vis, cmap="gray")
    axes[1].set_title("Mask")
    axes[1].axis("off")

    axes[2].imshow(prd_vis, cmap="gray" if prd_vis.ndim == 2 else None)
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    fig.tight_layout()
    save_path = os.path.join(output_dir, f"example_{index:05d}.png")
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# ---------------------------
# 6. DDP helpers
# ---------------------------


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    if not is_distributed():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not is_distributed():
        return 0
    return dist.get_rank()


def ddp_all_reduce_tensor(t: torch.Tensor) -> torch.Tensor:
    """
    All-reduce a tensor (sum) across ranks and return the result.
    """
    if not is_distributed():
        return t
    t = t.clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


# ---------------------------
# 7. Main testing routine
# ---------------------------


def run_test(args) -> Tuple[Dict, Optional[np.ndarray]]:
    device = torch.device(args.device)

    # DDP init (if launched with torchrun / mpirun etc.)
    if args.dist and dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)

    rank = get_rank()
    world_size = get_world_size()
    is_main = rank == 0

    # 1. Load model
    model = load_model(
        model_type=args.model_type,
        model_path=args.model_path,
        device=device,
        model_kwargs={},  # plug in architecture kwargs if needed
    )

    # 2. Build test dataloader
    indices = None
    if args.test_indices is not None and len(args.test_indices) > 0:
        indices = [int(i) for i in args.test_indices.split(",")]

    loader, sampler = build_dataloader(
        hdf5_path=args.hdf5_path,
        indices=indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=not args.no_normalize,
        multi_class=args.multi_class,
        has_target=not args.no_target,
        distributed=args.dist,
    )

    # 3. DDPM diffusion helper
    diffusion = DDPMDiffusion(
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
    )

    # Metric accumulators
    total_l2 = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n_samples = 0

    # Optional: store predictions for returning/saving
    all_preds: List[np.ndarray] = [] if args.save_predictions else []

    # For visualization: gather a fixed set of indices
    vis_indices = set()
    if args.vis_indices is not None and len(args.vis_indices) > 0:
        vis_indices = set(int(i) for i in args.vis_indices.split(","))

    os.makedirs(args.output_dir, exist_ok=True)

    # 3. Run inference
    model.eval()
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)  # (B,C,H,W)
            mask = batch["mask"].to(device)
            target = batch["target"].to(device)
            idxs = batch["index"]

            bsz = image.size(0)
            cond = mask  # conditioning on mask

            # Sample from DDPM starting at random noise x_T
            pred = diffusion.sample(
                model=model,
                shape=image.shape,
                cond=cond,
            )

            # 4. Compute metrics per batch
            # L2
            l2_val = mse_loss(pred, target).item()
            # PSNR / SSIM
            psnr_val = psnr(pred, target, max_val=1.0)
            ssim_val = ssim(pred, target, max_val=1.0, channel=image.size(1))

            total_l2 += l2_val * bsz
            total_psnr += psnr_val * bsz
            total_ssim += ssim_val * bsz
            n_samples += bsz

            # Save predictions in memory if requested
            if args.save_predictions:
                all_preds.append(pred.detach().cpu().numpy())

            # 5. Visualization for selected indices (only on main rank)
            if is_main and len(vis_indices) > 0:
                for i in range(bsz):
                    idx_int = int(idxs[i].item())
                    if idx_int in vis_indices:
                        save_example_figure(
                            output_dir=os.path.join(args.output_dir, "figures"),
                            index=idx_int,
                            image=image[i],
                            mask=mask[i],
                            pred=pred[i],
                        )

    # 7. DDP: aggregate metrics across ranks
    if is_distributed():
        total_l2_t = torch.tensor([total_l2], device=device)
        total_psnr_t = torch.tensor([total_psnr], device=device)
        total_ssim_t = torch.tensor([total_ssim], device=device)
        n_samples_t = torch.tensor([n_samples], device=device, dtype=torch.float32)

        total_l2_t = ddp_all_reduce_tensor(total_l2_t)
        total_psnr_t = ddp_all_reduce_tensor(total_psnr_t)
        total_ssim_t = ddp_all_reduce_tensor(total_ssim_t)
        n_samples_t = ddp_all_reduce_tensor(n_samples_t)

        total_l2 = total_l2_t.item()
        total_psnr = total_psnr_t.item()
        total_ssim = total_ssim_t.item()
        n_samples = int(n_samples_t.item() / world_size)  # average per-sample

    # 4. Aggregate metrics
    if n_samples > 0:
        avg_l2 = total_l2 / n_samples
        avg_psnr = total_psnr / n_samples
        avg_ssim = total_ssim / n_samples
    else:
        avg_l2 = avg_psnr = avg_ssim = float("nan")

    metrics = {
        "num_samples": n_samples,
        "l2": avg_l2,
        "psnr": avg_psnr,
        "ssim": avg_ssim,
        "timesteps": args.timesteps,
        "beta_start": args.beta_start,
        "beta_end": args.beta_end,
    }

    # 6. Save results on main rank
    if is_main:
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        if args.save_predictions and len(all_preds) > 0:
            preds_arr = np.concatenate(all_preds, axis=0)
            np.save(os.path.join(args.output_dir, "predictions.npy"), preds_arr)

    # Convert preds to numpy for return (main only)
    preds_np = None
    if args.save_predictions and len(all_preds) > 0 and is_main:
        preds_np = np.concatenate(all_preds, axis=0)

    return metrics, preds_np


# ---------------------------
# 8. CLI
# ---------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="DPM Conditional Diffusion Testing Script"
    )

    # Model
    parser.add_argument("--model_type", type=str, required=True,
                        help="Model class name (e.g., UNet, AttU_Net).")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model weights (checkpoint).")

    # Data
    parser.add_argument("--hdf5_path", type=str, required=True,
                        help="Path to test dataset HDF5 file.")
    parser.add_argument("--test_indices", type=str, default=None,
                        help="Comma-separated list of test indices (optional).")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for inference.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of DataLoader workers.")
    parser.add_argument("--no_normalize", action="store_true",
                        help="Disable [0,1] normalization of inputs/targets.")
    parser.add_argument("--multi_class", action="store_true",
                        help="Flag if dataset is multi-class (not used directly here).")
    parser.add_argument("--no_target", action="store_true",
                        help="If set, use image as target (no dedicated target dataset).")

    # DDPM config
    parser.add_argument("--timesteps", type=int, default=1000,
                        help="Number of DDPM timesteps.")
    parser.add_argument("--beta_start", type=float, default=1e-4,
                        help="Starting beta for linear schedule.")
    parser.add_argument("--beta_end", type=float, default=2e-2,
                        help="Ending beta for linear schedule.")

    # Device / DDP
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--dist", action="store_true",
                        help="Enable Distributed Data Parallel (DDP) testing.")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="Local rank for DDP (set automatically by torchrun).")

    # Output / visualization
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save metrics and visualizations.")
    parser.add_argument("--vis_indices", type=str, default=None,
                        help="Comma-separated list of indices to visualize.")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Save all predicted images as a NumPy file.")

    return parser.parse_args()


def main():
    args = parse_args()
    metrics, _ = run_test(args)
    # Print metrics summary (rank 0)
    if get_rank() == 0:
        print("Test metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
