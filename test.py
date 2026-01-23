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
- Save 3 subplots (original, mask, generated) for selected examples

7. DDP Handling
- Aggregate metrics across ranks if using multiple GPUs
- Only rank 0 saves plots and JSON files

8. Return
- Return metrics dictionary and optionally generated images
"""


import os
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from src.utils.diffusion import Diffusion


@torch.no_grad()
def test(model, test_loader, device, config, checkpoint_path):
    """
    Test trained DDPM model on test dataset.

    Handles multiple normalization methods and one-hot encoded masks.
    Supports DDP and aggregates metrics across GPUs.
    """
    # Get DDP info if available
    is_distributed = dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    world_size = dist.get_world_size() if is_distributed else 1

    # Load model weights
    if rank == 0:
        print(f"Loading model from {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        if rank == 0:
            print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Initialize diffusion with training parameters
    diff_cfg = config["diffusion"]
    diffusion = Diffusion(
        timesteps=diff_cfg["timesteps"],
        beta_schedule=diff_cfg["beta_schedule"],
        device=device,
    )

    # Get test configuration
    test_cfg = config["test"]
    sampling_method = test_cfg["sampling_method"]
    sampling_steps = test_cfg["sampling_steps"]
    ddim_eta = test_cfg["ddim_eta"]
    max_examples = test_cfg["num_examples"]
    
    # Get normalization method from dataset config
    normalize_method = config["data"]["normalize"]

    if rank == 0:
        print(f"\nTest Configuration:")
        print(f"  Sampling method: {sampling_method}")
        print(f"  Sampling steps: {sampling_steps}")
        if sampling_method == "ddim":
            print(f"  DDIM eta: {ddim_eta}")
        print(f"  Training timesteps: {diff_cfg['timesteps']}")
        print(f"  Normalization: {normalize_method}\n")

    # Metrics storage
    total_mse = torch.tensor(0.0, device=device)
    total_psnr = torch.tensor(0.0, device=device)
    total_ssim = torch.tensor(0.0, device=device)
    total_pixels = torch.tensor(0, device=device)

    examples = []

    pbar = tqdm(test_loader, desc="Testing") if rank == 0 else test_loader

    for batch_idx, (x0_true, cond) in enumerate(pbar):
        x0_true = x0_true.to(device)
        cond = cond.to(device)
        # Handle both single-channel and multi-channel (one-hot) masks
        if cond.dim() == 3:  # (B, H, W) - binary mask
            cond = cond.unsqueeze(1).to(device)  # -> (B, 1, H, W)            
        B, C, H, W = x0_true.shape

        # Generate samples
        if sampling_method == "ddim":
            x0_pred = diffusion.ddim_sample(
                model=model, shape=(B, C, H, W), cond=cond,
                num_steps=sampling_steps, eta=ddim_eta
            )
        elif sampling_method == "ddpm":
            x0_pred = diffusion.p_sample_loop(
                model=model, shape=(B, C, H, W), cond=cond,
                num_steps=sampling_steps
            )
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")

        # MSE per batch (sum over all pixels)
        mse = nn.functional.mse_loss(x0_pred, x0_true, reduction='sum')
        total_mse += mse
        total_pixels += B * C * H * W

        # Compute PSNR / SSIM per sample
        for i in range(B):
            gt = x0_true[i].cpu().numpy()  # (C, H, W)
            pr = x0_pred[i].cpu().numpy()
            gt = gt.transpose(1, 2, 0)  # (H, W, C)
            pr = pr.transpose(1, 2, 0)
            if gt.shape[2] == 1:
                gt = gt.squeeze(-1)
                pr = pr.squeeze(-1)
            
            # Normalize to [0, 1] for metrics based on normalization method
            if normalize_method == "global_minmax":
                # Images are in [-1, 1], map to [0, 1]
                gt_norm = (gt + 1.0) / 2.0
                pr_norm = (pr + 1.0) / 2.0
                # Clip to handle potential numerical issues
                gt_norm = np.clip(gt_norm, 0, 1)
                pr_norm = np.clip(pr_norm, 0, 1)
            else:
                # Adaptive normalization for other methods
                min_val = min(gt.min(), pr.min())
                max_val = max(gt.max(), pr.max())
                gt_norm = (gt - min_val) / (max_val - min_val + 1e-8)
                pr_norm = (pr - min_val) / (max_val - min_val + 1e-8)

            psnr = peak_signal_noise_ratio(gt_norm, pr_norm, data_range=1.0)
            if gt.ndim == 3:
                ssim = structural_similarity(gt_norm, pr_norm, channel_axis=-1, data_range=1.0)
            else:
                ssim = structural_similarity(gt_norm, pr_norm, data_range=1.0)
            total_psnr += psnr
            total_ssim += ssim

            # Save examples
            if rank == 0 and len(examples) < max_examples:
                examples.append({
                    'ground_truth': x0_true[i].cpu(),
                    'predicted': x0_pred[i].cpu(),
                    'condition': cond[i].cpu()
                })

    # Aggregate metrics across GPUs
    if is_distributed:
        dist.all_reduce(total_mse, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_psnr, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_ssim, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_pixels, op=dist.ReduceOp.SUM)

    # Compute averages
    avg_mse = (total_mse / total_pixels).item()  # per-pixel MSE
    avg_psnr = (total_psnr / total_pixels).item()
    avg_ssim = (total_ssim / total_pixels).item()

    # Save results
    if rank == 0:
        output_dir = config["output"]["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        metrics = {
            "mse": avg_mse,
            "psnr": avg_psnr,
            "ssim": avg_ssim,
            "num_samples": total_pixels.item() // (C*H*W),
            "sampling_method": sampling_method,
            "sampling_steps": sampling_steps,
            "training_timesteps": diff_cfg["timesteps"],
            "ddim_eta": ddim_eta if sampling_method=="ddim" else None,
            "normalization": normalize_method
        }

        metrics_path = os.path.join(output_dir, "test_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        print(f"\n{'='*50}")
        print("Test Results:")
        print(f"{'='*50}")
        print(f"Sampling: {sampling_method} ({sampling_steps} steps)")
        print(f"Normalization: {normalize_method}")
        print(f"MSE:  {avg_mse:.6f}")
        print(f"PSNR: {avg_psnr:.2f} dB")
        print(f"SSIM: {avg_ssim:.4f}")
        print(f"{'='*50}\n")

        # Save visualizations
        save_visualizations(examples, output_dir, normalize_method)
        print(f"Results saved to {output_dir}")


def save_visualizations(examples, output_dir, normalize_method="minmax"):
    """
    Save visualization examples with proper handling of multi-channel masks.
    """
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    for idx, example in enumerate(examples):
        gt = example['ground_truth'].numpy()
        pred = example['predicted'].numpy()
        cond = example['condition'].numpy()

        # Select middle frame for visualization
        mid_frame = gt.shape[0] // 2 if gt.shape[0] > 1 else 0
        gt_vis = gt[mid_frame] if gt.shape[0] > 1 else gt[0]
        pred_vis = pred[mid_frame] if pred.shape[0] > 1 else pred[0]
        
        # Handle multi-channel masks (one-hot encoded)
        if cond.shape[0] > 1:  # Multi-channel mask
            # Convert one-hot to class labels for visualization
            cond_vis = cond.argmax(0)  # (H, W) with class indices
            cond_cmap = 'tab10'  # Categorical colormap
            cond_vmin, cond_vmax = 0, cond.shape[0] - 1
        else:  # Single-channel mask
            cond_vis = cond.squeeze()
            cond_cmap = 'gray'
            cond_vmin, cond_vmax = 0, 1

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(gt_vis, cmap='gray', vmin=gt_vis.min(), vmax=gt_vis.max())
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')

        im1 = axes[1].imshow(cond_vis, cmap=cond_cmap, vmin=cond_vmin, vmax=cond_vmax)
        axes[1].set_title('Condition Mask')
        axes[1].axis('off')
        # Add colorbar for multi-class masks
        if cond.shape[0] > 1:
            plt.colorbar(im1, ax=axes[1], label='Class')

        axes[2].imshow(pred_vis, cmap='gray', vmin=pred_vis.min(), vmax=pred_vis.max())
        axes[2].set_title('Generated')
        axes[2].axis('off')

        plt.tight_layout()
        save_path = os.path.join(vis_dir, f"example_{idx+1}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Saved {len(examples)} visualization examples to {vis_dir}")