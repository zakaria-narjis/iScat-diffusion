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


def ddpm_sample(model, shape, cond, diffusion, device, num_steps=None):
    """
    Full denoising process from pure noise to clean image.
    
    Args:
        model: trained denoising model
        shape: tuple (B, C, H, W) for generated images
        cond: conditioning mask (B, cond_ch, H, W)
        diffusion: Diffusion object with noise schedules
        device: torch device
        num_steps: number of denoising steps (defaults to diffusion.timesteps)
    
    Returns:
        x0: denoised image (B, C, H, W)
    """
    model.eval()
    
    if num_steps is None:
        num_steps = diffusion.timesteps
    
    # start from pure noise
    x = torch.randn(shape, device=device)
    
    # denoise step by step from T to 0
    for t in reversed(range(num_steps)):
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        
        with torch.no_grad():
            # predict noise
            pred_noise = model(x, cond, t_batch)
            
            # extract relevant parameters
            alpha_t = diffusion.alphas[t]
            alpha_bar_t = diffusion.alphas_cumprod[t]
            beta_t = diffusion.betas[t]
            
            if t > 0:
                alpha_bar_t_prev = diffusion.alphas_cumprod[t - 1]
            else:
                alpha_bar_t_prev = torch.tensor(1.0, device=device)
            
            # compute mean of p(x_{t-1} | x_t)
            # using DDPM reverse process formula
            mean = (1 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise
            )
            
            if t > 0:
                # add noise (except for final step)
                # variance for reverse process
                variance = beta_t * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(variance) * noise
            else:
                x = mean
    
    return x


@torch.no_grad()
def test(model, test_loader, device, config, checkpoint_path):
    """
    Test trained DDPM model on test dataset.
    
    Args:
        model: model architecture (not loaded yet)
        test_loader: DataLoader for test dataset
        device: torch device
        config: configuration dictionary
        checkpoint_path: path to trained model weights
    """
    
    # Get DDP info if available
    is_distributed = dist.is_initialized()
    if is_distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    
    # Load model weights
    if rank == 0:
        print(f"Loading model from {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        if rank == 0:
            print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    # Load state dict
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Initialize diffusion
    from src.utils.diffusion import Diffusion
    diff_cfg = config["diffusion"]
    diffusion = Diffusion(
        timesteps=diff_cfg["timesteps"],
        beta_schedule=diff_cfg["beta_schedule"],
        device=device,
    )
    
    # Metrics storage
    total_mse = torch.tensor(0.0, device=device)
    total_psnr = torch.tensor(0.0, device=device)
    total_ssim = torch.tensor(0.0, device=device)
    total_samples = 0
    
    # Storage for visualization examples
    examples = []
    max_examples = config.get("test", {}).get("num_examples", 5)
    
    if rank == 0:
        pbar = tqdm(test_loader, desc="Testing")
    else:
        pbar = test_loader
    
    for batch_idx, (x0_true, cond) in enumerate(pbar):
        x0_true = x0_true.to(device)
        cond = cond.unsqueeze(1).to(device)  # (B, 1, H, W)
        
        B, C, H, W = x0_true.shape
        
        # Generate samples using full denoising process
        x0_pred = ddpm_sample(
            model=model,
            shape=(B, C, H, W),
            cond=cond,
            diffusion=diffusion,
            device=device,
            num_steps=diff_cfg.get("sampling_steps", diffusion.timesteps)
        )
        
        # Clamp predictions to valid range
        x0_pred = x0_pred.clamp(-1, 1)
        
        # Compute metrics
        mse = nn.functional.mse_loss(x0_pred, x0_true, reduction='sum')
        total_mse += mse
        
        # Compute PSNR and SSIM per sample
        for i in range(B):
            gt = x0_true[i].cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
            pr = x0_pred[i].cpu().numpy().transpose(1, 2, 0)
            
            # Handle single channel case
            if gt.shape[2] == 1:
                gt = gt.squeeze(-1)
                pr = pr.squeeze(-1)
            
            psnr = peak_signal_noise_ratio(gt, pr, data_range=2.0)
            ssim = structural_similarity(
                gt, pr, 
                channel_axis=-1 if len(gt.shape) == 3 else None,
                data_range=2.0
            )
            
            total_psnr += psnr
            total_ssim += ssim
            total_samples += 1
            
            # Save examples for visualization
            if rank == 0 and len(examples) < max_examples:
                examples.append({
                    'ground_truth': x0_true[i].cpu(),
                    'predicted': x0_pred[i].cpu(),
                    'condition': cond[i].cpu(),
                })
    
    # Aggregate metrics across GPUs
    if is_distributed:
        total_samples_tensor = torch.tensor(total_samples, device=device)
        dist.all_reduce(total_mse, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_psnr, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_ssim, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
        total_samples = total_samples_tensor.item()
    
    # Compute averages
    avg_mse = total_mse.item() / total_samples
    avg_psnr = total_psnr.item() / total_samples
    avg_ssim = total_ssim.item() / total_samples
    
    # Save results (only rank 0)
    if rank == 0:
        output_dir = config["output"]["output_dir"]
        
        # Save metrics as JSON
        metrics = {
            "mse": avg_mse,
            "psnr": avg_psnr,
            "ssim": avg_ssim,
            "num_samples": total_samples,
        }
        
        metrics_path = os.path.join(output_dir, "test_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"\n{'='*50}")
        print(f"Test Results:")
        print(f"{'='*50}")
        print(f"MSE:  {avg_mse:.6f}")
        print(f"PSNR: {avg_psnr:.2f} dB")
        print(f"SSIM: {avg_ssim:.4f}")
        print(f"{'='*50}\n")
        
        # Save visualizations
        save_visualizations(examples, output_dir)
        
        print(f"Results saved to {output_dir}")


def save_visualizations(examples, output_dir):
    """
    Save visualization plots for test examples.
    
    Args:
        examples: list of dicts with 'ground_truth', 'predicted', 'condition'
        output_dir: directory to save plots
    """
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    for idx, example in enumerate(examples):
        gt = example['ground_truth'].numpy().transpose(1, 2, 0)  # (H, W, C)
        pred = example['predicted'].numpy().transpose(1, 2, 0)
        cond = example['condition'].numpy().squeeze()  # (H, W)
        
        # Handle single channel
        if gt.shape[2] == 1:
            gt = gt.squeeze(-1)
            pred = pred.squeeze(-1)
        
        # Normalize to [0, 1] for visualization
        gt_vis = (gt + 1) / 2
        pred_vis = (pred + 1) / 2
        cond_vis = (cond + 1) / 2 if cond.min() < 0 else cond
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Ground truth
        if len(gt_vis.shape) == 2:
            axes[0].imshow(gt_vis, cmap='gray', vmin=0, vmax=1)
        else:
            axes[0].imshow(gt_vis)
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')
        
        # Condition (mask)
        axes[1].imshow(cond_vis, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Condition Mask')
        axes[1].axis('off')
        
        # Prediction
        if len(pred_vis.shape) == 2:
            axes[2].imshow(pred_vis, cmap='gray', vmin=0, vmax=1)
        else:
            axes[2].imshow(pred_vis)
        axes[2].set_title('Generated')
        axes[2].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(vis_dir, f"example_{idx+1}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {len(examples)} visualization examples to {vis_dir}")
