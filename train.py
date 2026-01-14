"""
DDPM Conditional Diffusion Training Script

Purpose:
- Set up distributed training (DDP) for conditional DDPM on ISCAT microscopy images.
- Load dataset, model, trainer, and handle experiment logging.
- Supports reproducibility (seed fixing) and distributed metric aggregation.

Main Steps:

0. Seed Initialization
- Set Python, NumPy, PyTorch, and CUDA seeds for reproducibility.
- Ensure deterministic behavior via cudnn flags.

1. Load and Parse Configuration
- Use YAML / OmegaConf for config.
- Resolve dataset-dependent parameters (image channels, input size, mask channels, etc.).
- Optional: log config to TensorBoard or save config.yaml in output folder.

2. DDP Setup
- Get rank, local rank, world size from environment variables.
- Initialize torch.distributed process group (NCCL backend).
- Set device to local GPU.

3. Output Directory / Logging
- Rank 0 creates experiment folder with timestamp.
- Save config.yaml for reproducibility.
- Broadcast output_dir to all ranks.

4. Dataset & DataLoaders
- Load datasets (train / val / test) with iScatDataset or custom Dataset.
- Split indices for train/val/test.
- Apply augmentations / normalization.
- Use `DistributedSampler` for DDP.
- Wrap DataLoader with batch size and num_workers.

5. Model Initialization
- Instantiate conditional U-Net or variant (AttU_Net, R2U_Net, etc.).
- If using DDPM:
    - Input channels = image + mask channels
    - Output channels = predicted noise (usually 1)
- Move model to device and wrap with DDP.
- Optional: sync batchnorm if multiple GPUs.

6. Trainer Setup
- Initialize DDPMTrainer with:
    - model
    - train_loader, val_loader
    - optimizer, scheduler
    - device, rank, world_size
    - optional : logging / writer (TensorBoard) 

7. Training Loop
- Call trainer.train() which internally:
    - Iterates over epochs
    - Calls train_epoch() and valid_epoch()
    - Aggregates metrics across ranks
    - Handles early stopping and checkpointing
    - Logs metrics

8. Post-Training / Testing
- After training, synchronize all ranks.
- Call `test_model()`:
    - Load trained weights
    - Run inference on test set
    - Save predicted images for a few examples
    - Save masks, generated images, and metrics (PSNR, SSIM, etc.)
- Save results as JSON and optionally visualize with plots.

9. Cleanup
- Delete model, datasets, trainer to free GPU memory.
- Destroy process group (dist.destroy_process_group()).
"""