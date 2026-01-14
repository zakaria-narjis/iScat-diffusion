# src/ddpm_trainer.py

"""
Trainer class for Conditional Diffusion (DDPM) on ISCAT Microscopy images.

## Responsibilities
- Manage training and validation loops for a conditional denoising diffusion probabilistic model.
- Support distributed training with PyTorch DDP.
- Handle learning rate scheduling, optimizer, checkpointing, and early stopping.
- Aggregate and log metrics and loss across ranks.

## Key Components

### Attributes
- model: the conditional U-Net or other diffusion model
- config: dict containing training, saving, and diffusion hyperparameters
- train_loader / val_loader: PyTorch DataLoader objects
- device: GPU device for this rank
- optimizer: AdamW optimizer for model parameters
- scheduler: learning rate scheduler
- output_dir / best_model_path: for saving checkpoints
- early stopping params: enabled, patience, monitored metric, mode
- diffusion-specific params: timesteps T, noise schedule, etc.

### Methods
1. __init__(...):
    - Initialize model, optimizer, scheduler
    - Initialize early stopping
    - Wrap model in DDP
    - Set up diffusion-specific hyperparameters from config

2. train_epoch():
    - Iterate over train_loader
    - For each batch:
        1. Sample random timesteps t ∈ [1, T]
        2. Add noise to clean images: x_t = sqrt(alpha_t) * x_0 + sqrt(1-alpha_t) * epsilon
        3. Forward pass through model: predict epsilon_theta(x_t, t, mask)
        4. Compute L2 loss between predicted and true noise
        5. Backprop and optimizer step
    - Aggregate loss across distributed ranks
    - Return average training loss

3. valid_epoch():
    - Set model to eval mode
    - Iterate over val_loader
    - For each batch:
        1. Sample timesteps t 
        2. Forward pass and compute L2 loss
    - Aggregate metrics across ranks
    - Return average validation loss (and optionally metrics like PSNR, SSIM)

4. train():
    - High-level loop over epochs
    - Call train_epoch() and valid_epoch()
    - Log metrics and learning rate
    - Check for best model and save
    - Handle early stopping across distributed ranks

5. check_best_metric(val_loss, optional_metrics):
    - Compare current metrics to previous best
    - Return True if current is the best

6. save_best_model():
    - Save state_dict to disk (optionally only on rank 0)
"""