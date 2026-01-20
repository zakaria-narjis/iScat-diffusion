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

# %%
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src.utils.diffusion import Diffusion


class DDPMTrainer:
    def __init__(
        self,
        model,
        config,
        train_loader,
        val_loader,
        device,
        rank,
        world_size,
    ):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.config = config

        # model
        self.model = model.to(device)
        self.model = DDP(self.model, device_ids=[device])

        self.train_loader = train_loader
        self.val_loader = val_loader

        # optimizer
        opt_cfg = config["training"]["optimizer"]["parameters"]
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg["weight_decay"],
            betas=tuple(opt_cfg["betas"]),
        )

        # scheduler
        sched_cfg = config["training"]["scheduler"]
        sched_type = sched_cfg["type"]

        if sched_type == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=sched_cfg["parameters"]["mode"],
                factor=sched_cfg["parameters"]["factor"],
                patience=sched_cfg["parameters"]["patience"],
            )
        elif sched_type == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_cfg["parameters"]["T_max"],
                eta_min=sched_cfg["parameters"]["eta_min"],
            )
        else:
            raise ValueError(f"Unknown scheduler type: {sched_type}")

        # diffusion
        diff_cfg = config["diffusion"]
        self.diffusion = Diffusion(
            timesteps=diff_cfg["timesteps"],
            beta_schedule=diff_cfg["beta_schedule"],
            device=device,
        )
        self.T = self.diffusion.timesteps

        # early stopping
        es_cfg = config["training"]["early_stopping"]
        self.early_stop_enabled = es_cfg["enabled"]
        self.patience = es_cfg["patience"]
        self.criterion = es_cfg["criterion"]  # "min" or "max"

        self.best_metric = float("inf") if self.criterion == "min" else -float("inf")
        self.bad_epochs = 0

        self.output_dir = config["output"]["output_dir"]
        self.best_model_path = os.path.join(self.output_dir, "best_model.pt")

        if self.rank == 0:
            os.makedirs(self.output_dir, exist_ok=True)

    def train_epoch(self):
        self.model.train()
        total_loss = torch.tensor(0.0, device=self.device)
        num_batches = 0

        for x0, cond in self.train_loader:
            x0 = x0.to(self.device)
            cond = cond.unsqueeze(1).to(self.device)  # (B, 1, H, W)

            B = x0.size(0)
            t = torch.randint(0, self.T, (B,), device=self.device)
            noise = torch.randn_like(x0)  # (B, C, H, W)

            x_t = self.diffusion.q_sample(x0, t, noise)  # (B, C, H, W)
            pred_noise = self.model(x_t, cond, t)

            loss = nn.functional.mse_loss(pred_noise, noise)

            self.optimizer.zero_grad()
            loss.backward()
            
            # Optional: gradient clipping for stability
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()

            total_loss += loss.detach()
            num_batches += 1

        # Aggregate across all processes
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        num_batches_tensor = torch.tensor(num_batches, device=self.device)
        dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)
        
        return total_loss.item() / num_batches_tensor.item()

    @torch.no_grad()
    def valid_epoch(self):
        self.model.eval()

        total_loss = torch.tensor(0.0, device=self.device)
        total_psnr = torch.tensor(0.0, device=self.device)
        total_ssim = torch.tensor(0.0, device=self.device)
        total_samples = 0

        for x0, cond in self.val_loader:
            x0 = x0.to(self.device)
            cond = cond.unsqueeze(1).to(self.device)  # FIXED: Added unsqueeze

            B = x0.size(0)
            t = torch.randint(0, self.T, (B,), device=self.device)
            noise = torch.randn_like(x0)

            x_t = self.diffusion.q_sample(x0, t, noise)
            pred_noise = self.model(x_t, cond, t)

            loss = nn.functional.mse_loss(pred_noise, noise, reduction='sum')
            total_loss += loss
            
            # Estimate x0 from predicted noise
            alpha_bar = self.diffusion.alphas_cumprod[t].view(-1, 1, 1, 1)
            x0_pred = (x_t - torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
            x0_pred = x0_pred.clamp(-1, 1)

            # Compute metrics per sample
            for i in range(B):
                gt = x0[i].cpu().numpy().transpose(1, 2, 0)
                pr = x0_pred[i].cpu().numpy().transpose(1, 2, 0)
                
                # Handle single channel
                if gt.shape[2] == 1:
                    gt = gt.squeeze(-1)
                    pr = pr.squeeze(-1)

                psnr = peak_signal_noise_ratio(gt, pr, data_range=2.0)
                ssim_val = structural_similarity(
                    gt, pr, 
                    channel_axis=-1 if len(gt.shape) == 3 else None,
                    data_range=2.0
                )
                
                total_psnr += psnr
                total_ssim += ssim_val
                total_samples += 1

        # Aggregate across processes
        total_samples_tensor = torch.tensor(total_samples, device=self.device)
        for tensor in (total_loss, total_psnr, total_ssim, total_samples_tensor):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        n = total_samples_tensor.item()
        return (
            total_loss.item() / n,
            total_psnr.item() / n,
            total_ssim.item() / n,
        )

    def train(self):
        num_epochs = self.config["training"]["num_epochs"]

        pbar = tqdm(
            range(num_epochs),
            desc="Training",
            disable=(self.rank != 0),
        )

        for epoch in pbar:
            # Set epoch for distributed sampler
            self.train_loader.sampler.set_epoch(epoch)
            
            train_loss = self.train_epoch()
            val_loss, psnr, ssim = self.valid_epoch()

            if self.config["training"]["scheduler"]["type"] == "ReduceLROnPlateau":
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]

            if self.rank == 0:
                pbar.set_postfix(
                    {
                        "train": f"{train_loss:.4f}",
                        "val": f"{val_loss:.4f}",
                        "psnr": f"{psnr:.2f}",
                        "ssim": f"{ssim:.4f}",
                        "lr": f"{lr:.2e}",
                    }
                )

            # Check for improvement
            improved = (
                val_loss < self.best_metric
                if self.criterion == "min"
                else val_loss > self.best_metric
            )

            # Synchronize improvement decision across all ranks
            improved_tensor = torch.tensor(int(improved), device=self.device)
            dist.all_reduce(improved_tensor, op=dist.ReduceOp.MAX)

            if improved_tensor.item():
                self.best_metric = val_loss
                self.bad_epochs = 0
                if self.rank == 0:
                    torch.save(self.model.module.state_dict(), self.best_model_path)
                    print(f"\n✓ Saved best model (val_loss: {val_loss:.4f})")
            else:
                self.bad_epochs += 1

            # Early stopping check
            stop_tensor = torch.tensor(
                int(self.early_stop_enabled and self.bad_epochs >= self.patience),
                device=self.device,
            )
            dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)

            if stop_tensor.item():
                if self.rank == 0:
                    pbar.close()
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
                break
        
        if self.rank == 0:
            print(f"\nTraining complete. Best val_loss: {self.best_metric:.4f}")