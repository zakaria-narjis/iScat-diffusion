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

# train.py
import os
import re
import random
import json
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from src.datasets.dataset import iScatDataset
from src.trainers.trainer import DDPMTrainer
from src.models.unet import U_Net  # adjust import if needed
from src.test import test  # placeholder, implemented later


# -----------------------------
# Utility functions
# -----------------------------

def sanitize_filename(name):
    return re.sub(r"[^\w\-_\. ]", "_", name)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def getdatetime():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def load_config(config_path):
    config = OmegaConf.load(config_path)
    return OmegaConf.to_container(config, resolve=True)


def create_dataloaders(
    train_dataset,
    valid_dataset,
    test_dataset,
    batch_size,
    world_size,
    config,
    rank,
):
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        valid_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=config["data"]["train_dataset"]["num_workers"],
        pin_memory=True,
    )

    val_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=config["data"]["valid_dataset"]["num_workers"],
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# -----------------------------
# Main
# -----------------------------

def main():
    # -------------------------------------------------
    # 1. DDP setup
    # -------------------------------------------------
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group(backend="nccl")

    # -------------------------------------------------
    # 2. Load config
    # -------------------------------------------------
    config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    config = load_config(config_path)

    # -------------------------------------------------
    # 3. Seed
    # -------------------------------------------------
    set_random_seed(config["seed"] + rank)

    # -------------------------------------------------
    # 4. Experiment directory
    # -------------------------------------------------
    if rank == 0:
        exp_name = sanitize_filename(
            f'{config["experiment_name"]}_{getdatetime()}'
        )
        output_dir = os.path.join("experiments", exp_name)
        os.makedirs(output_dir, exist_ok=True)

        # save resolved config
        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            OmegaConf.save(config=OmegaConf.create(config), f=f.name)
    else:
        output_dir = None

    # broadcast output dir
    obj_list = [output_dir]
    dist.broadcast_object_list(obj_list, src=0)
    output_dir = obj_list[0]

    config["output"]["output_dir"] = output_dir

    # -------------------------------------------------
    # 5. Datasets
    # -------------------------------------------------
    full_dataset = iScatDataset(
        dataset_folder_path=config["data"]["dataset_folder_path"],
        image_size=config["data"]["image_size"],
        z_chunk_size=config["data"]["z_chunk_size"],
        fluo_masks_indices=config["data"]["fluo_masks_indices"],
        seg_method=config["data"]["seg_method"],
        data_type=config["data"]["data_type"],
        normalize=config["data"]["normalize"],
        multi_class=config["data"]["multi_class"],
        apply_augmentation=True,
    )

    train_size = int(len(full_dataset) * config["training"]["train_split_size"])
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    test_dataset = iScatDataset(
        dataset_folder_path=config["data"]["dataset_folder_path"],
        image_size=config["data"]["image_size"],
        z_chunk_size=config["data"]["z_chunk_size"],
        fluo_masks_indices=config["data"]["fluo_masks_indices"],
        seg_method=config["data"]["seg_method"],
        data_type=config["data"]["data_type"],
        normalize=config["data"]["normalize"],
        multi_class=config["data"]["multi_class"],
        apply_augmentation=False,
    )

    # -------------------------------------------------
    # 6. DataLoaders
    # -------------------------------------------------
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=config["training"]["batch_size"],
        world_size=world_size,
        config=config,
        rank=rank,
    )

    # -------------------------------------------------
    # 7. Model
    # -------------------------------------------------
    model = U_Net(
        img_ch=len(config["data"]["z_chunk_size"]),
        output_ch=len(config["data"]["z_chunk_size"]),
    ).to(device)

    # -------------------------------------------------
    # 8. Trainer
    # -------------------------------------------------
    trainer = DDPMTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        rank=rank,
        world_size=world_size,
    )

    # -------------------------------------------------
    # 9. Training
    # -------------------------------------------------
    trainer.train()
    dist.barrier()

    # -------------------------------------------------
    # 10. Testing
    # -------------------------------------------------
    if rank == 0:
        test(
            model=model,
            test_loader=test_loader,
            device=device,
            config=config,
            checkpoint_path=os.path.join(output_dir, "best_model.pt"),
        )

    # -------------------------------------------------
    # 11. Cleanup
    # -------------------------------------------------
    del trainer, model, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
