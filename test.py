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