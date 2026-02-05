import torch
import os
from src.models.model_contrast import U_Net
from src.utils.diffusion import Diffusion

def load_ddpm_model(checkpoint_path, config, device="cuda"):
    """
    Loads the trained DDPM U-Net model.

    Args:
        checkpoint_path (str): Path to the .pt file.
        config (dict): The configuration dictionary.
        device (str or torch.device): 'cuda' or 'cpu'.

    Returns:
        model (torch.nn.Module): The loaded model in eval mode.
    """
    device = torch.device(device)
    
    # Extract model parameters directly from config (no defaults)
    img_ch = config["data"]["z_chunk_size"]
    mask_ch = config["data"]["mask_channels"]
    use_contrast = config["model"]["contrast_conditioning"]["enabled"]
    
    print(f"Initializing U_Net with img_ch={img_ch}, cond_ch={mask_ch}, contrast={use_contrast}")

    # Initialize model architecture
    model = U_Net(
        img_ch=img_ch,
        cond_ch=mask_ch,
        output_ch=img_ch,
        use_contrast_cond=use_contrast
    ).to(device)

    # Load weights
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    print(f"Loading weights from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.eval()
    return model


@torch.no_grad()
def run_inference(
    model, 
    noise_shape, 
    condition, 
    config, 
    device="cuda", 
    style_vals=None, 
    sampling_method=None, 
    sampling_steps=None
):
    """
    Runs diffusion inference to generate images.
    
    Strictly uses config keys if arguments are not provided (no .get defaults).
    Args:
        model (torch.nn.Module): The trained DDPM U-Net model.
        noise_shape (tuple): Shape of the input noise tensor for eg (B, C, H, W).
        condition (torch.Tensor): Conditioning mask tensor of shape (B, cond_ch, H, W)
        config (dict): Configuration dictionary.
        device (str or torch.device): 'cuda' or 'cpu'.
        style_vals (torch.Tensor, optional): Tensor of shape (B, 2) for contrast conditioning.
        sampling_method (str, optional): 'ddim' or 'ddpm'.
        sampling_steps (int, optional): Number of sampling steps.
    """
    device = torch.device(device)
    model.to(device)
    
    # Ensure inputs are on correct device
    cond = condition.to(device)
    
    # Handle mask dimensions: if (B, H, W) -> (B, 1, H, W)
    if cond.dim() == 3:
        cond = cond.unsqueeze(1)

    # Setup Diffusion Object
    diff_cfg = config["diffusion"]
    diffusion = Diffusion(
        timesteps=diff_cfg["timesteps"],
        beta_schedule=diff_cfg["beta_schedule"],
        cosine_smooth=diff_cfg["cosine_smooth"],
        device=device,
    )

    # Determine Sampling Settings
    # If arguments are None, force read from config keys
    if sampling_method is None:
        sampling_method = config["test"]["sampling_method"]
    
    if sampling_steps is None:
        sampling_steps = config["test"]["sampling_steps"]
    
    # Access ddim_eta directly from config (will raise KeyError if missing)
    ddim_eta = config["test"]["ddim_eta"]

    # Handle Contrast/Style Conditioning
    use_contrast = config["model"]["contrast_conditioning"]["enabled"]
    if use_contrast:
        if style_vals is None:
            raise ValueError(
                "Model was trained with contrast conditioning enabled. "
                "You must provide 'style_vals' (shape [B, 2]) containing target mean and std."
            )
        style_vals = style_vals.to(device)
    else:
        style_vals = None

    # Run Sampling
    B, C, H, W = noise_shape
    
    if sampling_method == "ddim":
        x_pred = diffusion.ddim_sample(
            model=model, 
            shape=(B, C, H, W), 
            cond=cond,
            style_vals=style_vals,
            num_steps=sampling_steps, 
            eta=ddim_eta,
        )
    elif sampling_method == "ddpm":
        # Passing x_T to p_sample_loop assuming your implementation supports it.
        # If your Diffusion.p_sample_loop does not take x_T/noise, 
        # remove the 'noise=x_T' argument below.
        x_pred = diffusion.p_sample_loop(
            model=model, 
            shape=(B, C, H, W), 
            cond=cond,
            style_vals=style_vals,
            num_steps=sampling_steps
        )
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")

    return x_pred