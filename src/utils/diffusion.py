import torch
import math


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2):
    """
    Linear schedule from DDPM paper.
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule from:
    https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


class Diffusion:
    def __init__(self, timesteps=1000, beta_schedule="linear", device="cpu"):
        self.timesteps = timesteps
        self.device = device

        # choose beta schedule
        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.betas = betas.to(device)

        # α_t = 1 - β_t
        self.alphas = 1.0 - self.betas

        # ᾱ_t = product_{i=1}^t α_i
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # useful precomputed terms
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # For reverse process
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # FIXED: Create tensor on the correct device
        alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0], device=device), 
            self.alphas_cumprod[:-1]
        ])
        
        self.posterior_variance = (
            self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x0, t, noise):
        """
        Forward diffusion process:
        x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise

        Args:
            x0: clean image (B, C, H, W)
            t: timestep indices (B,)
            noise: standard Gaussian noise (same shape as x0)
        
        Returns:
            x_t: noisy image at timestep t
        """
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        return sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
    
    @torch.no_grad()
    def p_sample(self, model, x_t, t, cond):
        """
        Single reverse diffusion step: sample x_{t-1} from x_t
        
        Args:
            model: denoising model
            x_t: noisy image at timestep t (B, C, H, W)
            t: current timestep (int)
            cond: conditioning mask (B, cond_ch, H, W)
        
        Returns:
            x_{t-1}: denoised image at timestep t-1
        """
        B = x_t.shape[0]
        t_batch = torch.full((B,), t, device=self.device, dtype=torch.long)
        
        # Predict noise
        pred_noise = model(x_t, cond, t_batch)
        
        # Get parameters for this timestep
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alphas_cumprod[t]
        beta_t = self.betas[t]
        
        # Compute mean of reverse distribution
        mean = (1 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise
        )
        
        if t > 0:
            # Sample from N(mean, variance) if not final step
            variance = self.posterior_variance[t]
            noise = torch.randn_like(x_t)
            x_t_minus_1 = mean + torch.sqrt(variance) * noise
        else:
            # No noise at t=0
            x_t_minus_1 = mean
        
        return x_t_minus_1
    
    @torch.no_grad()
    def p_sample_loop(self, model, shape, cond, num_steps=None):
        """
        Full reverse diffusion: generate samples from noise
        
        Args:
            model: denoising model
            shape: tuple (B, C, H, W)
            cond: conditioning mask (B, cond_ch, H, W)
            num_steps: number of denoising steps (default: self.timesteps)
        
        Returns:
            x_0: generated clean image
        """
        if num_steps is None:
            num_steps = self.timesteps
        
        # Start from pure noise
        x = torch.randn(shape, device=self.device)
        
        # Iteratively denoise
        for t in reversed(range(num_steps)):
            x = self.p_sample(model, x, t, cond)
        
        return x
    
    @torch.no_grad()
    def ddim_sample(self, model, shape, cond, num_steps=50, eta=0.0):
        """
        DDIM sampling for faster generation (optional enhancement)
        
        Args:
            model: denoising model
            shape: tuple (B, C, H, W)
            cond: conditioning mask
            num_steps: number of sampling steps (much less than training steps)
            eta: controls stochasticity (0 = deterministic, 1 = DDPM)
        
        Returns:
            x_0: generated clean image
        """
        # Select subset of timesteps
        step_size = self.timesteps // num_steps
        timesteps = torch.arange(0, self.timesteps, step_size, device=self.device)
        timesteps = torch.flip(timesteps, [0])
        
        x = torch.randn(shape, device=self.device)
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            
            # Predict noise
            pred_noise = model(x, cond, t_batch)
            
            alpha_bar_t = self.alphas_cumprod[t]
            
            # Predict x0
            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
            pred_x0 = pred_x0.clamp(-1, 1)
            
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_bar_t_prev = self.alphas_cumprod[t_prev]
                
                # Compute variance
                sigma_t = eta * torch.sqrt(
                    (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev)
                )
                
                # Direction pointing to x_t
                dir_xt = torch.sqrt(1 - alpha_bar_t_prev - sigma_t**2) * pred_noise
                
                # Random noise
                noise = torch.randn_like(x) if eta > 0 else 0
                
                # Update
                x = torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt + sigma_t * noise
            else:
                x = pred_x0
        
        return x