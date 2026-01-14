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

        # ᾱ_t = product_{i=1}^t α_i
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # useful precomputed terms
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x0, t, noise):
        """
        Forward diffusion process:
        x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise

        x0: clean image (B, C, H, W)
        t: timestep indices (B,)
        noise: standard Gaussian noise (same shape as x0)
        """
        # gather correct alpha_bar for each batch element
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        return sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
