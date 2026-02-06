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
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


class Diffusion:
    def __init__(self, timesteps=1000, beta_schedule="linear", device="cpu", 
                 betas=None, alphas=None, alphas_cumprod=None, cosine_smooth=0.008):
        """
        Initialize Diffusion object.
        
        Args:
            timesteps: number of diffusion timesteps
            beta_schedule: "linear" or "cosine"
            device: torch device
            betas: optional pre-computed betas (for loading from checkpoint)
            alphas: optional pre-computed alphas
            alphas_cumprod: optional pre-computed cumulative product of alphas
        """
        self.timesteps = timesteps
        self.device = device

        # Allow loading pre-computed schedules (for exact reproduction)
        if betas is not None:
            self.betas = betas.to(device)
        else:
            # Choose beta schedule
            if beta_schedule == "linear":
                betas = linear_beta_schedule(timesteps)
            elif beta_schedule == "cosine":
                betas = cosine_beta_schedule(timesteps, s=cosine_smooth)
            else:
                raise ValueError(f"Unknown beta schedule: {beta_schedule}")
            self.betas = betas.to(device)

        # α_t = 1 - β_t
        if alphas is not None:
            self.alphas = alphas.to(device)
        else:
            self.alphas = 1.0 - self.betas

        # ᾱ_t = product_{i=1}^t α_i
        if alphas_cumprod is not None:
            self.alphas_cumprod = alphas_cumprod.to(device)
        else:
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Useful precomputed terms
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # For reverse process
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # Posterior variance
        alphas_cumprod_prev = torch.cat([
            torch.ones(1, device=device, dtype=self.alphas_cumprod.dtype),
            self.alphas_cumprod[:-1]
        ])
        
        self.posterior_variance = (
            self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)
    def get_noise_schedule(self):
        """
        Return the noise schedule parameters for saving/loading.
        Useful for ensuring test-time diffusion uses exact same schedule as training.
        """
        return {
            'betas': self.betas,
            'alphas': self.alphas,
            'alphas_cumprod': self.alphas_cumprod,
        }

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
    def p_sample(self, model, x_t, t, cond, style_vals=None):
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
        pred_noise = model(x_t, cond, t_batch, style_vals)
        
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
    def p_sample_loop(self, model, shape, cond, style_vals=None, num_steps=None):
        """
        Full reverse diffusion: generate samples from noise using DDPM
        
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
            x = self.p_sample(model, x, t, cond, style_vals=style_vals)
        
        return x
    
    @torch.no_grad()
    def ddim_sample(self, model, shape, cond, style_vals=None, num_steps=50, eta=0.0):
        """
        DDIM sampling for faster generation.
        
        Args:
            model: denoising model
            shape: tuple (B, C, H, W)
            cond: conditioning mask (B, cond_ch, H, W)
            num_steps: number of sampling steps (much less than training steps)
            eta: controls stochasticity (0 = deterministic, 1 = DDPM-like)
        
        Returns:
            x_0: generated clean image
        """
        # Select subset of timesteps uniformly
        # We want to sample from T-1 down to 0. 

        # Actually, there is a dillema about test time sampling with ddim or few steps sampling method. 
        # The thing is that the scheduling we are using so far does not probably guarantee that the last step xT is pure noise (zero SNR).
        # Which means that the model is not trained to handle pure noise at the first step (T-1) of the reverse process.
        # And since in inference (test time) we will be starting from pure noise, this leads to a distribution shift between training and inference, which can cause poor sample quality.
        # This paper https://arxiv.org/pdf/2305.08891 talk about this issue and propose to  Rescale Schedule to Zero Terminal SNR they also gave the algorithm to do that.

        # In the same paper they argue sample steps selection  (discretization) also makes a difference. (Table2). 
        # So far we tried method 1 and method 4. And method 1 seems to be better.

        # Method 1: Uniform step selection (original DDIM , PNDM approach)
        # --- start old way ---
        # step_size = self.timesteps // num_steps
        # noise_timesteps_list = list(range(0, self.timesteps, step_size))
        # noise_timesteps_list = list(reversed(noise_timesteps_list))  # Start from highest timestep
        # --- end old way ---

        # more simple an clear
        noise_timesteps_list = torch.arange(0, self.timesteps , self.timesteps // num_steps) 
        noise_timesteps_list = torch.flip(noise_timesteps_list, dims=[0]).long().tolist()  # Start from highest timestep

        # Method 2: iDDPM
        # noise_timesteps_list = torch.round(torch.linspace(0, self.timesteps-1, num_steps)).long().tolist()
        # noise_timesteps_list = torch.flip(torch.tensor(noise_timesteps_list), dims=[0]).tolist()
        
        # Method 3: DPM trailing
        # noise_timesteps_list = torch.round(torch.arange(self.timesteps-1, 0, -self.timesteps / num_steps)).long().tolist()

        # Method 4: chatgpt
        # noise_timesteps_list = torch.linspace(
        #     self.timesteps-1, 0, num_steps, dtype=torch.long
        # ).tolist()

        # Start from pure noise
        x = torch.randn(shape, device=self.device)
        
        for i, t in enumerate(noise_timesteps_list):
            t_batch = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            
            # Predict noise
            pred_noise = model(x, cond, t_batch, style_vals)
            
            alpha_bar_t = self.alphas_cumprod[t]
            
            # Predict x0 from x_t and predicted noise
            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
            
            # Check if this is the last step
            if i < len(noise_timesteps_list) - 1:
                t_prev = noise_timesteps_list[i + 1]
                alpha_bar_t_prev = self.alphas_cumprod[t_prev]
                
                # Compute variance
                sigma_t = eta * torch.sqrt(
                    (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev)
                )
                
                # Direction pointing to x_t
                dir_xt = torch.sqrt(1 - alpha_bar_t_prev - sigma_t**2) * pred_noise
                
                # Random noise
                noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
               
                # Update: x_{t-1} = sqrt(alpha_bar_{t-1}) * pred_x0 + dir_xt + sigma_t * noise
                x = torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt + sigma_t * noise
            else:
                # Final step: return predicted x0
                x = pred_x0
        
        return x