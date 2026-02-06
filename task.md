# ISCAT Microscopy Conditional Diffusion Project

## Approach

We can start with **pixel-space conditional DDPM** for ISCAT microscopy images with **segmentation masks as conditions**.  Then maybe try Latent diffusion models LDM like stable diffusion or different conditioning approaches. 

- **Input:** noisy image \(x_t\) + segmentation mask  
- **Target:** predicted noise \(\epsilon\) at timestep \(t\)  
- **Goal:** train a model to denoise images conditioned on segmentation maps, generating realistic images from scratch noise.

---

## Training Phase

- For each sample:
  1. Sample a random timestep \(t \in [1, T]\)  
  2. Add noise to the clean image \(x_0\) using the closed-form formula:
     \[
     x_t = \sqrt{\alpha_t} x_0 + \sqrt{1 - \alpha_t} \epsilon
     \]
  3. Feed \(x_t\) + segmentation mask + timestep \(t\) into the network  
  4. Compute loss (\(L_2\) between predicted and true noise)  
  5. Backpropagate and update network weights  

> Each forward/backward pass only sees **one noisy image per sample**, but across batches and epochs the model learns all timesteps.

---

## Inference Phase

- Start from pure Gaussian noise \(x_T \sim \mathcal{N}(0, I)\)  
- Iteratively denoise from \(T \to 0\) using the trained network:
  1. At step \(t\), predict \(\hat{\epsilon} = \epsilon_\theta(x_t, t, c)\)  
  2. Compute \(x_{t-1}\) from \(x_t\) and \(\hat{\epsilon}\)  
- Repeat until \(x_0\) is generated  
- Optional: use **classifier-free guidance** to enforce strong conditioning (https://arxiv.org/pdf/2207.12598)

---

## Project Tasks

### 1. Dataset
- `dataset.py`  
  - PyTorch Dataset class  
  - Loads ISCAT images and corresponding segmentation masks  
  - Applies augmentations (flips, rotations, intensity scaling)  
  - Returns `(image, mask)` tensors

### 2. Model
- `model.py`  
  - Conditional U-Net  
  - Input: `x_t` + segmentation mask + timestep embedding(to add time embed either: residual addition or FiLM https://arxiv.org/pdf/1709.07871) 
  - Output: predicted noise \(\epsilon_\theta\)  
  - Optional attention at low-resolution layers  

### 3. Diffusion Utilities
- Functions to compute:
  - Forward noising (closed-form formula for \(x_t\))  
  - Reverse step helpers for inference  
  - Noise schedules (linear, cosine)  

### 4. Trainer Class
- `trainer.py`  
- `class Trainer:` with signature:  
  ```python
  class Trainer:
      def __init__(self, model, config, train_loader, val_loader, rank, world_size):
          ...

### Some resource when can look at
  - [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
  - [Denoising diffusion probabilistic models DPPM paper](https://arxiv.org/pdf/2006.11239)
  - [Latent diffusion model LDM](https://arxiv.org/abs/2112.10752)
  - [Denoising Diffusion Implicit Models DDIM](https://arxiv.org/abs/2010.02502) (More sample efficient than ddpm)
  - [batchnorm vs GroupNorm](https://apxml.com/courses/advanced-diffusion-architectures/chapter-2-advanced-unet-architectures/unet-normalization-techniques)
  - [DPM Solver++](https://arxiv.org/abs/2211.01095) ( other inference sampling method)


# Further ideas to try:
  - Use different samplers : (DPM Solver ++, ...) something that could be better than DDIM
  - use v-prediction rather than noise prediction (see what I wrote in diffusion.py line 199 in Diffusion.ddim_sample() function) 
  - force zero final step SNR  (https://arxiv.org/pdf/2305.08891) (see what I wrote in diffusion.py line 199 in Diffusion.ddim_sample() function) 
  - Use better weighting for the segmentation mask (condition)
  - The style condition (basically contrast condition) seems to give us control on how we want the contrast of the image to look like maybe we can enhance it better than than using just the mean and std of the whole image (C,H,W) we can compute the stats for each channel. Maybe there is something better than the mean and std.
  - Can we add more objectives to the training, so far we are using just MSE (noise, predicted_noise)? Can we emphasize more on the axial contrast variation of generated particles (since is the most important part for segmenation).
  - classifier free diffusion guidance (https://arxiv.org/abs/2207.12598) can Help to force the model to adhere more to the segmentation mask. Since sometimes we see ghost particles (particles that doesnt exist in the condition/segmentation mask but somehow produced by the model).


### Usefull Commands
  - tmux new -t ddpm (start a tmux session if its not already running) if its already running use tmux attach 
  - conda activate iscat
  - CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py 2>&1 | tee training.log
