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