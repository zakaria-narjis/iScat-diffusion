"""
Conditional U-Net for DDPM (ISCAT Microscopy)

## Purpose
- Predict noise ε_theta(x_t, t, c) given a noisy image x_t and a segmentation mask c.

## Conditioning
- Concatenate segmentation mask `c` with noisy image `x_t` along the channel dimension.
- No separate embedding for the mask is required; concatenation is sufficient.

## Timestep Embedding
- Scalar timestep `t` is converted to a vector embedding (sinusoidal or learned).
- Timestep embedding is added to feature maps in each encoder/decoder block via residual addition or FiLM modulation.
- Role: informs the model about the current noise level (timestep) to denoise appropriately.

## Output
- Single-channel predicted noise of shape (B, C, H, W)
- Loss: L2 between predicted and true noise

## Implementation Tasks
1. **Timestep Embedding Class**
   - Args: `dim` (embedding dimension), `max_timesteps` (T)
   - Method: `forward(t)` → returns embedding of shape `(B, dim)`

2. **Conditional U-Net**
   - Args: 
       - `img_ch` (input image channels, including mask channels)
       - `output_ch` (number of output channels, usually 1)
       - `time_embed_dim` (dimension of timestep embedding)
   - Forward Steps:
       1. Concatenate `x_t` and `mask` along channel dim.
       2. Compute timestep embedding from scalar `t`.
       3. Pass input through encoder blocks; add timestep embedding to each block.
       4. Pass through decoder blocks; add timestep embedding to each block.
       5. Return predicted noise of shape `(B, output_ch, H, W)`

3. **Integration with Trainer**
   - `train_epoch()`:
       1. Sample timestep t ∈ [1, T]
       2. Add noise to x0 → x_t
       3. Forward pass: `pred_noise = model(x_t, t, mask)`
       4. Compute L2 loss: `loss = ||pred_noise - true_noise||^2`
       5. Backprop and optimizer step
"""
