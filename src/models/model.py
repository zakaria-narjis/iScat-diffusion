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
import torch
import torch.nn as nn
from torch.nn import init


def init_weights(net, init_type="normal", gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)


def get_time_embedding(time_steps, temb_dim):
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"

    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2))
    )

    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, temb_dim):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, 1, 1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, 3, 1, 1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

        self.time_proj = nn.Linear(temb_dim, ch_out)

    def forward(self, x, t_emb):
        h = self.conv(x)

        # timestep injection
        t = self.time_proj(t_emb)[:, :, None, None]
        h = h + t

        return h



class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True
            ),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    def __init__(self, img_ch=3, cond_ch=1, output_ch=1, temb_dim=256):
        super().__init__()

        self.temb_dim = temb_dim

        # timestep MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(temb_dim, temb_dim * 4),
            nn.SiLU(),
            nn.Linear(temb_dim * 4, temb_dim),
        )

        self.Maxpool = nn.MaxPool2d(2, 2)

        self.Conv1 = conv_block(img_ch + cond_ch, 64, temb_dim)
        self.Conv2 = conv_block(64, 128, temb_dim)
        self.Conv3 = conv_block(128, 256, temb_dim)
        self.Conv4 = conv_block(256, 512, temb_dim)
        self.Conv5 = conv_block(512, 1024, temb_dim)

        self.Up5 = up_conv(1024, 512)
        self.Up_conv5 = conv_block(1024, 512, temb_dim)

        self.Up4 = up_conv(512, 256)
        self.Up_conv4 = conv_block(512, 256, temb_dim)

        self.Up3 = up_conv(256, 128)
        self.Up_conv3 = conv_block(256, 128, temb_dim)

        self.Up2 = up_conv(128, 64)
        self.Up_conv2 = conv_block(128, 64, temb_dim)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, 1)

    def forward(self, x, cond, t):
        """
        x    : (B, img_ch, H, W)
        cond : (B, cond_ch, H, W)  segmentation mask
        t    : (B,) timesteps
        returns:
            noise : (B, output_ch, H, W)
        """

        # concatenate conditioning
        x = torch.cat([x, cond], dim=1)

        # timestep embedding
        t_emb = get_time_embedding(t, self.temb_dim)
        t_emb = self.time_mlp(t_emb)

        # encoder
        x1 = self.Conv1(x, t_emb)
        x2 = self.Conv2(self.Maxpool(x1), t_emb)
        x3 = self.Conv3(self.Maxpool(x2), t_emb)
        x4 = self.Conv4(self.Maxpool(x3), t_emb)
        x5 = self.Conv5(self.Maxpool(x4), t_emb)

        # decoder
        d5 = self.Up5(x5)
        d5 = self.Up_conv5(torch.cat([x4, d5], dim=1), t_emb)

        d4 = self.Up4(d5)
        d4 = self.Up_conv4(torch.cat([x3, d4], dim=1), t_emb)

        d3 = self.Up3(d4)
        d3 = self.Up_conv3(torch.cat([x2, d3], dim=1), t_emb)

        d2 = self.Up2(d3)
        d2 = self.Up_conv2(torch.cat([x1, d2], dim=1), t_emb)
        output = self.Conv_1x1(d2)
        return output




