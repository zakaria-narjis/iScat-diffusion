import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

# ----------------------------
# Weight Initialization
# ----------------------------
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


# ----------------------------
# Time embedding
# ----------------------------
def get_time_embedding(time_steps, temb_dim):
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"
    factor = 10000 ** ((torch.arange(
        start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2))
    )
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb


# ----------------------------
# Safe GroupNorm
# ----------------------------
def make_groupnorm(channels, default_groups=8):
    """
    Returns a GroupNorm layer with num_groups adjusted
    so that num_groups divides channels.
    """
    groups = min(default_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


# ----------------------------
# ResBlock with FiLM (timestep + optional cond)
# ----------------------------
class ResBlockFiLM(nn.Module):
    def __init__(self, ch_in, ch_out, temb_dim, cond_dim=None, num_groups=8):
        super().__init__()

        self.norm1 = make_groupnorm(ch_in, num_groups)
        self.conv1 = nn.Conv2d(ch_in, ch_out, 3, padding=1)

        self.norm2 = make_groupnorm(ch_out, num_groups)
        self.conv2 = nn.Conv2d(ch_out, ch_out, 3, padding=1)

        self.act = nn.SiLU()

        # timestep → (scale, shift)
        self.time_proj = nn.Linear(temb_dim, 2 * ch_out)

        # optional conditioning (seg mask features)
        self.cond_proj = (
            nn.Conv2d(cond_dim, 2 * ch_out, 1) if cond_dim is not None else None
        )

        self.residual = (
            nn.Conv2d(ch_in, ch_out, 1) if ch_in != ch_out else nn.Identity()
        )

    def forward(self, x, t_emb, cond_feat=None):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        # timestep FiLM
        scale_t, shift_t = self.time_proj(t_emb).chunk(2, dim=1)
        scale_t = scale_t[:, :, None, None]
        shift_t = shift_t[:, :, None, None]

        h = self.norm2(h)

        if cond_feat is not None:
            # Resize cond_feat to match current spatial size
            cond_resized = F.interpolate(cond_feat, size=h.shape[2:], mode='bilinear')
            scale_c, shift_c = self.cond_proj(cond_resized).chunk(2, dim=1)
            h = h * (1 + scale_t + scale_c) + (shift_t + shift_c)
        else:
            h = h * (1 + scale_t) + shift_t

        h = self.act(h)
        h = self.conv2(h)

        return h + self.residual(x)


# ----------------------------
# Mask Encoder
# ----------------------------
# class MaskEncoder(nn.Module):
#     def __init__(self, cond_ch, base_ch=64):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(cond_ch, base_ch, 3, padding=1),
#             nn.SiLU(),
#             nn.Conv2d(base_ch, base_ch, 3, padding=1),
#         )

#     def forward(self, cond):
#         return self.net(cond)
    
class MaskEncoder(nn.Module):
    def __init__(self, cond_ch, base_ch=64):
        super().__init__()
        # Initial projection
        self.init_conv = nn.Conv2d(cond_ch, base_ch, 3, padding=1)
        
        # Increasing depth with residual layers
        self.layer1 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, 3, padding=1)
        )
        
        # Downsampling layers to match U-Net resolution steps
        self.down1 = nn.Conv2d(base_ch, base_ch, 4, stride=2, padding=1) # 256 -> 128
        self.down2 = nn.Conv2d(base_ch, base_ch, 4, stride=2, padding=1) # 128 -> 64
        self.down3 = nn.Conv2d(base_ch, base_ch, 4, stride=2, padding=1) # 64 -> 32

    def forward(self, cond):
        feat0 = self.init_conv(cond)
        feat1 = self.layer1(feat0)
        feat2 = self.down1(feat1)
        feat3 = self.down2(feat2)
        feat4 = self.down3(feat3)
        # You can return a list of features for different U-Net layers
        return [feat1, feat2, feat3, feat4]

# ----------------------------
# Upsample + Conv Block
# ----------------------------
class up_conv(nn.Module):
    """Upsampling conv block with GroupNorm"""
    def __init__(self, ch_in, ch_out, num_groups=8):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, 3, 1, 1, bias=True),
            make_groupnorm(ch_out, num_groups),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


# ----------------------------
# U-Net
# ----------------------------
class U_Net(nn.Module):
    def __init__(self, img_ch=3, cond_ch=1, output_ch=1, temb_dim=256, num_groups=8):
        super().__init__()

        self.temb_dim = temb_dim
        self.Maxpool = nn.MaxPool2d(2)

        # timestep embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(temb_dim, temb_dim * 4),
            nn.SiLU(),
            nn.Linear(temb_dim * 4, temb_dim),
        )

        # mask encoder
        self.mask_enc = MaskEncoder(cond_ch, base_ch=64)

        # encoder
        self.Conv1 = ResBlockFiLM(img_ch + cond_ch, 64, temb_dim, cond_dim=64)
        self.Conv2 = ResBlockFiLM(64, 128, temb_dim, cond_dim=64)
        self.Conv3 = ResBlockFiLM(128, 256, temb_dim, cond_dim=64)
        self.Conv4 = ResBlockFiLM(256, 512, temb_dim, cond_dim=64)
        self.Conv5 = ResBlockFiLM(512, 1024, temb_dim, cond_dim=64)

        # decoder
        self.Up5 = up_conv(1024, 512, num_groups)
        self.Up_conv5 = ResBlockFiLM(1024, 512, temb_dim, cond_dim=64)

        self.Up4 = up_conv(512, 256, num_groups)
        self.Up_conv4 = ResBlockFiLM(512, 256, temb_dim, cond_dim=64)

        self.Up3 = up_conv(256, 128, num_groups)
        self.Up_conv3 = ResBlockFiLM(256, 128, temb_dim, cond_dim=64)

        self.Up2 = up_conv(128, 64, num_groups)
        self.Up_conv2 = ResBlockFiLM(128, 64, temb_dim, cond_dim=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, 1)

    # def forward(self, x, cond, t):
    #     # raw concat (cheap spatial prior)
    #     x = torch.cat([x, cond], dim=1)

    #     # timestep embedding
    #     t_emb = self.time_mlp(get_time_embedding(t, self.temb_dim))

    #     # mask features
    #     cond_feat = self.mask_enc(cond)

    #     # encoder
    #     x1 = self.Conv1(x, t_emb, cond_feat)
    #     x2 = self.Conv2(self.Maxpool(x1), t_emb, cond_feat)
    #     x3 = self.Conv3(self.Maxpool(x2), t_emb, cond_feat)
    #     x4 = self.Conv4(self.Maxpool(x3), t_emb, cond_feat)
    #     x5 = self.Conv5(self.Maxpool(x4), t_emb, cond_feat)

    #     # decoder
    #     d5 = self.Up5(x5)
    #     d5 = self.Up_conv5(torch.cat([x4, d5], dim=1), t_emb, cond_feat)

    #     d4 = self.Up4(d5)
    #     d4 = self.Up_conv4(torch.cat([x3, d4], dim=1), t_emb, cond_feat)

    #     d3 = self.Up3(d4)
    #     d3 = self.Up_conv3(torch.cat([x2, d3], dim=1), t_emb, cond_feat)

    #     d2 = self.Up2(d3)
    #     d2 = self.Up_conv2(torch.cat([x1, d2], dim=1), t_emb, cond_feat)

    #     return self.Conv_1x1(d2)
    
    def forward(self, x, cond, t):
            # 1. Timestep embedding
            t_emb = self.time_mlp(get_time_embedding(t, self.temb_dim))

            # 2. Mask features (Now returns a list of 4 tensors)
            # [0]: 256x256, [1]: 128x128, [2]: 64x64, [3]: 32x32
            cond_list = self.mask_enc(cond)

            # 3. Initial Concat
            x = torch.cat([x, cond], dim=1)

            # 4. Encoder
            # Use cond_list[0] for the highest resolution blocks
            x1 = self.Conv1(x, t_emb, cond_list[0]) 
            
            # Use cond_list[1] for the 128x128 blocks
            x2 = self.Conv2(self.Maxpool(x1), t_emb, cond_list[1]) 
            
            # Use cond_list[2] for the 64x64 blocks
            x3 = self.Conv3(self.Maxpool(x2), t_emb, cond_list[2]) 
            
            # Use cond_list[3] for the 32x32 and deeper blocks
            x4 = self.Conv4(self.Maxpool(x3), t_emb, cond_list[3]) 
            x5 = self.Conv5(self.Maxpool(x4), t_emb, cond_list[3]) 

            # 5. Decoder
            # Match the resolution levels similarly for the upward path
            d5 = self.Up5(x5)
            d5 = self.Up_conv5(torch.cat([x4, d5], dim=1), t_emb, cond_list[3])

            d4 = self.Up4(d5)
            d4 = self.Up_conv4(torch.cat([x3, d4], dim=1), t_emb, cond_list[2])

            d3 = self.Up3(d4)
            d3 = self.Up_conv3(torch.cat([x2, d3], dim=1), t_emb, cond_list[1])

            d2 = self.Up2(d3)
            d2 = self.Up_conv2(torch.cat([x1, d2], dim=1), t_emb, cond_list[0])

            return self.Conv_1x1(d2)