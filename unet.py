import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, residual=False):
        super().__init__()
        self.residual = residual
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(2, out_ch),
            nn.GELU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(2, out_ch)
        )
        
        self.norm = nn.GroupNorm(2, out_ch)

    def forward(self, x):
        if self.residual:
            return F.gelu(self.norm(x + self.double_conv(x)))
        else:
            return F.gelu(self.double_conv(x))
        
        
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim=64):
        super().__init__()
        self.max_pool = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch=in_channels, out_ch=in_channels, residual=True),
            DoubleConv(in_ch=in_channels, out_ch=in_channels, residual=True),
            DoubleConv(in_ch=in_channels, out_ch=out_channels, residual=False)
            )
        
        self.embed_layer = nn.Sequential( # a projection Layer to feature maps dim , [b, embed_dim] -> [b, channels_dim] so each channel will get scalar pos info
            nn.Linear(time_dim, in_channels),
            nn.SiLU(),
        )
        
        self.norm = nn.GroupNorm(2, in_channels)
        
    def forward(self, x, t):
        emb = self.embed_layer(t)[:, :, None, None] #.repeat(1, 1, x.shape[-2], x.shape[-1])
        x = self.norm(x + emb * 0.7)
        x = self.max_pool(x)
        return x
    
    
class Up(nn.Module):
    def __init__(self, in_channel, out_channel, time_dim=256) -> None:
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv = nn.Sequential(
            DoubleConv(in_ch=in_channel, out_ch=in_channel),
            DoubleConv(in_ch=in_channel, out_ch=out_channel),
            DoubleConv(in_ch=out_channel, out_ch=out_channel)
        )

        self.emb_layer = nn.Linear(in_features=time_dim, out_features=in_channel)
        
        self.norm = nn.GroupNorm(2, in_channel)

    def forward(self, x, x_skip, t_emb):
        x = self.up(x)
        x = torch.concat([x_skip, x], dim=1)
        t_emb = self.emb_layer(t_emb)[:, :, None, None]
        x = self.norm(x + t_emb * 0.7)
        x = self.conv(x)
        return x
    
class SelfAttention(nn.Module):
    def __init__(self, in_channels, mlp_ratio=4):
        super().__init__()
        self.pre_norm = nn.LayerNorm(in_channels)
        self.attn_layer = nn.MultiheadAttention(embed_dim=in_channels,num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(in_channels)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=in_channels*mlp_ratio),
            nn.GELU(),
            nn.Linear(in_features=in_channels*mlp_ratio, out_features=in_channels),
        )
        self.norm2 = nn.LayerNorm(in_channels)
        
    def forward(self, x: torch.Tensor):
        n_batch, channels, width, height = x.size()
        # flatten each img channel to 1d
        x = x.view(n_batch, channels, width*height)
        # img_pixels in horizontal -> channel pixels in horizontal [1st channel pixel 1 , 2nd ch pix 2]
        x = x.transpose(1, 2) # b, imgsize, channels
        norm_x = self.pre_norm(x)
        
        residual_x = x        
        attn_out, _ = self.attn_layer(norm_x, norm_x, norm_x, need_weights=False)
        x = residual_x + attn_out
        x = self.norm1(x)
        
        residual_x = x
        mlp_out = self.mlp(x)
        x = residual_x + mlp_out
        x = self.norm2(x)
        
        return x.transpose(1, 2).contiguous().view(n_batch, channels, width, height) 

class UNet(nn.Module):
    def __init__(self, c_in=3, time_dim: int = 64, device='cuda', multiplier=1) -> None:
        super().__init__()
        self.device = device
        self.time_dim = time_dim

        self.inc = DoubleConv(c_in, 64 * multiplier)

        self.down1 = Down(64 * multiplier, 128 * multiplier, time_dim)
        self.sa1 = SelfAttention(128 * multiplier)

        self.down2 = Down(128 * multiplier, 256 * multiplier, time_dim)
        self.sa2 = SelfAttention(256 * multiplier)

        self.down3 = Down(256 * multiplier, 256 * multiplier, time_dim)
        self.sa3 = SelfAttention(256 * multiplier)

        self.bot1 = DoubleConv(256 * multiplier, 512 * multiplier)
        self.bot2 = DoubleConv(512 * multiplier, 512 * multiplier)
        self.bot3 = DoubleConv(512 * multiplier, 256 * multiplier)

        self.up1 = Up(512 * multiplier, 128 * multiplier, time_dim) # 512 cuz we have 256 skip connection
        self.sa4 = SelfAttention(128 * multiplier)

        self.up2 = Up(256 * multiplier, 64 * multiplier, time_dim) # 126 skip
        self.sa5 = SelfAttention(64 * multiplier)

        self.up3 = Up(128 * multiplier, 64 * multiplier, time_dim)
        self.sa6 = SelfAttention(64 * multiplier)

        self.outc = nn.Conv2d(64 * multiplier, c_in, kernel_size=1) # 1x1


    def pos_encoding(self, t, dim):
        device = t.device
        half_dim = dim // 2

        emb = torch.exp(
            torch.arange(half_dim, device=device) * (-math.log(10000) / (half_dim - 1))
        )

        emb = t * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
    
    
    
    def forward(self, x, t):
        t = t.float() / 1000.0 # or self.n_timesteps
        t = t.float().unsqueeze(-1)
        t_emb = self.pos_encoding(t, self.time_dim)


        x1 = self.inc(x) # (b, 3, 64, 64) -> (b, 64, 64, 64)

        x2 = self.down1(x1, t_emb) # 64 -> 32
        # x2 = self.sa1(x2) # 32 -> 32

        x3 = self.down2(x2, t_emb) # 32 -> 16
        x3 = self.sa2(x3) # 16 -> 16

        x4 = self.down3(x3, t_emb) # 16 -> 8
        x4 = self.sa3(x4) # 8 -> 8

        x4 = self.bot1(x4) # 8
        x4 = self.bot2(x4) # 8
        x4 = self.bot3(x4) 

        x = self.up1(x4, x3, t_emb) # 8 -> 16
        x = self.sa4(x) 

        x = self.up2(x, x2, t_emb) # 16 -> 32
        x = self.sa5(x) # 32

        x = self.up3(x, x1, t_emb) # 32 -> 64
        # x = self.sa6(x) # 64


        output = self.outc(x) # 1x1
        

        return output
    