import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1. Time Embedding
#    Paper: sinusoidal encoding → 2-layer MLP
#    (Ho et al. 2020 §3.3, Improved DDPM §A)
# ─────────────────────────────────────────────────────────────────────────────
def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    device  = timesteps.device
    half    = dim // 2
    freqs   = torch.exp(
        -math.log(10000) * torch.arange(half, device=device) / (half - 1)
    )                                            # (half,)
    args    = timesteps[:, None].float() * freqs[None]   # (B, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


class TimeEmbedding(nn.Module):
    """Sinusoidal embedding → Linear(dim, 4*dim) → SiLU → Linear(4*dim, 4*dim)."""
    def __init__(self, base_dim: int):
        super().__init__()
        self.base_dim = base_dim
        self.mlp = nn.Sequential(
            nn.Linear(base_dim, base_dim * 4),
            nn.SiLU(),
            nn.Linear(base_dim * 4, base_dim * 4),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(sinusoidal_embedding(t, self.base_dim))  # (B, 4*base_dim)


# ─────────────────────────────────────────────────────────────────────────────
# 2. ResBlock with Adaptive Group Norm (AdaGN)
#    Paper: "we condition on t by adding in the Transformer sinusoidal
#            position embedding into each residual block"
#    Improved DDPM: AdaGN replaces simple addition with scale-shift.
#
#    AdaGN(h, y) = y_s · GroupNorm(h) + y_b
#    where [y_s, y_b] = Linear(t_emb)
# ─────────────────────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_emb_dim: int, dropout: float = 0.1):
        super().__init__()

        # First conv path
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        # Time projection → scale + shift for AdaGN (output is out_ch * 2)
        self.t_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_ch * 2),
        )

        # Second conv path
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.drop  = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        # Skip: 1×1 conv if channel dims differ, else identity
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # First half
        h = self.conv1(F.silu(self.norm1(x)))

        # AdaGN: inject time
        scale, shift = self.t_proj(t_emb)[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift

        # Second half
        h = self.conv2(self.drop(F.silu(h)))
        return h + self.skip(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Self-Attention
#    Paper: multi-head at 16×16 and 8×8 resolutions.
#    GroupNorm before attention; residual connection after.
# ─────────────────────────────────────────────────────────────────────────────
class SelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).transpose(1, 2)   # (B, HW, C)
        h, _ = self.attn(h, h, h, need_weights=False)
        return x + h.transpose(1, 2).view(B, C, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Down / Up sampling
#    Paper: strided conv for down (not MaxPool), nearest+conv for up.
# ─────────────────────────────────────────────────────────────────────────────
class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


# ─────────────────────────────────────────────────────────────────────────────
# 5. UNet
#    Architecture (64×64 defaults, matching DDPM paper):
#
#    init_conv ─────────────────────────────────────────── out_conv
#        │                                                     ▲
#     Encoder                                              Decoder
#    (down levels)                                        (up levels)
#        │                                                     │
#        └──── skip connections (concat in decoder) ──────────┘
#                          │
#                     Bottleneck
#                  ResBlock─Attn─ResBlock
#
#    Channel progression: 128 → 256 → 256 → 256  (mults 1,2,2,2)
#    Attention: applied at resolutions in `attn_resolutions`
#    Num ResBlocks per level: `num_res_blocks` (enc) / `num_res_blocks+1` (dec)
# ─────────────────────────────────────────────────────────────────────────────
class UNet(nn.Module):
    def __init__(
        self,
        c_in:             int   = 3,
        base_channels:    int   = 128,
        channel_mults:    tuple = (1, 2, 2, 2),
        num_res_blocks:   int   = 2,
        attn_resolutions: tuple = (16, 8),
        dropout:          float = 0.1,
        time_dim:         int   = 128,
        input_size:       int   = 64,
        device:           str   = 'cuda',
    ):
        super().__init__()
        self.device          = device
        self.channel_mults   = channel_mults
        self.num_res_blocks  = num_res_blocks

        t_dim = time_dim * 4    # expanded time embedding dim

        # ── time embedding ────────────────────────────────────────────────────
        self.time_embed = TimeEmbedding(time_dim)

        # ── initial conv ──────────────────────────────────────────────────────
        self.init_conv = nn.Conv2d(c_in, base_channels, 3, padding=1)

        # ── encoder ───────────────────────────────────────────────────────────
        # enc_blocks: flat list of (ResBlock, Attn|Identity) pairs per block
        # downsamples: one per level (None for last level)
        self.enc_blocks  = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        # Track channel count at every skip-save point in the encoder.
        # The first entry is base_channels because forward() saves h after
        # init_conv before entering the encoder loop.
        self._enc_skip_chs: list[int] = [base_channels]

        in_ch = base_channels
        res   = input_size

        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.enc_blocks.append(ResBlock(in_ch, out_ch, t_dim, dropout))
                self.enc_blocks.append(SelfAttention(out_ch) if res in attn_resolutions else nn.Identity())
                self._enc_skip_chs.append(out_ch)
                in_ch = out_ch

            if level < len(channel_mults) - 1:   # not the last level → downsample
                self.downsamples.append(Downsample(in_ch))
                self._enc_skip_chs.append(in_ch)  # feature after downsample also skipped
                res //= 2
            else:
                self.downsamples.append(None)     # placeholder to keep index aligned

        # ── bottleneck ────────────────────────────────────────────────────────
        self.mid_res1 = ResBlock(in_ch, in_ch, t_dim, dropout)
        self.mid_attn = SelfAttention(in_ch)
        self.mid_res2 = ResBlock(in_ch, in_ch, t_dim, dropout)

        # ── decoder ───────────────────────────────────────────────────────────
        # Mirror of encoder; each block receives a skip connection (concat → 2× channels)
        self.dec_blocks = nn.ModuleList()
        self.upsamples  = nn.ModuleList()

        # pop() pulls from the tail, which corresponds to the deepest encoder
        # features first — exactly the order the decoder needs them.
        # Do NOT reverse: _enc_skip_chs is already in push order (shallow→deep).
        skip_chs = list(self._enc_skip_chs)

        for level, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult

            for _ in range(num_res_blocks + 1):   # +1 block vs encoder (for skip concat)
                skip_ch = skip_chs.pop()
                self.dec_blocks.append(ResBlock(in_ch + skip_ch, out_ch, t_dim, dropout))
                self.dec_blocks.append(SelfAttention(out_ch) if res in attn_resolutions else nn.Identity())
                in_ch = out_ch

            if level < len(channel_mults) - 1:    # not the last level → upsample
                self.upsamples.append(Upsample(in_ch))
                res *= 2
            else:
                self.upsamples.append(None)

        # ── output head ───────────────────────────────────────────────────────
        # Paper: GroupNorm → SiLU → Conv 1×1
        self.out_norm = nn.GroupNorm(32, in_ch)
        self.out_conv = nn.Conv2d(in_ch, c_in, 1)

        self._init_weights()

    # ── weight initialisation ────────────────────────────────────────────────
    def _init_weights(self):
        """Zero-init the last conv in each ResBlock (stabilises early training)."""
        for m in self.modules():
            if isinstance(m, ResBlock):
                nn.init.zeros_(m.conv2.weight)
                if m.conv2.bias is not None:
                    nn.init.zeros_(m.conv2.bias)

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Time embedding
        t_emb = self.time_embed(t)          # (B, 4*time_dim)

        # Initial conv
        h = self.init_conv(x)

        # ── Encoder ──
        skips: list[torch.Tensor] = [h]     # first skip = after init_conv
        eb = iter(self.enc_blocks)           # pairs: (ResBlock, Attn|Identity)

        for level, ds in enumerate(self.downsamples):
            for _ in range(self.num_res_blocks):
                res_blk = next(eb)
                attn    = next(eb)
                h = res_blk(h, t_emb)
                h = attn(h)
                skips.append(h)

            if ds is not None:
                h = ds(h)
                skips.append(h)

        # ── Bottleneck ──
        h = self.mid_res1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, t_emb)

        # ── Decoder ──
        db = iter(self.dec_blocks)

        for level, up in enumerate(self.upsamples):
            for _ in range(self.num_res_blocks + 1):
                h = torch.cat([h, skips.pop()], dim=1)   # concat skip
                res_blk = next(db)
                attn    = next(db)
                h = res_blk(h, t_emb)
                h = attn(h)

            if up is not None:
                h = up(h)

        # ── Output ──
        return self.out_conv(F.silu(self.out_norm(h)))