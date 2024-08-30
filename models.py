import torch
import torch.nn as nn
import math


class ResNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.silu = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        x = self.silu(self.conv1(x))
        x = self.silu(self.conv2(x))
        return x + residual


class AttnBlock(nn.Module):
    def __init__(self, in_ch, n_heads=8):
        super(AttnBlock, self).__init__()
        self.norm = nn.LayerNorm(in_ch)
        self.attn = nn.MultiheadAttention(embed_dim=in_ch, num_heads=n_heads)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(2, 0, 1)  # (HW, B, C)
        x = self.norm(x)
        x, _ = self.attn(x, x, x)
        x = x.permute(1, 2, 0).view(b, c, h, w)  # (B, C, H, W)
        return x


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.dim = dim

    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class TimestepMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TimestepMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        channels = [128, 128, 256, 256, 512]
        self.sinusoidal_embedding = SinusoidalPositionEmbedding(128)
        self.timestep_embed = TimestepMLP(128, 512, 512)

        self.down_blocks = nn.ModuleList()
        for i in range(len(channels)):
            in_ch = channels[i - 1] if i > 0 else 1  # Grayscale images
            out_ch = channels[i]
            self.down_blocks.append(
                nn.Sequential(
                    ResNetBlock(in_ch, out_ch),
                    ResNetBlock(out_ch, out_ch),
                    AttnBlock(out_ch),
                    nn.MaxPool2d(2),
                )
            )

        self.up_blocks = nn.ModuleList()
        for i in range(len(channels) - 1, -1, -1):
            in_ch = channels[i] * 2 if i < len(channels) - 1 else channels[i]
            out_ch = channels[i]
            self.up_blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    ResNetBlock(in_ch, out_ch),
                    ResNetBlock(out_ch, out_ch),
                    AttnBlock(out_ch),
                )
            )

        self.final_conv = nn.Conv2d(channels[0], 1, kernel_size=1)  # Grayscale output

    def forward(self, x, t):
        t_embed = self.sinusoidal_embedding(t)  # (batch_size, 128)
        t_embed = self.timestep_embed(t_embed).unsqueeze(-1).unsqueeze(-1)  # (batch_size, 512, 1, 1)

        skips = []
        for down in self.down_blocks:
            x = down(x)
            skips.append(x)
            x += t_embed

        for up in self.up_blocks:
            x = up(torch.cat([x, skips.pop()], dim=1))
            x += t_embed

        return self.final_conv(x)
