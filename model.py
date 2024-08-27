import torch
import torch.nn as nn
import math


class ResNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResNetBlock, self).__init__()
        self.c1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.silu = nn.SiLU()
        self.c2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.skip(x)
        x = self.silu(self.c1(x))
        x = self.c2(x)


class AttnBlock(nn.Module):
    def __init__(self, in_ch, n_heads=8):
        super(AttnBlock, self).__init__()
        self.norm = nn.LayerNorm(in_ch)
        self.attn = nn.MultiheadAttention(in_ch, n_heads)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(2, 0, 1)
        x, _ = self.attn(x, x, x)
        x = x.permute(1, 2, 0).view(b, c, h, w)


# Timestep embedding
def sin_embedding(timesteps, dim):
    half_dim = dim // 2
    emb = math.log(10000)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.silu = nn.SiLU()
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        x = self.silu(self.fc1(x))
        return self.fc2(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        channels = [128, 256, 512]

        # downsampling
        self.down_blocks = nn.ModuleList()
        for i in range(len(channels)):
            in_ch = channels[i - 1] if i > 0 else 1  # grayscale images
            out_ch = channels[i]
            self.down_blocks.append(
                nn.Sequential(
                    ResNetBlock(in_ch, out_ch),
                    ResNetBlock(out_ch, out_ch),
                    nn.MaxPool2d(2),
                )
            )

        # Attention Block
        self.attention = AttnBlock(channels[-1])

        # Upsampling
        self.up_blocks = nn.ModuleList()
        for i in range(len(channels) - 1, -1, -1):
            in_ch = channels[i] * 2 if i < len(channels) - 1 else channels[i]
            out_ch = channels[i]
            self.up_blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    ResNetBlock(in_ch, out_ch),
                    ResNetBlock(out_ch, out_ch),
                )
            )

        # Timestep embedding
        self.timestep_embed = MLP(128, channels[-1])

        # Final Convolution
        self.final_conv = nn.Conv2d(channels[0], 3, kernel_size=1)

    def forward(self, x, t):
        # Encode the timestep with sinusoidal embedding
        t_embed = sin_embedding(t, 128)
        t_embed = self.timestep_embed(t_embed).unsqueeze(-1).unsqueeze(-1)

        # Downsampling
        skips = []
        for down in self.down_blocks:
            x = down(x + t_embed)
            skips.append(x)

        # Attention
        x = self.attention(x)

        # Upsampling
        for i, up in enumerate(self.up_blocks):
            x = up(torch.cat([x, skips.pop()], dim=1))

        # Final Convolution
        return self.final_conv(x)


if __name__ == "__main__":
    pass
