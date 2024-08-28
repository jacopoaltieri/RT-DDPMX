import torch
import torch.nn as nn
import math

class ResNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResNetBlock, self).__init__()
        self.c1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.silu = nn.SiLU()
        self.c2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        
        # Handle residual connection
        self.skip = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.skip(x)
        x = self.silu(self.c1(x))
        x = self.silu(self.c2(x))  # Apply SiLU after second conv layer
        x += residual  # Add residual connection
        return x

class AttnBlock(nn.Module):
    def __init__(self, in_ch, n_heads=8):
        super(AttnBlock, self).__init__()
        self.norm = nn.LayerNorm(in_ch)
        self.attn = nn.MultiheadAttention(embed_dim=in_ch, num_heads=n_heads)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(2, 0, 1)  # (HW, B, C)
        x = self.norm(x)  # Apply LayerNorm before attention
        x, _ = self.attn(x, x, x)
        x = x.permute(1, 2, 0).view(b, c, h, w)  # (B, C, H, W)
        return x

def sin_embedding(timesteps, dim, device):
    half_dim = dim // 2
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=device)
        * -math.log(10000.0)
        / half_dim
    )
    emb = timesteps[:, None].to(device) * emb[None, :]
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
        channels = [128, 256, 512]  # Channels for each stage
        
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
        
        self.up_blocks = nn.ModuleList()
        for i in range(len(channels) - 1, -1, -1):
            in_ch = channels[i] * 2 if i < len(channels) - 1 else channels[i]
            out_ch = channels[i]
            self.up_blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    ResNetBlock(in_ch, out_ch),
                    ResNetBlock(out_ch, out_ch),
                )
            )

        self.timestep_embed = MLP(128, channels[-1])  # Output channels match the last downsampled layer

        self.final_conv = nn.Conv2d(channels[0], 1, kernel_size=1)  # Grayscale output

    def forward(self, x, t):
        t_embed = sin_embedding(t, 128, x.device)  # (batch_size, 128)
        t_embed = self.timestep_embed(t_embed)  # (batch_size, channels[-1])

        t_embed = t_embed.unsqueeze(-1).unsqueeze(-1)  # (batch_size, channels[-1], 1, 1)

        skips = []
        for down in self.down_blocks:
            x = down(x)
            skips.append(x)
            # Create AttnBlock with correct number of input channels
            attn_block = AttnBlock(x.size(1), n_heads=8).to(x.device)  # Move to correct device
            x = attn_block(x)
            x += t_embed[:x.size(0)]

        for i, up in enumerate(self.up_blocks):
            x = up(torch.cat([x, skips.pop()], dim=1))
            # Create AttnBlock with correct number of input channels
            attn_block = AttnBlock(x.size(1), n_heads=8).to(x.device)  # Move to correct device
            x = attn_block(x)

        x = self.final_conv(x)
        return x
