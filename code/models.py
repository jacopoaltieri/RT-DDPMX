import torch
import torch.nn as nn
import math
from torch.nn import functional as F


def Normalize(in_ch):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_ch, eps=1e-6, affine=True)


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

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

class Upsample(nn.Module):
    def __init__(self, in_ch, with_conv):
        super().__init__()
        self.with_conv = with_conv

        if self.with_conv:
            self.conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)

        return x


class Downsample(nn.Module):
    def __init__(self, in_ch, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_ch, in_ch, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResNetBlock(nn.Module):
    def __init__(
        self, in_ch, out_ch=None, conv_shortcut=False, dropout=0.0, temb_ch=512
    ):

        super().__init__()
        self.in_ch = in_ch
        out_ch = in_ch if out_ch is None else out_ch
        self.out_ch = out_ch
        self.use_conv_shortcut = conv_shortcut

        self.n1 = Normalize(in_ch)
        self.c1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.temb_proj = nn.Linear(temb_ch, out_ch)
        self.n2 = Normalize(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.c2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.silu = nn.SiLU()

        if self.in_ch != self.out_ch:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_ch, out_ch, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_ch, out_ch, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, temb):
        h = x
        h = self.n1(h)
        h = self.silu(h)
        h = self.c1(h)

        h = h + self.temb_proj(self.silu(temb))[:, :, None, None]

        h = self.n2(h)
        h = self.silu(h)
        h = self.dropout(h)
        h = self.c2(h)

        if self.in_ch != self.out_ch:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.in_ch = in_ch

        self.norm = Normalize(in_ch)
        self.q = torch.nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(
            in_ch, in_ch, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


# class EfficientAttention(nn.Module):
#     def __init__(self, in_ch, head_count=8):
#         super().__init__()
#         self.in_ch = in_ch
#         self.head_count = head_count

#         self.queries = nn.Conv2d(in_ch, in_ch, 1)
#         self.reprojection = nn.Conv2d(in_ch, in_ch, 1)

#     def forward(self, input_):
#         n, _, h, w = input_.size()
#         queries = self.queries(input_).reshape(n, self.in_ch, h * w)
#         head_channels = self.in_ch // self.head_count

#         attended_values = []
#         for i in range(self.head_count):
#             query = F.softmax(
#                 queries[:, i * head_channels : (i + 1) * head_channels, :], dim=1
#             )
#             context = query @ query.transpose(1, 2)
#             attended_value = (context.transpose(1, 2) @ query).reshape(
#                 n, head_channels, h, w
#             )
#             attended_values.append(attended_value)

#         aggregated_values = torch.cat(attended_values, dim=1)
#         reprojected_value = self.reprojection(aggregated_values)
#         attention = reprojected_value + input_

#         return attention


# class AttnBlock(nn.Module):
#     def __init__(self, in_ch, n_heads=8):
#         super(AttnBlock, self).__init__()
#         self.norm = nn.LayerNorm(in_ch)
#         self.attn = nn.MultiheadAttention(embed_dim=in_ch, num_heads=n_heads)

#     def forward(self, x):
#         b, c, h, w = x.shape
#         x = x.view(b, c, h * w).permute(2, 0, 1)  # (HW, B, C)
#         x = self.norm(x)
#         x, _ = self.attn(x, x, x)
#         x = x.permute(1, 2, 0).view(b, c, h, w)  # (B, C, H, W)
#         return x


# class SinusoidalPositionEmbedding(nn.Module):
#     def __init__(self, dim):
#         super(SinusoidalPositionEmbedding, self).__init__()
#         self.dim = dim

#     def forward(self, timesteps):
#         device = timesteps.device
#         half_dim = self.dim // 2
#         emb = math.log(10000) / (half_dim - 1)
#         emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
#         emb = timesteps[:, None] * emb[None, :]
#         return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


# class TimestepMLP(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(TimestepMLP, self).__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.SiLU(),
#             nn.Linear(hidden_dim, output_dim),
#         )

#     def forward(self, x):
#         return self.mlp(x)


class UNet(nn.Module):
    def __init__(self, 
                 in_ch=1, 
                 out_ch=1, 
                 ch=128, 
                 ch_mult=(1, 1, 2, 2, 4), 
                 num_res_blocks=2, 
                 attn_resolutions=[16], 
                 dropout=0.1, 
                 resolution=256, 
                 resamp_with_conv=True):
        super().__init__()
        
        self.ch = ch
        self.temb_ch = self.ch * 4
        self.temb_mlp = TimestepMLP(self.ch, self.temb_ch, self.temb_ch)

        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_ch = in_ch
        
        self.silu = nn.SiLU()

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch, self.temb_ch),
            torch.nn.Linear(self.temb_ch, self.temb_ch),
        ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_ch, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResNetBlock(in_ch=block_in,
                                         out_ch=block_out,
                                         temb_ch=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResNetBlock(in_ch=block_in,
                                       out_ch=block_in,
                                       temb_ch=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResNetBlock(in_ch=block_in,
                                       out_ch=block_in,
                                       temb_ch=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(ResNetBlock(in_ch=block_in + skip_in,
                                         out_ch=block_out,
                                         temb_ch=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t):
        assert x.shape[2] == x.shape[3] == self.resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb_mlp(temb)  # Pass through the MLP

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = self.silu(h)
        h = self.conv_out(h)
        return h
    
    
if __name__ == "__main__":
    # Create an instance of the UNet model
    model = UNet(in_ch=1, out_ch=1, resolution=512)

    # Generate a random input tensor of shape (batch_size, channels, height, width)
    x = torch.randn(4, 1, 512, 512)

    # Generate random timesteps tensor for testing
    t = torch.randint(0, 1000, (4,))  # Timesteps should be a 1D tensor with length equal to batch size

    # Forward pass through the model
    output = model(x, t)

    # Print the output shape to verify the model's forward pass
    print("Output shape:", output.shape)