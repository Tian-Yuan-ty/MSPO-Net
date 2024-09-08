import torch
from einops import rearrange
from torch import nn, einsum
from torch.nn.init import trunc_normal_

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size, padding=padding, groups=dim_in, stride=stride, bias=bias),
            nn.BatchNorm2d(dim_in),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias),
        )

    def forward(self, x):
        return self.net(x)


class AttentionCNN(nn.Module):
    def __init__(self, in_dim, dim_head=64, heads=8, residual_mode=True):
        super().__init__()
        proj_kernel = 3
        kv_proj_stride = 2
        inner_dim = dim_head * heads
        padding = proj_kernel // 2
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_q = DepthWiseConv2d(in_dim, inner_dim, proj_kernel, padding, stride=1)
        self.to_kv = DepthWiseConv2d(in_dim, inner_dim * 2, proj_kernel, padding, stride=kv_proj_stride)
        self.residual_mode = residual_mode

        self.norm = []
        for _ in range(heads):
            self.norm.append(nn.Sequential(
                LayerNorm(dim_head)))
        self.norm = nn.ModuleList(self.norm)
        if self.residual_mode:
            if in_dim != dim_head:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_dim, dim_head, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(dim_head),
                )
            else:
                self.downsample = nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        if self.residual_mode:
            residual = self.downsample(x)
        shape = x.shape
        b, n, _, y, h = *shape, self.heads
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=1))
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h=h), (q, k, v))
        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) d -> b h d x y', b=b, h=h, y=y)
        if self.residual_mode:
            for h in range(0, out.shape[1]):
                out[:, h, :, :, :] = out[:, h, :, :, :] + residual

        out_ = self.norm[0](out[:, 0, :, :, :]).unsqueeze(1)
        for h in range(1, out.shape[1]):
            out_ = torch.cat((out_, self.norm[h](out[:, h, :, :, :]).unsqueeze(1)), dim=1)
        return out_.view(out_.shape[0], out_.shape[1], out_.shape[2], -1).mean(dim=3)


class SeqAttention(nn.Module):
    def __init__(self, in_dim, head_dim, n_heads, residual_mode=False):
        super().__init__()
        self.sqa = AttentionCNN(in_dim, head_dim, n_heads, residual_mode)

    def forward(self, x):
        return self.sqa(x)
