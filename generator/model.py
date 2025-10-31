import torch
import torch.nn as nn


class PixelNorm(nn.Module):
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = list(range(1, x.ndim))
        return x * torch.rsqrt(torch.mean(x * x, dim=dims, keepdim=True) + self.eps)


class SelfAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        reduced_channels = max(1, channels // 8)
        self.query = nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False)
        self.key = nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False)
        self.value = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.scale = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        proj_query = self.query(x).view(batch, -1, height * width).permute(0, 2, 1)
        proj_key = self.key(x).view(batch, -1, height * width)
        attention = torch.bmm(proj_query, proj_key)
        attention = self.softmax(attention)

        proj_value = self.value(x).view(batch, channels, height * width)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch, channels, height, width)
        return self.scale * out + x


def dconv_bn_relu(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, 5, 2, padding=2, output_padding=1, bias=False),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
        PixelNorm(),
    )


class Generator(nn.Module):
    def __init__(self, in_dim: int = 100, dim: int = 64) -> None:
        super().__init__()

        self.project = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU(),
            PixelNorm(),
        )

        self.up1 = dconv_bn_relu(dim * 8, dim * 4)
        self.up2 = dconv_bn_relu(dim * 4, dim * 2)
        self.attn = SelfAttention(dim * 2)
        self.up3 = dconv_bn_relu(dim * 2, dim)
        self.to_rgb = nn.Sequential(
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.project(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.up1(y)
        y = self.up2(y)
        y = self.attn(y)
        y = self.up3(y)
        y = self.to_rgb(y)
        return y
