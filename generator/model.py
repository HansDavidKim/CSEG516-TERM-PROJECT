import torch
import torch.nn as nn
import torch.nn.functional as F


class Blur(nn.Module):
    """Applies a fixed 3x3 blur kernel to reduce checkerboard artefacts after upsampling."""

    def __init__(self) -> None:
        super().__init__()
        kernel_1d = torch.tensor([1.0, 2.0, 1.0])
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel_2d /= kernel_2d.sum()
        self.register_buffer("kernel", kernel_2d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.size(1)
        kernel = self.kernel.expand(channels, 1, 3, 3)
        return F.conv2d(x, kernel, padding=1, groups=channels)


class UpsampleBlock(nn.Module):
    """Nearest upsample -> blur -> conv -> InstanceNorm -> activation."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2.0, mode="nearest")
        self.blur = Blur()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True, eps=1e-5)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.blur(x)
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


class Generator(nn.Module):
    def __init__(self, in_dim: int = 100, dim: int = 64) -> None:
        super().__init__()

        self.project = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.LayerNorm(dim * 8 * 4 * 4),
            nn.ReLU(inplace=True),
        )

        self.block1 = UpsampleBlock(dim * 8, dim * 4)
        self.block2 = UpsampleBlock(dim * 4, dim * 2)
        self.block3 = UpsampleBlock(dim * 2, dim)

        self.to_rgb = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="nearest"),
            Blur(),
            nn.Conv2d(dim, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.project(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.block1(y)
        y = self.block2(y)
        y = self.block3(y)
        y = self.to_rgb(y)
        return y
