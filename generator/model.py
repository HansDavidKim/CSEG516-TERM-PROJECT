import numpy as np
import torch
import torch.nn as nn


def dconv_block(in_channels: int, out_channels: int, *, apply_norm: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
    ]
    if apply_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.01, inplace=True))
    return nn.Sequential(*layers)


class Generator(nn.Module):
    def __init__(self, in_dim: int = 100, dim: int = 64, use_batchnorm: bool = True) -> None:
        super().__init__()
        self.use_batchnorm = use_batchnorm

        linear_layers: list[nn.Module] = [nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False)]
        if use_batchnorm:
            linear_layers.append(nn.BatchNorm1d(dim * 8 * 4 * 4))
        linear_layers.append(nn.ReLU(inplace=True))
        self.project = nn.Sequential(*linear_layers)

        self.block1 = dconv_block(dim * 8, dim * 4, apply_norm=use_batchnorm)
        self.block2 = dconv_block(dim * 4, dim * 2, apply_norm=use_batchnorm)
        self.block3 = dconv_block(dim * 2, dim, apply_norm=use_batchnorm)

        self.to_rgb = nn.Sequential(
            nn.ConvTranspose2d(dim, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.project(x)
        y = y.view(x.size(0), -1, 4, 4)
        y = self.block1(y)
        y = self.block2(y)
        y = self.block3(y)
        y = self.to_rgb(y)
        return y

    def generate_from_latent(self, z: torch.Tensor | np.ndarray) -> torch.Tensor:
        """
        Generate images from latent vectors.
        Handles numpy arrays and adds batch dimension if necessary.
        """
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()
        
        if z.dim() == 1:
            z = z.unsqueeze(0)
            
        z = z.to(next(self.parameters()).device)
        
        return self.forward(z)
