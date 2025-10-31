import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        reshaped = x.permute(0, 2, 3, 1).contiguous()
        normalized = self.norm(reshaped)
        return normalized.permute(0, 3, 1, 2).contiguous()

def dconv_bn_relu(in_dim, out_dim):
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, 5, 2, padding=2, output_padding=1, bias=False),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
        LayerNorm2d(out_dim)
    )

class Generator(nn.Module):
    def __init__(self, in_dim=100, dim=64):
        super().__init__()
        
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias = False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU(),
            nn.LayerNorm(dim * 8 * 4 * 4)
        )

        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y
    
