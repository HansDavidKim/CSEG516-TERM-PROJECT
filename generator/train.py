from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from PIL import Image
from torch import nn
from torch.autograd import grad
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, utils as vutils
from tqdm.auto import tqdm

from generator.model import Generator
from utils import seed_everything


class Discriminator(nn.Module):
    """Minimal DCGAN-style discriminator for 64x64 RGB images."""

    def __init__(self, in_channels: int = 3, dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, dim, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(dim, dim * 2, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(dim * 2, dim * 4, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(dim * 4, dim * 8, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(dim * 8, 1, kernel_size=4, stride=1, padding=0)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return logits.view(-1)


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


class FlatImageDataset(Dataset):
    """Fallback dataset for folders that keep images directly under the root."""

    def __init__(self, root: Path | str, *, transform: Optional[transforms.Compose] = None) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset path '{self.root}' does not exist.")
        self.transform = transform
        self.images = sorted(
            path for path in self.root.rglob("*")
            if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
        )
        if not self.images:
            raise RuntimeError(f"No supported image files were found under '{self.root}'.")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0


def _select_device(explicit: Optional[str] = None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_dataset_root(data_root: str, split: Optional[str]) -> Path:
    base = Path(data_root)
    if split:
        candidate = base / split
        if candidate.exists():
            return candidate
    return base


def _has_class_subdirs(path: Path) -> bool:
    try:
        entries = list(path.iterdir())
    except OSError:
        return False

    for entry in entries:
        if not entry.is_dir():
            continue
        try:
            child_iter = entry.iterdir()
        except OSError:
            continue
        if any(child.is_file() and child.suffix.lower() in _IMAGE_EXTENSIONS for child in child_iter):
            return True
    return False


def _add_instance_noise(images: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0:
        return images
    noise = torch.randn_like(images) * std
    return torch.clamp(images + noise, -1.0, 1.0)


def _gradient_penalty(
    discriminator: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

    mixed_scores = discriminator(interpolated)
    gradients = grad(
        outputs=mixed_scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()


def _build_dataloader(
    data_root: str,
    *,
    split: Optional[str],
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset_path = _resolve_dataset_root(data_root, split)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path '{dataset_path}' does not exist.")

    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if _has_class_subdirs(dataset_path):
        dataset = datasets.ImageFolder(dataset_path, transform=transform)
    else:
        dataset = FlatImageDataset(dataset_path, transform=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )


def _save_checkpoint(
    path: Path,
    *,
    epoch: int,
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
    }
    torch.save(checkpoint, path)


def _save_samples(
    generator: Generator,
    *,
    latent_dim: int,
    count: int,
    epoch: int,
    directory: Path,
    device: torch.device,
) -> Path:
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(count, latent_dim, device=device)
        samples = generator(noise).cpu()

    epoch_dir = directory / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    for idx, sample in enumerate(samples):
        vutils.save_image(
            sample,
            epoch_dir / f"{idx:02d}.png",
            normalize=True,
            value_range=(-1, 1),
        )
    return epoch_dir


@dataclass
class TrainResult:
    epochs: int
    device: str
    last_checkpoint: str
    last_sample: Optional[str]
    final_g_loss: float
    final_d_loss: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "epochs": self.epochs,
            "device": self.device,
            "last_checkpoint": self.last_checkpoint,
            "last_sample": self.last_sample,
            "final_g_loss": self.final_g_loss,
            "final_d_loss": self.final_d_loss,
        }


def train(
    *,
    data_root: str,
    output_dir: str = "checkpoints/generator",
    epochs: int = 50,
    batch_size: int = 128,
    latent_dim: int = 100,
    learning_rate: float = 2e-4,
    beta1: float = 0.5,
    beta2: float = 0.999,
    num_workers: int = 4,
    seed: Optional[int] = 42,
    sample_every: int = 1,
    sample_count: int = 10,
    split: Optional[str] = "train",
    device: Optional[str] = None,
    base_dim: int = 64,
    critic_steps: int = 5,
    gp_weight: float = 10.0,
    drift: float = 0.001,
    instance_noise: float = 0.05,
) -> Dict[str, object]:
    if seed is not None:
        seed_everything(seed)

    torch.backends.cudnn.benchmark = True

    resolved_device = _select_device(device)

    dataloader = _build_dataloader(
        data_root,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    generator = Generator(in_dim=latent_dim, dim=base_dim).to(resolved_device)
    discriminator = Discriminator(dim=base_dim).to(resolved_device)

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, beta2))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, beta2))

    output_path = Path(output_dir)
    checkpoint_dir = output_path
    sample_dir = output_path / "samples"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    last_sample_path: Optional[Path] = None
    last_checkpoint_path = checkpoint_dir / "generator_last.pt"

    steps_per_epoch = len(dataloader)
    data_iterator = iter(dataloader)

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        d_updates = 0
        g_updates = 0
        current_d_loss = 0.0
        current_g_loss = 0.0
        last_batch_size = batch_size

        progress = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}/{epochs}", leave=False)
        for _ in progress:
            # Critic updates
            for _ in range(max(1, critic_steps)):
                try:
                    real_images, _ = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(dataloader)
                    real_images, _ = next(data_iterator)

                real_images = real_images.to(resolved_device, non_blocking=True)
                real_images = _add_instance_noise(real_images, instance_noise)

                b_size = real_images.size(0)
                noise = torch.randn(b_size, latent_dim, device=resolved_device)
                fake_images = generator(noise).detach()
                fake_images = _add_instance_noise(fake_images, instance_noise)

                optimizer_d.zero_grad(set_to_none=True)
                real_scores = discriminator(real_images)
                fake_scores = discriminator(fake_images)

                gp = _gradient_penalty(discriminator, real_images, fake_images, resolved_device) if gp_weight > 0 else torch.tensor(0.0, device=resolved_device)
                loss_d = fake_scores.mean() - real_scores.mean()
                if gp_weight > 0:
                    loss_d = loss_d + gp_weight * gp
                if drift > 0:
                    loss_d = loss_d + drift * (real_scores.pow(2).mean())

                loss_d.backward()
                optimizer_d.step()

                epoch_d_loss += loss_d.item()
                current_d_loss = loss_d.item()
                d_updates += 1
                last_batch_size = b_size

            # Generator update
            noise = torch.randn(last_batch_size, latent_dim, device=resolved_device)
            generated_images = generator(noise)
            generated_images = _add_instance_noise(generated_images, instance_noise)

            optimizer_g.zero_grad(set_to_none=True)
            gen_scores = discriminator(generated_images)
            loss_g = -gen_scores.mean()
            loss_g.backward()
            optimizer_g.step()

            epoch_g_loss += loss_g.item()
            current_g_loss = loss_g.item()
            g_updates += 1

            progress.set_postfix(d_loss=current_d_loss, g_loss=current_g_loss)

        avg_d_loss = epoch_d_loss / max(1, d_updates)
        avg_g_loss = epoch_g_loss / max(1, g_updates)
        tqdm.write(f"[Epoch {epoch:03d}] Critic: {avg_d_loss:.4f} | Generator: {avg_g_loss:.4f}")

        if sample_every > 0 and epoch % sample_every == 0:
            last_sample_path = _save_samples(
                generator,
                latent_dim=latent_dim,
                count=sample_count,
                epoch=epoch,
                directory=sample_dir,
                device=resolved_device,
            )

        _save_checkpoint(
            last_checkpoint_path,
            epoch=epoch,
            generator=generator,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
        )

    result = TrainResult(
        epochs=epochs,
        device=str(resolved_device),
        last_checkpoint=str(last_checkpoint_path),
        last_sample=str(last_sample_path) if last_sample_path else None,
        final_g_loss=avg_g_loss,
        final_d_loss=avg_d_loss,
    )
    return result.as_dict()
