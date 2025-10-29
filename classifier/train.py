import math
import os
import random

from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from config import classifier_config
from .utils import load_weight
from .models import VGG16, ResNet152, FaceNet

class CelebADirectoryDataset(Dataset):
    def __init__(
        self,
        root: Path | str,
        *,
        transform: transforms.Compose | None = None,
        class_to_idx: Dict[str, int] | None = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset split not found at: {self.root}")

        self.transform = transform
        self.class_to_idx = class_to_idx or self._build_class_index(self.root)
        self.samples = self._gather_samples(self.root, self.class_to_idx)
        if not self.samples:
            raise RuntimeError(f"No images were found under {self.root}")

    @staticmethod
    def _build_class_index(root: Path) -> Dict[str, int]:
        def sort_key(path: Path) -> Tuple[int, int | str]:
            name = path.name
            if name.isdigit():
                return (0, int(name))
            return (1, name)

        class_dirs = [p for p in root.iterdir() if p.is_dir()]
        if not class_dirs:
            raise RuntimeError(f"No identity folders found in {root}")
        ordered = sorted(class_dirs, key=sort_key)
        return {p.name: idx for idx, p in enumerate(ordered)}

    @staticmethod
    def _gather_samples(root: Path, class_to_idx: Dict[str, int]) -> Iterable[Tuple[Path, int]]:
        samples: list[Tuple[Path, int]] = []
        for class_name, label in class_to_idx.items():
            identity_dir = root / class_name
            if not identity_dir.exists():
                continue
            for image_path in sorted(identity_dir.glob("*.jpg")):
                samples.append((image_path, label))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def create_model(model_name: str, num_classes: int, weight_path: str | None) -> nn.Module:
    model_name = model_name.strip()
    if weight_path:
        try:
            model = load_weight(model_name, weight_path, num_classes)
            print(f"Loaded pretrained weights from {weight_path}")
            return model
        except AssertionError as exc:
            print(f"[WARN] {exc}. Initializing model from scratch.")

    if model_name == "VGG16":
        return VGG16(num_classes)
    if model_name == "ResNet152":
        return ResNet152(num_classes)
    if model_name in {"Face.evoLVe", "FaceNet"}:
        return FaceNet(num_classes)
    raise ValueError(f"Unsupported model_name '{model_name}'.")


def resolve_device(prefer_mps: bool = True) -> torch.device:
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_optimizer(
    model: nn.Module,
    *,
    optimizer_name: str,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    opt_name = optimizer_name.lower()
    if opt_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    if opt_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer '{optimizer_name}'.")


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    desc: str,
    mix_params: Dict[str, float] | None = None,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    running_loss = 0.0
    correct = 0.0
    total = 0

    progress = tqdm(dataloader, desc=desc, leave=False)
    for images, targets in progress:
        images = images.to(device)
        targets = targets.to(device)

        targets_a = targets
        targets_b = targets
        lam = 1.0
        mixed = False
        if is_train and mix_params:
            images, targets_a, targets_b, lam, mixed = apply_mixup_cutmix(
                images,
                targets,
                mixup_alpha=mix_params.get("mixup_alpha", 0.0),
                cutmix_alpha=mix_params.get("cutmix_alpha", 0.0),
                prob=mix_params.get("prob", 0.0),
                switch_prob=mix_params.get("switch_prob", 0.5),
            )

        with torch.set_grad_enabled(is_train):
            features, logits = model(images)
            if mixed:
                loss = lam * criterion(logits, targets_a) + (1.0 - lam) * criterion(logits, targets_b)
            else:
                loss = criterion(logits, targets)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        preds = logits.argmax(dim=1)
        running_loss += loss.item() * images.size(0)
        if mixed:
            correct += (
                lam * (preds == targets_a).sum().item()
                + (1.0 - lam) * (preds == targets_b).sum().item()
            )
        else:
            correct += (preds == targets).sum().item()
        total += images.size(0)

        avg_loss = running_loss / total
        accuracy = correct / total * 100.0
        progress.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{accuracy:.2f}%")

    return running_loss / total, correct / total * 100.0


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    if dataloader is None:
        return 0.0, 0.0
    return run_epoch(model, dataloader, criterion, device, optimizer=None, desc="Valid")


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(
            64,
            scale=(0.8, 1.0),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(interpolation=InterpolationMode.BICUBIC),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.02,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(72, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, eval_tf


def _rand_bbox(size: torch.Size, lam: float) -> Tuple[int, int, int, int]:
    _, _, height, width = size
    cut_ratio = math.sqrt(1.0 - lam)
    cut_height = int(height * cut_ratio)
    cut_width = int(width * cut_ratio)

    cy = random.randint(0, height - 1)
    cx = random.randint(0, width - 1)

    y1 = max(cy - cut_height // 2, 0)
    y2 = min(cy + cut_height // 2, height)
    x1 = max(cx - cut_width // 2, 0)
    x2 = min(cx + cut_width // 2, width)

    return y1, y2, x1, x2


def apply_mixup_cutmix(
    images: torch.Tensor,
    targets: torch.Tensor,
    *,
    mixup_alpha: float = 0.8,
    cutmix_alpha: float = 1.0,
    prob: float = 0.7,
    switch_prob: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, bool]:
    batch_size = images.size(0)
    if batch_size < 2 or random.random() > prob:
        return images, targets, targets, 1.0, False

    use_mixup = random.random() < switch_prob
    perm = torch.randperm(batch_size, device=images.device)

    if use_mixup and mixup_alpha > 0:
        lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()
        mixed = lam * images + (1.0 - lam) * images[perm]
        targets_a = targets
        targets_b = targets[perm]
        return mixed, targets_a, targets_b, lam, True

    if cutmix_alpha > 0:
        lam = torch.distributions.Beta(cutmix_alpha, cutmix_alpha).sample().item()
        y1, y2, x1, x2 = _rand_bbox(images.size(), lam)
        images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]

        adjusted_lam = 1.0 - ((y2 - y1) * (x2 - x1) / (images.size(2) * images.size(3)))
        targets_a = targets
        targets_b = targets[perm]
        return images, targets_a, targets_b, adjusted_lam, True

    return images, targets, targets, 1.0, False


def train(
    *,
    data_root: str = "dataset/private/celeba",
    model_name: str = "VGG16",
    pretrained: str | None = "pretrained/VGG16.tar",
    batch_size: int | None = None,
    learning_rate: float | None = None,
    optimizer_name: str | None = None,
    momentum: float | None = None,
    weight_decay: float | None = None,
    epochs: int | None = None,
    patience: int | None = None,
    num_workers: int | None = None,
    checkpoint_dir: str = "checkpoints",
    load_pretrained: bool = True,
    mixup_alpha: float = 0.8,
    cutmix_alpha: float = 1.0,
    mix_prob: float = 0.7,
    mix_switch_prob: float = 0.5,
) -> Dict[str, float | int | str]:
    device = resolve_device()
    if device.type != "mps":
        print(f"[WARN] MPS unavailable; falling back to {device}.")
    else:
        print("Using device: mps")

    cfg_batch = classifier_config["batch_size"]
    cfg_lr = classifier_config["learning_rate"]
    cfg_optimizer = classifier_config["optimizer"]
    cfg_momentum = classifier_config.get("momentum", 0.9)
    cfg_weight_decay = classifier_config.get("weight_decay", 0.0)
    cfg_epochs = classifier_config["epoch"]
    cfg_patience = classifier_config.get("patience", 5)

    batch_size = batch_size or cfg_batch
    learning_rate = learning_rate or cfg_lr
    optimizer_name = optimizer_name or cfg_optimizer
    momentum = momentum if momentum is not None else cfg_momentum
    weight_decay = weight_decay if weight_decay is not None else cfg_weight_decay
    epochs = epochs or cfg_epochs
    patience = patience if patience is not None else cfg_patience
    num_workers = num_workers if num_workers is not None else min(4, os.cpu_count() or 1)

    data_root_path = Path(data_root)
    train_root = data_root_path / "train"
    valid_root = data_root_path / "valid"
    test_root = data_root_path / "test"

    train_tf, eval_tf = get_transforms()
    train_dataset = CelebADirectoryDataset(train_root, transform=train_tf)
    class_to_idx = train_dataset.class_to_idx
    valid_dataset = CelebADirectoryDataset(valid_root, transform=eval_tf, class_to_idx=class_to_idx)
    test_dataset = CelebADirectoryDataset(test_root, transform=eval_tf, class_to_idx=class_to_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    num_classes = len(class_to_idx)
    weight_path = pretrained if load_pretrained else None
    model = create_model(model_name, num_classes, weight_path)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        model,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir_path / f"{model_name.lower()}_celeba_best.pt"

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    epoch_iterator = tqdm(range(1, epochs + 1), desc="Epochs")
    for epoch in epoch_iterator:
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            desc=f"Train [{epoch}/{epochs}]",
            mix_params={
                "mixup_alpha": mixup_alpha,
                "cutmix_alpha": cutmix_alpha,
                "prob": mix_prob,
                "switch_prob": mix_switch_prob,
            },
        )
        val_loss, val_acc = evaluate(model, valid_loader, criterion, device)

        epoch_iterator.set_postfix(
            train_loss=f"{train_loss:.4f}",
            train_acc=f"{train_acc:.2f}%",
            val_loss=f"{val_loss:.4f}",
            val_acc=f"{val_acc:.2f}%",
        )

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_name": model_name,
                    "state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "epoch": epoch,
                },
                ckpt_path,
            )
        else:
            patience_counter += 1

        if classifier_config.get("early_stopping", False) and patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}. Best epoch: {best_epoch}")
            break

    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")

    test_loss, test_acc = run_epoch(
        model,
        test_loader,
        criterion,
        device,
        optimizer=None,
        desc="Test",
    )
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.2f}%")
    print(f"Best checkpoint saved to {ckpt_path}")

    return {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "checkpoint_path": str(ckpt_path),
    }


if __name__ == "__main__":
    train()
