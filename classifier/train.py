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

# ============================================================
# ArcFace modules
# ============================================================

import torch.nn.functional as F

class ArcMarginProduct(nn.Module):
    """Implements the additive angular margin for ArcFace."""
    def __init__(self, in_features: int, out_features: int, s: float = 64.0, m: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))  # (B, C)
        cosine = cosine.clamp(-1.0, 1.0)
        theta = torch.acos(cosine)
        phi = torch.cos(theta + self.m)  # margin added
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = self.s * torch.where(one_hot.bool(), phi, cosine)
        return logits


class ArcFaceLoss(nn.Module):
    """Cross-entropy wrapper for ArcFace logits."""
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, labels)

# ============================================================

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

# ============================================================
# transforms & augmentation
# ============================================================

IMAGENET_MEAN = (0.5177433, 0.4284404, 0.3802497)
IMAGENET_STD = (0.3042383, 0.2845056, 0.2826854)

def get_transforms(level: str = "normal") -> Tuple[transforms.Compose, transforms.Compose]:
    level = level.lower()
    if level == "weak":
        train_tf = transforms.Compose([
            transforms.Resize(72, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    elif level == "normal":
        train_tf = transforms.Compose([
            transforms.Resize(72, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.05,0.05,0.05,0.01),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    elif level == "strong":
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.9,1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.08,0.08,0.08,0.02),
            transforms.RandomAffine(degrees=6, translate=(0.02,0.02), scale=(0.95,1.05)),
            transforms.RandomErasing(p=0.08, scale=(0.02,0.15)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        raise ValueError("augmentation level must be one of: weak, normal, strong")

    eval_tf = transforms.Compose([
        transforms.Resize(72, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, eval_tf

# ============================================================

def run_epoch(
    model: nn.Module,
    arc_head: ArcMarginProduct,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    desc: str,
) -> Tuple[float, Dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)
    arc_head.train(is_train)
    running_loss = 0.0
    total = 0
    topk = (1, 3, 5)
    correct_at_k: Dict[int, float] = {k: 0.0 for k in topk}

    def _topk_hits(logits: torch.Tensor, targets: torch.Tensor) -> Dict[int, float]:
        max_k = min(max(topk), logits.size(1))
        _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
        pred = pred.t()
        targets_expanded = targets.view(1, -1).expand_as(pred)
        matches = pred.eq(targets_expanded)
        return {k: matches[:k].reshape(-1).float().sum().item() for k in topk}

    progress = tqdm(dataloader, desc=desc, leave=False)
    for images, targets in progress:
        images, targets = images.to(device), targets.to(device)
        with torch.set_grad_enabled(is_train):
            features, _ = model(images)
            logits = arc_head(features, targets)
            loss = criterion(logits, targets)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        hits = _topk_hits(logits, targets)
        for k in topk:
            correct_at_k[k] += hits[k]
        total += images.size(0)
        avg_loss = running_loss / total
        metrics = {f"top{k}": correct_at_k[k] / total * 100.0 for k in topk}
        progress.set_postfix(loss=f"{avg_loss:.4f}", **metrics)
    return running_loss / total, {f"top{k}": correct_at_k[k] / total * 100.0 for k in topk}

# ============================================================

def train(
    *,
    data_root: str = "dataset/private/celeba",
    model_name: str = "VGG16",
    pretrained: str | None = None,
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
    augmentation_level: str | None = None,
) -> Dict[str, float | int | str]:
    device = resolve_device()
    print(f"Using device: {device}")

    cfg = classifier_config
    batch_size = batch_size or cfg["batch_size"]
    learning_rate = learning_rate or cfg["learning_rate"]
    optimizer_name = optimizer_name or cfg["optimizer"]
    momentum = momentum if momentum is not None else cfg.get("momentum", 0.9)
    weight_decay = weight_decay if weight_decay is not None else cfg.get("weight_decay", 0.0)
    epochs = epochs or cfg["epoch"]
    patience = patience if patience is not None else cfg.get("patience", 5)
    num_workers = num_workers or min(4, os.cpu_count() or 1)
    aug_level = (augmentation_level or cfg.get("augmentation_level", "normal")).lower()

    train_tf, eval_tf = get_transforms(aug_level)
    root = Path(data_root)
    train_ds = CelebADirectoryDataset(root / "train", transform=train_tf)
    class_to_idx = train_ds.class_to_idx
    val_ds = CelebADirectoryDataset(root / "valid", transform=eval_tf, class_to_idx=class_to_idx)
    test_ds = CelebADirectoryDataset(root / "test", transform=eval_tf, class_to_idx=class_to_idx)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    num_classes = len(class_to_idx)
    model = create_model(model_name, num_classes, pretrained if load_pretrained else None).to(device)
    emb_dim = getattr(model, "embedding_dim", 512)
    arc_head = ArcMarginProduct(emb_dim, num_classes, s=64.0, m=0.5).to(device)
    criterion = ArcFaceLoss()
    optimizer = build_optimizer(model, optimizer_name=optimizer_name,
                                learning_rate=learning_rate, momentum=momentum, weight_decay=weight_decay)

    best_loss = float("inf")
    patience_counter = 0
    ckpt_path = Path(checkpoint_dir) / f"{model_name.lower()}_arcface_best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss, _ = run_epoch(model, arc_head, train_dl, criterion, device, optimizer=optimizer, desc=f"Train {epoch}/{epochs}")
        val_loss, val_metrics = run_epoch(model, arc_head, val_dl, criterion, device, optimizer=None, desc="Valid")

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Top1: {val_metrics['top1']:.2f}%")

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save({"model": model.state_dict(), "arc_head": arc_head.state_dict(),
                        "epoch": epoch, "val_loss": val_loss, "val_metrics": val_metrics}, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping.")
                break

    print(f"Best Val Loss: {best_loss:.4f}")
    test_loss, test_metrics = run_epoch(model, arc_head, test_dl, criterion, device, optimizer=None, desc="Test")
    print(f"Test Top1: {test_metrics['top1']:.2f}% | Loss: {test_loss:.4f}")
    print(f"Checkpoint saved to {ckpt_path}")
    return {"best_loss": best_loss, "test_top1": test_metrics["top1"], "checkpoint": str(ckpt_path)}


if __name__ == "__main__":
    train()
