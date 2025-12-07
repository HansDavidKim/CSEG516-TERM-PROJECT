import os
import random
import tarfile
import zipfile
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import kagglehub
from datasets import Dataset, DatasetDict, load_dataset
import warnings, logging

from config import env_config

warnings.filterwarnings(
    "ignore",
    message="Looks like you're using an outdated `kagglehub` version",
    category=UserWarning,
)


def _extract_archive(archive_path: Path) -> None:
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as archive:
            archive.extractall(archive_path.parent)
        return

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(archive_path.parent)


def _extract_archives_in_dir(path: Path) -> None:
    for candidate in path.rglob("*"):
        if not candidate.is_file():
            continue
        name_lower = candidate.name.lower()
        if name_lower.endswith(_ZIP_SUFFIXES) or name_lower.endswith(_TAR_SUFFIXES):
            _extract_archive(candidate)


def _resolve_single_root(path: Path) -> Path:
    current = path
    while current.is_dir():
        try:
            entries = [
                entry for entry in current.iterdir()
                if entry.name not in {".DS_Store", "__MACOSX"} and entry.is_dir()
            ]
        except OSError:
            break
        file_with_images = _contains_images(current)
        if len(entries) == 1 and not file_with_images:
            current = entries[0]
            continue
        break
    return current


def _contains_images(path: Path) -> bool:
    try:
        iterator = path.iterdir()
    except OSError:
        return False
    return any(
        entry.is_file() and entry.suffix.lower() in _IMAGE_EXTENSIONS
        for entry in iterator
    )


def _dir_has_label_dirs(path: Path) -> bool:
    try:
        subdirs = [entry for entry in path.iterdir() if entry.is_dir()]
    except OSError:
        return False
    if not subdirs:
        return False
    return any(_contains_images(subdir) for subdir in subdirs)


def _discover_split_dirs(root: Path) -> Dict[str, Path]:
    split_dirs: Dict[str, Path] = {}
    try:
        entries = list(root.iterdir())
    except OSError as exc:
        raise ValueError(f"Unable to read contents of '{root}': {exc}") from exc

    for entry in entries:
        if not entry.is_dir():
            continue
        canonical = _SPLIT_NAME_MAP.get(entry.name.lower())
        if canonical and _dir_has_label_dirs(entry):
            split_dirs[canonical] = entry

    if split_dirs:
        return split_dirs

    if _dir_has_label_dirs(root):
        return {"train": root}

    for entry in entries:
        if entry.is_dir() and _dir_has_label_dirs(entry):
            return {"train": entry}

    raise ValueError(f"Could not find image folders inside '{root}'.")


def _prepare_downloaded_dataset(path: Path) -> Path:
    base_dir = path
    if path.is_file():
        _extract_archive(path)
        base_dir = path.parent
    elif path.is_dir():
        _extract_archives_in_dir(path)

    return _resolve_single_root(base_dir)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
_SPLIT_NAME_MAP = {
    "train": "train",
    "training": "train",
    "valid": "valid",
    "validation": "valid",
    "val": "valid",
    "test": "test",
    "testing": "test",
}
_TAR_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz", ".tar.xz", ".txz")
_ZIP_SUFFIXES = (".zip",)

### ==================== Universal Helper Functions ====================
def seed_everything(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

### ==================== Kaggle Helper Functions ====================
def login_kaggle():
    os.environ['KAGGLE_USERNAME'] = env_config['KAGGLE_USERNAME']
    os.environ['KAGGLE_KEY'] = env_config['KAGGLE_KEY']

def load_kaggle_dataset(dataset: str):
    download_path = Path(kagglehub.dataset_download(dataset))
    print("Path to dataset files:", download_path)

    try:
        _ = dataset.split('/')[1]
    except IndexError as exc:
        raise ValueError("Dataset handle must look like 'owner/dataset'.") from exc

    dataset_root = _prepare_downloaded_dataset(download_path)

    try:
        split_dirs = _discover_split_dirs(dataset_root)
    except ValueError as exc:
        if _contains_images(dataset_root):
            hf_split = load_dataset(
                "imagefolder",
                data_dir=str(dataset_root),
                split="train",
                drop_labels=True,
            )
            if 'identity' not in hf_split.column_names:
                hf_split = hf_split.add_column('identity', [0] * len(hf_split))
            return hf_split
        raise

    loaded: Dict[str, Dataset] = {}
    for split_name, split_dir in split_dirs.items():
        hf_split = load_dataset("imagefolder", data_dir=str(split_dir), split="train")
        if 'label' in hf_split.column_names and 'identity' not in hf_split.column_names:
            hf_split = hf_split.rename_column('label', 'identity')
        loaded[split_name] = hf_split

    if len(loaded) == 1:
        return next(iter(loaded.values()))
    return DatasetDict(loaded)

def configure_logging() -> None:
    """Suppress KaggleHub's outdated-version chatter while keeping other logs intact."""
    kagglehub_logger = logging.getLogger("kagglehub.clients")
    kagglehub_logger.setLevel(logging.ERROR)
    kagglehub_logger.propagate = False

if __name__ == '__main__':
    login_kaggle()
    load_kaggle_dataset("annasvoboda/pubfig83")
