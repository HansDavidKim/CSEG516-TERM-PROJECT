import os
import random
from collections import Counter

from pathlib import Path

from PIL import ImageOps, Image
from datasets import load_dataset, concatenate_datasets, DatasetDict
from tqdm.auto import tqdm

from utils import seed_everything


def _log(message: str, verbose: bool):
    if verbose:
        print(message)


def _progress(iterable, verbose: bool, desc: str, total: int | None = None):
    if not verbose:
        return iterable
    return tqdm(iterable, desc=desc, total=total, leave=False)


def _remap_identity(batch, identity_map):
    return {'identity': [identity_map[i] for i in batch['identity']]}


def _filter_identity_lt(batch, threshold):
    return [identity < threshold for identity in batch['identity']]


def _filter_identity_ge(batch, threshold):
    return [identity >= threshold for identity in batch['identity']]


def _preprocess_images_batch(batch):
    return {'image': [preprocess_image(img) for img in batch['image']]}


def _map_identity(dataset, identity_map, num_proc=None, verbose: bool = False, prefix: str = 'map_identity'):
    fn_kwargs = {'identity_map': identity_map}
    preview = list(identity_map.items())[:5]
    _log(
        f"[{prefix}] remapping {len(identity_map)} identities; sample mapping: {preview}",
        verbose
    )

    if isinstance(dataset, DatasetDict):
        return DatasetDict({
            split_name: split_dataset.map(
                _remap_identity,
                batched=True,
                fn_kwargs=fn_kwargs,
                num_proc=num_proc
            )
            for split_name, split_dataset in dataset.items()
        })

    return dataset.map(
        _remap_identity,
        batched=True,
        fn_kwargs=fn_kwargs,
        num_proc=num_proc
    )

def unify_format(dataset):
    if isinstance(dataset, DatasetDict):
        for split_name, split_dataset in dataset.items():
            if 'celeb_id' in split_dataset.column_names:
                dataset[split_name] = split_dataset.rename_column('celeb_id', 'identity')

    elif hasattr(dataset, 'column_names') and 'celeb_id' in dataset.column_names:
        dataset = dataset.rename_column('celeb_id', 'identity')

    return dataset

def sort_identity(dataset, num_proc: int | None = None, verbose: bool = False):
    if isinstance(dataset, DatasetDict):
        counts = Counter()
        for split_dataset in dataset.values():
            counts.update(split_dataset['identity'])

        identity_map = {identity: idx for idx, (identity, _) in enumerate(counts.most_common())}

        mapped = _map_identity(dataset, identity_map, num_proc=num_proc, verbose=verbose, prefix='sort_identity')
        for split_name, split_dataset in mapped.items():
            mapped[split_name] = split_dataset.sort('identity')
        return mapped

    counts = Counter(dataset['identity'])
    identity_map = {identity: idx for idx, (identity, _) in enumerate(counts.most_common())}

    mapped = _map_identity(dataset, identity_map, num_proc=num_proc, verbose=verbose, prefix='sort_identity')
    return mapped.sort('identity')

def shuffle_identity(dataset, num_proc: int | None = None, verbose: bool = False):
    if isinstance(dataset, DatasetDict):
        identities = set()
        for split_dataset in dataset.values():
            identities.update(split_dataset['identity'])

        identity_list = sorted(identities)
        shuffled = identity_list[:]
        random.shuffle(shuffled)
        identity_map = {original: shuffled[idx] for idx, original in enumerate(identity_list)}

        return _map_identity(dataset, identity_map, num_proc=num_proc, verbose=verbose, prefix='shuffle_identity')

    identity_list = sorted(set(dataset['identity']))
    shuffled = identity_list[:]
    random.shuffle(shuffled)
    identity_map = {original: shuffled[idx] for idx, original in enumerate(identity_list)}

    return _map_identity(dataset, identity_map, num_proc=num_proc, verbose=verbose, prefix='shuffle_identity')

def combine_dataset(dataset):
    try:
        combined = concatenate_datasets([
            dataset['train'],
            dataset['valid'],
            dataset['test']
        ])
    except:
        combined = dataset
    
    return combined

def extract_public(dataset, num_identities: int, num_proc: int | None = None):
    def split_identity(ds):
        public = ds.filter(
            _filter_identity_lt,
            batched=True,
            fn_kwargs={'threshold': num_identities},
            num_proc=num_proc
        )
        private = ds.filter(
            _filter_identity_ge,
            batched=True,
            fn_kwargs={'threshold': num_identities},
            num_proc=num_proc
        )
        return public, private

    if isinstance(dataset, DatasetDict):
        public_splits = {}
        private_splits = {}
        for split_name, split_dataset in dataset.items():
            public_split, private_split = split_identity(split_dataset)
            public_splits[split_name] = public_split
            private_splits[split_name] = private_split
        return DatasetDict(public_splits), DatasetDict(private_splits)

    return split_identity(dataset)

def split_dataset(dataset, seed: int = 42, stratify_column: str | None = 'identity'):
    if isinstance(dataset, DatasetDict) and {'train', 'valid', 'test'}.issubset(set(dataset.keys())):
        return dataset

    stratify = stratify_column if stratify_column and stratify_column in dataset.column_names else None

    def _train_test_split(ds, test_size):
        if stratify:
            try:
                return ds.train_test_split(
                    test_size=test_size,
                    seed=seed,
                    stratify_by_column=stratify
                )
            except ValueError:
                pass
        return ds.train_test_split(test_size=test_size, seed=seed)

    first_split = _train_test_split(dataset, 0.2)
    valid_test_split = _train_test_split(first_split['test'], 0.5)

    return DatasetDict({
        'train': first_split['train'],
        'valid': valid_test_split['train'],
        'test': valid_test_split['test'],
    })

def preprocess_image(image: Image.Image)->Image.Image:
    crop_side = min(image.size)
    cropped = ImageOps.fit(image, (crop_side, crop_side), centering=(0.5, 0.5))
    return cropped.resize((64, 64), Image.Resampling.LANCZOS)

def save_dataset(
        dataset,
        file_path: str,
        max_images_per_identity: int | None = 20,
        max_images_per_split: int | None = None,
        verbose: bool = False,
    ):
    base_path = Path(file_path)
    base_path.mkdir(parents=True, exist_ok=True)

    def ensure_pil(image):
        if isinstance(image, Image.Image):
            return image
        return Image.fromarray(image)

    def should_stop_split(saved_count: int) -> bool:
        return max_images_per_split is not None and saved_count >= max_images_per_split

    if isinstance(dataset, DatasetDict):
        for split_name, split_dataset in dataset.items():
            split_dir = base_path / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            identity_counts = Counter()
            saved_in_split = 0
            _log(f"[save_dataset] Start writing split '{split_name}' to {split_dir}", verbose)
            iterator = _progress(
                split_dataset,
                verbose,
                desc=f"[save_dataset] {split_name}",
                total=len(split_dataset) if verbose else None
            )
            for example in iterator:
                if should_stop_split(saved_in_split):
                    break

                identity = example['identity']
                if max_images_per_identity is not None and identity_counts[identity] >= max_images_per_identity:
                    continue

                image = ensure_pil(example['image'])
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                identity_dir = split_dir / str(identity)
                identity_dir.mkdir(parents=True, exist_ok=True)
                filename = identity_dir / f"{identity_counts[identity]:06d}.jpg"
                image.save(filename)
                identity_counts[identity] += 1
                saved_in_split += 1
            _log(f"[save_dataset] Saved {saved_in_split} images for split '{split_name}'", verbose)
        return

    identity_counts = Counter()
    saved = 0
    _log(f"[save_dataset] Start writing dataset to {base_path}", verbose)
    iterator = _progress(
        dataset,
        verbose,
        desc="[save_dataset] dataset",
        total=len(dataset) if verbose else None
    )
    for idx, example in enumerate(iterator):
        if should_stop_split(saved):
            break

        identity = example.get('identity')
        if identity is not None and max_images_per_identity is not None:
            if identity_counts[identity] >= max_images_per_identity:
                continue

        image = ensure_pil(example['image'])
        if image.mode != 'RGB':
            image = image.convert('RGB')

        filename = base_path / f"{idx:06d}.jpg"
        image.save(filename)
        if identity is not None:
            identity_counts[identity] += 1
        saved += 1
    _log(f"[save_dataset] Saved {saved} images", verbose)

def preprocess_dataset(
        dataset_name: str, 
        num_identities: int, 
        seed: int=42,
        num_proc: int | None = None,
        max_images_per_identity: int | None = 20,
        max_images_per_split: int | None = None,
        verbose: bool = True,
    ):
    if num_proc is None:
        cpu_count = os.cpu_count() or 1
        num_proc = max(1, cpu_count // 2)

    _log(f"[preprocess_dataset] Start preprocessing '{dataset_name}'", verbose)
    _log(f"[preprocess_dataset] Using num_proc={num_proc}", verbose)

    try:
        _log("[preprocess_dataset] Loading dataset", verbose)
        data = load_dataset(dataset_name)
        _log("[preprocess_dataset] Dataset loaded", verbose)
        data = unify_format(data)
    except:
        print(f'{dataset_name} is not available on huggingface.')
        exit(1)

    _log("[preprocess_dataset] Combining splits", verbose)
    data = combine_dataset(data)
    seed_everything(seed)
    _log(f"[preprocess_dataset] Seeded with {seed}", verbose)

    if 'celeba' in dataset_name:
        _log("[preprocess_dataset] Sorting identities by frequency", verbose)
        data = sort_identity(data, num_proc=num_proc, verbose=verbose)
    else:
        _log("[preprocess_dataset] Shuffling identities", verbose)
        data = shuffle_identity(data, num_proc=num_proc, verbose=verbose)

    _log("[preprocess_dataset] Preprocessing images", verbose)
    data = data.map(
        _preprocess_images_batch,
        batched=True,
        num_proc=num_proc
    )
    _log("[preprocess_dataset] Image preprocessing complete", verbose)

    _log("[preprocess_dataset] Extracting public/private splits", verbose)
    public, private = extract_public(data, num_identities, num_proc=num_proc)
    _log("[preprocess_dataset] Splitting public dataset into train/valid/test", verbose)
    public = split_dataset(public, seed=seed)

    dataset_id = dataset_name.split('/')[-1]
    save_dataset(
        public,
        f'dataset/public/{dataset_id}',
        max_images_per_identity=max_images_per_identity,
        max_images_per_split=max_images_per_split,
        verbose=verbose,
    )
    save_dataset(
        private,
        f'dataset/private/{dataset_id}',
        max_images_per_identity=max_images_per_identity,
        max_images_per_split=max_images_per_split,
        verbose=verbose,
    )
    _log("[preprocess_dataset] Dataset saved successfully", verbose)

    return public, private

if __name__ == '__main__':
    preprocess_dataset('flwrlabs/celeba', 1000, verbose=True)
