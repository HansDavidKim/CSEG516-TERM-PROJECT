import random
import numpy as np
import torch


def set_random_seed(seed: int, deterministic: bool):
    """Seed all major RNGs so experiments become reproducible.

    Args:
        seed: Base value applied to Python, NumPy, and PyTorch generators.
        deterministic: If True, force CuDNN to run only deterministic kernels
            (disables heuristics that pick the fastest algorithm).

    Note:
        Even with deterministic mode enabled, some third-party CUDA ops may
        remain stochastic. This helper covers the standard PyTorch stack.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Keep multi-GPU kernels in sync.

    if deterministic:
        # Trade execution speed for reproducible CuDNN behavior.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
