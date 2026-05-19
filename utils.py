import os
import random
import numpy as np
import torch

from config import DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, FIGURES_DIR, CIFAR10C_DIR

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic setting improves reproducibility.
    # It may reduce speed slightly.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_dirs():
    for path in [DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, FIGURES_DIR, CIFAR10C_DIR]:
        os.makedirs(path, exist_ok=True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
