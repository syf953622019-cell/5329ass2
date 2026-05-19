import os
import torch

PROJECT_DIR = "/content/drive/MyDrive/COMP5329ASS2"

DATA_DIR = os.path.join(PROJECT_DIR, "data")
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
CIFAR10C_DIR = os.path.join(PROJECT_DIR, "CIFAR-10-C")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

SELECTED_CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "motion_blur",
    "brightness",
    "contrast",
    "jpeg_compression",
]

SEVERITIES = [1, 2, 3, 4, 5]

EXPERIMENT_CONFIGS = [
    {
        "experiment_name": "sgd_seed42",
        "optimizer_name": "sgd",
        "lr": 0.1,
        "rho": None,
        "seed": 42,
    },
    {
        "experiment_name": "adamw_seed42",
        "optimizer_name": "adamw",
        "lr": 0.001,
        "rho": None,
        "seed": 42,
    },
    {
        "experiment_name": "sam_rho001_seed42",
        "optimizer_name": "sam",
        "lr": 0.1,
        "rho": 0.01,
        "seed": 42,
    },
    {
        "experiment_name": "sam_rho005_seed42",
        "optimizer_name": "sam",
        "lr": 0.1,
        "rho": 0.05,
        "seed": 42,
    },
    {
        "experiment_name": "sam_rho010_seed42",
        "optimizer_name": "sam",
        "lr": 0.1,
        "rho": 0.10,
        "seed": 42,
    },
]
