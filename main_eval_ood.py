import os
import pandas as pd
import torch
import torch.nn as nn

from config import (
    DEVICE,
    CHECKPOINT_DIR,
    RESULTS_DIR,
    CIFAR10C_DIR,
    EXPERIMENT_CONFIGS,
    SELECTED_CORRUPTIONS,
    SEVERITIES,
)
from utils import make_dirs
from models import get_resnet18_cifar10
from data import get_cifar10c_loader
from train_utils import evaluate


def evaluate_cifar10c(
    model,
    cifar10c_root,
    corruptions,
    severities,
    batch_size=128,
):
    criterion = nn.CrossEntropyLoss()
    results = []

    for corruption in corruptions:
        for severity in severities:
            loader = get_cifar10c_loader(
                root=cifar10c_root,
                corruption=corruption,
                severity=severity,
                batch_size=batch_size,
            )

            loss, acc, probs, labels = evaluate(
                model, loader, criterion, DEVICE
            )

            results.append({
                "corruption": corruption,
                "severity": severity,
                "loss": loss,
                "accuracy": acc,
            })

            print(
                f"{corruption:20s} | severity {severity} | "
                f"loss {loss:.4f} | acc {acc:.4f}"
            )

    return pd.DataFrame(results)


def main(batch_size=128):
    make_dirs()

    for config in EXPERIMENT_CONFIGS:
        experiment_name = config["experiment_name"]

        print(f"\nEvaluating {experiment_name} on CIFAR-10-C")

        model = get_resnet18_cifar10().to(DEVICE)
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{experiment_name}_best.pt")
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        model.eval()

        ood_df = evaluate_cifar10c(
            model=model,
            cifar10c_root=CIFAR10C_DIR,
            corruptions=SELECTED_CORRUPTIONS,
            severities=SEVERITIES,
            batch_size=batch_size,
        )

        ood_df["experiment"] = experiment_name
        ood_df["optimizer"] = config["optimizer_name"]
        ood_df["rho"] = config["rho"]

        save_path = os.path.join(RESULTS_DIR, f"{experiment_name}_cifar10c_results.csv")
        ood_df.to_csv(save_path, index=False)

        print(f"Saved OOD results to {save_path}")


if __name__ == "__main__":
    main(batch_size=128)
