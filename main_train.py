import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from config import DEVICE, CHECKPOINT_DIR, RESULTS_DIR, EXPERIMENT_CONFIGS
from utils import set_seed, make_dirs
from data import get_cifar10_loaders
from models import get_resnet18_cifar10
from optimizers import get_optimizer
from train_utils import train_one_epoch_standard, train_one_epoch_sam, evaluate


def run_training(
    optimizer_name,
    experiment_name,
    seed=42,
    epochs=30,
    batch_size=128,
    lr=0.1,
    weight_decay=5e-4,
    rho=0.05,
):
    set_seed(seed)
    make_dirs()

    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size)

    model = get_resnet18_cifar10().to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    optimizer = get_optimizer(
        model=model,
        optimizer_name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
        rho=rho if rho is not None else 0.05,
    )

    if optimizer_name == "sam":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer.base_optimizer,
            T_max=epochs,
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
        )

    best_test_acc = 0.0
    logs = []

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs} | Experiment: {experiment_name}")

        if optimizer_name == "sam":
            train_loss, train_acc = train_one_epoch_sam(
                model, train_loader, optimizer, criterion, DEVICE
            )
        else:
            train_loss, train_acc = train_one_epoch_standard(
                model, train_loader, optimizer, criterion, DEVICE
            )

        test_loss, test_acc, _, _ = evaluate(
            model, test_loader, criterion, DEVICE
        )

        scheduler.step()

        logs.append({
            "experiment": experiment_name,
            "optimizer": optimizer_name,
            "rho": rho if optimizer_name == "sam" else None,
            "seed": seed,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        })

        print(
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{experiment_name}_best.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")

    log_df = pd.DataFrame(logs)
    log_path = os.path.join(RESULTS_DIR, f"{experiment_name}_training_log.csv")
    log_df.to_csv(log_path, index=False)

    print(f"\nBest clean test accuracy for {experiment_name}: {best_test_acc:.4f}")

    return model, log_df


def main(epochs=30, batch_size=128):
    for config in EXPERIMENT_CONFIGS:
        run_training(
            optimizer_name=config["optimizer_name"],
            experiment_name=config["experiment_name"],
            seed=config["seed"],
            epochs=epochs,
            batch_size=batch_size,
            lr=config["lr"],
            rho=config["rho"],
        )


if __name__ == "__main__":
    main(epochs=30, batch_size=128)
