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
from data import get_cifar10_loaders, get_cifar10c_loader
from train_utils import evaluate
from metrics import (
    expected_calibration_error,
    compute_sharpness_proxy,
    extract_features,
    compute_feature_stability,
    compute_effective_rank,
    compute_class_separation,
    bootstrap_ci,
)


def run_summary_analysis(batch_size=128):
    make_dirs()

    _, test_loader = get_cifar10_loaders(batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()

    summary_results = []

    for config in EXPERIMENT_CONFIGS:
        experiment_name = config["experiment_name"]

        print(f"\nRunning final analysis for {experiment_name}")

        model = get_resnet18_cifar10().to(DEVICE)
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{experiment_name}_best.pt")
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        model.eval()

        clean_loss, clean_acc, probs, labels = evaluate(
            model, test_loader, criterion, DEVICE
        )

        ece = expected_calibration_error(probs, labels)

        sharpness_result = compute_sharpness_proxy(
            model,
            test_loader,
            criterion,
            DEVICE,
            epsilon=0.01,
            n_samples=5,
            seed=config["seed"],
        )

        clean_features, clean_labels = extract_features(model, test_loader, DEVICE)

        effective_rank = compute_effective_rank(clean_features)

        separation_result = compute_class_separation(clean_features, clean_labels)

        # Main representation stability uses Gaussian noise severity 3.
        corrupted_loader = get_cifar10c_loader(
            root=CIFAR10C_DIR,
            corruption="gaussian_noise",
            severity=3,
            batch_size=batch_size,
        )

        corrupted_features, _ = extract_features(model, corrupted_loader, DEVICE)

        stability_result = compute_feature_stability(
            clean_features,
            corrupted_features,
        )

        ood_path = os.path.join(RESULTS_DIR, f"{experiment_name}_cifar10c_results.csv")
        ood_df = pd.read_csv(ood_path)
        mean_cifar10c_acc = ood_df["accuracy"].mean()

        ci_lower, ci_upper = bootstrap_ci(ood_df["accuracy"].values)

        summary_results.append({
            "experiment": experiment_name,
            "optimizer": config["optimizer_name"],
            "rho": config["rho"],
            "clean_accuracy": clean_acc,
            "clean_loss": clean_loss,
            "mean_cifar10c_accuracy": mean_cifar10c_acc,
            "robustness_gap": clean_acc - mean_cifar10c_acc,
            "cifar10c_ci_lower": ci_lower,
            "cifar10c_ci_upper": ci_upper,
            "ece": ece,
            "sharpness": sharpness_result["sharpness"],
            "sharpness_std": sharpness_result["sharpness_std"],
            "sharpness_trials": sharpness_result["sharpness_trials"],
            "effective_rank": effective_rank,
            "class_separation_ratio": separation_result["separation_ratio"],
            "feature_stability_cosine": stability_result["mean_cosine_similarity"],
        })

    summary_df = pd.DataFrame(summary_results)
    save_path = os.path.join(RESULTS_DIR, "final_summary_results.csv")
    summary_df.to_csv(save_path, index=False)

    print(f"Saved summary results to {save_path}")
    print(summary_df)

    return summary_df


def run_corruption_wise_feature_stability(batch_size=128):
    _, test_loader = get_cifar10_loaders(batch_size=batch_size)

    rows = []

    for config in EXPERIMENT_CONFIGS:
        experiment_name = config["experiment_name"]
        print(f"\nCorruption-wise feature stability for {experiment_name}")

        model = get_resnet18_cifar10().to(DEVICE)
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{experiment_name}_best.pt")
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        model.eval()

        clean_features, _ = extract_features(model, test_loader, DEVICE)

        for corruption in SELECTED_CORRUPTIONS:
            for severity in SEVERITIES:
                corrupted_loader = get_cifar10c_loader(
                    root=CIFAR10C_DIR,
                    corruption=corruption,
                    severity=severity,
                    batch_size=batch_size,
                )

                corrupted_features, _ = extract_features(
                    model,
                    corrupted_loader,
                    DEVICE,
                )

                stability = compute_feature_stability(
                    clean_features,
                    corrupted_features,
                )

                rows.append({
                    "experiment": experiment_name,
                    "optimizer": config["optimizer_name"],
                    "rho": config["rho"],
                    "corruption": corruption,
                    "severity": severity,
                    "feature_stability_cosine": stability["mean_cosine_similarity"],
                    "feature_stability_std": stability["std_cosine_similarity"],
                })

    stability_df = pd.DataFrame(rows)
    save_path = os.path.join(RESULTS_DIR, "corruption_wise_feature_stability.csv")
    stability_df.to_csv(save_path, index=False)

    print(f"Saved corruption-wise feature stability to {save_path}")

    return stability_df


if __name__ == "__main__":
    run_summary_analysis(batch_size=128)
    run_corruption_wise_feature_stability(batch_size=128)
