import os
import pandas as pd
import torch

from config import (
    DEVICE,
    CHECKPOINT_DIR,
    RESULTS_DIR,
    CIFAR10C_DIR,
    EXPERIMENT_CONFIGS,
)
from models import get_resnet18_cifar10
from data import get_cifar10_loaders, get_cifar10c_loader
from layer_analysis import (
    extract_layerwise_features,
    compute_layerwise_stability,
    linear_cka,
)


def main(batch_size=128):
    _, test_loader = get_cifar10_loaders(batch_size=batch_size)

    layerwise_rows = []
    cka_rows = []

    for config in EXPERIMENT_CONFIGS:
        experiment_name = config["experiment_name"]

        print(f"\nLayer-wise representation analysis for {experiment_name}")

        model = get_resnet18_cifar10().to(DEVICE)
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{experiment_name}_best.pt")
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        model.eval()

        clean_features_dict, _ = extract_layerwise_features(
            model,
            test_loader,
            DEVICE,
        )

        corrupted_loader = get_cifar10c_loader(
            root=CIFAR10C_DIR,
            corruption="gaussian_noise",
            severity=3,
            batch_size=batch_size,
        )

        corrupted_features_dict, _ = extract_layerwise_features(
            model,
            corrupted_loader,
            DEVICE,
        )

        stability = compute_layerwise_stability(
            clean_features_dict,
            corrupted_features_dict,
        )

        for layer_name, values in stability.items():
            layerwise_rows.append({
                "experiment": experiment_name,
                "optimizer": config["optimizer_name"],
                "rho": config["rho"],
                "layer": layer_name,
                "mean_cosine_similarity": values["mean_cosine_similarity"],
                "std_cosine_similarity": values["std_cosine_similarity"],
            })

            cka_score = linear_cka(
                clean_features_dict[layer_name],
                corrupted_features_dict[layer_name],
            )

            cka_rows.append({
                "experiment": experiment_name,
                "optimizer": config["optimizer_name"],
                "rho": config["rho"],
                "layer": layer_name,
                "cka_clean_corrupted": cka_score,
            })

    layerwise_df = pd.DataFrame(layerwise_rows)
    cka_df = pd.DataFrame(cka_rows)

    layerwise_path = os.path.join(RESULTS_DIR, "layerwise_feature_stability.csv")
    cka_path = os.path.join(RESULTS_DIR, "layerwise_cka_clean_corrupted.csv")

    layerwise_df.to_csv(layerwise_path, index=False)
    cka_df.to_csv(cka_path, index=False)

    print(f"Saved layer-wise stability to {layerwise_path}")
    print(f"Saved CKA results to {cka_path}")


if __name__ == "__main__":
    main(batch_size=128)
