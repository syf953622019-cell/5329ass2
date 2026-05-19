import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import RESULTS_DIR, FIGURES_DIR, EXPERIMENT_CONFIGS


def save_bar(df, x, y, title, ylabel, filename, ylim=None):
    plt.figure(figsize=(8, 4))
    plt.bar(df[x], df[y])
    plt.xlabel("Experiment")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=30, ha="right")

    if ylim is not None:
        plt.ylim(*ylim)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300)
    plt.show()


def plot_summary_bars():
    summary_path = os.path.join(RESULTS_DIR, "final_summary_results.csv")
    summary_df = pd.read_csv(summary_path)

    save_bar(
        summary_df,
        "experiment",
        "clean_accuracy",
        "Clean CIFAR-10 Accuracy",
        "Clean Accuracy",
        "clean_accuracy_comparison.png",
        ylim=(0, 1),
    )

    save_bar(
        summary_df,
        "experiment",
        "mean_cifar10c_accuracy",
        "Mean CIFAR-10-C Accuracy",
        "Mean CIFAR-10-C Accuracy",
        "cifar10c_accuracy_comparison.png",
        ylim=(0, 1),
    )

    save_bar(
        summary_df,
        "experiment",
        "robustness_gap",
        "Robustness Gap",
        "Clean-OOD Accuracy Gap",
        "robustness_gap.png",
    )

    save_bar(
        summary_df,
        "experiment",
        "sharpness",
        "Sharpness Proxy",
        "Sharpness Proxy",
        "sharpness_comparison.png",
    )

    save_bar(
        summary_df,
        "experiment",
        "feature_stability_cosine",
        "Representation Stability under Gaussian Noise",
        "Feature Cosine Similarity",
        "feature_stability_comparison.png",
        ylim=(0, 1),
    )


def plot_cifar10c_by_severity():
    experiment_names = [item["experiment_name"] for item in EXPERIMENT_CONFIGS]

    all_ood_results = []

    for experiment_name in experiment_names:
        df = pd.read_csv(os.path.join(RESULTS_DIR, f"{experiment_name}_cifar10c_results.csv"))
        all_ood_results.append(df)

    ood_df = pd.concat(all_ood_results, ignore_index=True)

    severity_summary = (
        ood_df
        .groupby(["experiment", "severity"])["accuracy"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(8, 5))

    for experiment_name in experiment_names:
        subset = severity_summary[severity_summary["experiment"] == experiment_name]

        plt.plot(
            subset["severity"],
            subset["accuracy"],
            marker="o",
            label=experiment_name,
        )

    plt.xlabel("Corruption Severity")
    plt.ylabel("Mean CIFAR-10-C Accuracy")
    plt.title("OOD Robustness across Corruption Severity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "cifar10c_severity_accuracy.png"), dpi=300)
    plt.show()


def plot_corruption_wise_accuracy():
    experiment_names = [item["experiment_name"] for item in EXPERIMENT_CONFIGS]

    all_ood_results = []

    for experiment_name in experiment_names:
        df = pd.read_csv(os.path.join(RESULTS_DIR, f"{experiment_name}_cifar10c_results.csv"))
        all_ood_results.append(df)

    ood_df = pd.concat(all_ood_results, ignore_index=True)

    corruption_summary = (
        ood_df
        .groupby(["experiment", "corruption"])["accuracy"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(10, 5))

    for experiment_name in experiment_names:
        subset = corruption_summary[corruption_summary["experiment"] == experiment_name]

        plt.plot(
            subset["corruption"],
            subset["accuracy"],
            marker="o",
            label=experiment_name,
        )

    plt.xlabel("Corruption Type")
    plt.ylabel("Mean Accuracy")
    plt.title("Corruption-wise OOD Robustness")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "corruption_wise_accuracy.png"), dpi=300)
    plt.show()


def plot_mechanism_scatter():
    summary_df = pd.read_csv(os.path.join(RESULTS_DIR, "final_summary_results.csv"))

    plt.figure(figsize=(6, 4))
    plt.scatter(summary_df["sharpness"], summary_df["mean_cifar10c_accuracy"])

    for _, row in summary_df.iterrows():
        plt.text(row["sharpness"], row["mean_cifar10c_accuracy"], row["experiment"], fontsize=8)

    plt.xlabel("Sharpness Proxy")
    plt.ylabel("Mean CIFAR-10-C Accuracy")
    plt.title("Sharpness vs OOD Robustness")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "sharpness_vs_ood_accuracy.png"), dpi=300)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.scatter(summary_df["feature_stability_cosine"], summary_df["mean_cifar10c_accuracy"])

    for _, row in summary_df.iterrows():
        plt.text(row["feature_stability_cosine"], row["mean_cifar10c_accuracy"], row["experiment"], fontsize=8)

    plt.xlabel("Feature Stability Cosine Similarity")
    plt.ylabel("Mean CIFAR-10-C Accuracy")
    plt.title("Representation Stability vs OOD Robustness")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "stability_vs_ood_accuracy.png"), dpi=300)
    plt.show()


def plot_layerwise_analysis():
    layer_path = os.path.join(RESULTS_DIR, "layerwise_feature_stability.csv")
    cka_path = os.path.join(RESULTS_DIR, "layerwise_cka_clean_corrupted.csv")

    if os.path.exists(layer_path):
        layer_df = pd.read_csv(layer_path)

        plt.figure(figsize=(8, 5))

        for experiment_name in layer_df["experiment"].unique():
            subset = layer_df[layer_df["experiment"] == experiment_name]

            plt.plot(
                subset["layer"],
                subset["mean_cosine_similarity"],
                marker="o",
                label=experiment_name,
            )

        plt.xlabel("Network Layer")
        plt.ylabel("Clean-Corrupted Feature Cosine Similarity")
        plt.title("Layer-wise Representation Stability")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "layerwise_representation_stability.png"), dpi=300)
        plt.show()

    if os.path.exists(cka_path):
        cka_df = pd.read_csv(cka_path)

        plt.figure(figsize=(8, 5))

        for experiment_name in cka_df["experiment"].unique():
            subset = cka_df[cka_df["experiment"] == experiment_name]

            plt.plot(
                subset["layer"],
                subset["cka_clean_corrupted"],
                marker="o",
                label=experiment_name,
            )

        plt.xlabel("Network Layer")
        plt.ylabel("Linear CKA")
        plt.title("Layer-wise Clean-Corrupted Representation Similarity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "layerwise_cka.png"), dpi=300)
        plt.show()


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    plot_summary_bars()
    plot_cifar10c_by_severity()
    plot_corruption_wise_accuracy()
    plot_mechanism_scatter()
    plot_layerwise_analysis()


if __name__ == "__main__":
    main()
