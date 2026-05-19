import os
import pandas as pd

from config import RESULTS_DIR, EXPERIMENT_CONFIGS


def main():
    experiment_names = [item["experiment_name"] for item in EXPERIMENT_CONFIGS]

    all_ood_results = []

    for experiment_name in experiment_names:
        path = os.path.join(RESULTS_DIR, f"{experiment_name}_cifar10c_results.csv")
        df = pd.read_csv(path)
        df["experiment"] = experiment_name
        all_ood_results.append(df)

    ood_df = pd.concat(all_ood_results, ignore_index=True)

    stability_path = os.path.join(RESULTS_DIR, "corruption_wise_feature_stability.csv")
    stability_df = pd.read_csv(stability_path)

    merged_df = pd.merge(
        ood_df,
        stability_df,
        on=["experiment", "corruption", "severity"],
        how="inner",
    )

    merged_df.rename(columns={"accuracy": "ood_accuracy"}, inplace=True)

    pearson_corr = merged_df["ood_accuracy"].corr(
        merged_df["feature_stability_cosine"],
        method="pearson",
    )

    spearman_corr = merged_df["ood_accuracy"].corr(
        merged_df["feature_stability_cosine"],
        method="spearman",
    )

    print("Pearson correlation between OOD accuracy and feature stability:", pearson_corr)
    print("Spearman correlation between OOD accuracy and feature stability:", spearman_corr)

    save_path = os.path.join(RESULTS_DIR, "ood_accuracy_feature_stability_merged.csv")
    merged_df.to_csv(save_path, index=False)

    summary_path = os.path.join(RESULTS_DIR, "mechanism_correlation_summary.csv")
    pd.DataFrame([
        {
            "metric_pair": "ood_accuracy_vs_feature_stability",
            "pearson": pearson_corr,
            "spearman": spearman_corr,
        }
    ]).to_csv(summary_path, index=False)

    print(f"Saved merged analysis to {save_path}")
    print(f"Saved correlation summary to {summary_path}")


if __name__ == "__main__":
    main()
