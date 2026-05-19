import torch
import torch.nn as nn
import torch.nn.functional as F

from train_utils import evaluate


def expected_calibration_error(probs, labels, n_bins=15):
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        in_bin = confidences.gt(lower) * confidences.le(upper)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()


@torch.no_grad()
def add_weight_perturbation(model, epsilon=0.01, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    perturbations = []

    for p in model.parameters():
        if p.requires_grad:
            noise = torch.randn_like(p)
            noise = epsilon * noise / (torch.norm(noise) + 1e-12)
            p.add_(noise)
            perturbations.append(noise)
        else:
            perturbations.append(None)

    return perturbations


@torch.no_grad()
def remove_weight_perturbation(model, perturbations):
    for p, noise in zip(model.parameters(), perturbations):
        if noise is not None:
            p.sub_(noise)


def compute_sharpness_proxy(
    model,
    loader,
    criterion,
    device,
    epsilon=0.01,
    n_samples=5,
    seed=42,
):
    """
    Sharpness proxy:
    average increase in loss after applying multiple small random weight perturbations.
    """
    model.eval()

    clean_loss, clean_acc, _, _ = evaluate(model, loader, criterion, device)

    sharpness_values = []
    perturbed_losses = []
    perturbed_accs = []

    for i in range(n_samples):
        perturbations = add_weight_perturbation(
            model,
            epsilon=epsilon,
            seed=seed + i,
        )

        perturbed_loss, perturbed_acc, _, _ = evaluate(model, loader, criterion, device)

        remove_weight_perturbation(model, perturbations)

        perturbed_losses.append(perturbed_loss)
        perturbed_accs.append(perturbed_acc)
        sharpness_values.append(perturbed_loss - clean_loss)

    sharpness_tensor = torch.tensor(sharpness_values)
    loss_tensor = torch.tensor(perturbed_losses)
    acc_tensor = torch.tensor(perturbed_accs)

    return {
        "clean_loss": clean_loss,
        "perturbed_loss": loss_tensor.mean().item(),
        "perturbed_loss_std": loss_tensor.std(unbiased=False).item(),
        "sharpness": sharpness_tensor.mean().item(),
        "sharpness_std": sharpness_tensor.std(unbiased=False).item(),
        "clean_acc": clean_acc,
        "perturbed_acc": acc_tensor.mean().item(),
        "perturbed_acc_std": acc_tensor.std(unbiased=False).item(),
        "sharpness_trials": n_samples,
    }

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.feature_layers = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        x = self.feature_layers(x)
        x = torch.flatten(x, 1)
        return x


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()

    feature_extractor = ResNetFeatureExtractor(model).to(device)
    feature_extractor.eval()

    all_features = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        features = feature_extractor(images)

        all_features.append(features.cpu())
        all_labels.append(labels.cpu())

    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_features, all_labels


def compute_feature_stability(clean_features, corrupted_features):
    clean_features = F.normalize(clean_features, dim=1)
    corrupted_features = F.normalize(corrupted_features, dim=1)

    cosine_similarity = (clean_features * corrupted_features).sum(dim=1)

    return {
        "mean_cosine_similarity": cosine_similarity.mean().item(),
        "std_cosine_similarity": cosine_similarity.std().item(),
    }


def compute_effective_rank(features):
    features = features - features.mean(dim=0, keepdim=True)

    _, singular_values, _ = torch.svd(features)

    probs = singular_values / singular_values.sum()
    entropy = -(probs * torch.log(probs + 1e-12)).sum()

    effective_rank = torch.exp(entropy)

    return effective_rank.item()


def compute_class_separation(features, labels, num_classes=10):
    class_centers = []
    intra_distances = []

    for c in range(num_classes):
        class_features = features[labels == c]
        center = class_features.mean(dim=0)

        class_centers.append(center)

        intra_distance = torch.norm(class_features - center, dim=1).mean()
        intra_distances.append(intra_distance)

    class_centers = torch.stack(class_centers)

    inter_distances = []

    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            distance = torch.norm(class_centers[i] - class_centers[j])
            inter_distances.append(distance)

    mean_intra = torch.stack(intra_distances).mean()
    mean_inter = torch.stack(inter_distances).mean()

    separation_ratio = mean_inter / (mean_intra + 1e-12)

    return {
        "mean_intra_distance": mean_intra.item(),
        "mean_inter_distance": mean_inter.item(),
        "separation_ratio": separation_ratio.item(),
    }


def bootstrap_ci(values, n_bootstrap=1000, ci=95, seed=42):
    import numpy as np

    rng = np.random.default_rng(seed)
    values = np.array(values)

    boot_means = []

    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means.append(sample.mean())

    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)

    return lower, upper
