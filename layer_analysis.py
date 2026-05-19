import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetLayerFeatureExtractor(nn.Module):
    """
    Extract features from multiple ResNet-18 layers.
    We use global average pooling to convert each feature map to a vector.
    """

    def __init__(self, model):
        super().__init__()

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool

    def forward(self, x):
        features = {}

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        features["layer1"] = torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)

        x = self.layer2(x)
        features["layer2"] = torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)

        x = self.layer3(x)
        features["layer3"] = torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)

        x = self.layer4(x)
        features["layer4"] = torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)

        x = self.avgpool(x)
        features["penultimate"] = torch.flatten(x, 1)

        return features


@torch.no_grad()
def extract_layerwise_features(model, loader, device):
    model.eval()

    extractor = ResNetLayerFeatureExtractor(model).to(device)
    extractor.eval()

    all_features = {
        "layer1": [],
        "layer2": [],
        "layer3": [],
        "layer4": [],
        "penultimate": [],
    }

    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        features = extractor(images)

        for layer_name in all_features.keys():
            all_features[layer_name].append(features[layer_name].cpu())

        all_labels.append(labels.cpu())

    for layer_name in all_features.keys():
        all_features[layer_name] = torch.cat(all_features[layer_name], dim=0)

    all_labels = torch.cat(all_labels, dim=0)

    return all_features, all_labels


def compute_layerwise_stability(clean_features_dict, corrupted_features_dict):
    results = {}

    for layer_name in clean_features_dict.keys():
        clean_features = F.normalize(clean_features_dict[layer_name], dim=1)
        corrupted_features = F.normalize(corrupted_features_dict[layer_name], dim=1)

        cosine_similarity = (clean_features * corrupted_features).sum(dim=1)

        results[layer_name] = {
            "mean_cosine_similarity": cosine_similarity.mean().item(),
            "std_cosine_similarity": cosine_similarity.std().item(),
        }

    return results


def linear_cka(X, Y):
    """
    Linear CKA between two representation matrices.
    X: [n_samples, feature_dim]
    Y: [n_samples, feature_dim]
    """
    X = X.float()
    Y = Y.float()

    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    hsic = torch.norm(X.T @ Y, p="fro") ** 2
    var1 = torch.norm(X.T @ X, p="fro")
    var2 = torch.norm(Y.T @ Y, p="fro")

    cka = hsic / (var1 * var2 + 1e-12)

    return cka.item()
