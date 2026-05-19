import torch.nn as nn
from torchvision.models import resnet18


def get_resnet18_cifar10(num_classes=10):
    """
    Standard ResNet-18 is designed for ImageNet-sized images.
    CIFAR-10 images are 32x32, so we use a smaller first conv layer
    and remove the initial maxpool.
    """
    model = resnet18(weights=None)

    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
