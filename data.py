import os
import numpy as np
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader

from config import DATA_DIR, CIFAR10_MEAN, CIFAR10_STD


def get_cifar10_loaders(batch_size=128, num_workers=2):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_set = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


class CIFAR10CDataset(Dataset):
    """
    CIFAR-10-C contains 50,000 images per corruption type.
    Each severity level contains 10,000 images.
    severity=1 -> index 0:10000
    severity=2 -> index 10000:20000
    ...
    severity=5 -> index 40000:50000
    """

    def __init__(self, root, corruption, severity, transform=None):
        self.root = root
        self.corruption = corruption
        self.severity = severity
        self.transform = transform

        data_path = os.path.join(root, corruption + ".npy")
        label_path = os.path.join(root, "labels.npy")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Missing corruption file: {data_path}")

        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Missing labels file: {label_path}")

        data = np.load(data_path)
        labels = np.load(label_path)

        start = (severity - 1) * 10000
        end = severity * 10000

        self.data = data[start:end]
        self.labels = labels[start:end]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = int(self.labels[idx])

        image = torchvision.transforms.ToPILImage()(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_cifar10c_loader(root, corruption, severity, batch_size=128, num_workers=2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    dataset = CIFAR10CDataset(
        root=root,
        corruption=corruption,
        severity=severity,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader
