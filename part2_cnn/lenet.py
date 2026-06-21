import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def get_fashion_mnist_loaders(batch_size=64, augment=True):
    mean, std = (0.2860,), (0.3530,)

    train_transforms = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    test_transforms = [transforms.ToTensor(), transforms.Normalize(mean, std)]

    if augment:
        train_transforms = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(28, padding=4),
        ] + train_transforms

    train_ds = torchvision.datasets.FashionMNIST(
        root=DATA_DIR, train=True, download=True,
        transform=transforms.Compose(train_transforms)
    )
    test_ds = torchvision.datasets.FashionMNIST(
        root=DATA_DIR, train=False, download=True,
        transform=transforms.Compose(test_transforms)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


class LeNet5(nn.Module):
    def __init__(self, num_classes=10, pool_type="avg",
                 filter_scale=1.0, use_1x1=False,
                 conv1_padding=2, conv1_stride=1, conv2_stride=1):
        super().__init__()
        c1 = max(1, int(6 * filter_scale))
        c2 = max(1, int(16 * filter_scale))

        Pool = nn.AvgPool2d if pool_type == "avg" else nn.MaxPool2d

        self.use_1x1 = use_1x1
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, c1, kernel_size=5, padding=conv1_padding,
                               stride=conv1_stride)
        self.pool1 = Pool(2, 2)
        self.conv1x1 = nn.Conv2d(c1, c1, kernel_size=1) if use_1x1 else nn.Identity()
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=5, stride=conv2_stride)
        self.pool2 = Pool(2, 2)

        self._flat_size = self._get_flat_size()

        self.fc1 = nn.Linear(self._flat_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def _get_flat_size(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 28, 28)
            return self._features(dummy).shape[1]

    def _features(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.conv1x1(x)
        x = self.pool2(self.relu(self.conv2(x)))
        return x.view(x.size(0), -1)

    def forward(self, x):
        x = self._features(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class MLPBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet5().to(device)
    print(model)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,}")
    x = torch.randn(2, 1, 28, 28).to(device)
    print("Output shape:", model(x).shape)
