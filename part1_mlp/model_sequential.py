import torch
import torch.nn as nn


def build_sequential_mlp(device):
    model = nn.Sequential(
        nn.Linear(30, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(32, 2),
    ).to(device)
    return model


def apply_init(model, strategy="xavier"):
    for layer in model:
        if isinstance(layer, nn.Linear):
            if strategy == "gaussian":
                nn.init.normal_(layer.weight, mean=0, std=0.01)
            elif strategy == "constant":
                nn.init.constant_(layer.weight, val=0.1)
            elif strategy == "xavier":
                nn.init.xavier_uniform_(layer.weight)
            elif strategy == "kaiming":
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = build_sequential_mlp(device)
    print(m)
    x = torch.randn(4, 30).to(device)
    print("Output shape:", m(x).shape)
