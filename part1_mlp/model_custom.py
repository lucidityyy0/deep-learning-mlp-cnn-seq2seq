import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim=30, hidden=[64, 32], out_dim=2, p=0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def apply_init(model, strategy="xavier"):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if strategy == "gaussian":
                nn.init.normal_(m.weight, mean=0, std=0.01)
            elif strategy == "constant":
                nn.init.constant_(m.weight, val=0.1)
            elif strategy == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif strategy == "kaiming":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP().to(device)
    print(model)
    x = torch.randn(8, 30).to(device)
    print("Output shape:", model(x).shape)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
