import os
import time
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from lenet import LeNet5, MLPBaseline, get_fashion_mnist_loaders, CLASSES

SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(model, train_loader, test_loader, epochs=20, lr=1e-3, model_name="lenet"):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses, test_accs = [], []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * len(y)
        train_losses.append(ep_loss / len(train_loader.dataset))

        acc = evaluate(model, test_loader)
        test_accs.append(acc)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d} | Loss: {train_losses[-1]:.4f} | Test Acc: {acc:.4f}")

    elapsed = time.time() - t0
    torch.save(model.state_dict(), os.path.join(SAVED_MODELS_DIR, f"{model_name}.pth"))
    return train_losses, test_accs, elapsed


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        preds = model(X).argmax(1)
        correct += (preds == y).sum().item()
        total += len(y)
    return correct / total


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def run_experiment(config, train_loader, test_loader, epochs=20, name="exp"):
    print(f"\n--- Experiment: {name} ---")
    model = LeNet5(**config).to(device)
    print(f"  Params: {count_params(model):,}")
    losses, accs, t = train_model(model, train_loader, test_loader, epochs=epochs, model_name=name)
    best_acc = max(accs)
    print(f"  Best Test Acc: {best_acc:.4f} | Time: {t:.1f}s")
    return {"name": name, "params": count_params(model), "best_acc": best_acc, "train_time": t,
            "losses": losses, "accs": accs}


def visualise_feature_maps(model, test_loader, n_maps=6):
    activation = {}

    def hook(name):
        def fn(m, i, o):
            activation[name] = o.detach()
        return fn

    model.conv1.register_forward_hook(hook("conv1"))
    model.eval()
    sample, _ = next(iter(test_loader))
    sample = sample[:1].to(device)
    with torch.no_grad():
        _ = model(sample)

    fmaps = activation["conv1"][0].cpu()
    n = min(n_maps, fmaps.shape[0])
    fig, axes = plt.subplots(1, n, figsize=(14, 2))
    for i, ax in enumerate(axes):
        ax.imshow(fmaps[i], cmap="viridis")
        ax.axis("off")
        ax.set_title(f"Filter {i+1}")
    plt.suptitle("Conv1 feature maps")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "feature_maps_conv1.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(sample[0, 0].cpu(), cmap="gray")
    ax.axis("off")
    ax.set_title("Input")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "feature_maps_input.png"))
    plt.close()


def plot_comparison(results_list, fname="cnn_experiment_comparison.png"):
    names = [r["name"] for r in results_list]
    accs = [r["best_acc"] for r in results_list]
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(names, accs, color="steelblue")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel("Best Test Accuracy")
    ax.set_title("CNN experiment comparison (Fashion-MNIST)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, fname)
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def main(epochs=20):
    set_seed(42)
    train_loader, test_loader = get_fashion_mnist_loaders(batch_size=64)
    results = []

    r = run_experiment({}, train_loader, test_loader, epochs=epochs, name="baseline_lenet")
    results.append(r)

    r = run_experiment({"conv1_padding": 0}, train_loader, test_loader, epochs=epochs, name="padding_valid")
    results.append(r)

    r = run_experiment({"conv1_stride": 2}, train_loader, test_loader, epochs=epochs, name="stride2")
    results.append(r)

    r = run_experiment({"pool_type": "max"}, train_loader, test_loader, epochs=epochs, name="maxpool")
    results.append(r)

    r = run_experiment({"filter_scale": 0.5}, train_loader, test_loader, epochs=epochs, name="filters_half")
    results.append(r)

    r = run_experiment({"filter_scale": 2.0}, train_loader, test_loader, epochs=epochs, name="filters_double")
    results.append(r)

    r = run_experiment({"use_1x1": True}, train_loader, test_loader, epochs=epochs, name="with_1x1")
    results.append(r)

    print("\n--- MLP Baseline (784, 256, 128, 10) ---")
    mlp = MLPBaseline().to(device)
    print(f"  Params: {count_params(mlp):,}")
    losses_mlp, accs_mlp, t_mlp = train_model(mlp, train_loader, test_loader, epochs=epochs, model_name="mlp_baseline")
    results.append({"name": "mlp_baseline", "params": count_params(mlp),
                    "best_acc": max(accs_mlp), "train_time": t_mlp,
                    "losses": losses_mlp, "accs": accs_mlp})
    print(f"  Best Test Acc: {max(accs_mlp):.4f} | Time: {t_mlp:.1f}s")

    baseline_model = LeNet5().to(device)
    baseline_model.load_state_dict(torch.load(
        os.path.join(SAVED_MODELS_DIR, "baseline_lenet.pth"), map_location=device,
        weights_only=False
    ))
    visualise_feature_maps(baseline_model, test_loader)

    df = pd.DataFrame([{
        "Experiment": r["name"],
        "Params": r["params"],
        "Best Test Acc": f"{r['best_acc']:.4f}",
        "Train Time (s)": f"{r['train_time']:.1f}",
    } for r in results])
    df.to_csv(os.path.join(RESULTS_DIR, "cnn_comparison.csv"), index=False)
    print("\n=== Experiment Summary ===")
    print(df.to_string(index=False))

    plot_comparison(results)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(results[0]["losses"], label="Baseline Train Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("LeNet baseline training loss")
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "lenet_baseline_loss.png"))
    plt.close()


if __name__ == "__main__":
    main()
