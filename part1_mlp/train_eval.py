import os
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)

from data_prep import load_and_prepare
from model_sequential import build_sequential_mlp, apply_init as seq_init
from model_custom import MLP, apply_init as custom_init

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


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc, np.array(all_preds), np.array(all_labels)


def train(model, train_loader, val_loader, criterion, optimizer,
          epochs=100, patience=10, model_name="best_mlp"):
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion)

        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        train_accs.append(tr_acc)
        val_accs.append(va_acc)

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(SAVED_MODELS_DIR, f"{model_name}.pth"))
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} "
                  f"| Val Loss: {va_loss:.4f} Acc: {va_acc:.4f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    return train_losses, val_losses, train_accs, val_accs


def plot_loss_curves(train_losses, val_losses, title="Loss Curves", fname="loss_curves.png"):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, fname)
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def plot_confusion_matrix(labels, preds, title="Confusion Matrix", fname="confusion_matrix.png"):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Malignant", "Benign"],
                yticklabels=["Malignant", "Benign"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, fname)
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def print_metrics(labels, preds, split="Test"):
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="binary")
    rec = recall_score(labels, preds, average="binary")
    f1 = f1_score(labels, preds, average="binary")
    print(f"\n--- {split} Metrics ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def run_init_comparison(train_loader, val_loader, test_loader, criterion, epochs=100):
    strategies = ["gaussian", "constant", "xavier", "kaiming"]
    results = {}
    for strat in strategies:
        print(f"\n=== Init strategy: {strat.upper()} ===")
        model = MLP().to(device)
        custom_init(model, strat)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        tr_losses, va_losses, _, _ = train(
            model, train_loader, val_loader, criterion, optimizer,
            epochs=epochs, patience=10, model_name=f"mlp_{strat}"
        )
        model.load_state_dict(torch.load(
            os.path.join(SAVED_MODELS_DIR, f"mlp_{strat}.pth"),
            map_location=device, weights_only=False
        ))
        _, _, preds, labels = evaluate(model, test_loader, criterion)
        metrics = print_metrics(labels, preds, split=f"Test ({strat})")
        results[strat] = metrics
        plot_loss_curves(tr_losses, va_losses,
                         title=f"Loss curves ({strat})",
                         fname=f"loss_curves_{strat}.png")
    return results


def main():
    set_seed(42)
    train_loader, val_loader, test_loader, class_weights, _ = load_and_prepare()
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    print("\n=== Sequential Model ===")
    model_seq = build_sequential_mlp(device)
    seq_init(model_seq, "xavier")
    optimizer = torch.optim.Adam(model_seq.parameters(), lr=1e-3, weight_decay=1e-4)
    tr_l, va_l, _, _ = train(
        model_seq, train_loader, val_loader, criterion, optimizer,
        epochs=100, patience=10, model_name="best_mlp_sequential"
    )
    plot_loss_curves(tr_l, va_l, title="Sequential MLP loss curves",
                     fname="loss_curves_sequential.png")

    model_seq.load_state_dict(torch.load(
        os.path.join(SAVED_MODELS_DIR, "best_mlp_sequential.pth"), map_location=device, weights_only=False
    ))
    model_seq.eval()
    _, _, preds, labels = evaluate(model_seq, test_loader, criterion)
    print_metrics(labels, preds, "Test (Sequential)")
    plot_confusion_matrix(labels, preds,
                          title="Sequential MLP confusion matrix",
                          fname="confusion_matrix_sequential.png")

    print("\n=== Initialisation Strategy Comparison ===")
    run_init_comparison(train_loader, val_loader, test_loader, criterion)

    print("\n=== Model Save / Reload Demo ===")
    model_reload = MLP().to(device)
    model_reload.load_state_dict(torch.load(
        os.path.join(SAVED_MODELS_DIR, "mlp_xavier.pth"), map_location=device, weights_only=False
    ))
    model_reload.eval()
    _, _, preds_r, labels_r = evaluate(model_reload, test_loader, criterion)
    print_metrics(labels_r, preds_r, "Test (Reloaded Xavier model)")


if __name__ == "__main__":
    main()
