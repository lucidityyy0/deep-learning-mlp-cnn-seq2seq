import os
import time
import math
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class RecurrentModel(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hid_dim=256,
                 n_layers=2, dropout=0.5, cell="lstm"):
        super().__init__()
        self.cell_type = cell
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        Cell = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = Cell(
            emb_dim, hid_dim, num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hid_dim, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        out, _ = self.rnn(x)
        return self.fc(out)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def train_epoch(model, loader, criterion, optimizer, grad_clip=1.0):
    model.train()
    total_loss, total_tokens = 0.0, 0
    grad_norms = []

    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        inp = tgt[:, :-1]
        lbl = tgt[:, 1:]
        if inp.shape[1] == 0:
            continue

        optimizer.zero_grad()
        logits = model(inp)
        B, T, V = logits.shape
        loss = criterion(logits.reshape(B * T, V), lbl.reshape(B * T))
        loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        grad_norms.append(norm.item())
        optimizer.step()

        mask = lbl.reshape(-1) != criterion.ignore_index
        total_loss += loss.item() * mask.sum().item()
        total_tokens += mask.sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20))
    return avg_loss, perplexity, grad_norms


@torch.no_grad()
def evaluate_lm(model, loader, criterion):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        inp = tgt[:, :-1]
        lbl = tgt[:, 1:]
        if inp.shape[1] == 0:
            continue
        logits = model(inp)
        B, T, V = logits.shape
        loss = criterion(logits.reshape(B * T, V), lbl.reshape(B * T))
        mask = lbl.reshape(-1) != criterion.ignore_index
        total_loss += loss.item() * mask.sum().item()
        total_tokens += mask.sum().item()
    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss, math.exp(min(avg_loss, 20))


def train_and_compare(train_loader, val_loader, tgt_vocab, epochs=15):
    cells = ["rnn", "lstm", "gru"]
    summary = []
    all_train_ppl = {}
    all_val_ppl = {}
    all_grad_norms = {}

    pad_idx = tgt_vocab.pad_idx
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    for cell in cells:
        print(f"\n=== Cell type: {cell.upper()} ===")
        model = RecurrentModel(
            vocab_size=len(tgt_vocab), emb_dim=128, hid_dim=256,
            n_layers=2, dropout=0.5, cell=cell
        ).to(device)
        print(f"  Parameters: {model.count_params():,}")
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

        tr_ppls, va_ppls = [], []
        all_gnorms = []
        t0 = time.time()

        for epoch in range(1, epochs + 1):
            tr_loss, tr_ppl, gnorms = train_epoch(model, train_loader, criterion, optimizer)
            va_loss, va_ppl = evaluate_lm(model, val_loader, criterion)
            tr_ppls.append(tr_ppl)
            va_ppls.append(va_ppl)
            all_gnorms.extend(gnorms)
            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:2d} | Train PPL: {tr_ppl:.2f} | Val PPL: {va_ppl:.2f}")

        elapsed = time.time() - t0
        torch.save(model.state_dict(),
                   os.path.join(SAVED_MODELS_DIR, f"rnn_{cell}.pth"))
        summary.append({
            "Cell": cell.upper(),
            "Params": model.count_params(),
            "Final Train PPL": f"{tr_ppls[-1]:.2f}",
            "Final Val PPL": f"{va_ppls[-1]:.2f}",
            "Train Time (s)": f"{elapsed:.1f}",
        })
        all_train_ppl[cell] = tr_ppls
        all_val_ppl[cell] = va_ppls
        all_grad_norms[cell] = all_gnorms

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for cell in cells:
        axes[0].plot(all_train_ppl[cell], label=cell.upper())
        axes[1].plot(all_val_ppl[cell], label=cell.upper())
    for ax, title in zip(axes, ["Train Perplexity", "Val Perplexity"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel("Perplexity")
        ax.set_title(title); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "rnn_lstm_gru_perplexity.png"))
    plt.close()

    plt.figure(figsize=(10, 3))
    gnorms = all_grad_norms.get("lstm", [])
    plt.plot(gnorms[:200], alpha=0.7)
    plt.axhline(1.0, color="red", linestyle="--", label="Clip threshold")
    plt.xlabel("Step"); plt.ylabel("Grad Norm")
    plt.title("LSTM gradient norms and clipping")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "gradient_norms_lstm.png"))
    plt.close()

    df = pd.DataFrame(summary)
    print("\n=== RNN / LSTM / GRU Comparison ===")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(RESULTS_DIR, "rnn_comparison.csv"), index=False)
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_prep_seq import get_loaders
    random.seed(42)
    torch.manual_seed(42)
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab, _ = get_loaders()
    train_and_compare(train_loader, val_loader, tgt_vocab, epochs=5)
