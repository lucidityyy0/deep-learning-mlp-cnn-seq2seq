import os
import math
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TF_RATIO = 0.5


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.emb = nn.Embedding(src_vocab_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers,
                           dropout=dropout if n_layers > 1 else 0.0,
                           batch_first=True)

    def forward(self, src):
        embedded = self.emb(src)
        outputs, (h, c) = self.rnn(embedded)
        return h, c


class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.emb = nn.Embedding(tgt_vocab_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers,
                           dropout=dropout if n_layers > 1 else 0.0,
                           batch_first=True)
        self.fc = nn.Linear(hid_dim, tgt_vocab_size)

    def forward(self, input_token, h, c):
        x = self.emb(input_token.unsqueeze(1))
        out, (h, c) = self.rnn(x, (h, c))
        logits = self.fc(out.squeeze(1))
        return logits, h, c


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, sos_idx, eos_idx, pad_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

    def forward(self, src, tgt, teacher_forcing_ratio=TF_RATIO):
        B, T_tgt = tgt.shape
        h, c = self.encoder(src)
        vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(B, T_tgt - 1, vocab_size).to(device)

        input_t = tgt[:, 0]
        for t in range(T_tgt - 1):
            logit, h, c = self.decoder(input_t, h, c)
            outputs[:, t, :] = logit
            use_teacher = torch.rand(1).item() < teacher_forcing_ratio
            input_t = tgt[:, t + 1] if use_teacher else logit.argmax(-1)

        return outputs


def build_seq2seq(src_vocab_size, tgt_vocab_size, sos_idx, eos_idx, pad_idx,
                  emb_dim=128, hid_dim=256, n_layers=2, dropout=0.5):
    enc = Encoder(src_vocab_size, emb_dim, hid_dim, n_layers, dropout)
    dec = Decoder(tgt_vocab_size, emb_dim, hid_dim, n_layers, dropout)
    model = Seq2Seq(enc, dec, sos_idx, eos_idx, pad_idx).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Seq2Seq model parameters: {total:,}")
    return model


def train_seq2seq(model, train_loader, val_loader, epochs=15, lr=5e-4):
    criterion = nn.CrossEntropyLoss(ignore_index=model.pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    tr_losses, va_losses = [], []
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss, ep_tok = 0.0, 0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            logits = model(src, tgt)
            B, T, V = logits.shape
            lbl = tgt[:, 1:].reshape(B * T)
            loss = criterion(logits.reshape(B * T, V), lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            mask = lbl != model.pad_idx
            ep_loss += loss.item() * mask.sum().item()
            ep_tok += mask.sum().item()

        tr_ppl = math.exp(min(ep_loss / max(ep_tok, 1), 20))

        model.eval()
        va_loss, va_tok = 0.0, 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                logits = model(src, tgt, teacher_forcing_ratio=0.0)
                B, T, V = logits.shape
                lbl = tgt[:, 1:].reshape(B * T)
                loss = criterion(logits.reshape(B * T, V), lbl)
                mask = lbl != model.pad_idx
                va_loss += loss.item() * mask.sum().item()
                va_tok += mask.sum().item()
        va_ppl = math.exp(min(va_loss / max(va_tok, 1), 20))

        tr_losses.append(tr_ppl)
        va_losses.append(va_ppl)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d} | Train PPL: {tr_ppl:.2f} | Val PPL: {va_ppl:.2f}")

        if va_ppl < best_val:
            best_val = va_ppl
            torch.save(model.state_dict(),
                       os.path.join(SAVED_MODELS_DIR, "best_seq2seq.pth"))

    plt.figure(figsize=(8, 4))
    plt.plot(tr_losses, label="Train PPL")
    plt.plot(va_losses, label="Val PPL")
    plt.xlabel("Epoch"); plt.ylabel("Perplexity")
    plt.title("Seq2Seq LSTM perplexity")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "seq2seq_perplexity.png"))
    plt.close()
    return tr_losses, va_losses


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_prep_seq import get_loaders
    random.seed(42)
    torch.manual_seed(42)
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab, _ = get_loaders()
    model = build_seq2seq(
        src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab),
        sos_idx=tgt_vocab.sos_idx, eos_idx=tgt_vocab.eos_idx,
        pad_idx=tgt_vocab.pad_idx,
    )
    train_seq2seq(model, train_loader, val_loader, epochs=5)
