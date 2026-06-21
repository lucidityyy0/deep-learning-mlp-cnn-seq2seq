import os
import re
import unicodedata
import random
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SPECIAL = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


class Vocab:
    def __init__(self, name):
        self.name = name
        self.word2idx = {t: i for i, t in enumerate(SPECIAL)}
        self.idx2word = {i: t for i, t in enumerate(SPECIAL)}
        self.word_freq = Counter()

    def add_sentence(self, tokens):
        for t in tokens:
            self.word_freq[t] += 1

    def build(self, max_size=None):
        most_common = self.word_freq.most_common(max_size)
        for word, _ in most_common:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, tokens):
        unk = self.word2idx[UNK_TOKEN]
        return [self.word2idx.get(t, unk) for t in tokens]

    def decode(self, indices):
        return [self.idx2word.get(i, UNK_TOKEN) for i in indices]

    def __len__(self):
        return len(self.word2idx)

    @property
    def pad_idx(self): return self.word2idx[PAD_TOKEN]
    @property
    def sos_idx(self): return self.word2idx[SOS_TOKEN]
    @property
    def eos_idx(self): return self.word2idx[EOS_TOKEN]


def unicode_to_ascii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def preprocess(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-z.!?]+", " ", s)
    return s.strip()


def load_pairs(max_pairs=20000, max_len=10):
    fra_path = os.path.join(DATA_DIR, "fra.txt")
    if not os.path.exists(fra_path):
        print("fra.txt not found, using synthetic fallback corpus.")
        return _synthetic_corpus()

    pairs = []
    with open(fra_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            eng, fra = preprocess(parts[0]), preprocess(parts[1])
            eng_tok = eng.split()
            fra_tok = fra.split()
            if len(eng_tok) <= max_len and len(fra_tok) <= max_len:
                pairs.append((fra_tok, eng_tok))
            if len(pairs) >= max_pairs:
                break
    print(f"Loaded {len(pairs)} sentence pairs (max_len={max_len})")
    return pairs


def _synthetic_corpus():
    templates = [
        (["je", "suis", "etudiant"], ["i", "am", "a", "student"]),
        (["il", "fait", "beau"], ["it", "is", "nice"]),
        (["ou", "est", "la", "gare"], ["where", "is", "the", "station"]),
        (["je", "parle", "francais"], ["i", "speak", "french"]),
        (["merci", "beaucoup"], ["thank", "you", "very", "much"]),
        (["bonjour"], ["hello"]),
        (["au", "revoir"], ["goodbye"]),
        (["je", "mange"], ["i", "eat"]),
        (["il", "dort"], ["he", "sleeps"]),
        (["elle", "chante"], ["she", "sings"]),
    ]
    pairs = []
    for _ in range(2000):
        pairs.append(random.choice(templates))
    print(f"Synthetic corpus: {len(pairs)} pairs")
    return pairs


def build_vocabs(pairs, src_max=None, tgt_max=None):
    src_vocab = Vocab("fra")
    tgt_vocab = Vocab("eng")
    for src, tgt in pairs:
        src_vocab.add_sentence(src)
        tgt_vocab.add_sentence(tgt)
    src_vocab.build(src_max)
    tgt_vocab.build(tgt_max)
    print(f"Src vocab size: {len(src_vocab)}, Tgt vocab size: {len(tgt_vocab)}")
    return src_vocab, tgt_vocab


class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab):
        self.data = []
        for src, tgt in pairs:
            src_ids = src_vocab.encode(src) + [src_vocab.eos_idx]
            tgt_ids = [tgt_vocab.sos_idx] + tgt_vocab.encode(tgt) + [tgt_vocab.eos_idx]
            self.data.append((
                torch.tensor(src_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long),
            ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch, src_pad, tgt_pad):
    srcs, tgts = zip(*batch)
    srcs_pad = pad_sequence(srcs, batch_first=True, padding_value=src_pad)
    tgts_pad = pad_sequence(tgts, batch_first=True, padding_value=tgt_pad)
    return srcs_pad, tgts_pad


def get_loaders(batch_size=32, max_pairs=20000, max_len=10, seed=42):
    random.seed(seed)
    pairs = load_pairs(max_pairs=max_pairs, max_len=max_len)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]

    src_vocab, tgt_vocab = build_vocabs(train_pairs)

    train_ds = TranslationDataset(train_pairs, src_vocab, tgt_vocab)
    val_ds = TranslationDataset(val_pairs, src_vocab, tgt_vocab)
    test_ds = TranslationDataset(test_pairs, src_vocab, tgt_vocab)

    def collate(batch):
        return collate_fn(batch, src_vocab.pad_idx, tgt_vocab.pad_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    src_b, tgt_b = next(iter(train_loader))
    print(f"Batch src shape: {src_b.shape}, tgt shape: {tgt_b.shape}")

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab, test_pairs


if __name__ == "__main__":
    get_loaders()
