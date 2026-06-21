import os
import math
import torch
import torch.nn.functional as F

try:
    from sacrebleu.metrics import BLEU
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def greedy_decode(model, src_tensor, tgt_vocab, max_len=20):
    model.eval()
    h, c = model.encoder(src_tensor)
    input_t = torch.tensor([tgt_vocab.sos_idx], device=device)
    output_tokens = []

    for _ in range(max_len):
        logit, h, c = model.decoder(input_t, h, c)
        pred = logit.argmax(-1)
        token = tgt_vocab.idx2word.get(pred.item(), "<unk>")
        if token == "<eos>":
            break
        if token not in ("<sos>", "<pad>"):
            output_tokens.append(token)
        input_t = pred

    return output_tokens


@torch.no_grad()
def beam_search_decode(model, src_tensor, tgt_vocab, beam_size=3, max_len=20):
    model.eval()
    h, c = model.encoder(src_tensor)
    sos = tgt_vocab.sos_idx
    eos = tgt_vocab.eos_idx

    beams = [(0.0, [sos], h, c)]
    completed = []

    for _ in range(max_len):
        new_beams = []
        for score, tokens, h_b, c_b in beams:
            if tokens[-1] == eos:
                completed.append((score, tokens))
                continue
            input_t = torch.tensor([tokens[-1]], device=device)
            logit, h_new, c_new = model.decoder(input_t, h_b, c_b)
            log_probs = F.log_softmax(logit, dim=-1).squeeze()
            top_k = log_probs.topk(beam_size)
            for lp, idx in zip(top_k.values, top_k.indices):
                new_score = score - lp.item()
                new_beams.append((new_score, tokens + [idx.item()], h_new, c_new))
        new_beams.sort(key=lambda x: x[0])
        beams = new_beams[:beam_size]

    completed += [(s, t) for s, t, _, _ in beams]
    completed.sort(key=lambda x: x[0])
    best_tokens = completed[0][1] if completed else [sos]

    words = []
    for idx in best_tokens:
        word = tgt_vocab.idx2word.get(idx, "<unk>")
        if word in ("<sos>", "<eos>", "<pad>"):
            continue
        words.append(word)
    return words


def compute_bleu(hypotheses, references):
    if not SACREBLEU_AVAILABLE:
        print("sacrebleu not installed, BLEU skipped.")
        return None
    bleu = BLEU()
    result = bleu.corpus_score(hypotheses, [references])
    return result


def compute_perplexity(model, loader, tgt_vocab):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx)
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            inp = tgt[:, :-1]
            lbl = tgt[:, 1:]
            if inp.shape[1] == 0:
                continue
            logits = model(inp)
            B, T, V = logits.shape
            loss = criterion(logits.reshape(B * T, V), lbl.reshape(B * T))
            mask = lbl.reshape(-1) != tgt_vocab.pad_idx
            total_loss += loss.item() * mask.sum().item()
            total_tokens += mask.sum().item()
    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(min(avg_loss, 20))


def evaluate_translation(model, test_pairs, src_vocab, tgt_vocab,
                         beam_size=3, n_samples=200):
    greedy_hyps, beam_hyps, refs = [], [], []

    for src_tok, tgt_tok in test_pairs[:n_samples]:
        src_ids = src_vocab.encode(src_tok) + [src_vocab.eos_idx]
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

        greedy_out = greedy_decode(model, src_tensor, tgt_vocab)
        beam_out = beam_search_decode(model, src_tensor, tgt_vocab, beam_size=beam_size)

        greedy_hyps.append(" ".join(greedy_out))
        beam_hyps.append(" ".join(beam_out))
        refs.append(" ".join(tgt_tok))

    print("\n--- Sample translations (greedy) ---")
    for i in range(min(5, len(refs))):
        src_str = " ".join(test_pairs[i][0])
        print(f"  SRC : {src_str}")
        print(f"  REF : {refs[i]}")
        print(f"  HYP : {greedy_hyps[i]}")
        print()

    bleu_greedy = compute_bleu(greedy_hyps, refs)
    bleu_beam = compute_bleu(beam_hyps, refs)
    print(f"BLEU (greedy, k=1): {bleu_greedy}")
    print(f"BLEU (beam,   k={beam_size}): {bleu_beam}")

    with open(os.path.join(RESULTS_DIR, "translation_samples.txt"), "w") as f:
        f.write(f"BLEU greedy: {bleu_greedy}\n")
        f.write(f"BLEU beam  : {bleu_beam}\n\n")
        for pair, ref, hyp_g, hyp_b in zip(test_pairs[:n_samples], refs, greedy_hyps, beam_hyps):
            f.write(f"SRC   : {' '.join(pair[0])}\n")
            f.write(f"REF   : {ref}\n")
            f.write(f"GREEDY: {hyp_g}\n")
            f.write(f"BEAM  : {hyp_b}\n\n")

    return bleu_greedy, bleu_beam


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_prep_seq import get_loaders
    from seq2seq import build_seq2seq, Seq2Seq

    train_loader, val_loader, test_loader, src_vocab, tgt_vocab, test_pairs = get_loaders()
    model = build_seq2seq(
        src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab),
        sos_idx=tgt_vocab.sos_idx, eos_idx=tgt_vocab.eos_idx,
        pad_idx=tgt_vocab.pad_idx,
    )
    ckpt = os.path.join(os.path.dirname(__file__), "..", "saved_models", "best_seq2seq.pth")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
        print("Loaded best_seq2seq.pth")
    evaluate_translation(model, test_pairs, src_vocab, tgt_vocab)
