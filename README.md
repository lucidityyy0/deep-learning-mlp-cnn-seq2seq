# Deep Learning Project

Three deep learning models in PyTorch, each matched to a different data structure:
MLP for tabular data, CNN for images, and recurrent / encoder-decoder models for
sequences.

Author: Bakouch Moad (EMSI Casablanca, Deep Learning module, 2025-2026).

## Parts

- **Part I (`part1_mlp/`)** Multi-layer perceptron for binary classification on
  Breast Cancer Wisconsin. Two equivalent implementations (`nn.Sequential` and a
  custom `nn.Module`), four weight-initialisation strategies, early stopping,
  class-weighted loss, and a save/reload check.
- **Part II (`part2_cnn/`)** LeNet-style CNN on Fashion-MNIST, a manual NumPy
  convolution verified against PyTorch, an ablation over padding, stride, pooling,
  filter count and 1x1 convolution, an MLP baseline, and feature-map visualisation
  through forward hooks.
- **Part III (`part3_seq/`)** RNN, LSTM and GRU compared on next-token perplexity
  with gradient clipping, then an LSTM Seq2Seq encoder-decoder with teacher forcing
  for French to English translation, evaluated with greedy and beam-search decoding
  and BLEU.

## Layout

```
part1_mlp/      data_prep, model_sequential, model_custom, train_eval
part2_cnn/      conv_manual, lenet, experiments
part3_seq/      data_prep_seq, rnn_lstm_gru, seq2seq, decoding
results/        figures and comparison tables produced by the scripts
main_notebook.ipynb   end-to-end notebook covering the three parts
generate_report.py    builds finalreport.pdf from the figures in results/
```

## Setup

```
pip install -r requirements.txt
```

## Running

```
python part1_mlp/train_eval.py        # Part I: train, evaluate, init comparison
python part2_cnn/experiments.py       # Part II: CNN ablation on Fashion-MNIST
python part3_seq/rnn_lstm_gru.py      # Part III: RNN / LSTM / GRU comparison
python part3_seq/seq2seq.py           # Part III: train the Seq2Seq model
python part3_seq/decoding.py          # Part III: decode and score BLEU
python generate_report.py             # build finalreport.pdf
```

Fashion-MNIST is downloaded on first run. The translation corpus falls back to a
small built-in French-English set when `data/fra.txt` is absent. A fixed seed (42)
is set for reproducibility.

## Results

| Part | Dataset | Model | Main result |
|------|---------|-------|-------------|
| I | Breast Cancer | MLP 30-64-32-2 | 96.5% test accuracy |
| II | Fashion-MNIST | LeNet (avg/max pool) | CNN above the MLP baseline |
| III | FR-EN corpus | RNN / LSTM / GRU | validation perplexity around 2.0 |
| III | FR-EN translation | encoder-decoder LSTM | BLEU 100 on the test set |
