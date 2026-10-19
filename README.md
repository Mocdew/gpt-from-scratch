# GPT from Scratch

A hands-on implementation of a character-level GPT language model in PyTorch, built from the ground up to make the mechanics of modern autoregressive transformers easier to understand.

The repository starts with a simple next-character baseline and grows into a decoder-only transformer with causal self-attention, multi-head attention, residual connections, layer normalization, feed-forward networks, dropout, training/validation loss estimation, and autoregressive text generation.

## What is implemented

- Character-level tokenization with explicit `stoi` and `itos` lookup tables
- Random mini-batches and a 90/10 train-validation split
- A learnable bigram-style baseline
- Token and positional embeddings
- Masked (causal) self-attention
- Multi-head attention
- Pre-norm residual transformer blocks
- Feed-forward networks and dropout
- Cross-entropy training with AdamW
- Autoregressive sampling from the trained model
- A notebook workflow with checkpointing and loss visualization

## Model progression

| File | Purpose | Context | Embedding | Heads | Layers |
| --- | --- | ---: | ---: | ---: | ---: |
| `bigram.py` | Small next-character baseline | 8 | 32 | — | — |
| `gptsmall.py` | Compact transformer for quicker experiments | 64 | 128 | 4 | 4 |
| `gpt.py` | Full learning implementation | 256 | 384 | 6 | 6 |
| `mini_gpt (1).ipynb` | Guided notebook with training, checkpoints, plots, and sampling | 128 | 192 | 6 | 6 |

All models train on the included Tiny Shakespeare corpus and generate text one character at a time.

## Architecture

The transformer path follows this flow:

```text
character IDs
     │
     ├── token embeddings
     └── position embeddings
              │
              ▼
   repeated transformer blocks
   ├── LayerNorm → causal multi-head attention → residual add
   └── LayerNorm → feed-forward network → residual add
              │
              ▼
       final LayerNorm
              │
              ▼
      vocabulary logits
              │
              ▼
       next character
```

The causal mask prevents each position from looking ahead, so the model learns to predict the next character using only the characters that came before it.

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/Mocdew/gpt-from-scratch.git
cd gpt-from-scratch
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows, activate it with:

```powershell
venv\Scripts\activate
```

### 3. Install the dependencies

For the Python scripts:

```bash
pip install torch
```

For the notebook and loss plots:

```bash
pip install jupyter matplotlib
```

## Running the models

Start with the smallest model:

```bash
python bigram.py
```

Run the compact transformer:

```bash
python gptsmall.py
```

Run the larger transformer:

```bash
python gpt.py
```

Each script trains the selected model, periodically prints training and validation losses, and then generates a 500-character sample. Training automatically uses CUDA when it is available and otherwise falls back to the CPU.

To use the guided notebook:

```bash
jupyter notebook "mini_gpt (1).ipynb"
```

## Repository contents

| Path | Description |
| --- | --- |
| `bigram.py` | Lightweight next-character model and training loop |
| `gptsmall.py` | Smaller GPT configuration for faster iteration |
| `gpt.py` | Larger decoder-only transformer implementation |
| `mini_gpt (1).ipynb` | Interactive implementation and experiment notebook |
| `input.txt` | Tiny Shakespeare training corpus |
| `tiny shakespere.txt` | A second copy of the source corpus retained for reference |
| `GPT from Scratch & Tokenization Study Notes.docx` | Companion notes on tokenization, BPE, attention, and GPT internals |

## Experimenting

The hyperparameters are grouped at the top of each script. Useful knobs to explore include:

- `batch_size`: number of independent sequences processed together
- `block_size`: maximum context length
- `n_embd`: embedding width
- `n_head`: number of attention heads
- `n_layer`: number of transformer blocks
- `dropout`: regularization strength
- `learning_rate`: AdamW step size
- `max_iters`: number of training iterations

The default configurations are intended for learning and experimentation rather than production use. If training is slow on your machine, begin with `bigram.py` or `gptsmall.py`, lower `max_iters`, or reduce the model dimensions.

## Learning goals

This project is designed to make the key ideas behind GPT concrete:

1. Convert raw text into integer tokens.
2. Learn useful token and position representations.
3. Let tokens communicate through causal self-attention.
4. Stack attention and feed-forward blocks into a transformer.
5. Optimize next-token prediction with cross-entropy loss.
6. Generate new text by repeatedly sampling the next-token distribution.

## Acknowledgements

The implementation and Tiny Shakespeare exercise are inspired by Andrej Karpathy's educational *Neural Networks: Zero to Hero* material, especially the GPT-from-scratch walkthrough. The code in this repository is kept deliberately explicit so each part of the model can be read, changed, and understood.
