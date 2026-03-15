# Batch Normalisation vs Layer Normalisation

## Why Normalisation Exists

Deep networks develop a problem during training called **internal covariate shift**: as weights in early layers update, the distribution of values flowing into later layers keeps changing. The later layers are constantly trying to learn from a moving target.

Normalisation fixes this by rescaling activations to have roughly zero mean and unit variance at certain points in the network.

Without it: training is slower, learning rates must be smaller, and deep networks are harder to optimise.

---

## What Normalisation Does (The Operation)

Both methods do the same basic thing: for a set of values, subtract the mean and divide by the standard deviation.

```text
values:   [2, 4, 6, 8]
mean:     5
std:      2.24

normalised: [(2-5)/2.24, (4-5)/2.24, (6-5)/2.24, (8-5)/2.24]
           = [-1.34, -0.45, +0.45, +1.34]
```

After normalisation, values are centred at 0 and spread around ±1. The network then applies learned scale (gamma) and shift (beta) parameters to let it rescale if needed.

---

## Batch Normalisation (BatchNorm)

Normalises **across the batch dimension** — i.e., for each feature position, computes the mean and std across all examples in the mini-batch.

```text
Mini-batch of 4 sentences, each with a hidden layer of 3 values:

           Feature 1   Feature 2   Feature 3
Example A:    0.5          1.2         -0.3
Example B:    1.1          0.8          0.4
Example C:   -0.2          2.1          0.1
Example D:    0.8          0.5         -0.7

BatchNorm normalises DOWN each column (across examples A, B, C, D).
Mean and std are computed per feature across the batch.
```

**The problem with BatchNorm in NLP:**
- Batch size might be small → noisy statistics
- Sequences have different lengths → padding messes up the mean
- At inference time with batch size = 1, there's nothing to normalise across → must use running averages from training, which is a mismatch

---

## Layer Normalisation (LayerNorm)

Normalises **across the feature dimension** — i.e., for each example independently, computes the mean and std across all its features.

```text
Single example (one token's hidden state):
   [0.5, 1.2, -0.3, 0.8, -1.1, 2.0]

LayerNorm normalises ACROSS these 6 values for this one example.
Mean = 0.52,  Std = 0.97

Normalised: [-0.02, 0.70, -0.84, 0.29, -1.67, 1.52]
```

Each example is normalised independently — batch size doesn't matter.

---

## Side-by-Side Comparison

| Property | BatchNorm | LayerNorm |
|---|---|---|
| Normalises across | Batch (across examples) | Features (within one example) |
| Requires large batch? | Yes — unstable with small batches | No — works with batch size = 1 |
| Works at inference with single sample? | Awkward (needs stored running stats) | Yes, naturally |
| Used in | CNNs (ResNet, etc.) | Transformers (BERT, GPT, etc.) |
| Sensitive to sequence length? | Yes | No |

```text
BatchNorm:                    LayerNorm:

[ex1_f1  ex1_f2  ex1_f3]     [ex1_f1  ex1_f2  ex1_f3]
[ex2_f1  ex2_f2  ex2_f3]  ↓  [ex2_f1  ex2_f2  ex2_f3]  →
[ex3_f1  ex3_f2  ex3_f3]     [ex3_f1  ex3_f2  ex3_f3]  →

↓ = normalise down columns    → = normalise across rows
```

---

## Why Transformers Use LayerNorm

You've already seen LayerNorm in your transformer notes (file 07):

```text
input → Self-Attention → Add & LayerNorm → Feed-Forward → Add & LayerNorm → output
```

The reasons:
1. Sentences in a batch have different lengths — BatchNorm's statistics would be polluted by padding
2. Autoregressive generation (file 03 in Phase 3) runs with one token at a time — batch size = 1, so BatchNorm is unusable
3. LayerNorm is simpler and more stable for the sequential, variable-length nature of text

---

## Key Points

| | BatchNorm | LayerNorm |
|---|---|---|
| **What** | Normalise each feature across a batch | Normalise each example across its features |
| **Where** | After conv layers in CNNs | After attention/FFN in Transformers |
| **Why** | Stabilises CNN training | Needed because text batches are variable-length and auto-regressive inference is single-sample |
