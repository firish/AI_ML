# Activation Functions

## Why They Exist

Without an activation function, a stack of layers is mathematically equivalent to a single layer — no matter how many layers you add. You can prove this:

```text
Layer 1: output = X * W1 + b1
Layer 2: output = (X * W1 + b1) * W2 + b2
                = X * (W1*W2) + (b1*W2 + b2)
                = X * W_combined + b_combined   ← still just one linear transformation
```

Activation functions introduce **non-linearity**, which is what lets deep networks learn curves, corners, and complex patterns — not just straight lines.

---

## ReLU (Rectified Linear Unit)

**The most common activation in hidden layers.**

```text
ReLU(x) = max(0, x)

x = -3  → 0
x = -1  → 0
x =  0  → 0
x =  1  → 1
x =  4  → 4
```

You've already seen this in the CNN notes (file 09). It zeroes out negative values and passes positive ones through unchanged.

**Why it works:** Simple and fast. Gradients don't shrink for positive values, which helps backprop work well in deep networks. The main weakness is "dead neurons" — if a neuron's output is always negative, its gradient is always 0 and it never updates.

---

## Sigmoid

Maps any number to the range (0, 1).

```math
sigmoid(x) = 1 / (1 + e^(-x))
```

```text
x = -5  → 0.007
x = -1  → 0.269
x =  0  → 0.500
x =  1  → 0.731
x =  5  → 0.993
```

**Used for:** Binary classification output (is this spam? yes/no), gating mechanisms inside LSTMs.

**Not used in hidden layers** of modern networks because for large |x|, the gradient is nearly zero — the "vanishing gradient" problem. Gradients stop flowing and deep layers stop learning.

---

## Softmax

Takes a vector of numbers and converts them to a probability distribution (all positive, sums to 1).

```math
softmax(x_i) = e^(x_i) / Σ_j e^(x_j)
```

Toy example:

```text
Raw scores (logits): [2.0, 1.0, 0.1]

e^2.0 = 7.39
e^1.0 = 2.72
e^0.1 = 1.10
Sum   = 11.21

Softmax: [7.39/11.21, 2.72/11.21, 1.10/11.21]
       = [0.659,       0.243,       0.098]
       Sums to 1.000 ✓
```

**Used for:** The final layer of a classification network — converts raw scores into "probability that the answer is class X."

Also used inside Transformer attention:
```text
attention_weights = softmax(Q * K^T / sqrt(d_k))
```
(You've seen this in file 07 — softmax there normalises attention scores so they sum to 1.)

---

## Tanh

Maps any number to (-1, 1). Similar to sigmoid but centred at zero.

```text
x = -2  → -0.964
x = -1  → -0.762
x =  0  →  0.000
x =  1  →  0.762
x =  2  →  0.964
```

**Used for:** Historically popular in RNNs. Largely replaced by ReLU in modern networks. Still used in some gating mechanisms (LSTMs, GRUs).

---

## GELU (Gaussian Error Linear Unit)

A smooth version of ReLU used in transformers (BERT, GPT).

```text
GELU(x) ≈ x * sigmoid(1.702 * x)

x = -2  → -0.045
x = -1  → -0.159
x =  0  →  0.000
x =  1  →  0.841
x =  2  →  1.955
```

Unlike ReLU, GELU doesn't hard-zero negatives — it slightly passes small negative values through. In practice this works better for transformers, possibly because the soft gating is more gradient-friendly.

---

## Summary: When to Use Which

| Activation | Output range | Used for | Notes |
|---|---|---|---|
| **ReLU** | [0, ∞) | Hidden layers in CNNs, MLPs | Default choice for hidden layers |
| **GELU** | (-∞, ∞) | Hidden layers in transformers | Used in BERT, GPT feed-forward blocks |
| **Sigmoid** | (0, 1) | Binary classification output | Avoid in hidden layers (vanishing gradients) |
| **Softmax** | (0, 1), sums to 1 | Multi-class classification output, attention weights | Always at the end, not hidden layers |
| **Tanh** | (-1, 1) | Some gating mechanisms | Mostly historical |

---

## The Vanishing Gradient Problem (Brief)

Sigmoid and tanh both squash their output into a small range. When you differentiate them at large |x|, the gradient is nearly zero. In a deep network, multiplying many near-zero gradients together (via backprop's chain rule) means the gradient at the input layer becomes vanishingly small — early layers stop learning.

**ReLU's gradient for positive values = 1** exactly. It doesn't shrink. This is the primary reason ReLU replaced sigmoid in hidden layers.

(Residual connections, which you've already seen in ResNet and Transformers, are the other fix — they give gradients a direct path that bypasses these small-gradient layers.)
