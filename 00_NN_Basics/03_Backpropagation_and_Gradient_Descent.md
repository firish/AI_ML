# Backpropagation and Gradient Descent

## The Problem

After the forward pass, we have a loss. We need to answer: **which weights should increase and which should decrease to make the loss smaller?**

Doing this by trial and error (try every combination of weights) is impossibly slow — even a small network has millions of weights. We need a smarter method.

---

## Gradients: The Compass

A **gradient** tells you, for each weight: "if I increase this weight by a tiny amount, does the loss go up or down, and by how much?"

```text
weight w has gradient = +2.0
→ increasing w by a tiny bit increases the loss
→ so we should DECREASE w

weight w has gradient = -0.5
→ increasing w by a tiny bit decreases the loss
→ so we should INCREASE w
```

The gradient is just the slope of the loss function at the current weight value.

---

## Gradient Descent: Walking Downhill

Imagine the loss as a hilly landscape. You're standing somewhere on it, and you want to reach the lowest valley. Gradient descent says: **always take a step in the downhill direction.**

```text
Update rule:
    new_weight = old_weight - (learning_rate * gradient)
```

The minus sign: we go *opposite* to the gradient direction (downhill, not uphill).

**Toy example** — a single weight:

```text
Initial weight: w = 3.0
Gradient:      dL/dw = +2.0   (loss increases if w increases)
Learning rate: lr = 0.1

new_w = 3.0 - (0.1 * 2.0) = 3.0 - 0.2 = 2.8
```

Repeat this for every weight in the network, every step.

---

## Learning Rate: Step Size

The learning rate controls how big each step is.

```text
lr = 0.01  → tiny steps, slow but stable
lr = 0.1   → medium steps, usually fine
lr = 10.0  → giant steps, likely to overshoot the valley and oscillate
```

```text
Loss landscape (side view):

                 *
              *     *
           *           *
        *                 *
   valley ← we want here
```

Too small: converges very slowly.
Too large: bounces around and never settles.

---

## Backpropagation: How Gradients Are Computed

Computing the gradient for a weight deep in the network requires the **chain rule** from calculus. Backpropagation is just the efficient algorithm for applying the chain rule layer by layer, from the output back to the input.

### The Chain Rule (The One Piece of Math You Need)

If A affects B, and B affects C, then:

```text
How much does A affect C?  =  (how much A affects B)  ×  (how much B affects C)

In notation:   dC/dA = dC/dB × dB/dA
```

That's it. Backprop is just applying this rule repeatedly through each layer.

### Walk-through: Computing Gradients for Every Weight

A tiny network: 1 input, 1 hidden neuron, 1 output neuron, no activation (to keep the math visible).

```text
Network:
    x=2.0 —[w1=0.5]→ h —[w2=3.0]→ output → loss
                                      ↕
                                  target=4.0
```

**Step 1: Forward pass**

```text
h      = x * w1      = 2.0 * 0.5  = 1.0
output = h * w2      = 1.0 * 3.0  = 3.0
loss   = (output - target)²  = (3.0 - 4.0)²  = 1.0
```

**Step 2: Backward pass — output layer (w2)**

Start from the loss and work backwards. How much does w2 affect the loss?

```text
d(loss)/d(output) = 2 * (output - target) = 2 * (3.0 - 4.0) = -2.0
     "the loss decreases if we increase the output" ✓ (output is 3, target is 4)

d(output)/d(w2) = h = 1.0
     "output = h * w2, so increasing w2 by 1 increases output by h"

Chain rule:
d(loss)/d(w2) = d(loss)/d(output) × d(output)/d(w2)
              = -2.0 × 1.0
              = -2.0
     "increasing w2 decreases the loss" → so we should increase w2 ✓
```

**Step 3: Backward pass — hidden layer (w1)**

Now the chain goes one step deeper. How much does w1 affect the loss?

```text
d(loss)/d(output) = -2.0             (already computed)
d(output)/d(h)    = w2 = 3.0         (output = h * w2)
d(h)/d(w1)        = x = 2.0          (h = x * w1)

Chain rule (one more link in the chain):
d(loss)/d(w1) = d(loss)/d(output) × d(output)/d(h) × d(h)/d(w1)
              = -2.0 × 3.0 × 2.0
              = -12.0
```

**Step 4: Update both weights**

```text
lr = 0.1

w2_new = w2 - lr * d(loss)/d(w2) = 3.0 - 0.1 * (-2.0)  = 3.0 + 0.2  = 3.2  ↑
w1_new = w1 - lr * d(loss)/d(w1) = 0.5 - 0.1 * (-12.0) = 0.5 + 1.2  = 1.7  ↑
```

Both weights increase, which makes the output larger, which is correct — it was 3.0 and needs to get to 4.0.

### The Pattern for Any Network

No matter how deep the network is, the process is the same:

```text
Forward:  x → Layer 1 → Layer 2 → ... → Layer N → loss

Backward: Start at loss.
          For each layer from N back to 1:
              gradient for this layer's weights =
                  (gradient flowing in from the right) × (this layer's local derivative)
              pass the gradient leftward to the next layer
```

Each layer only needs two things:
1. **The gradient flowing in from the layer above** (how much its output affected the loss)
2. **Its own local derivative** (how much its weights affected its output)

Multiply them. That's the gradient for that layer. Pass the signal backward. Repeat.

In a real framework (PyTorch, TensorFlow), you never write this manually — `loss.backward()` does the entire backward pass automatically. But this is what it's doing under the hood.

---

## Mini-batch Gradient Descent

Computing the exact gradient requires looking at ALL training examples — expensive. In practice:

| Variant | What it computes gradient over | Pro | Con |
|---------|-------------------------------|-----|-----|
| Batch GD | Whole dataset | Stable, exact | Very slow per step |
| Stochastic GD (SGD) | 1 random example | Fast per step | Noisy, unstable |
| **Mini-batch GD** | 32-512 random examples | Best of both | Standard in practice |

Mini-batch size is typically 32, 64, 128, or 256 — a tunable hyperparameter.

---

## What Optimisers Are

Plain gradient descent works but can be slow. **Optimisers** are smarter update rules:

| Optimiser | Key idea | Common use |
|-----------|----------|------------|
| SGD | Plain gradient descent | Simple tasks |
| SGD + Momentum | Remembers direction from previous steps, smooths out noise | CNNs |
| **Adam** | Adapts learning rate per weight, uses momentum | Most transformers, default choice |
| AdamW | Adam + weight decay (regularisation baked in) | BERT, GPT, standard LLM training |

When in doubt: use **AdamW**.

---

## Key Points

| Term | Plain English |
|------|--------------|
| **Gradient** | The slope — which way does the loss go if I increase this weight? |
| **Gradient descent** | Move each weight in the direction that reduces loss |
| **Learning rate** | How big a step to take each time |
| **Backpropagation** | Efficient algorithm for computing gradients throughout the network |
| **Mini-batch** | Compute gradient on a small random subset of data each step |
| **Epoch** | One full pass through the training data |
| **Optimiser** | A smarter version of gradient descent (Adam is the standard) |
