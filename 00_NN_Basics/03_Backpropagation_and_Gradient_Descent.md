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

You don't need to derive it yourself. The key intuition:

```text
Forward pass:   input → Layer 1 → Layer 2 → output → loss
                                                       ↓
Backward pass:  ← gradients flow back through each layer ←
                each layer passes blame to the layer before it
```

Each layer asks: "given that my output was slightly off, how much was each of my inputs responsible?" That answer becomes the gradient for the weights in that layer, and the signal passed back to the layer before it.

```text
Concrete flow for a 2-layer network:

Forward:
    x → [Layer 1, weights W1] → h → [Layer 2, weights W2] → output → loss

Backward:
    d(loss)/d(W2)  ← computed first  (output layer, easy)
    d(loss)/d(W1)  ← computed second (multiply through Layer 2's gradient)
```

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
