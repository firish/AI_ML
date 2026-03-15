# Loss Functions — Measuring How Wrong the Model Is

## The Core Idea

A **loss function** (also called a cost function) is a single number that says: *"how far off is the network's prediction from the correct answer?"*

- Loss = 0 → perfect predictions
- Loss is large → predictions are way off
- The entire goal of training is to **minimise** this number by adjusting weights

---

## Loss for Regression: Mean Squared Error (MSE)

Use when the output is a continuous number (e.g., predict house price, predict a vector coordinate).

```text
Predictions: [200k, 350k, 180k]
True values: [210k, 300k, 190k]

Errors:       [-10k, +50k, -10k]
Squared:      [100, 2500, 100]
Mean:         (100 + 2500 + 100) / 3 = 900

MSE = 900
```

Squaring serves two purposes:
1. Makes negatives and positives both count (an error of -10 is just as bad as +10)
2. Penalises large errors more heavily than small ones (50k error → 2500, not 50)

---

## Loss for Classification: Cross-Entropy

Use when the output is a class probability (e.g., "is this a cat, dog, or bird?").

The network outputs probabilities for each class — e.g., after a softmax layer:

```text
True label: "cat" (index 0)

Network output (probabilities): [0.70, 0.20, 0.10]
                                  cat   dog   bird
```

Cross-entropy loss for this one example:
```math
loss = -log(probability assigned to the CORRECT class)
     = -log(0.70)
     = 0.357
```

Why log? Because we want the model to be *confident* in the right answer, not just correct:

```text
Prediction 0.70 for cat → loss = -log(0.70) = 0.357   (okay)
Prediction 0.99 for cat → loss = -log(0.99) = 0.010   (great)
Prediction 0.10 for cat → loss = -log(0.10) = 2.303   (very wrong)
```

The loss explodes as the model becomes more confidently wrong.

---

## What "Learning" Means

Training is just a loop:

```text
repeat many times:
    1. Forward pass  → get prediction
    2. Compute loss  → how wrong?
    3. Backward pass → compute gradients (which direction makes loss smaller?)
    4. Update weights → nudge them in that direction
```

Each full pass through the training dataset is called an **epoch**.

---

## Loss During Training

A healthy training run looks like this:

```text
Epoch  1: loss = 2.40
Epoch  5: loss = 1.20
Epoch 10: loss = 0.65
Epoch 20: loss = 0.31
Epoch 50: loss = 0.08
```

- If loss stops decreasing: the model has converged (or is stuck)
- If training loss is low but validation loss is high: the model is overfitting (file 06)
- If loss oscillates or explodes: learning rate is too high (file 03)

---

## Which Loss to Use

| Task | Output type | Loss function |
|------|-------------|---------------|
| Predict a number (price, temperature) | Single float | MSE |
| Classify into one of N classes | Probabilities over N classes | Cross-entropy |
| Similarity between two vectors (word2vec, CLIP) | Dot product score | InfoNCE / contrastive loss |
| Match two things (translation, summarisation) | Token probabilities | Cross-entropy per token |

You've already seen contrastive loss (CLIP, file 10) and InfoNCE (SimCLR, file 08). They are just specialised versions of the same idea: penalise the model when it assigns low probability to the correct pairing.
