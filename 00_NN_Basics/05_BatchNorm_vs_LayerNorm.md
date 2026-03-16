# Batch Normalisation vs Layer Normalisation

## First — What Problem Are We Solving?

Imagine a neuron in Layer 5 of a deep network. During training:
- Layer 1's weights change → Layer 2 gets different inputs
- Layer 2's weights change → Layer 3 gets different inputs
- ...and so on down to Layer 5

So Layer 5 is trying to learn, but the numbers coming into it keep shifting every training step. It's like trying to hit a target that moves every time you adjust your aim.

Concretely, here's what the hidden values flowing into Layer 5 might look like across training steps:

```text
Step 100:  values arriving at Layer 5 → [0.2, 0.5, 0.1, 0.8]    range: 0 to 1
Step 200:  values arriving at Layer 5 → [15, 42, 8, 31]          range: 0 to 50
Step 300:  values arriving at Layer 5 → [-200, -50, -180, -90]   range: -200 to 0
```

The scale and centre keep shifting. Layer 5's weights were tuned for values around 0-1, and suddenly it's getting values in the hundreds. Training becomes unstable.

**Normalisation = force these values back to a consistent range before each layer processes them.**

---

## The Normalisation Operation (Both Types Do This)

Take a set of numbers, make them centred at 0 with spread ~1.

```text
Before: [2, 4, 6, 8]

Step 1: Find the mean
    mean = (2 + 4 + 6 + 8) / 4 = 5

Step 2: Subtract the mean (centres at 0)
    [2-5, 4-5, 6-5, 8-5] = [-3, -1, +1, +3]

Step 3: Find the standard deviation (how spread out the values are)
    std = sqrt(( (-3)^2 + (-1)^2 + 1^2 + 3^2 ) / 4) = sqrt(20/4) = sqrt(5) = 2.24

Step 4: Divide by std (scales spread to ~1)
    [-3/2.24, -1/2.24, +1/2.24, +3/2.24] = [-1.34, -0.45, +0.45, +1.34]

After: [-1.34, -0.45, +0.45, +1.34]   ← centred at 0, spread around +/-1
```

No matter what the original values were (0.001 or 50,000), after normalisation they're always in a tidy range around 0.

**But wait — what if the network actually NEEDS a different range?** After normalisation, two learned parameters (gamma and beta) let it rescale:

```text
final = gamma * normalised + beta

gamma and beta are learned during training, just like weights.
If the network decides it needs values centred at 5 with spread 2, it learns gamma=2, beta=5.
```

So normalisation doesn't remove information — it just gives the network a clean, stable starting point.

---

## The Key Question: Which Numbers Do You Average Over?

Both BatchNorm and LayerNorm do the exact same operation above. The ONLY difference is: **which set of numbers do you compute the mean and std from?**

Let's set up a concrete scenario. You're training a text model. Your mini-batch has 3 sentences, each represented as a hidden vector of 4 features:

```text
               Feature 1   Feature 2   Feature 3   Feature 4
Sentence A:       0.5         1.2        -0.3         0.8
Sentence B:       1.1         0.8         0.4        -0.2
Sentence C:      -0.2         2.1         0.1         0.6
```

This is a 3x4 grid. BatchNorm and LayerNorm just differ in which direction they compute the mean/std.

---

## BatchNorm: Normalise Down Each Column

BatchNorm asks: "For Feature 1, what is the mean and std across all sentences in this batch?"

```text
               Feature 1   Feature 2   Feature 3   Feature 4
Sentence A:       0.5         1.2        -0.3         0.8
Sentence B:       1.1         0.8         0.4        -0.2
Sentence C:      -0.2         2.1         0.1         0.6
                   ↓           ↓           ↓           ↓
                normalise   normalise   normalise   normalise
                this col    this col    this col    this col

Feature 1 column: [0.5, 1.1, -0.2]
    mean = 0.47, std = 0.53
    normalised = [0.06, 1.19, -1.26]

Feature 2 column: [1.2, 0.8, 2.1]
    mean = 1.37, std = 0.54
    normalised = [-0.31, -1.05, 1.35]

...and so on for each feature column.
```

Each feature gets its own mean and std, computed across the batch.

**When it works well:** CNNs processing images. Every image in the batch is 224x224 pixels — same size, same structure. The batch statistics are meaningful.

**When it breaks down:**
- **Small batch sizes:** If your batch is only 2 sentences, the mean/std of 2 numbers is very noisy — not a reliable estimate
- **Variable-length sequences:** Sentence A has 5 tokens, Sentence B has 20 tokens. The shorter ones are padded with zeros. Those zeros pollute the mean
- **Inference with 1 input:** At test time you often process one sentence at a time. A column of 1 number has no meaningful mean/std. BatchNorm must fall back to running averages saved during training — a hack

---

## LayerNorm: Normalise Across Each Row

LayerNorm asks: "For Sentence A, what is the mean and std across all its features?"

```text
               Feature 1   Feature 2   Feature 3   Feature 4
Sentence A:       0.5         1.2        -0.3         0.8     → normalise this row
Sentence B:       1.1         0.8         0.4        -0.2     → normalise this row
Sentence C:      -0.2         2.1         0.1         0.6     → normalise this row

Sentence A row: [0.5, 1.2, -0.3, 0.8]
    mean = 0.55, std = 0.55
    normalised = [-0.09, 1.18, -1.55, 0.45]

Sentence B row: [1.1, 0.8, 0.4, -0.2]
    mean = 0.525, std = 0.48
    normalised = [1.20, 0.57, -0.26, -1.51]

Sentence C row: [-0.2, 2.1, 0.1, 0.6]
    mean = 0.65, std = 0.85
    normalised = [-1.00, 1.71, -0.65, -0.06]
```

Each sentence is normalised **independently**. Sentence A doesn't need to know anything about Sentences B or C.

**Why this is better for text/transformers:**
- Batch size = 1? Fine — you normalise across that one sentence's features
- Sentences have different lengths? Fine — each is normalised on its own
- Want to generate one token at a time (like GPT)? Fine — normalise that one token's hidden state across its features

---

## Visual Summary

```text
The same 3x4 grid, two different directions:

BatchNorm (down columns ↓):         LayerNorm (across rows →):

  f1   f2   f3   f4                   f1   f2   f3   f4
[ 0.5  1.2 -0.3  0.8 ]             [ 0.5  1.2 -0.3  0.8 ] ← normalise together
[ 1.1  0.8  0.4 -0.2 ]             [ 1.1  0.8  0.4 -0.2 ] ← normalise together
[-0.2  2.1  0.1  0.6 ]             [-0.2  2.1  0.1  0.6 ] ← normalise together
  ↑    ↑    ↑    ↑
  normalise each column
  (needs the whole batch)            (each row is independent)
```

---

## Where They Appear

**BatchNorm in CNNs (like ResNet from your file 09):**
```text
Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU → ...
```
Images in a batch are all the same size, batches are typically 32-256, so column stats are stable.

**LayerNorm in Transformers (like BERT from your file 07):**
```text
Input → Self-Attention → Add & LayerNorm → Feed-Forward → Add & LayerNorm → Output
```
Text sequences vary in length, and at inference the model generates one token at a time (batch size = 1).

---

## Key Takeaway

| | BatchNorm | LayerNorm |
|---|---|---|
| **Normalises** | Each feature across the batch (down columns) | Each example across its features (across rows) |
| **Needs the batch?** | Yes — unstable if batch is small | No — each example is independent |
| **Works at inference with 1 input?** | Awkward (uses stored running stats) | Yes, naturally |
| **Used in** | CNNs (images, fixed-size inputs) | Transformers (text, variable-length, autoregressive) |
| **Why it exists** | Both stabilise training by keeping values in a consistent range |
