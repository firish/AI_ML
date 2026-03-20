## Activations, Optimizers, and Losses — Reference for Encoders & Decoders

This file expands on the basics from files 02, 03, and 04. Those files introduced what these things are. This file covers the **specific variants used by real models** and the intuition for why each model chose what it did.

---

## Part 1: Activation Functions in Practice

### Quick Recap

An activation function introduces non-linearity after each linear transformation. Without it, stacking layers is mathematically equivalent to one layer (see file 04).

### ReLU — The CNN Default

```text
ReLU(x) = max(0, x)

Used by: ResNet, all CNNs, simple MLPs
Where:   Hidden layers (after convolution or linear layers)

Why it works:
    - Gradient = 1 for positive inputs → no vanishing gradient
    - Gradient = 0 for negative inputs → sparse activations (many neurons "off")
    - Very fast to compute (just a comparison)

The problem (dead neurons):
    If a neuron's input is always negative, gradient is always 0.
    That neuron permanently stops learning.
    Happens more often with high learning rates.
```

### GELU — The Transformer Default

```text
GELU(x) = x × Φ(x)    where Φ(x) is the standard normal CDF
         ≈ x × sigmoid(1.702 × x)    (fast approximation)

Used by: BERT, GPT-2, GPT-3, RoBERTa, ViT
Where:   Inside the FFN block of every transformer layer

    x = -2.0  →  -0.045    (small negative allowed through)
    x = -1.0  →  -0.159    (some negative signal preserved)
    x =  0.0  →   0.000
    x =  1.0  →   0.841
    x =  2.0  →   1.955

Why transformers use GELU instead of ReLU:
    GELU is smooth (no hard cutoff at 0).
    Small negative values get a small negative output — this acts as a
    soft gate that preserves more information.
    ReLU's hard zero kills all negative signal, which loses nuance.

    Intuition: GELU weights each input by "how likely it is to be positive"
    under a Gaussian distribution. Positive values pass through almost fully,
    negative values are mostly suppressed but not killed.
```

### SwiGLU — The Modern LLM Default

```text
SwiGLU(x) = Swish(x × W₁) ⊙ (x × W₂)

    where Swish(x) = x × sigmoid(x)
    and ⊙ = element-wise multiplication

Used by: LLaMA, LLaMA 2/3, Mistral, Gemma, PaLM, GPT-4 (likely)
Where:   Replaces GELU inside the FFN block

Standard FFN (BERT, GPT-2):
    FFN(x) = GELU(x × W₁) × W₂
    Two weight matrices, one activation

SwiGLU FFN (LLaMA):
    FFN(x) = [Swish(x × W_gate) ⊙ (x × W_up)] × W_down
    THREE weight matrices, gated activation

    W_gate: produces the "gate" (what to keep)
    W_up:   produces the "value" (the actual information)
    Multiply gate × value → keep only what matters
    W_down: project back to model dimension

Why it's better:
    The gating mechanism lets the network learn WHAT to pass through.
    GELU applies the same smooth function everywhere.
    SwiGLU has a learned gate — "this feature matters, that one doesn't."

    Empirically: SwiGLU gives ~1-2% better results at the same compute.
    The cost: one extra weight matrix per layer (more parameters for same dims).
    Fix: reduce hidden dim slightly to match parameter count.
```

### Swish (SiLU)

```text
Swish(x) = x × sigmoid(x)

    x = -2.0  →  -0.238
    x = -1.0  →  -0.269
    x =  0.0  →   0.000
    x =  1.0  →   0.731
    x =  2.0  →   1.762

Used by: EfficientNet, some ViT variants
Similar to GELU but slightly different curve.
In practice, GELU ≈ Swish — the difference is minimal.
Swish is used as the activation INSIDE SwiGLU (hence the name).
```

### Softmax — Not a Hidden Activation

```text
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)

NOT used in hidden layers. Used at two specific points:

1. Attention weights:
    scores = Q · Kᵀ / √d
    weights = softmax(scores)    → sums to 1, each weight is a "how much to attend"

2. Output prediction head:
    logits = hidden × W_vocab    → one raw score per vocabulary token
    probs = softmax(logits)      → probability distribution over vocabulary

Temperature scaling (used during generation):
    softmax(x_i / T)

    T = 0.1: very sharp (one token gets ~100% probability) → deterministic
    T = 1.0: normal distribution → balanced
    T = 2.0: very flat (probabilities spread out) → creative/random
```

### Sigmoid — Gating and Binary Decisions

```text
sigmoid(x) = 1 / (1 + exp(-x))     → output in (0, 1)

Not used in hidden layers (vanishing gradient).

Where it appears in modern models:
    1. Inside SwiGLU:  Swish(x) = x × sigmoid(x)
    2. Inside SigLIP:  sigmoid loss instead of softmax loss
    3. Binary classification: P(yes) = sigmoid(logit)
    4. Gating in MoE:  router sometimes uses sigmoid to select experts
```

### Summary: Which Activation Where

```text
| Model family        | FFN activation | Why                                    |
| ------------------- | -------------- | -------------------------------------- |
| CNNs (ResNet)       | ReLU           | Simple, proven, fast                   |
| Early transformers  | GELU           | Smooth, no dead neurons                |
| (BERT, GPT-2/3)     |                |                                        |
| Modern LLMs         | SwiGLU         | Gated: learns what to pass through     |
| (LLaMA, Mistral)    |                | ~1-2% better at same compute           |
| Attention scores    | Softmax        | Normalize to probability distribution  |
| Prediction head     | Softmax        | Token probabilities over vocabulary    |
```

---

## Part 2: Optimizers

### Why Not Plain SGD?

```text
Plain SGD: w_new = w - lr × gradient

Problems:
    1. Same learning rate for ALL weights
       Some weights need big updates (rarely activated)
       Some need small updates (frequently activated)

    2. Noisy gradients
       Mini-batch gradients jump around. SGD follows every jump.
       Progress is zig-zaggy and slow.

    3. Saddle points
       The loss landscape is high-dimensional.
       Many flat regions where gradients ≈ 0.
       SGD gets stuck.
```

### SGD with Momentum

```text
v = β × v_prev + gradient          (β ≈ 0.9)
w_new = w - lr × v

Intuition: a ball rolling downhill.
    - v accumulates direction from previous steps
    - If gradients consistently point the same way → v grows → faster convergence
    - If gradients flip direction → v dampens → less oscillation

    Without momentum:    ↗ ↘ ↗ ↘ ↗ ↘ ↗ (zig-zag)
    With momentum:       ↗ → → → → → →  (smooth path)

Used by: CNNs (ResNet training), some classic models.
Mostly replaced by Adam for transformers.
```

### Adam (Adaptive Moment Estimation)

```text
The default optimizer for deep learning. Combines two ideas:

Idea 1: Momentum (first moment — m)
    m = β₁ × m_prev + (1 - β₁) × gradient        (β₁ = 0.9)
    Running average of gradient direction.
    Smooths out noise.

Idea 2: Adaptive learning rate (second moment — v)
    v = β₂ × v_prev + (1 - β₂) × gradient²       (β₂ = 0.999)
    Running average of gradient MAGNITUDE (squared).
    Tracks how big gradients typically are for each weight.

Update:
    w_new = w - lr × m / (√v + ε)

    ε = 1e-8 (tiny number to prevent division by zero)
```

**What this means intuitively:**

```text
For a weight with consistently large gradients:
    v is large → √v is large → effective lr is SMALL
    "You're already getting strong signal, take careful steps"

For a weight with consistently small gradients:
    v is small → √v is small → effective lr is LARGE
    "You're barely getting signal, take bigger steps to make progress"

Each weight automatically gets its own effective learning rate.
This is why it's called "adaptive."
```

**Bias correction** (a detail that matters):

```text
At the start of training, m and v are initialised to 0.
First few steps: m and v are biased toward 0 (haven't accumulated enough).

Fix: divide by (1 - β^t) where t = step number.
    m_corrected = m / (1 - β₁ᵗ)
    v_corrected = v / (1 - β₂ᵗ)

At step 1: (1 - 0.9¹) = 0.1 → divides by 0.1 → 10× amplification (compensating for near-zero start)
At step 100: (1 - 0.9¹⁰⁰) ≈ 1.0 → no effect (m has accumulated enough)
```

### AdamW (Adam with Decoupled Weight Decay)

```text
THE standard for training transformers. Used by BERT, GPT, LLaMA, Claude, everything.

The problem with Adam + L2 regularization:
    L2 regularization adds λ × w² to the loss.
    Adam sees this as "just another gradient" and adapts to it.
    The adaptive learning rate partly cancels out the regularization effect.
    The regularization becomes weaker than intended.

AdamW's fix:
    Don't add weight decay to the loss (don't let Adam see it).
    Instead, apply it directly to the weight AFTER Adam's update:

    Standard Adam + L2:
        gradient_with_reg = gradient + λ × w     ← Adam adapts to this
        w_new = w - lr × Adam_update(gradient_with_reg)

    AdamW:
        w_new = w - lr × Adam_update(gradient) - lr × λ × w
                     ↑ Adam handles gradient         ↑ weight decay applied separately

    The weight decay is "decoupled" — Adam can't interfere with it.

Why it matters:
    Better generalization (less overfitting).
    More predictable regularization strength.
    λ (weight decay coefficient) typically = 0.01 or 0.1.
```

### Learning Rate Schedules

The learning rate isn't constant during training — it follows a **schedule**:

```text
Warmup + Cosine Decay (most common for LLMs):

Learning rate
    ▲
    │         ╭──────╮
    │        ╱        ╲
    │       ╱          ╲
    │      ╱            ╲
    │     ╱              ╲
    │    ╱                ╲
    │   ╱                  ╲
    │  ╱                    ╲────
    │ ╱
    └──────────────────────────→ training steps
      ↑ warmup   peak        decay to ~0

Warmup phase (first ~2000 steps):
    lr increases linearly from 0 to peak_lr.
    Why: at the start, weights are random → gradients are huge and chaotic.
    A high lr would cause instability. Start slow, ramp up.

Cosine decay (rest of training):
    lr decreases following a cosine curve from peak to near-zero.
    Why: early in training, big updates are helpful (exploring the loss landscape).
    Later, small updates are better (fine-tuning into a good minimum).
```

```text
Typical values:
    Peak lr = 3e-4 (for small models) to 1e-4 (for large models)
    Warmup = 1-5% of total training steps
    Min lr = 0.1 × peak_lr (don't go all the way to zero)
```

### Which Optimizer Does Each Model Use?

```text
| Model / Task             | Optimizer | lr (peak)  | Weight decay | Schedule         |
| ------------------------ | --------- | ---------- | ------------ | ---------------- |
| ResNet (ImageNet)        | SGD+Mom   | 0.1        | 1e-4         | Step decay       |
| BERT pre-training        | AdamW     | 1e-4       | 0.01         | Linear warmup+decay|
| GPT-2 pre-training       | AdamW     | 2.5e-4     | 0.01         | Cosine decay     |
| GPT-3 pre-training       | AdamW     | 6e-5       | 0.1          | Cosine decay     |
| LLaMA pre-training       | AdamW     | 3e-4       | 0.1          | Cosine decay     |
| LLaMA fine-tuning (LoRA) | AdamW     | 1e-4       | 0.01         | Cosine decay     |
| CLIP training            | AdamW     | 5e-4       | 0.2          | Cosine decay     |
| DINOv2                   | AdamW     | 2e-4       | 0.04         | Cosine decay     |
```

**Pattern:** everything uses AdamW with cosine decay. The only things that change are the peak learning rate and weight decay strength. CNNs are the exception (SGD with momentum).

---

## Part 3: Loss Functions for Real Models

### Cross-Entropy (for Classification and Language Models)

Already covered in file 02. Here's how it connects to real models:

```text
Cross-Entropy loss = -log(P(correct answer))

    If P(correct) = 0.99 → loss = -log(0.99) = 0.01   (great)
    If P(correct) = 0.50 → loss = -log(0.50) = 0.69   (bad)
    If P(correct) = 0.01 → loss = -log(0.01) = 4.60   (terrible)
```

**Where it's used:**

```text
Decoder pre-training (GPT, LLaMA):
    At each position, predict the next token.
    Target = actual next token (one-hot over vocabulary)
    Loss = cross-entropy between softmax output and target.

    Input:  "The cat sat on"
    Pos 0 predicts "cat"  → -log(P("cat"))
    Pos 1 predicts "sat"  → -log(P("sat"))
    Pos 2 predicts "on"   → -log(P("on"))
    Total loss = average of all positions.

    This single loss trains every weight in the model.

Encoder pre-training (BERT MLM):
    Same loss, but only on masked tokens (15% of positions).

    Input:  "The [MASK] sat on the mat"
    Only the [MASK] position contributes to loss.
    Loss = -log(P("cat")) at the masked position.

Image classification (ResNet, ViT supervised):
    Output = softmax over 1000 ImageNet classes.
    Target = correct class (e.g., "cat")
    Loss = -log(P("cat")).
```

### Contrastive Loss / InfoNCE

The loss used when training for **similarity** — encoders (Sentence-BERT), multimodal (CLIP), self-supervised (SimCLR).

```text
Setup:
    Positive pair: two things that SHOULD be close (anchor, positive)
    Negative pairs: things that SHOULD be far (anchor vs everything else in batch)

InfoNCE loss:
    L = -log[ exp(sim(anchor, positive) / τ) / Σ_j exp(sim(anchor, neg_j) / τ) ]

    τ = temperature (controls sharpness)
```

**Intuition — it's just cross-entropy in disguise:**

```text
Think of it as a classification problem:
    "Among all items in the batch, which one is the positive?"

    sim(anchor, item_0) = 0.9     ← this is the positive
    sim(anchor, item_1) = 0.2
    sim(anchor, item_2) = 0.1
    sim(anchor, item_3) = 0.3

    Apply softmax: [0.62, 0.10, 0.08, 0.12]
    Target: index 0 (the positive)
    Loss = -log(0.62) = 0.48

    Same as cross-entropy where the "classes" are "which batch item is the match?"
```

**Where it's used:**

```text
| Model          | Positive pair                        | Negative pairs             |
| -------------- | ------------------------------------ | -------------------------- |
| Sentence-BERT  | (anchor text, semantically similar)  | Other texts in batch       |
| SimCLR         | (augmentation 1, augmentation 2)     | Other images in batch      |
| MoCo           | (aug 1, aug 2)                       | Queue of past embeddings   |
| CLIP           | (image, matching caption)            | Other images/captions      |
| CLAP           | (audio, matching caption)            | Other audio/captions       |
| DINO           | (global crop, local crop)            | (uses teacher, no explicit negatives) |
```

### Triplet Loss

A simpler contrastive loss — works with explicit triplets instead of batches.

```text
L = max(0, sim(anchor, negative) - sim(anchor, positive) + margin)

    margin ≈ 0.2

Intuition:
    "The positive should be closer than the negative, by at least margin."

    If sim(anchor, positive) = 0.8 and sim(anchor, negative) = 0.3:
        L = max(0, 0.3 - 0.8 + 0.2) = max(0, -0.3) = 0    ✓ already good

    If sim(anchor, positive) = 0.5 and sim(anchor, negative) = 0.6:
        L = max(0, 0.6 - 0.5 + 0.2) = max(0, 0.3) = 0.3    ✗ needs to learn

Where it's used:
    Face recognition (FaceNet)
    Product image similarity (custom fine-tuning)
    Domain-specific encoder fine-tuning

Why not InfoNCE instead?
    InfoNCE uses the whole batch as negatives → more signal per step.
    Triplet loss uses one negative at a time → simpler but less efficient.
    Most modern work prefers InfoNCE. Triplet loss is older but still used
    for fine-tuning on small datasets where you have explicit labeled pairs.
```

### MSE (Mean Squared Error) — For Reconstruction

```text
L = (1/N) × Σ (predicted - actual)²

Where it's used in the encoder/decoder world:

    MAE (Masked Autoencoder):
        Mask 75% of image patches.
        Model predicts the missing pixel values.
        Loss = MSE between predicted pixels and actual pixels.

    BYOL (self-supervised):
        Student predicts teacher's representation.
        Loss = MSE between student output and teacher output.

    Autoencoders (general):
        Compress input → latent space → reconstruct input.
        Loss = MSE between reconstruction and original.

NOT used for classification or language modelling.
Cross-entropy works much better for discrete predictions (tokens, classes).
```

### KL Divergence — Measuring Distribution Mismatch

```text
KL(P || Q) = Σ P(x) × log(P(x) / Q(x))

Measures: "how different is distribution Q from distribution P?"

    KL = 0: distributions are identical.
    KL > 0: some mismatch (larger = more different).

Where it's used:

    Knowledge distillation:
        Teacher model outputs: P = [0.7, 0.2, 0.1]    (soft targets)
        Student model outputs: Q = [0.6, 0.3, 0.1]    (student's prediction)
        Loss = KL(P || Q) → push student to match teacher's distribution.

        Why KL instead of cross-entropy?
        The teacher's soft targets contain information beyond the correct answer.
        P = [0.7, 0.2, 0.1] says "it's probably a cat, but slightly dog-like."
        Hard label says just "cat." The soft distribution is richer.

    RLHF / alignment:
        KL penalty between the fine-tuned model and the base model.
        "Don't drift too far from the pre-trained model's knowledge."
        Prevents the model from gaming the reward signal.

    VAEs (Variational Autoencoders):
        Push the latent distribution toward a standard normal.
```

### Summary: Loss Function → Model

```text
| Loss function   | Intuition                                    | Used by                        |
| --------------- | -------------------------------------------- | ------------------------------ |
| Cross-entropy   | "Be confident about the right answer"        | GPT, BERT, LLaMA, ResNet      |
|                 |                                              | (classification & LM)          |
| InfoNCE /       | "Pick your partner from the crowd"           | CLIP, SimCLR, Sentence-BERT,  |
| contrastive     |                                              | CLAP, MoCo                     |
| Triplet         | "Positive closer than negative, by margin"   | FaceNet, product search,       |
|                 |                                              | fine-tuning encoders           |
| MSE             | "Reconstruct the original exactly"           | MAE, BYOL, autoencoders       |
| KL divergence   | "Match this probability distribution"        | Distillation, RLHF, VAE       |
```

---

## Part 4: Putting It Together — What Each Model Uses

```text
| Model           | Type         | Activation | Optimizer | Loss          | lr Schedule     |
| --------------- | ------------ | ---------- | --------- | ------------- | --------------- |
| ResNet          | CNN encoder  | ReLU       | SGD+Mom   | Cross-entropy | Step decay      |
| BERT            | Text encoder | GELU       | AdamW     | Cross-entropy (MLM) | Linear decay |
| Sentence-BERT   | Text encoder | GELU       | AdamW     | InfoNCE       | Linear warmup   |
| GPT-2/3         | Decoder      | GELU       | AdamW     | Cross-entropy (next-token) | Cosine decay |
| LLaMA 1/2/3     | Decoder      | SwiGLU     | AdamW     | Cross-entropy (next-token) | Cosine decay |
| Mistral/Mixtral | Decoder      | SwiGLU     | AdamW     | Cross-entropy (next-token) | Cosine decay |
| ViT             | Img encoder  | GELU       | AdamW     | Cross-entropy | Cosine decay    |
| DINOv2          | Img encoder  | GELU       | AdamW     | DINO loss (cross-entropy variant) | Cosine decay |
| SimCLR          | Img encoder  | ReLU       | LARS      | InfoNCE       | Cosine decay    |
| CLIP            | Multimodal   | GELU       | AdamW     | InfoNCE (symmetric) | Cosine decay |
| MAE             | Img encoder  | GELU       | AdamW     | MSE (pixel reconstruction) | Cosine decay |
```

**The trends:**
- Early models (ResNet, 2015): ReLU + SGD + cross-entropy
- Transformer era (BERT/GPT, 2018-2020): GELU + AdamW + cross-entropy
- Modern LLMs (LLaMA, 2023+): SwiGLU + AdamW + cross-entropy
- Embedding models: same architecture activations/optimizers, but contrastive loss instead of cross-entropy
- Self-supervised vision: same activations/optimizers, but InfoNCE or MSE reconstruction loss
