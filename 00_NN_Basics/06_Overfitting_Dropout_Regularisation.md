# Overfitting, Dropout, and Regularisation

## The Core Problem: Memorisation vs Generalisation

A neural network has two jobs:
1. Learn the *patterns* in training data
2. Apply those patterns to *new, unseen data*

**Overfitting** is when it nails job 1 but fails job 2 — it has memorised the training examples instead of learning the underlying pattern.

```text
Training data:
    "The cat sat on the mat"  → positive
    "A dog ran in the park"   → positive
    "This is terrible"        → negative

Overfitted model might learn:
    "if the sentence contains 'mat' → positive"
    (true for training, useless for new data)

Well-fitted model learns:
    "positive words like 'sat', 'ran' → positive"
    (generalises to new sentences)
```

---

## Spotting Overfitting in Training Curves

Always track loss on both training data and a **held-out validation set** (data the model never trains on).

```text
Epoch    Train Loss    Val Loss
  5         0.80         0.85    ← both decreasing: good
 10         0.50         0.55    ← still good
 20         0.25         0.32    ← small gap: fine
 40         0.10         0.55    ← gap growing: overfitting starts
 60         0.05         0.80    ← clear overfitting: memorising training data
```

The model gets better at training data but *worse* at new data. This is the sign.

---

## Underfitting

The opposite problem: model is too simple or hasn't trained long enough.

```text
Epoch    Train Loss    Val Loss
  5         0.80         0.82
 10         0.78         0.80
 20         0.75         0.77    ← both still high: underfitting
```

Both losses are high and neither is decreasing much. The model hasn't learned enough.

```text
Underfitting ←——————————————→ Overfitting
(too simple,               (too complex,
 not enough training)       too much training)
              ↑
         Sweet spot
```

---

## Fix 1: Dropout

During training, randomly zero out a fraction of neurons in a layer at each step.

```text
Layer output before dropout: [0.5, 1.2, -0.3, 0.8, 0.6, -1.1]
Dropout rate = 0.3 (30% dropped)
Random mask:                  [ 1,   0,   1,   0,   1,   1 ]

After dropout:               [0.5, 0.0, -0.3, 0.0, 0.6, -1.1]
```

**Why this helps:** The network can't rely on any single neuron always being there. It's forced to learn redundant, distributed representations — multiple paths to the same answer. This makes it more robust and less likely to memorise.

**Important:** Dropout is only active during *training*. At inference time, all neurons are active, and their outputs are scaled down by (1 - dropout_rate) to compensate.

```text
Typical dropout rates:
  - 0.1–0.2 for transformers (light regularisation)
  - 0.5 for dense layers in older classifiers
```

---

## Fix 2: Weight Decay (L2 Regularisation)

Add a small penalty to the loss for having large weights.

```text
Loss_total = Loss_original + λ * Σ(w²)

where λ is a small number like 0.01
```

Large weights mean the model is placing heavy bets on specific features — a sign of memorisation. Penalising large weights forces the model to spread its bets more evenly.

**AdamW** (the standard LLM optimiser) has weight decay built in — one reason it's preferred over plain Adam.

---

## Fix 3: Early Stopping

Track validation loss during training. Stop when it starts increasing, even if training loss is still decreasing.

```text
                Train loss
         ↘
           ↘
             ↘_______________
Val loss      ↘
  ↘             ↘
    ↘              ↘___↗ ← stop here (val loss starts rising)
      ↘_______________
                      ↑
                 Best model checkpoint
```

Save the model weights at the point of lowest validation loss. This is the "early stopping checkpoint."

---

## Fix 4: More Data

Often the most effective fix. If the model has more examples to learn from, memorising becomes impractical and generalisation becomes necessary.

This is why large language models train on trillions of tokens — at that scale, memorisation is essentially impossible and the model must learn true patterns.

---

## Summary: What to Reach For

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Train loss low, val loss high | Overfitting | Dropout, weight decay, more data, early stopping |
| Both losses high | Underfitting | Bigger model, more training, lower learning rate |
| Val loss starts rising mid-training | Overfitting during training | Early stopping, add dropout |
| Very noisy val loss curve | Small validation set | Get more validation data |

---

## What You've Already Seen

- **Residual connections** (ResNet file 09, Transformer file 07): also help with generalisation by making training more stable, reducing the chance the model gets stuck in a bad solution
- **Dropout** appears in BERT and GPT training — dropout rate 0.1 on attention weights and FFN layers
- **Weight decay (AdamW)** is the default optimiser for all major LLMs
