## How Fine-Tuning Actually Works — What Changes, What Stays, and Why

File 09 covers WHEN and WHICH method to use (LoRA, QLoRA, full fine-tuning). This file explains the **mechanics** — what literally happens inside the model when you fine-tune it.

---

## What Does the Model Look Like After Pre-Training?

After pre-training (file 05), the model has two parts:

```text
1. The transformer body (e.g., 12 layers for GPT-1, 32 for LLaMA 7B):
    Takes token IDs → produces a hidden vector (e.g., 768-dim) at every position.
    This is where all the "understanding English" lives.
    Weights: ~110M params (GPT-1) or ~7B params (LLaMA 7B). FULLY TRAINED.

2. The pre-training head (W_vocab):
    A single linear layer: hidden_dim → vocab_size
    Example: 768 → 40,000 (GPT-1) or 4096 → 32,000 (LLaMA 7B)

    Takes the hidden vector at any position → produces a score for
    every token in the vocabulary → softmax → next-token probabilities.

    This is just a matrix multiplication:
        logits = h × W_vocab    (768-dim × 768×40,000 = 40,000 scores)
        probs = softmax(logits)

Together during pre-training:
    tokens → [transformer body] → hidden vectors → [W_vocab] → next-token probs
                                                      ↑
                                          this head predicted the NEXT TOKEN
```

---

## Adapting for a Classification Task (e.g., Sentiment)

### Step 1: Add a Task Head

```text
The pre-trained model knows how to predict the next token.
But now we want it to predict: positive or negative (2 classes).

We ADD a new linear layer — the "task head":
    W_task: hidden_dim → num_classes
    Example: 768 → 2  (for binary sentiment)

    This is a tiny matrix: 768 × 2 = 1,536 new parameters.
    Initialized randomly. Knows nothing yet.

The model now has TWO heads sitting on top of the same transformer:

    tokens → [transformer body] → hidden ─┬─→ [W_task]  → 2 class probs
                12 layers                  │      ↑ NEW (randomly initialized)
                ~110M params               │
                KEPT from pre-training     └─→ [W_vocab] → 40,000 token probs
                                                  ↑ KEPT from pre-training

    W_task is the only new component. Everything else is reused.
```

### Step 2: Pick the Right Hidden Vector

```text
The transformer produces a hidden vector at EVERY token position.
Which one do we feed into the task head?

Input: [Start] "This movie was fantastic" [Extract]

Position:   [Start]  This  movie  was  fantastic  [Extract]
Hidden:      h₀      h₁    h₂    h₃     h₄       h₅

GPT-1 uses h₅ — the hidden state at the LAST token ([Extract]).

Why the last token?
    This is a DECODER with causal attention. Each position can only
    see tokens to its LEFT:
        h₀ sees: [Start] only
        h₁ sees: [Start], This
        h₂ sees: [Start], This, movie
        ...
        h₅ sees: [Start], This, movie, was, fantastic, [Extract]
                 ↑ the ONLY position that has seen the ENTIRE input

    h₅ is the only vector that summarizes the complete input.
    That's why it's the one fed into the task head.

    (BERT uses the FIRST token [CLS] instead, because BERT's
     bidirectional attention means every position sees everything.
     GPT's causal attention means only the LAST position has full context.)
```

### Step 3: The Full Forward Pass (Toy Example with Numbers)

```text
Task: sentiment classification (positive / negative)
Input: "This movie was fantastic"

── FORWARD PASS ──

Step 1: Tokenize and add special tokens
    [Start] This movie was fantastic [Extract]
    → token IDs: [40001, 1212, 3415, 873, 12045, 40002]

Step 2: Through the 12 transformer layers
    Each token gets a 768-dim hidden vector.
    We care about the LAST position ([Extract]):
        h₅ = [0.82, -0.45, 0.13, ..., 0.67]    (768 values)

Step 3a: TASK HEAD — classify sentiment
    logits_task = h₅ × W_task
                = [768-dim vector] × [768 × 2 matrix]
                = [2.3, -1.1]     (2 scores: one per class)

    probs_task = softmax([2.3, -1.1])
               = [0.97, 0.03]     (97% positive, 3% negative)

    True label: positive (index 0)
    Task loss (L₂) = -log(0.97) = 0.03    (low — good prediction)

Step 3b: PRE-TRAINING HEAD — also predict next tokens (auxiliary)
    SIMULTANEOUSLY, the old W_vocab head still predicts the next token
    at EVERY position (same as during pre-training):

    h₀ ([Start])    × W_vocab → predict "This"?     loss₁ = -log(P("This"))
    h₁ (This)       × W_vocab → predict "movie"?    loss₂ = -log(P("movie"))
    h₂ (movie)      × W_vocab → predict "was"?      loss₃ = -log(P("was"))
    h₃ (was)        × W_vocab → predict "fantastic"? loss₄ = -log(P("fantastic"))
    h₄ (fantastic)  × W_vocab → predict "[Extract]"? loss₅ = -log(P("[Extract]"))

    LM loss (L₁) = average(loss₁ + loss₂ + loss₃ + loss₄ + loss₅) = 1.2

── COMBINED LOSS ──

Step 4: Combine both losses
    L₃ = L₂ (task loss) + λ × L₁ (LM loss)
       = 0.03            + 0.5 × 1.2
       = 0.03            + 0.6
       = 0.63

── BACKPROPAGATION ──

Step 5: Gradients flow from L₃ backward through:
    → W_task              (learns what hidden directions = which class)
    → W_vocab             (stays good at language modelling)
    → All 12 transformer layers   (gets nudged by BOTH losses)
    → Embeddings          (gets nudged by BOTH losses)

    EVERYTHING gets updated. The entire model shifts slightly.
```

---

## What Each Part Is Learning During Fine-Tuning

```text
W_task (NEW, 768 × 2, started random):
    Starts knowing nothing. Learns:
    "When the 768-dim vector has high values in dims 42, 156, 389
     → that means positive. When dims 7, 201, 550 are high → negative."

    The transformer already "understands" the sentence.
    h₅ already ENCODES "this is positive" — the information is in there.
    W_task just learns HOW TO READ that encoding and map it to a label.
    It's learning a simple linear boundary in 768-dimensional space.

    Analogy: the transformer is a person who understands the movie review.
    W_task is teaching them which button to press (positive / negative)
    to express their understanding.

Transformer body (KEPT, 12 layers, ~110M params):
    Already understands English. Gets GENTLY nudged:
    "Adjust your representations so that h₅ is even more useful
     for this specific classification task."

    The learning rate is 6.25e-5 — that's 25× SMALLER than
    pre-training's 2.5e-4. This means tiny updates.
    Don't destroy what you learned from 7,000 books.
    Just adjust slightly.

    Like fine-tuning a guitar that's already in tune — small turns,
    not restringing.

W_vocab (KEPT, 768 × 40,000):
    Already knows how to predict next tokens.
    Gets slightly updated to stay good at language modelling
    on this domain (movie reviews vs books).
```

---

## Why the Auxiliary LM Loss Helps

```text
Without auxiliary loss (λ = 0):
    Only the task loss sends gradients through the transformer.
    Problem: the sentiment dataset might have only 10,000 examples.
    The 110M parameter transformer starts "forgetting" general English
    because it's only being rewarded for 2-class classification.
    It overfits to the small labelled dataset.

    Epoch 1: "this is great" → positive ✓, still good at English
    Epoch 3: "this is great" → positive ✓, worse at English
    Epoch 10: "this is great" → positive ✓, can barely form sentences

With auxiliary loss (λ = 0.5):
    The LM loss says: "while you're learning sentiment, ALSO keep
    being good at predicting the next word."

    This is a REGULARIZER — it prevents the transformer from drifting
    too far from its pre-trained knowledge. Two simultaneous objectives
    keep the model grounded.

    Like a musician learning a new song while also practicing scales.
    The scales keep their fundamentals sharp.

The paper found:
    Auxiliary loss helps on LARGE fine-tuning datasets (MNLI, QQP).
    Doesn't help much on small datasets.
    λ = 0.5 was a good default across tasks.
```

---

## Handling Structured Inputs (Pairs, Triplets)

The pre-trained model processes ONE contiguous sequence. But many tasks have structured inputs. The solution: **flatten everything into a single sequence with delimiter tokens.**

### Classification (single text → label)

```text
    [Start] This movie was fantastic [Extract]
            └───────────────────────┘
                    input text

    → Transformer → take h at [Extract] → W_task (768→2) → softmax → label
    Straightforward.
```

### Entailment (two sentences → relationship label)

```text
    Task: "Does sentence A imply sentence B?"
    Labels: entailment / contradiction / neutral

    [Start] The dog is sleeping [Delim] An animal is resting [Extract]
            └─── premise ──────┘        └─── hypothesis ─────┘

    The [Delim] token tells the model "boundary between two pieces."
    → Take h at [Extract] → W_task (768 → 3) → softmax → 3-class probs

    Example:
        h_extract = [0.55, -0.12, 0.89, ..., 0.33]    (768-dim)
        logits = h_extract × W_task = [3.2, -0.8, 0.1]
        softmax → [0.92, 0.02, 0.06]
                   ↑ entailment (correct — sleeping dog IS a resting animal)
```

### Similarity (two sentences → same meaning or not?)

```text
    Problem: "A then B" and "B then A" should give the SAME answer.
    But a decoder processes left-to-right, so order matters.

    Solution: process BOTH orderings, ADD the representations:

    Pass 1: [Start] text1 [Delim] text2 [Extract] → h₁ = [0.45, -0.23, ...]
    Pass 2: [Start] text2 [Delim] text1 [Extract] → h₂ = [0.38, -0.19, ...]

    Combined: h₁ + h₂ = [0.83, -0.42, ...]    (element-wise addition)
    → W_task → softmax → similar or not

    Adding both orderings makes the result symmetric:
    same answer regardless of which text is first.
```

### Multiple Choice / Question Answering

```text
    Context: "The sky turned dark and rain began to fall."
    Question: "What happened to the weather?"
    Answers: A) "It got sunny"  B) "A storm arrived"  C) "Snow fell"

Why can't we just do classification (768 → 3)?
    Classification works when the classes are FIXED and KNOWN ahead of time
    (e.g., always "positive" or "negative"). But multiple-choice answers
    are DIFFERENT for every question. The model needs to READ each answer
    option and judge whether it fits the context. You can't hardcode
    "A storm arrived" as class index 1 — it's a different answer every time.

    So instead of asking "which class?", we ask a simpler question
    for each option: "how well does THIS answer fit?" — a single score.

The approach: score each option separately, then compare.

    Step 1: Build a separate input sequence per answer option.
            Each sequence contains the SAME context + question,
            but a DIFFERENT answer glued on the end.

        Option A: [Start] The sky turned dark and rain began to fall
                  [Delim] What happened to the weather? $ It got sunny [Extract]

        Option B: [Start] The sky turned dark and rain began to fall
                  [Delim] What happened to the weather? $ A storm arrived [Extract]

        Option C: [Start] The sky turned dark and rain began to fall
                  [Delim] What happened to the weather? $ Snow fell [Extract]

        The $ symbol is just a separator between question and answer
        (part of the input formatting, not a special token).

    Step 2: Run EACH sequence through the SAME transformer (3 forward passes).
            Extract the hidden vector at [Extract] from each:

        Option A → h_A = [0.12, -0.45, 0.88, ..., 0.33]   (768-dim)
        Option B → h_B = [0.67, -0.11, 0.52, ..., 0.91]   (768-dim)
        Option C → h_C = [0.34, -0.28, 0.71, ..., 0.55]   (768-dim)

        Each hidden vector encodes: "the transformer read the context,
        the question, AND this specific answer together."
        h_B encodes the fact that "a storm arrived" fits naturally
        after "the sky turned dark and rain began to fall."

    Step 3: W_task maps each 768-dim vector to a SINGLE number (768 → 1).
            This is just a dot product — "how good is this option?"

        score_A = h_A × W_task = 0.3    (low — bad fit)
        score_B = h_B × W_task = 2.1    (high — good fit)
        score_C = h_C × W_task = 0.7    (medium — okay fit)

        W_task is a vector of 768 weights. It learns:
        "when these dimensions are high, the answer fits the context."
        It's the SAME W_task applied to all three options.

    Step 4: Softmax ACROSS the three scores to get probabilities.

        softmax([0.3, 2.1, 0.7]) = [0.10, 0.62, 0.28]
                                           ↑ B wins (correct)

        Then cross-entropy loss against the true label (B).

Why 768 → 1 and not 768 → 3?
    Because the NUMBER of answer options can vary between questions.
    Some questions have 2 options, some have 4, some have 5.
    By scoring each option independently (768 → 1), the same W_task
    works regardless of how many options there are.

    Compare:
        Classification (sentiment):  768 → 2  (always 2 classes)
        Entailment:                  768 → 3  (always 3 classes)
        Multiple choice:             768 → 1  (per option, any number of options)

Cost: 3 answer options = 3 forward passes through the transformer.
    This is 3× slower than classification (which needs just 1 pass).
    But there's no other way — the model must READ each answer to judge it.
```

---

## The Special Tokens

```text
[Start], [Delim], [Extract] are NEW tokens added to the vocabulary.

    Each gets its own embedding vector (768-dim), randomly initialized.
    These embeddings are learned during fine-tuning.

    [Start]:   signals "beginning of input"
    [Delim]:   signals "boundary between two pieces"
    [Extract]: signals "extract the representation here for the task head"

Total new parameters for a 3-class task:
    [Start] embedding:   768 values
    [Delim] embedding:   768 values
    [Extract] embedding: 768 values
    W_task:              768 × 3 = 2,304 values
    ────────────────────────────────────
    Total:               4,608 new parameters

    Compare to the 110,000,000 pre-trained parameters.
    Fine-tuning adds 0.004% new parameters. Everything else is reused.
```

---

## How This Connects to Modern Fine-Tuning (File 09)

```text
GPT-1 style fine-tuning = "FULL fine-tuning":
    Update ALL 110M parameters + the new task head.
    Works great at 110M scale. Becomes impractical at 7B-70B scale.

    At LLaMA 7B scale:
        Full fine-tuning needs ~112 GB of GPU memory (file 09).
        You'd need 1-2 A100 GPUs just for one model.

    This is exactly why LoRA, QLoRA, and prompt tuning were invented:
        Instead of updating ALL weights, update a tiny fraction.
        Same idea (task-specific adaptation), much cheaper.

The evolution:
    GPT-1 (2018):  Full fine-tuning (update all 110M params)
    LoRA (2021):   Low-rank adapters (update ~0.1% of params)
    QLoRA (2023):  4-bit quantized model + LoRA (fit 65B on 1 GPU)

    All three do the same thing conceptually:
    "Nudge the pre-trained model's representations so they're more
     useful for this specific task."
    They just differ in HOW MANY weights they nudge.

GPT-1's contribution:
    Proved that the two-stage paradigm (pre-train → fine-tune) works.
    Every modern fine-tuning method is a descendant of this idea.
```
