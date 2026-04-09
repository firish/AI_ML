## GPT-1 — Improving Language Understanding by Generative Pre-Training

**Authors:** Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever (OpenAI, 2018)

---

## 1. The Problem

```text
In 2018, NLP was stuck in a frustrating pattern:

    For EVERY new task (sentiment analysis, question answering, entailment...),
    you needed:
        1. A large LABELLED dataset specific to that task
        2. A custom architecture designed for that task
        3. Train from scratch (or from word embeddings like Word2Vec)

    Problems:
        - Labelled data is expensive and scarce for most tasks
        - Each task needs its own model → no transfer between tasks
        - Word embeddings (Word2Vec, GloVe) only transfer WORD-level info,
          not sentence-level understanding

    What existed before GPT-1:
        - Word2Vec/GloVe: static word vectors, no context
        - ELMo (2018, same year): contextual embeddings from LSTMs,
          but used as FEATURES fed into task-specific models
        - ULMFiT (Howard & Ruder, 2018): fine-tuning LSTMs on text,
          but LSTMs struggle with long-range dependencies

    The two open questions nobody had answered:
        1. What's the best OBJECTIVE for learning transferable representations?
           (language modelling? translation? something else?)
        2. What's the best way to TRANSFER those representations to new tasks?
           (feature extraction? fine-tuning? new architecture?)
```

---

## 2. The Solution (The Big Idea)

```text
GPT-1's answer to both questions:

    1. Objective: next-token prediction (language modelling) on unlabelled text
    2. Transfer: fine-tune the ENTIRE model on the downstream task

    Step 1: Pre-train a transformer decoder on a huge corpus of
            unlabelled text using next-token prediction.
            (No labels needed. Just predict the next word.)

    Step 2: Fine-tune that same model on each specific task
            with a small labelled dataset.
            (Minimal architecture changes — just add one linear layer.)

    This is the "pre-train then fine-tune" paradigm.
    It seems obvious now. In 2018, it was NOT obvious.

Why this was new:
    - ELMo pre-trained, but then FROZE the representations and used them
      as input features to a separate task-specific model.
    - GPT-1 fine-tunes the WHOLE model end-to-end. The pre-trained weights
      are the starting point, not a frozen feature extractor.
    - ELMo used LSTMs. GPT-1 used a transformer decoder.
      Transformers handle long-range dependencies better.
```

---

## 3. The Architecture

```text
GPT-1 is a 12-layer transformer DECODER (not encoder, not encoder-decoder).

    ┌──────────────────────────────────────┐
    │         Prediction Head              │
    │    (linear → softmax over vocab)     │
    ├──────────────────────────────────────┤
    │                                      │
    │   Transformer Decoder Block × 12     │
    │                                      │
    │   Each block:                        │
    │     1. Masked multi-head self-attn   │
    │        (12 heads, d=768)             │
    │     2. LayerNorm + residual          │
    │     3. FFN (768 → 3072 → 768)       │
    │     4. LayerNorm + residual          │
    │                                      │
    ├──────────────────────────────────────┤
    │   Token Embeddings + Position Embeds │
    │   (learned, not sinusoidal)          │
    └──────────────────────────────────────┘

Specs:
    Layers:          12
    Hidden dim:      768
    Attention heads: 12
    FFN inner dim:   3072 (4× hidden)
    Context window:  512 tokens
    Vocab:           40,000 (BPE)
    Parameters:      ~117M
    Activation:      GELU (not ReLU — one of the first to use GELU)

Why DECODER (not encoder like BERT)?
    Decoders use CAUSAL (masked) attention — each token only sees
    tokens to its LEFT. This is what makes next-token prediction
    possible: you can't predict the next word if you can see it.

    BERT (published a few months later) used an ENCODER with
    bidirectional attention + masking. Different trade-off.
    GPT-1 bet on generation. BERT bet on understanding.
```

---

## 4. Stage 1: Unsupervised Pre-Training

```text
Training objective: predict the next token.

    Given tokens: u₁, u₂, ..., uₙ
    Maximize: Σ log P(uᵢ | uᵢ₋ₖ, ..., uᵢ₋₁)

    This is just: "given the previous k tokens, predict the next one."
    Standard language modelling. Cross-entropy loss.

Training data: BooksCorpus
    - 7,000+ unpublished books (Adventure, Fantasy, Romance)
    - ~800M words
    - Crucially: LONG contiguous text (not shuffled sentences)
    - The model learns to condition on long-range context

    Why books and not Wikipedia?
        Wikipedia is good for facts, but sentences are relatively
        independent. Books have long narrative arcs, character
        development, cause-and-effect across paragraphs.
        This forces the model to learn long-range dependencies.

Training details:
    - Optimizer:         Adam (lr = 2.5e-4, cosine schedule)
    - Batch size:        64
    - Sequence length:   512 tokens
    - Epochs:            100
    - Warmup:            linear over first 2000 steps
    - Dropout:           0.1 (on residual, embedding, and attention)
    - Weight init:       N(0, 0.02)
    - Tokenizer:         BPE, 40,000 merges

What does the model learn?
    After pre-training, the model has learned:
    - Grammar and syntax (how English sentences work)
    - Semantics (what words mean in context)
    - World knowledge (facts embedded in the books)
    - Long-range coherence (tracking topics across paragraphs)

    All from next-token prediction alone. No labels.
```

---

## 5. Stage 2: Supervised Fine-Tuning

```text
After pre-training, adapt to specific tasks with labelled data.

The key insight: MINIMAL architecture changes.

    Pre-trained model stays almost identical.
    Only addition: ONE linear layer on top (the "task head").

    Input tokens → 12 transformer layers → last token's hidden state
    → linear layer → softmax → task prediction

Fine-tuning objective (clever trick):

    L₃ = L₂(task loss) + λ × L₁(language modelling loss)

    They keep the language modelling objective AS AN AUXILIARY LOSS
    during fine-tuning, weighted by λ = 0.5.

    Why? Two benefits:
        1. Improves generalization (regularization effect)
        2. Faster convergence

    The model simultaneously learns the task AND keeps practicing
    next-token prediction, which prevents it from "forgetting"
    what it learned during pre-training.

Fine-tuning is fast:
    - Learning rate: 6.25e-5 (much smaller than pre-training)
    - Batch size: 32
    - Epochs: 3 (just 3!)
    - The model already knows English. Fine-tuning just teaches it
      the specific task format.
```

---

## 6. The Input Transformation Trick

```text
Problem: the pre-trained model processes ONE contiguous sequence.
But many tasks have STRUCTURED inputs (pairs, triplets):

    - Entailment: (premise, hypothesis) → entailment/contradiction/neutral
    - Similarity: (sentence A, sentence B) → similar or not
    - QA: (context, question, answer choices) → which answer

Solution: flatten everything into a single sequence with delimiters.

    Classification:
        [Start] text [Extract] → Transformer → Linear → label
        (straightforward — single text input)

    Entailment:
        [Start] premise [Delim] hypothesis [Extract] → Transformer → Linear
        (concatenate with a delimiter token between them)

    Similarity:
        Process BOTH orderings:
        [Start] text1 [Delim] text2 [Extract] → Transformer → vec₁
        [Start] text2 [Delim] text1 [Extract] → Transformer → vec₂
        Add: vec₁ + vec₂ → Linear → prediction
        (similarity is symmetric, so process both directions)

    Multiple Choice (QA):
        For each answer option k:
        [Start] context [Delim] question $ answer_k [Extract] → Transformer → score_k
        Softmax over all scores → pick highest

    The [Start], [Delim], and [Extract] tokens are randomly initialized
    and learned during fine-tuning. These are the ONLY new parameters
    (besides the final linear layer).

Why this is elegant:
    ZERO changes to the transformer architecture.
    Every task becomes: "process this sequence, extract last token's
    hidden state, apply a linear layer." The pre-trained model
    doesn't need to know what task it's doing.
```

---

## 7. Results

### SOTA on 9 out of 12 benchmarks

```text
Natural Language Inference (NLI):
    Task: given two sentences, is the second entailed by, contradicted
          by, or neutral to the first?

    | Dataset  | Previous SOTA | GPT-1  | Improvement |
    |----------|---------------|--------|-------------|
    | MNLI-m   | 80.6          | 82.1   | +1.5        |
    | MNLI-mm  | 80.1          | 81.4   | +1.3        |
    | SNLI     | 89.3          | 89.9   | +0.6        |
    | SciTail  | 83.3          | 88.3   | +5.0        |
    | QNLI     | 82.3          | 88.1   | +5.8        |

    Note: previous SOTA used ENSEMBLES of 5-9 models.
    GPT-1 is a SINGLE model beating ensembles.

Question Answering:
    | Dataset     | Previous SOTA | GPT-1 | Improvement |
    |-------------|---------------|-------|-------------|
    | Story Cloze | 77.6          | 86.5  | +8.9        |
    | RACE        | 53.3          | 59.0  | +5.7        |

Classification:
    | Dataset | Previous SOTA | GPT-1 | Improvement |
    |---------|---------------|-------|-------------|
    | CoLA    | 35.0          | 45.4  | +10.4       |
    | SST-2   | 93.2          | 91.3  | -1.9 (lost) |

    CoLA = linguistic acceptability ("is this sentence grammatical?")
    The +10.4 jump on CoLA was massive — the model learned English
    grammar from pre-training, not from 8,500 labelled examples.

Overall GLUE benchmark: 72.8 vs 68.9 (previous best)
```

---

## 8. Key Analysis Findings

### More layers transferred = better performance

```text
They tested transferring 1, 3, 6, 9, 12 layers from pre-training:

    Layers transferred → Accuracy
    0 (no pre-training)    ~60%
    1 (embeddings only)    ~68%
    3                      ~72%
    6                      ~76%
    9                      ~80%
    12 (all layers)        ~82%

    Every layer helps. This means every layer in the pre-trained
    model contains USEFUL information for downstream tasks.
    It's not just the embeddings — the deep layers matter.
```

### Transformer beats LSTM

```text
They compared: same framework, same training, but LSTM instead of Transformer.

    Transformer average score: 74.7
    LSTM average score:        69.1

    5.6 point drop just from swapping the architecture.
    The transformer's attention mechanism captures long-range
    dependencies that LSTMs struggle with.

    This is why GPT-1 used a transformer instead of an LSTM like
    ELMo and ULMFiT. The architecture choice mattered.
```

### Pre-training is essential

```text
    With pre-training:      74.7 average
    Without pre-training:   59.9 average

    14.8 point drop without pre-training.
    The model architecture alone isn't enough.
    What it learns from reading books is what makes it work.
```

### Zero-shot behaviour emerges

```text
Even WITHOUT fine-tuning, the pre-trained model shows some ability
on downstream tasks, using heuristic approaches:

    Sentiment analysis:
        Append "very" to input, check if the model predicts
        "positive" or "negative" with higher probability.
        → Performance improves steadily during pre-training.

    This is the first hint of what GPT-2 and GPT-3 would later
    exploit: the pre-trained model already "knows" how to do tasks,
    even before fine-tuning. GPT-1 noticed this. GPT-2 and GPT-3
    pushed it much further.
```

---

## 9. Ablation: What Matters Most

```text
Table 5 from the paper (ablation results):

    | Setting                      | Avg Score |
    |------------------------------|-----------|
    | Full model                   | 74.7      |
    | Without auxiliary LM loss    | 73.7      |  ← λ=0 hurts on large datasets
    | Without pre-training         | 59.9      |  ← biggest drop (14.8 points)
    | LSTM instead of Transformer  | 69.1      |  ← 5.6 point drop

Rankings of what matters:
    1. Pre-training (14.8 point impact)
    2. Transformer architecture (5.6 point impact)
    3. Auxiliary LM loss during fine-tuning (1.0 point impact)

The lesson: the DATA the model saw during pre-training matters
more than anything else. Architecture is second. Training tricks
are a distant third.
```

---

## 10. What GPT-1 Got Right (That Led to GPT-2, GPT-3, ChatGPT)

```text
1. Decoder-only transformers for language modelling ✓
    Every subsequent GPT model uses this exact architecture,
    just bigger. GPT-4 is still a decoder-only transformer.

2. Pre-train on unlabelled text, then fine-tune ✓
    This became THE standard paradigm for NLP.
    BERT (published months later) used the same two-stage approach,
    just with an encoder and a different pre-training game (MLM).

3. Next-token prediction as the universal training objective ✓
    GPT-1 showed this works. GPT-2 showed it scales.
    GPT-3 showed it's all you need (skip fine-tuning, use prompting).

4. Minimal task-specific architecture ✓
    Just add a linear layer. Don't redesign the model for each task.
    This philosophy culminated in GPT-3 where you don't even fine-tune —
    you just prompt.

5. Zero-shot abilities emerge from pre-training ✓
    GPT-1 noticed it. GPT-2 made it the headline.
    GPT-3 made it practical.
```

---

## 11. What GPT-1 Didn't Do (Limitations at the Time)

```text
1. Small scale
    117M parameters, BooksCorpus only (~800M words).
    Compare to GPT-3: 175B parameters, 300B tokens.
    The ideas were right, but the scale was too small to show
    the full potential.

2. Still required fine-tuning
    GPT-1 needed labelled data and task-specific fine-tuning.
    GPT-2 would show you can skip this (zero-shot).
    GPT-3 would show few-shot prompting works at scale.

3. Lost to BERT on some tasks
    BERT (released months later) dominated NLU benchmarks.
    Bidirectional attention (encoder) is genuinely better for
    understanding tasks. GPT-1's causal attention can't see
    future context, which hurts for classification/NLI.

    But GPT bet on GENERATION, not understanding.
    This bet paid off enormously in GPT-2, GPT-3, ChatGPT.

4. No in-context learning
    GPT-1 couldn't learn from examples in the prompt.
    That ability only emerged at GPT-3 scale (175B params).

5. Small context (512 tokens)
    Modern models: 128K-1M tokens.
```

---

## 12. The Historical Significance

```text
GPT-1 (June 2018) vs BERT (October 2018):

    These two papers, published 4 months apart, defined
    the two competing approaches for the next 5 years:

    GPT-1 (decoder):
        - Causal attention (see left only)
        - Next-token prediction
        - Good at GENERATION
        - Bet on: scale up → emergent abilities → general intelligence

    BERT (encoder):
        - Bidirectional attention (see everything)
        - Masked language modelling
        - Good at UNDERSTANDING
        - Bet on: better representations → better features → better task performance

    BERT won the short game (2018-2022):
        Dominated NLU benchmarks, became the default for search,
        classification, similarity.

    GPT won the long game (2022-now):
        Scale + generation + RLHF → ChatGPT → the entire AI revolution.

    GPT-1 is where this bet was placed.

The lineage:
    GPT-1 (2018):  "pre-train a decoder, fine-tune on tasks" (117M, 800M words)
    GPT-2 (2019):  "just pre-train bigger, zero-shot works"  (1.5B, 8B words)
    GPT-3 (2020):  "scale massively, few-shot from prompts"  (175B, 300B tokens)
    ChatGPT (2022): GPT-3.5 + RLHF alignment                (175B, aligned)
    GPT-4 (2023):  multimodal, MoE, frontier quality         (~1.8T total)

    Every step followed directly from GPT-1's core insight:
    train a decoder on next-token prediction, and scale.
```

---

## 13. One-Line Summary

```text
GPT-1 proved that a transformer decoder pre-trained on next-token prediction
learns general language representations that transfer to downstream tasks with
minimal fine-tuning — the idea that created ChatGPT, Claude, and the entire
modern AI landscape.
```
