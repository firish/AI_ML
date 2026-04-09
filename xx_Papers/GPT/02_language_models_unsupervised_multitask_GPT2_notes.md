## GPT-2 — Language Models are Unsupervised Multitask Learners

**Authors:** Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever (OpenAI, 2019)

---

## 1. The Problem

```text
GPT-1 proved: pre-train a decoder, fine-tune on tasks → great results.

But GPT-1 still needed:
    1. A labelled dataset for each task
    2. A task-specific head (linear layer) added on top
    3. 3 epochs of fine-tuning per task

The question GPT-2 asks:
    "What if we DON'T fine-tune at all?
     What if the pre-trained model can just DO tasks
     from reading the prompt — zero-shot?"

Why this matters:
    - Labelled data is expensive and limits what tasks you can do
    - Fine-tuning means N separate model copies for N tasks
    - If zero-shot works, one model handles everything

The bet: a language model trained on enough diverse data will
    learn to perform tasks as a side effect of learning to predict
    the next token. No explicit supervision needed.
```

---

## 2. The Core Insight

```text
GPT-2's key argument (in one sentence):

    "A language model that's good enough at predicting the next token
     has IMPLICITLY learned to do many tasks, because doing those tasks
     is necessary to predict what comes next in natural text."

Why?
    The internet contains text that LOOKS like tasks:

    Translation examples:
        "The French word for 'cat' is 'chat'."
        → A model predicting the next token here is doing translation.

    QA examples:
        "Q: Who wrote Romeo and Juliet? A: William Shakespeare"
        → A model predicting "William Shakespeare" is doing QA.

    Summarization examples:
        "[long article]... TL;DR: [summary]"
        → A model predicting the summary is doing summarization.

    The paper actually found these patterns in the training data.
    Table 1 shows naturally occurring English↔French translation
    pairs found throughout WebText.

The theoretical framing:
    Instead of modelling p(output | input) for each task separately,
    model p(output | input, task) — where the TASK is specified
    in natural language as part of the input.

    Example:
        p("William Shakespeare" | "Who wrote Romeo and Juliet?", "answer the question")

    A language model implicitly learns this because the task
    description, input, and output all appear as one contiguous
    text sequence during training.
```

---

## 3. Training Data: WebText

```text
GPT-1 trained on BooksCorpus (7,000 books, ~800M words).
GPT-2 needed something much bigger and more diverse.

The problem with existing web scrapes:
    Common Crawl is massive but "mostly unintelligible" garbage.
    Too much noise, not enough quality.

The solution: WebText
    - Scraped all outbound links from Reddit posts with ≥3 karma
    - Why Reddit karma? It's a human quality filter:
      "Did other users find this link interesting, educational,
       or just funny?" If yes → probably decent content.
    - Took 45 million links
    - Extracted text using Dragnet + Newspaper content extractors
    - Removed Wikipedia (to avoid overlap with test benchmarks)
    - De-duplicated + heuristic cleaning
    - Cut off at links created before Dec 2017

    Result: ~8 million documents, 40 GB of text
    Compare: BooksCorpus was ~800M words ≈ ~5 GB

    The diversity is key: WebText contains articles, stories,
    how-tos, reviews, forum posts, Q&A, translations...
    This diversity is what makes zero-shot possible.
```

---

## 4. Input Representation: Byte-Level BPE

```text
GPT-1 used standard BPE on Unicode code points (vocab = 40,000).

Problem with Unicode-level BPE:
    Unicode has 130,000+ symbols. You'd need a base vocabulary
    of 130,000 BEFORE any merges. That's already bigger than
    most final vocabularies (32K-64K).

Problem with byte-level models:
    Raw bytes (base vocab = 256) are too fine-grained.
    Byte-level LMs are not competitive on word-level benchmarks.

GPT-2's solution: Byte-level BPE
    Start with 256 byte tokens (covers ANY Unicode string).
    Apply BPE merges on top of those bytes.
    Final vocab: 50,257 tokens.

    Key trick: prevent BPE from merging across character categories.
    Without this, BPE creates tokens like "dog." "dog!" "dog?"
    — wasting vocabulary slots on redundant variations.
    Exception: spaces CAN merge with words (so "the" and " the"
    become separate tokens, which is useful).

    Benefits:
    - Can tokenize ANY Unicode string (no <UNK> tokens ever)
    - WebText has <UNK> only 26 times in 40 billion bytes
    - Works for any language, emoji, code, etc.
    - Gets the compression efficiency of word-level BPE
      with the generality of byte-level models

    This is the tokenizer that GPT-2, GPT-3, and many later
    models adopted. It's the "tiktoken" tokenizer.
```

---

## 5. The Architecture

```text
Same architecture as GPT-1 (transformer decoder), scaled up.

Four model sizes trained:

    | Name     | Params | Layers | Hidden dim | Heads |
    |----------|--------|--------|------------|-------|
    | Small    | 117M   | 12     | 768        | 12    |  ← same as GPT-1
    | Medium   | 345M   | 24     | 1024       | 16    |  ← same size as BERT-Large
    | Large    | 762M   | 36     | 1280       | 20    |
    | XL       | 1542M  | 48     | 1600       | 25    |  ← "GPT-2"

    GPT-2 (the name) usually refers to the 1.5B model.
    That's 13× the parameters of GPT-1.

Changes from GPT-1:
    1. Layer Normalization moved to INPUT of each sub-block
       (pre-norm instead of post-norm):
           GPT-1: x → Attention → LayerNorm → FFN → LayerNorm
           GPT-2: x → LayerNorm → Attention → LayerNorm → FFN

       Why? Pre-norm is more stable during training for deep models.
       With 48 layers, post-norm can have gradient issues.

    2. Additional LayerNorm after the final self-attention block
       (before the prediction head).

    3. Residual layer weights scaled by 1/√N at initialization
       (N = number of residual layers).

       Why? Without this, the residual stream's variance grows
       with depth. At 48 layers, the activations would explode.
       Scaling by 1/√48 ≈ 0.144 keeps them controlled.

    4. Context window: 512 → 1024 tokens (2× GPT-1)

    5. Vocabulary: 40,000 → 50,257 (byte-level BPE)

    6. Batch size: 512 (vs GPT-1's 64)

What stayed the same:
    - Decoder-only transformer (causal attention)
    - Next-token prediction objective
    - GELU activation
    - Learned position embeddings (not sinusoidal)
```

---

## 6. Key Claim: No Fine-Tuning

```text
GPT-1:  pre-train → fine-tune on labelled data → evaluate
GPT-2:  pre-train → evaluate directly (zero-shot)

Zero-shot means:
    Give the model a task described in natural language.
    No gradient updates. No labelled examples. No task head.
    Just read the prompt and generate.

How tasks are specified (examples):

    Summarization:
        Input: "[article text] TL;DR:"
        The model generates the summary after "TL;DR:"

    Translation:
        Input: "translate to French: cheese ="
        Or give examples: "sea otter = loutre de mer, cheese ="
        The model generates "fromage"

    Reading comprehension:
        Input: "[document] Q: [question] A:"
        The model generates the answer

    Question answering:
        Just ask the question. The model generates an answer.

No task-specific head needed.
No [Start], [Delim], [Extract] tokens.
No W_task matrix.
The model's own language generation IS the task solver.
```

---

## 7. Results: Language Modelling

```text
GPT-2 was evaluated zero-shot on 8 language modelling benchmarks.
Zero-shot = no training or fine-tuning on these datasets AT ALL.

SOTA on 7 out of 8 datasets:

    | Dataset       | Metric | Previous SOTA | GPT-2  | Better? |
    |---------------|--------|---------------|--------|---------|
    | LAMBADA       | PPL    | 99.8          | 8.63   | ✓ (12×) |
    | LAMBADA       | ACC    | 59.23         | 63.24  | ✓       |
    | CBT-CN        | ACC    | 85.7          | 93.30  | ✓       |
    | CBT-NE        | ACC    | 82.3          | 89.05  | ✓       |
    | WikiText-2    | PPL    | 39.14         | 18.34  | ✓       |
    | PTB           | PPL    | 46.54         | 35.76  | ✓       |
    | WikiText-103  | PPL    | 18.3          | 17.48  | ✓       |
    | 1BW           | PPL    | 21.8          | 42.16  | ✗       |

    PPL = perplexity (lower is better)
    ACC = accuracy (higher is better)

    The one failure: 1 Billion Word Benchmark (1BW).
    Why? 1BW shuffles sentences and destroys all long-range structure.
    GPT-2 was trained on coherent documents — shuffled text is
    out-of-distribution for it. Also 1BW is the largest dataset,
    and its heavy preprocessing removes the patterns GPT-2 learned.

Key result: LAMBADA
    LAMBADA tests long-range dependencies: predict the final word
    of a sentence that requires 50+ tokens of context.
    GPT-2 crushed it: perplexity from 99.8 → 8.63 (12× better).
    Accuracy: 19% → 63.24%.
    This shows GPT-2 learned to track context across long passages.

Performance scales with model size:
    Every task improves as models get bigger (117M → 1542M).
    The log-linear scaling curves (Figure 1) show no sign of
    plateauing — suggesting even bigger models would do even better.
    This observation directly motivated GPT-3.
```

---

## 8. Results: Zero-Shot Task Performance

```text
The headline results — GPT-2 doing tasks it was never trained for:

READING COMPREHENSION (CoQA):
    55 F1 — matches or beats 3 of 4 supervised baselines
    (baselines trained on 127,000+ labelled QA pairs)
    GPT-2 used 0 labelled examples.
    Method: condition on document + conversation history + "A:" token

SUMMARIZATION (CNN/Daily Mail):
    Prompt: append "TL;DR:" after the article
    ROUGE-L: 26.58 (vs 38.34 for supervised SOTA)
    Not great quantitatively, but qualitatively the summaries
    are coherent. Removing "TL;DR:" drops ROUGE by 6.4 points,
    showing the model understands the task hint.

TRANSLATION (WMT-14 English→French):
    5 BLEU — worse than word-by-word substitution
    But English→French with just 10MB of French in training data!
    On French→English: 11.5 BLEU — much better (strong English LM helps)
    Outperforms several unsupervised MT baselines.

WINOGRAD SCHEMA (commonsense reasoning):
    70.70% accuracy — 7 point improvement over previous SOTA
    Task: resolve ambiguous pronouns using world knowledge
    Example: "The trophy doesn't fit in the suitcase because IT is too big."
              (IT = trophy or suitcase?)

QUESTION ANSWERING (Natural Questions):
    4.1% exact match — not practically useful
    BUT: the smallest model (117M) gets < 1% (barely above
    the "always guess the most common answer type" baseline).
    GPT-2 (1542M) answers 5.3× more questions correctly.
    On its most confident 1% of answers: 63.1% accuracy.
    Table 5 shows impressive examples: correctly answers
    "Who wrote the origin of species?" → "Charles Darwin" (83.4%)
    "Who came up with the theory of relativity?" → "Albert Einstein" (76.4%)
```

---

## 9. The Scaling Story

```text
The most important finding in the paper (for the future of AI):

    Performance improves LOG-LINEARLY with model size
    across ALL tasks.

    117M → 345M → 762M → 1542M
    Every step up = consistent improvement on every benchmark.

    And crucially: the curves are NOT flattening out.
    Even at 1542M, the model is still UNDERFITTING WebText.
    Both train and test perplexity keep improving with size.

    Figure 4:
        117M: ~16 PPL (train), ~13 PPL (test)
        345M: ~12 PPL (train), ~10 PPL (test)
        762M: ~9 PPL (train), ~8 PPL (test)
        1542M: ~7 PPL (train), ~6 PPL (test)

    Train and test curves move together — no overfitting.
    The model wants to be BIGGER.

This is the observation that led to:
    - GPT-3 (175B) — 100× bigger → few-shot from prompts works
    - Scaling laws (Kaplan et al., 2020) — formalizing this
    - The entire "just make it bigger" era of AI (2020-2024)

GPT-2 showed the trend. GPT-3 proved it was real.
```

---

## 10. Generalization vs Memorization

```text
Important concern: is GPT-2 just memorizing WebText?

The paper does careful analysis (Section 4):

    Overlap analysis using Bloom filters:
        Built Bloom filters containing 8-grams of WebText training tokens.
        Checked what percentage of test sets overlap with training data.

    Results:
        WebText train vs test benchmarks: 1-6% overlap (avg 3.2%)
        Most benchmarks have LARGER overlap with their OWN
        training splits (avg 5.9%) than with WebText.

    Example: 1 Billion Word Benchmark has 13.2% overlap
    with its own training set.

    On LAMBADA specifically:
        Only 1.2% overlap. Excluding overlapping examples
        barely changes results (63.2% → 62.9% accuracy).

    On CoQA:
        ~15% of documents in the news domain appear in WebText.
        Gain from overlap: only 0.5-1.0 F1.

    Key evidence against memorization:
        Figure 4 shows train and test perplexity both improve
        as model gets bigger AND they move together.
        If the model were memorizing, train loss would drop
        while test loss stayed flat or rose.
        Instead: test loss tracks train loss closely.
        Even GPT-2 at 1.5B is still UNDERFITTING WebText.

Text memorization analysis (Appendix):
    GPT-2 occasionally memorizes verbatim text that appears
    many times in training (e.g., Gettysburg Address appears
    ~40 times in WebText → model can reproduce it).
    But Figure 5 shows GPT-2's generated samples overlap with
    training data LESS than actual held-out WebText test articles do.
    The model is generating, not copying.
```

---

## 11. What Changed from GPT-1 to GPT-2

```text
A direct comparison:

    | Aspect              | GPT-1              | GPT-2              |
    |---------------------|--------------------|--------------------|
    | Parameters          | 117M               | 1,542M (13×)       |
    | Layers              | 12                 | 48 (4×)            |
    | Hidden dim          | 768                | 1600 (2×)          |
    | Context window      | 512                | 1024 (2×)          |
    | Training data       | BooksCorpus (5GB)  | WebText (40GB, 8×) |
    | Vocab               | 40,000 (BPE)       | 50,257 (byte BPE)  |
    | LayerNorm position  | Post-norm          | Pre-norm            |
    | Residual init scale | N(0, 0.02)         | N(0, 0.02)/√N      |
    | Evaluation          | Fine-tune then eval| Zero-shot eval      |
    | Task heads needed?  | Yes (W_task)       | No                  |

    The architecture is the SAME (decoder-only transformer).
    The differences are:
        1. Scale (13× params, 8× data)
        2. Small stability fixes (pre-norm, residual scaling)
        3. Better tokenizer (byte-level BPE)
        4. The ambition: don't fine-tune at all

    GPT-1 asked: "Can pre-training help?"
    GPT-2 asked: "Can pre-training be ENOUGH?"
```

---

## 12. The Controversy: Staged Release

```text
GPT-2 was famous for something other than its results:
OpenAI initially REFUSED to release the full model.

Timeline:
    Feb 2019: Paper published, only 117M model released
    Reason: "too dangerous" — could generate convincing fake text
    May 2019: 345M released
    Aug 2019: 762M released
    Nov 2019: Full 1.5B released

The "talking unicorn" example (Table 13 in appendix):
    Given a 2-sentence prompt about scientists discovering
    unicorns in the Andes, GPT-2 generated a coherent
    multi-paragraph news article with quotes from fictional
    scientists, specific details, and a narrative arc.

    This was genuinely shocking in 2019.
    No model had produced text this coherent before.

Impact of the staged release:
    - Started the AI safety debate around language models
    - Made GPT-2 incredibly famous (Streisand effect)
    - Set precedent for staged/restricted model releases
    - Community largely concluded the fears were overblown
      (the model could be fine-tuned but wasn't uniquely dangerous)
    - Dario Amodei (co-author, later founded Anthropic)
      continued thinking about AI safety as a core concern
```

---

## 13. What GPT-2 Got Right

```text
1. Scale improves capabilities ✓
    The log-linear scaling was real and continued to GPT-3/4.
    "Just make it bigger" became the dominant strategy.

2. Zero-shot abilities are real ✓
    GPT-1 hinted at it. GPT-2 demonstrated it convincingly.
    GPT-3 made it practical with few-shot prompting.

3. Task specification through natural language ✓
    "TL;DR:", "Q: ... A:", "translate to French:"
    This is the ancestor of modern prompting and instruction following.

4. Diverse training data matters ✓
    Moving from books-only to web-diverse text made zero-shot possible.
    Every subsequent model uses web-diverse training data.

5. Byte-level BPE is the right tokenizer ✓
    GPT-2's tokenizer (or variants of it) is used by
    GPT-3, GPT-4, and many other models.
```

---

## 14. What GPT-2 Couldn't Do Yet

```text
1. Few-shot learning
    GPT-2 is zero-shot only. It can't learn from examples
    in the prompt. That ability required GPT-3's scale (175B).

2. Follow instructions reliably
    "TL;DR:" works but is fragile. Remove the hint and
    performance drops 6.4 ROUGE points on summarization.
    True instruction following needed RLHF (ChatGPT, 2022).

3. Competitive on most practical tasks
    Translation: 5 BLEU (useless in practice)
    QA: 4.1% exact match (useless in practice)
    Summarization: below supervised baselines
    The zero-shot ABILITY existed but wasn't PRACTICAL yet.

4. Handle the One Billion Word Benchmark
    Shuffled sentences broke the model. GPT-2 relies on
    document-level coherence, not sentence-level patterns.

5. Not memorize famous text
    Gettysburg Address and other frequently-repeated content
    could be reproduced verbatim. Early sign of the
    memorization problem that persists in modern LLMs.
```

---

## 15. The Historical Arc: GPT-1 → GPT-2 → GPT-3

```text
GPT-1 (2018, 117M):
    "Pre-train a decoder, fine-tune on tasks."
    Proved: pre-training transfers knowledge.
    Still needed: labelled data + fine-tuning for each task.

GPT-2 (2019, 1.5B):
    "Don't fine-tune. Just make it bigger and more diverse."
    Proved: zero-shot abilities emerge from scale.
    Still needed: much more scale for practical zero-shot.

GPT-3 (2020, 175B):
    "Scale 100× more. Use few-shot examples in the prompt."
    Proved: in-context learning works at scale.
    Still needed: alignment (model often unhelpful/harmful).

ChatGPT (2022, GPT-3.5 + RLHF):
    "Align to human preferences with RLHF."
    Proved: alignment makes models practically useful.
    Changed the world.

Each step follows directly from the previous one.
GPT-2's contribution: proving that scale → emergent zero-shot abilities,
and that diverse web data is the fuel that makes it happen.
```

---

## 16. One-Line Summary

```text
GPT-2 showed that scaling a decoder-only language model to 1.5B parameters
and training on diverse web text produces emergent zero-shot task abilities —
the finding that launched the scaling era and led directly to GPT-3.
```
