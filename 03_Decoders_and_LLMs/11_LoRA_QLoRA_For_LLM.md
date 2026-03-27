## Fine-Tuning, LoRA, and Adaptation

### When Do You Need Fine-Tuning?

You have a pre-trained model (LLaMA, Mistral, etc.) that's already been through SFT and RLHF. When is that not enough?

```text
Prompt engineering alone works when:
    ✓ The task is general (Q&A, summarisation, coding)
    ✓ A few examples in the prompt get you 90% of the way
    ✓ You don't need domain-specific vocabulary or style

Fine-tuning is worth it when:
    ✗ The model doesn't know your domain (medical, legal, proprietary data)
    ✗ You need a very specific output format consistently
    ✗ You want to distill a large model into a smaller, faster one
    ✗ You need the model to stop doing something it keeps doing
    ✗ Prompt engineering gets you to 80% but you need 95%
    ✗ You're paying too much for API calls (fine-tuned small model replaces large model)
```

---

## The Spectrum of Adaptation

```text
Cheapest / simplest                                      Most expensive / powerful
──────────────────────────────────────────────────────────────────────────────────
Prompt         Few-shot      Prompt       LoRA /     Full
engineering    examples      tuning       QLoRA      fine-tuning
                             (soft                   (update all
                              prompts)                weights)

No training    No training   Train ~10K   Train ~1%  Train 100%
needed         needed        params       of params  of params

Minutes        Minutes       Hours        Hours      Days-weeks

$0             $0            ~$10         $10-1000   $1000-100K+
```

---

## Full Fine-Tuning

### How It Works

Take the pre-trained model. Continue training on your data. Update ALL parameters.

```text
Same process as pre-training / SFT:
    1. Prepare dataset: (input, desired_output) pairs
    2. Forward pass through the model
    3. Compute loss (cross-entropy on output tokens)
    4. Backprop through ALL layers
    5. Update ALL weights with AdamW

LLaMA 7B: update all 7 billion parameters
LLaMA 70B: update all 70 billion parameters
```

### Why It's Expensive

```text
Memory required during training (much more than inference!):

    For each parameter, training stores:
    1. The weight itself              (2 bytes, FP16)
    2. The gradient                   (2 bytes, FP16)
    3. Adam optimizer state: m        (4 bytes, FP32)
    4. Adam optimizer state: v        (4 bytes, FP32)
    5. Master weight copy             (4 bytes, FP32)
    Total: ~16 bytes per parameter

    LLaMA 7B:  7B × 16 bytes = 112 GB     (1-2 A100s)
    LLaMA 70B: 70B × 16 bytes = 1.12 TB   (8-16 A100s)

    Plus: activations stored for backprop, KV cache, data batches
    Real total is ~20 bytes per parameter.
```

### When to Use Full Fine-Tuning

```text
✓ You have a lot of domain data (millions of examples)
✓ You have the GPU budget (multiple A100s/H100s)
✓ You need maximum quality for a specific domain
✓ You're a company building a core product on this model

✗ Don't use for small datasets (overfits immediately)
✗ Don't use if you can get away with LoRA (much cheaper, nearly as good)
```

### Catastrophic Forgetting

```text
The risk: fine-tuning on domain data can ERASE general knowledge.

Before fine-tuning:
    "What is the capital of France?" → "Paris"
    "Summarise this medical report:" → decent summary

After fine-tuning on medical data:
    "What is the capital of France?" → "The patient presents with..."
    "Summarise this medical report:" → excellent summary

The model "forgot" general knowledge because the fine-tuning
data only contained medical text.

Mitigations:
    1. Mix domain data with general data (e.g., 50/50)
    2. Use a low learning rate (small weight changes)
    3. Use LoRA instead (doesn't modify base weights)
    4. Train for fewer steps (stop before forgetting kicks in)
```

---

## LoRA — Low-Rank Adaptation

### The Key Insight

When you fine-tune a model, the weight CHANGES (ΔW) are **low-rank** — they can be approximated by the product of two much smaller matrices.

```text
Full fine-tuning:
    W_new = W_original + ΔW

    W_original: 4096 × 4096 = 16.7M parameters
    ΔW:         4096 × 4096 = 16.7M parameters to learn

LoRA:
    Instead of learning the full ΔW, decompose it:
    ΔW ≈ A × B

    A: 4096 × 16  = 65,536 parameters
    B: 16 × 4096  = 65,536 parameters
    Total: 131,072 parameters (vs 16.7M — that's 0.8%!)

    W_new = W_original + A × B

    The rank r = 16 (the inner dimension of A and B).
    "Low-rank" = the update only spans 16 directions in 4096-d space.
```

### Why Low-Rank Works

```text
Intuition: fine-tuning doesn't need to change EVERYTHING about the model.
It needs to shift the model's behaviour in a few key directions.

Example: fine-tuning for medical text
    The model already knows language, grammar, reasoning.
    It just needs to:
    - Recognise medical terminology better
    - Output in clinical report format
    - Be more precise about dosages and symptoms

    These are a small number of "directions" in weight space.
    A rank-16 update captures them. You don't need rank-4096.

Empirically:
    LoRA with r=16 achieves 95-100% of full fine-tuning quality
    on most tasks, while training <1% of parameters.
```

### How LoRA Training Works

```text
Step 1: Freeze ALL original weights (no gradients, no updates)
Step 2: Add A and B matrices to selected layers
Step 3: Train ONLY A and B

    Original weight W (frozen):
    ┌───────────────────┐
    │                   │
    │    W (4096×4096)  │  ← frozen, no gradient
    │                   │
    └───────────────────┘

    LoRA adapter (trainable):
    ┌──────┐   ┌──────┐
    │A     │   │     B│
    │4096  │ × │ 16   │   ← only these get gradients
    │× 16  │   │×4096 │
    └──────┘   └──────┘

    Forward pass:
    output = x × W + x × A × B
           = x × W + x × ΔW
              ↑         ↑
           frozen    trainable (small)

Initialisation:
    A: random (Gaussian)
    B: all zeros
    → At the start, A × B = 0, so the model behaves exactly
      as the original. Training gradually learns useful ΔW.
```

### Which Layers Get LoRA?

```text
You choose which weight matrices to attach LoRA adapters to.

Common choices:
    Q, K, V, O attention projections  → most impact
    FFN up/down projections            → helps with domain knowledge
    All of the above                   → maximum quality, more params

Typical configuration:
    LoRA on Q, V projections only
    r = 16 or 32
    α = 32 (scaling factor: ΔW = (α/r) × A × B, controls update magnitude)

LLaMA 7B with LoRA (r=16, Q+V only):
    Trainable params: ~4M (vs 7B total = 0.06%)
    Memory: ~14 GB (model in FP16 + small LoRA gradients)
    → Fits on a single consumer GPU
```

### LoRA Alpha and Rank

```text
Two hyperparameters:

Rank (r):
    r = 4:   very small adapter, fast, minimal quality gain
    r = 16:  standard, good balance
    r = 64:  larger adapter, better quality, more memory
    r = 256: approaching full fine-tuning territory

    Higher rank = more parameters = more capacity to learn
    But diminishing returns — r=16 to r=64 is the sweet spot.

Alpha (α):
    Scaling factor for the LoRA update.
    output = x × W + (α / r) × x × A × B

    α = r:   update magnitude ~1 (standard)
    α = 2r:  update magnitude ~2 (stronger adaptation)

    Common practice: set α = 2 × r (e.g., r=16, α=32)
    Then (α / r) = 2, which roughly normalises the update scale
    to be similar regardless of r.
```

---

## QLoRA — LoRA on a Quantized Model

### The Breakthrough

```text
LoRA makes training cheap (few trainable params).
Quantization makes the MODEL small (4-bit weights).
Combine them: fine-tune a 70B model on a single GPU.

QLoRA:
    1. Quantize the base model to 4-bit (NF4 format)
    2. Add LoRA adapters in FP16/BF16
    3. Train only the LoRA adapters
    4. Backprop through the quantized model using a trick:
       dequantize weights on-the-fly for the forward/backward pass,
       but store them in 4-bit.
```

### Memory Comparison

```text
| Method           | LLaMA 7B    | LLaMA 70B     |
| ---------------- | ----------- | ------------- |
| Full fine-tuning | 112+ GB     | 1.12+ TB      |
|                  | (2× A100)   | (16× A100)    |
| LoRA (FP16)      | 14 GB       | 140 GB        |
|                  | (1× A100)   | (2× A100)     |
| QLoRA (4-bit)    | 6 GB        | 35 GB         |
|                  | (1× RTX 4090)| (1× A100)    |

QLoRA made it possible for researchers and hobbyists
to fine-tune large models on consumer hardware.
```

### NF4 — The Quantization Format

```text
QLoRA introduced NF4 (NormalFloat 4-bit):

Normal INT4:
    Evenly-spaced values: -8, -7, -6, ..., 0, ..., 6, 7
    But neural network weights follow a NORMAL distribution
    (bell curve, clustered around 0).
    Most weights are near 0, few are large.
    Evenly-spaced values waste precision on rare large values.

NF4:
    Values are spaced to match a normal distribution.
    More values near 0 (where most weights are).
    Fewer values far from 0 (where few weights are).
    Result: lower quantization error for the same 4 bits.
```

---

## Prompt Tuning / Prefix Tuning

### The Idea

Don't change the model at all. Instead, learn a set of **soft prompt vectors** that are prepended to the input.

```text
Standard prompting:
    Input: [system prompt tokens] + [user message tokens]
    All tokens are real words, chosen by you.

Prompt tuning:
    Input: [learned_vec₁, learned_vec₂, ..., learned_vec₂₀] + [user message tokens]
    The first 20 "tokens" are continuous vectors in embedding space.
    They don't correspond to any real word.
    They're optimised by gradient descent to steer the model's behaviour.

    Only these 20 vectors are trained. The entire model is frozen.
    Trainable params: 20 × 4096 = 81,920 (vs 7 billion)
```

### Why It Works

```text
The learned vectors function as a "soft instruction" that the
model's attention can read. They push the model into a particular
mode of operation without changing any weights.

Think of it as: instead of writing a perfect prompt in English,
you write a prompt in the model's "native language" (embedding space).
The model can read it more efficiently than English instructions.
```

### Prompt Tuning vs LoRA

```text
| Aspect          | Prompt Tuning          | LoRA                   |
| --------------- | ---------------------- | ---------------------- |
| What's trained  | ~100 soft vectors      | Adapter matrices       |
| Params          | ~400K                  | ~4-40M                 |
| Quality         | Decent for simple tasks| Near full fine-tune    |
| Flexibility     | One task per prompt set| One task per adapter   |
| Switching tasks | Swap prompt vectors    | Swap adapter weights   |
| Model modified  | No                     | No (additive adapters) |

Prompt tuning is simpler but weaker.
LoRA is the standard choice for most fine-tuning needs.
```

---

## Knowledge Distillation

### The Idea

Train a small "student" model to mimic a large "teacher" model.

```text
Teacher: GPT-4 (huge, expensive, slow)
Student: LLaMA 7B (small, cheap, fast)

Goal: make the 7B model behave like GPT-4 on your specific task.

Step 1: Run your data through the teacher.
    Input: "What is photosynthesis?"
    Teacher output: "Photosynthesis is the process by which..."
    Teacher probabilities: P("Photosynthesis")=0.4, P("The")=0.2, P("It")=0.15, ...

Step 2: Train the student to match the teacher's OUTPUT DISTRIBUTION
    (not just the top answer, but the full probability distribution)

    Student loss = α × KL(teacher_probs || student_probs)
                 + (1-α) × cross_entropy(student_probs, hard_target)

    α ≈ 0.7 (mostly learn from teacher)

Why distributions, not just the top answer?
    Teacher's P = [0.4, 0.2, 0.15, 0.1, ...]
    Hard label = [1.0, 0.0, 0.0, 0.0, ...]

    The distribution contains "dark knowledge":
    P("The")=0.2 tells the student "'The' is a reasonable start too."
    P("Banana")=0.001 tells the student "definitely not 'Banana'."
    Hard labels throw away all this relative information.
```

### When to Use Distillation

```text
✓ You need a small, fast model for production
✓ The teacher model (API) is too expensive to call at scale
✓ You want GPT-4 quality on a specific task but at LLaMA-7B speed/cost
✓ Edge deployment (mobile, embedded — needs a small model)

Examples:
    GPT-4 → Phi-3 (Microsoft): distill general capabilities into 3.8B model
    Large model → domain-specific small model: common in production
    Alpaca (Stanford): GPT-3.5 → LLaMA 7B (early open-source success)
```

---

## Adapter Methods Compared

```text
| Method            | Trainable params | Memory needed | Quality    | Use case                |
| ----------------- | --------------- | ------------- | ---------- | ----------------------- |
| Full fine-tuning  | 100%            | ~20× model    | Best       | Max quality, big budget |
| LoRA              | 0.1-1%          | ~1× model     | Near-best  | Standard choice         |
| QLoRA             | 0.1-1%          | ~0.25× model  | Near-best  | Consumer GPU            |
| Prompt tuning     | <0.01%          | ~1× model     | Decent     | Simple tasks            |
| Distillation      | 100% (student)  | ~1× student   | Good       | Compress large → small  |
```

---

## Practical Decision Tree

```text
"I want better performance on my task"
    │
    ├── "I don't want to train anything"
    │       → Better prompt engineering
    │       → Few-shot examples in the prompt
    │       → RAG (retrieve relevant context)
    │
    ├── "I have a small dataset (<1K examples)"
    │       → Few-shot prompting first
    │       → If not enough: LoRA with low rank (r=8), high dropout
    │       → Watch for overfitting (validate on held-out set)
    │
    ├── "I have a medium dataset (1K-100K examples)"
    │       → LoRA or QLoRA (standard choice)
    │       → r=16-64 depending on task complexity
    │       → This covers most real-world use cases
    │
    ├── "I have a large dataset (100K+ examples)"
    │       → LoRA with high rank (r=128+)
    │       → Or full fine-tuning if you have the GPUs
    │
    ├── "I need to deploy a small, fast model"
    │       → Distill from a large model
    │       → Or QLoRA a small model (7B)
    │
    └── "I need to switch between multiple tasks"
            → LoRA: train separate adapters per task
            → Swap adapters at inference time (hot-swap in <1 second)
            → Base model stays the same, just load different A,B matrices
```

---

## Merging LoRA Adapters

```text
After training, you can MERGE the LoRA adapter back into the base model:

    W_merged = W_original + (α/r) × A × B

    This produces a regular model with no adapter overhead.
    Inference speed = same as the original model.
    No need to load A, B separately.

Or keep them separate:
    Load base model once.
    Swap adapters for different tasks without reloading the model.

    Request from customer A → load medical_adapter.bin
    Request from customer B → load legal_adapter.bin
    Same base model, different behaviour.
```

---

## Summary

```text
1. Full fine-tuning: update all weights. Best quality but expensive
   (16-20 bytes/param of memory). Risk of catastrophic forgetting.

2. LoRA: freeze base model, train tiny A×B adapter matrices.
   0.1-1% of params, 95%+ of full fine-tuning quality.
   THE standard method for most fine-tuning tasks.

3. QLoRA: quantize base model to 4-bit + LoRA on top.
   Fine-tune 70B models on a single GPU. Democratised fine-tuning.

4. Prompt tuning: learn soft prompt vectors, model completely frozen.
   Simplest, but weakest.

5. Distillation: train small model to mimic large model.
   For compression and deployment.

6. Start with prompting. Try LoRA if that's not enough.
   Full fine-tuning is the last resort, not the first choice.
```
