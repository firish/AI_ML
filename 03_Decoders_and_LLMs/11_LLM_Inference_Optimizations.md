## Inference Optimizations

### Why This Matters

Training happens once. Inference happens millions of times. A model that costs $100M to train might cost $10M/day to serve if inference isn't optimized. Every optimization here directly reduces cost and latency.

```text
The bottlenecks during inference:

1. Memory: the model weights + KV cache must fit in GPU memory
   LLaMA 70B at FP16 = 140GB → needs 2× A100 (80GB each)

2. Compute: each generated token requires a forward pass
   through all layers. At 70B params, that's 70B multiply-adds per token.

3. Memory bandwidth: the main bottleneck for generation.
   Each token reads all model weights from memory once.
   A100 bandwidth: 2TB/s. Reading 140GB: 70ms per token → ~14 tokens/sec.
   The GPU is WAITING for data, not computing.
   This is why generation is "memory-bandwidth-bound," not "compute-bound."
```

---

## 1. Quantization — Smaller Numbers, Smaller Model

### The Idea

Model weights are stored as numbers. Use fewer bits per number → model takes less memory → faster to load → more tokens per second.

```text
Full precision (FP32): 32 bits per weight
    Range: ±3.4 × 10³⁸, ~7 decimal digits of precision
    70B model: 70B × 4 bytes = 280 GB

Half precision (FP16 / BF16): 16 bits per weight
    Range: ±65,504 (FP16) or ±3.4 × 10³⁸ (BF16)
    70B model: 70B × 2 bytes = 140 GB
    Standard for training and inference today.

INT8: 8 bits per weight
    Range: -128 to 127 (256 discrete values)
    70B model: 70B × 1 byte = 70 GB
    ~1 GPU instead of 2

INT4: 4 bits per weight
    Range: -8 to 7 (16 discrete values)
    70B model: 70B × 0.5 bytes = 35 GB
    Fits on a single consumer GPU (RTX 4090: 24GB with some tricks)
```

### How Quantization Works

```text
FP16 weight: 0.0237, -0.1842, 0.0891, 0.3012, -0.0534, ...
    Each weight can be any real number. High precision.

INT8 quantization:
    1. Find the range of weights in a layer: min = -0.5, max = 0.5
    2. Map this range to -128..127:
        scale = (max - min) / 255 = 0.00392
        zero_point = round(-min / scale) = 128

    3. Quantize: int8_weight = round(fp16_weight / scale) + zero_point
        0.0237  → round(0.0237 / 0.00392) + 128 = round(6.05) + 128 = 134
        -0.1842 → round(-0.1842 / 0.00392) + 128 = round(-47.0) + 128 = 81

    4. Dequantize (during computation):
        fp16_weight ≈ (int8_weight - zero_point) × scale
        134 → (134 - 128) × 0.00392 = 0.0235   (original: 0.0237, error: 0.0002)
        81  → (81 - 128) × 0.00392 = -0.1842    (exact!)

    Small rounding errors, but the model's output barely changes.
```

### Why It Works (and When It Doesn't)

```text
Why it works:
    Neural network weights are ROBUST to small perturbations.
    A weight of 0.0237 vs 0.0235 changes the output by a negligible amount.
    Over billions of weights, the errors are random and tend to cancel out.

    FP16 → INT8: virtually no quality loss (<0.5% on benchmarks)
    FP16 → INT4: small quality loss (1-3% on benchmarks)

When it fails:
    Some weights are OUTLIERS — much larger than the rest.
    If one weight is 10.0 and the rest are -0.5 to 0.5,
    the INT8 range must cover -0.5 to 10.0 → poor resolution
    for the small weights (most of the model).

    Fix: mixed precision — keep outlier weights in higher precision,
    quantize the rest. This is what LLM.int8() does.
```

### Quantization Methods

```text
GPTQ (post-training, weight-only):
    Quantize weights after training is done.
    Uses a small calibration dataset to minimise quantization error.
    Very fast quantization (minutes).
    Good quality at INT4 (4-bit).
    Used by: TheBloke's quantized models on HuggingFace.

AWQ (Activation-Aware Weight Quantization):
    Identifies "important" weights (those that produce large activations)
    and keeps them at higher precision.
    Slightly better quality than GPTQ at the same bit width.
    Used by: vLLM, TGI.

GGUF (llama.cpp format):
    Quantizes model for CPU inference (not just GPU).
    Multiple precision levels: Q4_0, Q4_K_M, Q5_K_M, Q8_0.
    Can run a 7B model on a laptop CPU.
    Used by: llama.cpp, Ollama, LM Studio.

QLoRA (quantized fine-tuning):
    Quantize base model to 4-bit, add LoRA adapters in FP16.
    Fine-tune only the adapters → train a 70B model on one GPU.
    (More details in file 09 — Fine-tuning.)
```

### What Fits Where

```text
| Model      | FP16       | INT8     | INT4     | Hardware needed            |
| ---------- | ---------- | -------- | -------- | -------------------------- |
| 7B         | 14 GB      | 7 GB    | 3.5 GB   | 1× RTX 3090/4090 (24GB)   |
| 13B        | 26 GB      | 13 GB   | 6.5 GB   | 1× RTX 4090 or 1× A100    |
| 34B        | 68 GB      | 34 GB   | 17 GB    | 1× A100 (80GB) or 2× 4090 |
| 70B        | 140 GB     | 70 GB   | 35 GB    | 1× A100 or 2× A100 (INT4) |
| 405B       | 810 GB     | 405 GB  | 203 GB   | 8× A100 or 4× H100        |

Note: this is model weights only. KV cache adds more memory at runtime.
```

---

## 2. Flash Attention — Fast, Memory-Efficient Attention

### The Problem

Standard attention computes the full n×n attention matrix:

```text
Standard attention:
    1. S = Q · Kᵀ          → n×n matrix (stored in HBM)
    2. P = softmax(S)      → n×n matrix (stored in HBM)
    3. O = P · V           → n×d matrix

    Memory: O(n²) — for 128K tokens, that's 128K × 128K × 2 bytes = 32 GB
    Per layer. Per head.

GPU memory hierarchy:
    SRAM (on-chip):   ~20 MB     very fast    (can't fit n×n matrix)
    HBM (main VRAM):  80 GB     ~10× slower  (where the matrix goes)

    The bottleneck: reading/writing the n×n matrix from HBM.
    Most time is spent moving data, not computing.
```

### The Flash Attention Fix

```text
Key insight: you don't need the full n×n matrix at once.
Compute attention in TILES, streaming through SRAM.

Algorithm (simplified):
    1. Split Q into blocks: Q₁, Q₂, ..., Q_B  (each fits in SRAM)
    2. Split K, V into blocks: K₁, K₂, ..., K_B
    3. For each Q block:
        For each K, V block:
            Compute partial attention in SRAM (fast)
            Accumulate result
        Write final output to HBM (one write, not many)

    Never materialise the full n×n matrix.
    Compute exactly the same result (not approximate).

Result:
    Memory: O(n) instead of O(n²)
    Speed: 2-4× faster (fewer HBM reads/writes)
    Output: bit-for-bit identical to standard attention
```

```text
Versions:
    Flash Attention 1 (2022): tiling + recomputation
    Flash Attention 2 (2023): better parallelism across heads
    Flash Attention 3 (2024): H100-optimized, FP8 support

Used by: PyTorch (F.scaled_dot_product_attention), HuggingFace,
         vLLM, TGI — essentially everything modern.
```

---

## 3. Speculative Decoding — Draft and Verify

### The Problem

Generation is sequential — one token at a time. Each token requires reading the entire model from memory. For a 70B model, that's 70B weights read per token, even though the computation is simple.

```text
The bottleneck is MEMORY BANDWIDTH, not compute.
The GPU is starving — waiting for data from HBM most of the time.

What if we could generate multiple tokens per "read" of the model?
```

### How Speculative Decoding Works

```text
Use TWO models:
    Draft model:  small, fast (e.g., 1B params)
    Target model: large, accurate (e.g., 70B params)

Step 1: Draft model generates K tokens quickly (e.g., K=5)
    "The" → "cat" → "sat" → "on" → "the" → "mat"
    Fast because the draft model is tiny.

Step 2: Target model VERIFIES all K tokens in ONE forward pass
    Feed "The cat sat on the mat" through the 70B model.
    At each position, check: does the 70B model agree with the draft?

    Position 0: "The" → 70B predicts "cat" (P=0.12) → draft said "cat" ✓ accept
    Position 1: "The cat" → 70B predicts "sat" (P=0.15) → draft said "sat" ✓ accept
    Position 2: "The cat sat" → 70B predicts "on" (P=0.25) → draft said "on" ✓ accept
    Position 3: "The cat sat on" → 70B predicts "the" (P=0.30) → draft said "the" ✓ accept
    Position 4: "The cat sat on the" → 70B predicts "mat" (P=0.14) → draft said "mat" ✓ accept

    All 5 accepted! We generated 5 tokens with 1 big-model forward pass.

Step 3: If any token is rejected, discard it and everything after.
    Accept tokens up to the rejection point.
    Resample from the target model's distribution at the rejection point.
    Start a new draft from there.
```

**Key property:** the output is statistically identical to running the target model alone. We're not sacrificing quality — just using the draft model's guesses to skip ahead when it's right.

```text
Acceptance rate depends on how similar the draft and target models are:
    Good draft model: 70-90% acceptance rate → 2-3× speedup
    Bad draft model:  30-50% acceptance rate → minimal speedup

    The draft model should be from the same family:
    LLaMA 1B drafting for LLaMA 70B → high acceptance rate
    Random small model drafting for LLaMA 70B → low acceptance rate
```

---

## 4. Continuous Batching

### The Problem with Naive Batching

```text
Traditional batching: wait for all requests in a batch to finish.

Batch of 4 requests:
    Request A: "Hi"        → generates 3 tokens  (done at step 3)
    Request B: "Tell me..." → generates 50 tokens (done at step 50)
    Request C: "What is..." → generates 10 tokens (done at step 10)
    Request D: "Code..."    → generates 100 tokens (done at step 100)

Naive: wait until request D finishes (step 100) before starting new requests.
    Request A's GPU slot is wasted for 97 steps!
```

### Continuous Batching (Iteration-Level Scheduling)

```text
Continuous batching: as soon as one request finishes, slot in a new one.

Step 1:  [A, B, C, D]      → generate 1 token each
Step 2:  [A, B, C, D]      → generate 1 token each
Step 3:  [A, B, C, D]      → A finishes! Slot in new request E
Step 4:  [E, B, C, D]      → generate 1 token each
...
Step 10: [E, B, C, D]      → C finishes! Slot in request F
Step 11: [E, B, F, D]      → continue...

GPU stays fully utilized. No wasted slots.
Throughput increase: 2-10× depending on workload variance.

Used by: vLLM, TGI, TensorRT-LLM — all modern serving frameworks.
```

---

## 5. PagedAttention (vLLM)

### The KV Cache Memory Problem

```text
Each request needs its own KV cache.
But we don't know in advance how long the response will be.

Naive approach: pre-allocate max_length KV cache for every request.
    Max length = 4096 tokens, KV cache per token = 2.6 MB (LLaMA 70B)
    Pre-allocate: 4096 × 2.6 MB = 10.7 GB per request
    But the average response is only 200 tokens → 0.5 GB
    95% of pre-allocated memory is WASTED.

    With 80GB GPU: can only batch 7 requests (80GB / 10.7GB)
    But if average usage is 0.5GB: could fit 160 requests!
```

### The Fix: Paging (Like OS Virtual Memory)

```text
PagedAttention: allocate KV cache in small PAGES (blocks), on demand.

    Block size = 16 tokens of KV cache
    Each block ≈ 42 KB (for LLaMA 70B)

    Request starts → allocate 1 block (16 tokens)
    Generates 16 tokens → allocate another block
    Response ends at 50 tokens → only 4 blocks used (64 tokens allocated)
                                 vs 4096 tokens pre-allocated naively

    Memory waste: <1 block per request (instead of thousands of tokens)

Benefit:
    2-4× more concurrent requests on the same GPU
    → 2-4× higher throughput
    → proportionally lower cost per token

Used by: vLLM (the most popular LLM serving framework)
```

---

## 6. Model Parallelism — Splitting Across GPUs

When a model doesn't fit on one GPU:

### Tensor Parallelism (TP)

```text
Split individual weight matrices across GPUs.

Example: FFN weight W₁ (4096 × 11008) on 4 GPUs:
    GPU 0: W₁[:, 0:2752]       (4096 × 2752)
    GPU 1: W₁[:, 2752:5504]    (4096 × 2752)
    GPU 2: W₁[:, 5504:8256]    (4096 × 2752)
    GPU 3: W₁[:, 8256:11008]   (4096 × 2752)

Each GPU computes its shard of the output.
Then they exchange partial results (all-reduce).

Requires FAST inter-GPU communication (NVLink: 900 GB/s).
Only works within a single machine (cross-machine is too slow).

TP=4: split each layer across 4 GPUs → 4× less memory per GPU.
```

### Pipeline Parallelism (PP)

```text
Different layers on different GPUs.

LLaMA 70B (80 layers) on 4 GPUs:
    GPU 0: layers 0-19   + embeddings
    GPU 1: layers 20-39
    GPU 2: layers 40-59
    GPU 3: layers 60-79  + prediction head

Token flows through GPU 0 → 1 → 2 → 3 like an assembly line.

Works across machines (lower bandwidth needed — only
activations between layers, not partial matrix results).

Problem: pipeline bubbles — GPU 0 is idle while GPU 3 processes.
Fix: micro-batching — split batch into micro-batches and pipeline them.
```

### Typical Configurations

```text
| Model   | GPUs | Strategy                                    |
| ------- | ---- | ------------------------------------------- |
| 7B      | 1    | No parallelism needed (INT4: fits on 4090)  |
| 70B     | 2    | TP=2 on one machine                         |
| 70B     | 4    | TP=4 for faster inference                   |
| 405B    | 8    | TP=8 on one machine                         |
| 405B    | 16   | TP=8 × PP=2 across 2 machines               |
```

---

## 7. Other Optimizations

### Prefix Caching

```text
Many requests share the same system prompt.
Cache the KV values for the system prompt ONCE.
Reuse for every request with the same prompt.

    System prompt: "You are a helpful assistant..." (200 tokens)
    Without prefix caching: compute 200 tokens per request
    With prefix caching: skip 200 tokens, start from user message

    Saves: 200 tokens × latency per token × number of requests
```

### FP8 Training and Inference

```text
FP8 (8-bit floating point) — newer H100 GPUs support this natively.

    FP16: 16 bits, good precision
    FP8:  8 bits, acceptable precision for many operations

    2× faster matrix multiplication than FP16.
    Requires careful scaling to avoid overflow.
    Used during both training (recent) and inference.
```

### Kernel Fusion

```text
Instead of running LayerNorm, then attention, then ReLU as
separate GPU operations (each reading/writing to memory):

Fuse them into ONE GPU kernel:
    Read input once → LayerNorm → attention → ReLU → write output once

    Fewer memory reads/writes → faster.
    This is what frameworks like TensorRT-LLM and FlashAttention do.
```

---

## The Full Optimization Stack

```text
Layer                What it does                         Typical speedup
──────────────────────────────────────────────────────────────────────────
Quantization         Fewer bits per weight                2-4× memory reduction
Flash Attention      Tiled attention, O(n) memory         2-4× attention speedup
Speculative decode   Draft + verify                       2-3× generation speed
Continuous batching  No wasted GPU slots                  2-10× throughput
PagedAttention       Dynamic KV cache allocation          2-4× concurrent requests
Tensor parallelism   Split layers across GPUs             Linear with GPU count
Kernel fusion        Fewer memory round-trips             1.2-1.5× overall
Prefix caching       Reuse shared prompt KV cache         Skip system prompt cost
GQA (from file 07)   Fewer KV heads to cache              3× KV cache reduction

These stack multiplicatively:
    Quantized INT4 + Flash Attention + continuous batching + PagedAttention
    → a 70B model serving thousands of requests on 2 GPUs
    → what seemed impossible in 2022 is routine in 2025
```

---

## Practical: Running Models Locally

```text
"I want to run an LLM on my laptop"

    Option 1: Ollama (simplest)
        brew install ollama
        ollama run llama3:8b          ← downloads and runs INT4 quantized
        ollama run mistral:7b
        Runs on CPU or Apple Silicon GPU. Just works.

    Option 2: llama.cpp (most flexible)
        Download GGUF quantized model from HuggingFace
        ./main -m model.gguf -p "Hello"
        Very optimized for CPU inference.
        Supports Q4_0, Q4_K_M, Q5_K_M, Q8_0 quantization levels.

    Option 3: vLLM (for serving)
        pip install vllm
        python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3-8B
        OpenAI-compatible API. Continuous batching + PagedAttention built in.

What you can run:
    MacBook (16GB RAM):   7B Q4 model   (decent quality, 10-20 tokens/sec)
    MacBook (32GB RAM):   13B Q4 model  (good quality, 8-15 tokens/sec)
    RTX 4090 (24GB):      13B FP16 or 34B Q4 (good speed)
    A100 (80GB):          70B Q4 or 34B FP16 (production quality)
```
