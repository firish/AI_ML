## Key Concepts: Quantization

---

## 0. Recap — How Model Weights Are Stored

### Number formats on GPUs

```text
Every weight in a neural network is a floating-point number.
The format determines how many bits are used to store it:

FP32 (32 bits = 4 bytes):
    1 bit sign | 8 bits exponent | 23 bits mantissa
    Range: ±3.4 × 10³⁸, precision: ~7 decimal digits
    This is standard training precision.

FP16 (16 bits = 2 bytes):
    1 bit sign | 5 bits exponent | 10 bits mantissa
    Range: ±65504, precision: ~3.3 decimal digits
    Half the memory of FP32. Used widely for inference.

BF16 (16 bits = 2 bytes):
    1 bit sign | 8 bits exponent | 7 bits mantissa
    Same range as FP32 (same exponent bits), less precision.
    Preferred for training because the range prevents overflow.
    Used by most modern models (LLaMA, Mistral, etc.).

INT8 (8 bits = 1 byte):
    No exponent, no mantissa. Just an integer from -128 to 127.
    4× smaller than FP32. Can't represent 0.357 — only whole numbers.

INT4 (4 bits = 0.5 bytes):
    Integer from -8 to 7 (or 0 to 15 unsigned).
    8× smaller than FP32. Only 16 possible values.
```

### Why this matters for inference

```text
LLaMA 2 70B has 70 billion parameters.

    FP32: 70B × 4 bytes = 280 GB  ← doesn't fit on any single GPU
    FP16: 70B × 2 bytes = 140 GB  ← needs 2× A100 80GB
    INT8: 70B × 1 byte  = 70 GB   ← fits on 1× A100 80GB
    INT4: 70B × 0.5 bytes = 35 GB ← fits on 1× A100 40GB or consumer GPUs

Plus the KV cache, activations, and framework overhead on top of this.
```

---

## 1. The Problem: Memory Is the Bottleneck (Again)

### LLM inference is memory-bound

```text
During decode (generating one token at a time):

    Each token requires reading the ENTIRE model's weights from HBM.
    The computation per weight is tiny — one multiply-add.
    The GPU finishes the math before the next weights arrive from memory.

    This is the same memory-bound problem as Flash Attention,
    but here it's about model weights, not attention matrices.

Example: LLaMA 2 7B in FP16 (14 GB weights)
    Generating one token reads ~14 GB from HBM.
    A100 HBM bandwidth: ~2 TB/s.
    Time to read weights: 14 GB / 2 TB/s = 7 ms per token ≈ 143 tokens/sec

    The actual matrix multiplications take less time than this.
    The GPU is waiting for weights to arrive, not computing.

    If we halve the weight size (INT8): 7 GB / 2 TB/s = 3.5 ms per token ≈ 286 tok/s
    Halve again (INT4): 3.5 GB / 2 TB/s = 1.75 ms per token ≈ 571 tok/s

    Smaller weights → less data to move → directly faster inference.
```

### The dual benefit

```text
Quantization helps in two ways:

    1. Fits on fewer/cheaper GPUs (memory capacity)
       FP16 70B model needs 140 GB → multiple expensive GPUs
       INT4 70B model needs 35 GB → single consumer GPU

    2. Faster token generation (memory bandwidth)
       Less data to read per token → GPU spends less time waiting
       This is often the bigger win for serving at scale
```

---

## 2. How Quantization Works

### The basic idea

```text
Map a range of floating-point values to a small set of integers,
along with a scale factor to convert back.

Example: quantizing a weight vector to INT8

    Original FP16 weights: [0.23, -1.47, 0.89, -0.02, 1.51, -0.78]

    Step 1: Find the range
        min = -1.47, max = 1.51

    Step 2: Compute scale (maps float range to integer range)
        scale = (max - min) / (2⁸ - 1) = 2.98 / 255 = 0.01169

    Step 3: Compute zero point (maps 0.0 to an integer)
        zero_point = round(-min / scale) = round(125.7) = 126

    Step 4: Quantize — convert each float to an integer
        q(x) = round(x / scale) + zero_point

        0.23  → round(0.23 / 0.01169) + 126 = round(19.7) + 126 = 146
        -1.47 → round(-1.47 / 0.01169) + 126 = round(-125.7) + 126 = 0
        0.89  → round(0.89 / 0.01169) + 126 = round(76.1) + 126 = 202
        ...

    Stored: integers [146, 0, 202, 124, 255, 59] + scale (0.01169) + zero_point (126)
            ^^^^^^^^                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            1 byte each                              small overhead, shared per group

    Step 5: Dequantize — recover approximate float during compute
        x ≈ (q - zero_point) × scale
        146 → (146 - 126) × 0.01169 = 0.234    (original: 0.23)  ← close
        0   → (0 - 126) × 0.01169 = -1.473     (original: -1.47) ← close
```

### What's lost

```text
INT8 has 256 possible values. FP16 has 65536.
Quantization is lossy — you can't perfectly represent every weight.

    Original:   0.23
    Quantized:  146
    Recovered:  0.234
    Error:      0.004

For a single weight, this is tiny.
For 70 billion weights, these small errors compound.
The question is: does the model's output quality survive?

Empirically: yes, with careful quantization.
    FP16 → INT8: almost no quality loss on most benchmarks
    FP16 → INT4: small but measurable quality loss (1-3% on benchmarks)
    FP16 → INT3: noticeable degradation, usable for some tasks
    FP16 → INT2: broken for most tasks
```

---

## 3. Granularity: Per-Tensor vs Per-Channel vs Per-Group

### Why granularity matters

```text
The scale and zero_point from the example above are computed over some
set of weights. The smaller the set, the more accurate the quantization,
but the more overhead (more scales to store).

Per-tensor (one scale for the entire weight matrix):
    Weights: (4096, 4096) = 16M values, ONE scale and zero_point
    Problem: if one row has values in [-0.01, 0.01] and another has [-5, 5],
    the small row gets almost no resolution (all values map to ~0).
    Cheap but inaccurate.

Per-channel (one scale per row or column):
    Weights: (4096, 4096) → 4096 scales
    Each row gets its own range → much better accuracy
    Standard for INT8 quantization.

Per-group (one scale per group of e.g. 128 values):
    Weights: (4096, 4096) → 4096 × 32 = 131K scales (groups of 128)
    Even finer granularity → best accuracy
    Standard for INT4 quantization (e.g., GPTQ uses group_size=128).
    Overhead: 131K × 2 bytes = 256 KB per matrix (tiny vs 8 MB of INT4 weights).
```

---

## 4. What Are Activations?

```text
Activations are the intermediate values that flow through the network
during a forward pass — the tensors that aren't weights.

Think of it this way:

    x (input) → Linear layer → y (output)

    Linear.weight — the stored weight matrix (static, lives on disk/HBM)
    x             — the activation coming in (dynamic, computed at runtime)
    y             — also an activation going out

In a transformer, activations are everywhere:

    token embeddings          ← activation
          ↓
    LayerNorm output          ← activation
          ↓
    Q = x @ W_q               ← Q is an activation, W_q is a weight
    K = x @ W_k               ← same
    V = x @ W_v               ← same
          ↓
    scores = Q @ K.T          ← activation
    attn = softmax(scores)    ← activation
    out = attn @ V            ← activation
          ↓
    MLP hidden states         ← activations

The key distinction:

    Weights      — known before inference, static, loaded from HBM once
    Activations  — computed at runtime, depend on the actual input tokens
```

---

## 5. Weight-Only vs Weight + Activation Quantization

### Two approaches

```text
Weight-only quantization:
    Quantize: model weights (stored as INT4/INT8)
    Keep:     activations in FP16/BF16

    During inference:
        Load INT4 weight from memory (fast — small)
        Dequantize to FP16 on the fly
        Multiply with FP16 activation
        Result in FP16

    The win is memory bandwidth — you read 4× less data from HBM.
    The compute still happens in FP16.
    This is what GPTQ, AWQ, and most consumer quantization does.

Weight + activation quantization:
    Quantize: both weights AND activations to INT8

    During inference:
        INT8 weight × INT8 activation = INT8 matmul
        GPU has dedicated INT8 tensor cores — 2× throughput of FP16

    The win is BOTH bandwidth AND compute.
    But activations are harder to quantize — they have outliers.

    This is what LLM.int8() and SmoothQuant address.
```

### Why activations are harder

```text
Weights are fixed after training — you can analyze them carefully.
Activations change with every input.

The problem: activation outliers.

    Most activations in a layer: values in [-1, 1]
    A few activations:           values like 50 or -80
    These outliers appear in specific channels consistently.

    If you set the INT8 range to [-80, 50]:
        The 99% of values in [-1, 1] get mapped to just a few integers.
        Almost all precision is wasted on the outlier range.

    SmoothQuant (Xiao et al., 2022) fixes this:
        Move the difficulty from activations to weights.
        Scale down outlier activation channels by a factor s.
        Scale up the corresponding weight channels by 1/s.
        Mathematically equivalent: (X/s) × (sW) = X × W
        But now activations are smoother → easier to quantize.
```

---

## 6. Common Quantization Methods

### Post-Training Quantization (PTQ) — no retraining needed

```text
GPTQ (Frantar et al., 2022):
    What:   Weight-only, typically INT4, per-group (group_size=128)
    How:    Quantize weights one layer at a time.
            For each layer, find the INT4 values that minimize
            the output error on a small calibration dataset (~128 examples).
            Uses approximate second-order information (Hessian) to decide
            which weights can tolerate more rounding error.
    Speed:  Quantizes a 70B model in a few hours on one GPU.
    Quality: Near-lossless for INT4. The standard for local/consumer LLM use.
    Used by: TheBloke models on HuggingFace, llama.cpp GPTQ models.

AWQ — Activation-Aware Weight Quantization (Lin et al., 2023):
    What:   Weight-only, INT4, per-group
    How:    Observes which weight channels matter most by looking at
            activation magnitudes. Protects important channels by scaling
            them up before quantization (larger values → less rounding error).
    vs GPTQ: Often slightly better quality at same bit width.
              Faster quantization process.
    Used by: vLLM supports AWQ models natively.

LLM.int8() (Dettmers et al., 2022):
    What:   Weight + activation INT8, mixed-precision
    How:    Identifies outlier features (the few channels with large activations).
            Processes outlier channels in FP16, everything else in INT8.
    Trade:  Slight overhead from mixed-precision, but preserves quality.
    Used by: HuggingFace bitsandbytes library.

SmoothQuant (Xiao et al., 2022):
    What:   Weight + activation INT8
    How:    Smooths activations by migrating quantization difficulty to weights
            (the channel scaling trick described above).
    Win:    Full INT8 matmul → uses INT8 tensor cores → 2× compute throughput.
    Used by: NVIDIA TensorRT-LLM, FasterTransformer.
```

### Quantization-Aware Training (QAT) — retrains the model

```text
QAT simulates quantization during training:
    Forward pass:  quantize weights → compute → measure loss
    Backward pass: use straight-through estimator (gradients flow through
                   the quantization step as if it weren't there)
    The model learns to be robust to quantization error.

    Better quality than PTQ at aggressive bit widths (INT4, INT3).
    But requires full training infrastructure and compute.
    Rarely used for LLMs because PTQ is good enough and cheaper.
```

---

## 7. Practical Numbers

```text
LLaMA 2 7B inference on a single GPU:

    Format    Model size   Tokens/sec (A100)   Quality (MMLU)
    FP16      14 GB        ~140 tok/s           baseline
    INT8      7 GB         ~230 tok/s           -0.1% vs FP16
    INT4      3.5 GB       ~350 tok/s           -1.5% vs FP16

LLaMA 2 70B — what fits where:

    Format    Model size   Hardware needed
    FP16      140 GB       2× A100 80GB
    INT8      70 GB        1× A100 80GB
    INT4      35 GB        1× A100 40GB, or RTX 4090 (24GB with offloading)
    INT4      35 GB        2× RTX 3090 (24GB each)

The big picture:
    INT8:  the safe default for serving. Almost no quality loss.
    INT4:  the choice for fitting on smaller hardware. Small quality trade-off.
    INT4 with GPTQ/AWQ: the standard for running LLMs locally.
```

---

## Summary

```text
LLM inference is memory-bound — the GPU waits for weights to arrive from HBM.
Smaller weights = less data to move = directly faster inference.

Quantization maps float weights to lower-bit integers (INT8, INT4)
with a scale factor per group to recover approximate values.

    Weight-only (GPTQ, AWQ): store weights in INT4, compute in FP16.
        Win: 4× less memory, faster HBM reads. Standard for local use.

    Weight + activation (SmoothQuant): both in INT8.
        Win: 2× compute throughput from INT8 tensor cores. Standard for serving.

    Quality: INT8 is near-lossless. INT4 loses 1-3% on benchmarks.
    Granularity: per-group (128 values) gives the best accuracy/overhead trade-off.

Every production LLM deployment uses some form of quantization.
```
