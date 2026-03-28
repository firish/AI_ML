## TurboQuant — Online Vector Quantization with Near-Optimal Distortion Rate

**Authors:** Amir Zandieh, Vahab Mirrokni, et al. — Google Research, Google DeepMind, KAIST, NYU (ICLR 2026, arXiv April 2025, blog March 2026)

```text
TurboQuant combines two earlier papers by the same team:

    1. QJL (June 2024, AAAI 2025)    → see 02_QJL_notes.md
    2. PolarQuant (Feb 2025, AISTATS 2026) → see 03_PolarQuant_notes.md
```

---

## The Problem

```text
The KV cache is the #1 memory bottleneck for serving LLMs.

    KV cache size = layers × seq_length × 2 (K+V) × hidden_dim × batch × bytes

    For a 70B model serving 512 users with 4K context:
        ≈ 512 GB of KV cache — 4× the model weights themselves.

Traditional quantization (KIVI etc.) compresses values to fewer bits
but stores per-block constants (scale + zero_point) in full precision.
At 3-bit, this overhead adds 1-2 bits per value — defeating the purpose.

Both QJL and PolarQuant independently solved the overhead problem.
TurboQuant's contribution: combining them, because they fix
DIFFERENT things.
```

---

## Why Combine? The Complementarity

```text
PolarQuant alone:
    ✓ Minimizes reconstruction error (MSE)
    ✗ Can introduce BIAS in dot products (Q·K)

    MSE ≠ dot product accuracy. Example:
        True K = [1.0, 0.0], Quantized K = [0.9, 0.1]
        MSE is small (0.02), but Q·K with Q=[0.0, 1.0] goes from 0.0 to 0.1.
        Attention scores depend on dot products, not MSE.

QJL alone:
    ✓ Unbiased dot product estimation
    ✗ Higher MSE (sign bits are very coarse)

TurboQuant = PolarQuant (Stage 1) + QJL (Stage 2):
    Stage 1: PolarQuant at 3 bits → good MSE, but biased dot products
    Stage 2: QJL on the RESIDUAL at 1 bit → corrects the dot product bias

    Total: ~3.5 effective bits, zero overhead, zero bias.
    Better than either method alone on both MSE and dot product accuracy.
```

---

## Results

```text
Models: Llama-3.1-8B-Instruct, Gemma, Mistral

Accuracy: zero measurable loss on LongBench, Needle-in-a-Haystack,
    ZeroSCROLLS, RULER, L-Eval at 3.5-bit. Matches FP16 baseline.

Memory: 6× KV cache compression (512 GB → ~85 GB for the 70B example).

Speed: up to 8× on Q×Kᵀ attention on H100. ~1.2× end-to-end.

    | Aspect           | KIVI          | TurboQuant      |
    |------------------|---------------|-----------------|
    | Compression      | 2.6×          | 6×              |
    | Calibration      | Per-model     | None (universal)|
    | Accuracy at 3-bit| Some loss     | Zero loss       |
    | Overhead         | Has constants | Zero            |

Theoretical: MSE within √(3π/2) ≈ 2.7× of Shannon's lower bound.
    No algorithm can ever do more than 2.7× better. Near-optimal by proof.
```

---

## Why People Are Excited

```text
1. 6× less memory for serving → fewer GPUs, cheaper inference
2. Data-oblivious → works on any model instantly, no calibration
3. Provably near-optimal → hard to beat by much
4. Compounds with weight quantization (GPTQ/AWQ) → extreme total compression
5. Chip stocks crashed (Samsung -5%, SK Hynix -6%) → market believes it

Limitations:
    - Only tested up to 8B models (not 70B/405B yet)
    - 1.2× end-to-end speedup, not 8× (attention isn't the only bottleneck)
    - The rotation adds a small per-token compute cost
```

---

## How It Fits in the Pipeline

```text
Pre-training:  trillions of tokens, months, millions of dollars
Fine-tuning:   LoRA/QLoRA makes training affordable
Serving:       TurboQuant makes inference affordable
                ↑ compresses KV cache (not weights, not activations)

Weight quantization (GPTQ/AWQ): cheaper to STORE the model
KV cache quantization (TurboQuant): cheaper to RUN the model
These are independent and stack.
```

---

## One-Line Summary

```text
TurboQuant combines PolarQuant (MSE-optimal compression via polar coordinates)
with QJL (dot-product-accurate 1-bit correction) to achieve 6× KV cache compression
at 3.5 bits with zero overhead, zero calibration, and zero accuracy loss.
```
