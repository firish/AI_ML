## QJL — 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead

**Authors:** Amir Zandieh, Majid Daliri, Insu Han (June 2024, AAAI 2025)

---

## The Problem

```text
Attention computes: Score = softmax(Q × Kᵀ) × V

The Key vectors for all previous tokens are stored in the KV cache.
At 16 bits per value, this cache dominates memory during inference.

Existing fix: quantize K from 16-bit to 3-bit.
But every method (KIVI, KVQuant) needs per-block CONSTANTS
(scale + zero_point) stored in full precision.

    At 3-bit with block size 32:
    Data:     32 values × 3 bits = 96 bits
    Overhead: 2 constants × 32 bits = 64 bits
    Your "3-bit" is actually ~5-bit. Overhead eats the savings.

Goal: compress Keys to ~3 bits with ZERO overhead.
```

---

## The Solution

```text
Core idea: don't quantize the Key vectors directly.
Instead, quantize a SKETCH of them that preserves dot products.

The algorithm (3 lines):

    1. Generate a random Gaussian matrix S (m × d), once, shared forever.
       (d = embedding dim like 4096, m = compressed dim)

    2. For each Key vector k:
       Compress:  k̃ = sign(S × k)     → m values of {+1, -1}
       Store:     k̃ (1 bit each) + ||k||₂ (the norm, one number)

    3. At attention time, for query q:
       Estimate:  q·k ≈ (√(π/2) / m) × ||k||₂ × (S×q) · k̃

Why this works — the Johnson-Lindenstrauss (JL) lemma:
    Multiplying two vectors by the SAME random matrix S preserves
    their inner product (in expectation). This is a classic result.

    QJL's twist: quantize ONE of the projected vectors to just sign bits
    (+1/-1) and leave the other unquantized. They prove this STILL gives
    an unbiased estimator of the original inner product.

    This is perfect for attention:
        Keys are cached (quantize to 1-bit signs → tiny storage)
        Queries are fresh each step (keep full precision → no storage cost)
        Asymmetric: cheap where it matters, precise where it's free.

The "zero overhead" part:
    No scale factors. No zero points. No per-block constants.
    Just sign bits + one norm per key vector.
    The random matrix S is shared globally — not stored per vector.
```

---

## Practical Detail: Outlier Channels

```text
Key embeddings have OUTLIERS — a few channels (coordinates) in deeper
layers have 10-100× larger magnitudes than the rest.

    Layer 0:  magnitudes roughly uniform across channels
    Layer 31: ~4 channels have magnitudes 10-30× larger

These outliers dominate the norm and distort the quantization.

Fix: separate the outlier channels (identified during prompt encoding).
    Outliers: quantize with more bits (or keep full precision) — only ~4 channels
    Inliers:  quantize with QJL as usual — the other ~4092 channels

    Cost: negligible (4 extra FP16 values per key vector).
    Benefit: much lower distortion since norms are now reasonable.
```

---

## Key Results

```text
Models: Llama-2-7B, Llama-3-8B on A100 GPU

Accuracy at 3 bits (vs 16-bit baseline):

    | Model      | Method   | Bits | LambadaOAI | HellaSwag | PIQA  | MMLU  |
    |------------|----------|------|------------|-----------|-------|-------|
    | Llama-2-7B | Baseline | 16   | 73.90      | 57.18     | 78.07 | 41.85 |
    | Llama-2-7B | KIVI     | 3    | 73.88      | 57.13     | 78.07 | 41.81 |
    | Llama-2-7B | QJL      | 3    | 73.88      | 57.14     | 78.07 | 41.78 |
    | Llama-3-8B | Baseline | 16   | 75.59      | 60.17     | 79.65 | 62.09 |
    | Llama-3-8B | QJL      | 3    | 75.61      | 60.13     | 79.87 | 62.12 |

    → Essentially zero accuracy loss at 3 bits.
    → QJL matches KIVI on accuracy but without the overhead.
    → KIVI can't even run on Llama-3 (doesn't support BF16). QJL can.

Long-context (LongBench, seq len up to 31.5K):
    QJL beats KIVI on 4 of 6 QA datasets at 3 bits.
    Both beat KVQuant (which needs 4.3 bits).

Memory: 5× reduction in KV cache (16-bit → ~3-bit + norm).
Speed: marginal overhead vs exact baseline during prompting.
    Both QJL and KIVI are similarly fast at generation.
    KVQuant is significantly slower (heavy preprocessing).
```

---

## Implications

```text
1. Proved that 1-bit quantization can preserve dot products (with the
   asymmetric trick). This is theoretically surprising — sign bits
   alone carry enough information when paired with an unquantized query.

2. Zero overhead means compression ratio is EXACTLY what the bit-width
   says. 3 bits = 5.3× compression, period. No hidden costs.

3. Data-oblivious: works on any model without calibration data.
   Just plug in the random matrix and go.

4. Became Stage 2 of TurboQuant (2025): PolarQuant handles the main
   compression, QJL corrects the residual bias for 1 extra bit.
   The two papers were designed to compose.

Limitation: only compresses Keys, not Values.
    Values use standard per-token quantization (simpler problem).
    This is fine — Keys are the harder target because attention
    needs accurate dot products (Q×K), while Values just get
    weighted-summed (less sensitive to small errors).
```
