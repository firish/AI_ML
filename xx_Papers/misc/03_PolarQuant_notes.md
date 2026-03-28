## PolarQuant — Quantizing KV Caches with Polar Transformation

**Authors:** Insu Han (KAIST), Praneeth Kacham, Vahab Mirrokni, Amir Zandieh (Google Research), Amin Karbasi (Yale) — Feb 2025, AISTATS 2026

---

## The Problem

```text
Same overhead problem as QJL targets, but attacks it differently.

Traditional KV cache quantization (KIVI etc.):
    Group values into blocks → compute min/max → store scale + zero_point
    These constants are stored in full precision (16-bit) per block.
    At 3-bit quantization, overhead adds ~1-2 bits per value.

QJL's fix: project to random subspace, keep only sign bits.
    Works well, but optimizes for DOT PRODUCT preservation (good for Keys).

PolarQuant's fix: what if we quantize in a DIFFERENT coordinate system
    where the overhead problem disappears entirely?
```

---

## The Solution

```text
Core idea: convert vectors from Cartesian to POLAR coordinates,
then quantize the angles (not the x,y,z values).

Why polar?
    A vector in Cartesian: [x₁, x₂, ..., x_d]  — d values, all different ranges
    A vector in polar:     (radius, angle₁, angle₂, ..., angle_{d-1})

    The radius is one number (the vector's length).
    The angles describe DIRECTION only.

    After random preconditioning (multiply by random rotation matrix S),
    something remarkable happens to the angles:
        - They all concentrate tightly around π/4
        - The distribution is KNOWN analytically (a sin^n function)
        - Higher recursion levels → sharper concentration

    This means:
        - No per-block scale/zero_point needed (distribution is known)
        - A PRECOMPUTED codebook works for all vectors
        - Zero overhead.

The recursive polar transform:
    For a d-dimensional vector (d must be power of 2):

    Level 1: pair up coordinates (x₁,x₂), (x₃,x₄), ...
             → d/2 radii + d/2 angles

    Level 2: pair up the radii from level 1
             → d/4 radii + d/4 angles

    Level 3: pair up again → d/8 radii + d/8 angles
    ...
    Level log₂(d): → 1 final radius + the tree of angles

    In practice: only 4 levels (not full recursion).
    For d=128: get d/16=8 radii + 15d/16=120 angle values.

Quantizing the angles:
    Each angle's distribution is known → precompute optimal bin
    boundaries using Lloyd-Max algorithm (like k-means in 1D).

    Bit allocation per level:
        Level 1: 4 bits (range is [0, 2π) — widest)
        Levels 2-4: 2 bits each (range is [0, π/2] — tighter concentration)

    Total: ~3.875 bits per coordinate (with 16-bit radii).
    → 4.1× compression of the KV cache.
```

---

## Key Results

```text
Model: Llama-3.1-8B-Instruct on RTX A6000 (48GB)

Needle-in-a-Haystack (4K–104K context):
    | Method      | Score |
    |-------------|-------|
    | Exact 16-bit| 0.995 |
    | PolarQuant  | 0.991 |  ← best among all compressed methods
    | KIVI        | 0.984 |
    | PyramidKV   | 0.891 |
    | SnapKV      | 0.858 |

LongBench (average across SQA, MQA, Sum, Few, Syn, Code):
    | Method              | Average |
    |---------------------|---------|
    | Exact 16-bit        | 48.63   |
    | PolarQuant-R online | 48.37   |  ← closest to exact
    | PolarQuant-R offline| 48.29   |
    | KIVI                | 46.70   |
    | PolarQuant (no rot) | 48.11   |

    PolarQuant-R (with random preconditioning) nearly matches
    the uncompressed baseline at 4× compression.

Runtime (16K prompt, generate 1024 tokens):
    | Method              | Prefill (s) | Generation (s) |
    |---------------------|-------------|----------------|
    | Exact 16-bit        | 2.934       | 38.374         |
    | KIVI                | 3.590       | 49.564         |
    | PolarQuant-R offline| 3.364       | 44.097         |
    | PolarQuant online   | 11.623      | 43.652         |

    Offline variant: 14% faster generation than KIVI.
    Online variant: slow prefill (clustering overhead), but can
    use offline codebook to avoid this.
```

---

## Implications

```text
1. Polar coordinates are a better quantization basis than Cartesian.
   Angles concentrate after rotation → known distribution → no overhead.
   This is the core insight that TurboQuant builds on.

2. Random preconditioning is essential.
   Without it (PolarQuant-R vs PolarQuant), angles have outliers
   and wider spread. With it, angles cluster sharply around π/4
   at higher levels. Figure 2 in the paper shows this dramatically.

3. Asymptotically optimal: Theorem 1 proves the error bound
   uses O(log(1/ε)) bits per coordinate — matching the best
   possible for worst-case vectors.

4. Compresses both Keys AND Values (unlike QJL which only does Keys).
   Same polar transform works for both.

5. Limitation vs QJL: PolarQuant optimizes MSE (reconstruction error),
   not dot product accuracy directly. Attention needs accurate Q·K
   dot products, and MSE-optimal ≠ dot-product-optimal.
   This is exactly the gap QJL fills → hence TurboQuant = both.

The lineage:
    QJL (2024):      1-bit sketching, zero overhead, dot-product accurate
    PolarQuant (2025): polar coords, zero overhead, MSE optimal
    TurboQuant (2025): PolarQuant (Stage 1, MSE) + QJL (Stage 2, dot-product correction)
```
