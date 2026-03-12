# Scalar Quantization (SQ) — Notes

## 0. The Problem SQ Solves

IVF tells you **which** vectors to search. But each vector is still stored at full precision.

**A single 768-dim vector in float32:**
```
768 dimensions × 4 bytes = 3,072 bytes per vector
```

**1 billion vectors:**
```
1B × 3,072 bytes = ~3 TB
```

That doesn't fit in RAM. You need compression.

**SQ is the simplest possible compression:** reduce the precision of each number.

---

## 1. The Core Idea (One Sentence)

Replace each float32 value with a smaller representation (float16, int8, or int4) by mapping the range of values to fewer bits.

That's it. No clever tricks. Just use less precise numbers.

---

## 2. How SQ Works: int8 Example

### Step 1: Find the range

Scan all vectors. For each dimension `d`, find:
```
min_d = minimum value across all vectors in dimension d
max_d = maximum value across all vectors in dimension d
```

### Step 2: Map float → int

For a value `v` in dimension `d`:
```
quantized = round(255 * (v - min_d) / (max_d - min_d))
```

This maps the continuous range `[min_d, max_d]` to integers `[0, 255]` (1 byte).

### Step 3: Store the int (plus the range for decoding)

Store:
- The quantized int8 value (1 byte per dimension)
- The `min_d` and `max_d` per dimension (stored once, shared across all vectors)

### Decoding (reconstructing approximate float)

```
v_approx = min_d + quantized * (max_d - min_d) / 255
```

---

## 3. Concrete Example

Suppose dimension 5 has range `[-0.8, 1.2]` across all vectors.

A vector has value `0.3` in dimension 5:

**Encode:**
```
quantized = round(255 * (0.3 - (-0.8)) / (1.2 - (-0.8)))
          = round(255 * 1.1 / 2.0)
          = round(140.25)
          = 140
```

**Decode:**
```
v_approx = -0.8 + 140 * 2.0 / 255
         = -0.8 + 1.098
         = 0.298
```

Original: `0.3`, Reconstructed: `0.298`. Close enough for distance comparisons.

---

## 4. Memory Savings

| Precision | Bytes per dim | 768-dim vector | 1B vectors |
|---|---|---|---|
| float32 | 4 | 3,072 B | ~3 TB |
| float16 | 2 | 1,536 B | ~1.5 TB |
| int8 (SQ8) | 1 | 768 B | ~768 GB |
| int4 (SQ4) | 0.5 | 384 B | ~384 GB |

**SQ8 gives 4x compression** with minimal accuracy loss. That's the most common variant.

---

## 5. Distance Computation with SQ

Two options:

### A. Decode then compute (simpler)

Decompress both vectors to float, compute distance normally.

**Pro:** Exact same distance formula.
**Con:** You're decompressing on every comparison — slower.

### B. Compute directly on quantized values (faster)

For L2 distance between quantized vectors, you can compute directly on int8 values and scale the result.

**Pro:** Integer arithmetic is faster, especially on SIMD/GPU.
**Con:** Slightly more implementation complexity.

In practice, option B is what production systems use.

---

## 6. When SQ Is Good Enough

SQ works well when:
- You need moderate compression (4x is enough)
- Accuracy loss must be minimal
- You want simple implementation
- Dataset fits in RAM after 4x compression

**SQ8 typically loses < 1% recall** compared to float32. For many applications, that's perfectly fine.

---

## 7. When SQ Is Not Enough

**The problem:** SQ compresses each value independently. It doesn't exploit structure across dimensions.

A 768-dim vector compressed to int8 is still **768 bytes**. For 1 billion vectors, that's still ~768 GB.

**What if you need 10–100x compression?**

SQ can't do that without destroying accuracy. Going from int8 to int4 helps, but int2 or int1 is too lossy for most use cases.

**This is where Product Quantization (PQ) comes in.** PQ doesn't just reduce precision — it replaces entire subvectors with compact codes, achieving 10–100x compression while keeping distances useful.

---

## Key Takeaways

1. **SQ = reduce precision per float** — map continuous values to fewer bits
2. **SQ8 (int8) is the sweet spot** — 4x compression, <1% recall loss
3. **Simple to implement** — just min/max scaling per dimension
4. **Doesn't exploit cross-dimension structure** — each value compressed independently
5. **Not enough for billion-scale** — still too many bytes per vector
6. **Foundation concept** — understand SQ before PQ, since PQ builds on the idea of "approximate representation"

---

**Next:** Product Quantization (PQ) — compress entire subvectors into tiny codes for 10–100x compression.
