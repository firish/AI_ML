## Key Concepts: Flash Attention

---

## 0. Recap — How Standard Attention Uses the GPU

### The attention computation

```text
Given Q, K, V matrices (each n × d, where n = sequence length, d = head dim):

    Step 1: S = Q × Kᵀ         → (n, n) attention scores
    Step 2: P = softmax(S)      → (n, n) attention weights
    Step 3: O = P × V           → (n, d) output

Three matrix operations, straightforward.
The problem isn't the math — it's WHERE each step happens on the GPU.
```

### GPU memory hierarchy

A GPU has two levels of memory that matter here:

```text
┌─────────────────────────────────┐
│           GPU                   │
│                                 │
│   ┌───────────────┐            │
│   │     SRAM      │            │
│   │  (on-chip)    │            │
│   │               │            │
│   │  ~20 MB       │            │
│   │  ~19 TB/s     │            │
│   └───────┬───────┘            │
│           │ ← data moves here  │
│   ┌───────┴───────┐            │
│   │      HBM      │            │
│   │  (GPU DRAM)   │            │
│   │               │            │
│   │  ~40-80 GB    │            │
│   │  ~2 TB/s      │            │
│   └───────────────┘            │
└─────────────────────────────────┘

SRAM: tiny but fast. This is where the GPU cores actually compute.
HBM:  large but slow. This is where tensors (Q, K, V, etc.) live.

The speed gap: SRAM is ~10x faster than HBM.
Every computation requires moving data from HBM → SRAM, computing, then writing results back SRAM → HBM.
```

### How standard attention uses this hierarchy

```text
Step 1: S = Q × Kᵀ
    Read Q (n, d) from HBM → SRAM
    Read K (n, d) from HBM → SRAM
    Compute S = Q × Kᵀ
    Write S (n, n) back to HBM          ← full n×n matrix written to slow memory

Step 2: P = softmax(S)
    Read S (n, n) from HBM → SRAM       ← read the full n×n matrix back
    Compute row-wise softmax
    Write P (n, n) back to HBM          ← write full n×n matrix again

Step 3: O = P × V
    Read P (n, n) from HBM → SRAM       ← read n×n matrix AGAIN
    Read V (n, d) from HBM → SRAM
    Compute O = P × V
    Write O (n, d) back to HBM

Total HBM reads/writes: the n×n matrix is written once, read twice, written once more.
That's 4 passes over an n×n matrix through slow HBM memory.
```

---

## 1. The Problem: Memory Bandwidth, Not Compute

### The bottleneck is not multiplication — it's moving data

```text
For sequence length n = 4096, head dim d = 64:

    Q, K, V: each (4096, 64) = 262K elements
    S and P: each (4096, 4096) = 16.7M elements  ← 64× larger

The actual matrix multiplications are fast — GPUs are built for that.
The slow part: shuttling 16.7M elements back and forth between HBM and SRAM,
multiple times, for the intermediate S and P matrices.

The GPU cores sit idle waiting for data to arrive from HBM.
This is called being "memory-bound" — the bottleneck is bandwidth, not compute.
```

### The O(n²) memory problem

```text
Standard attention MUST store the full S matrix (n × n) in HBM:

    n = 2048:   S is 2048²  = 4M entries   = 8 MB  (FP16)
    n = 4096:   S is 4096²  = 16M entries  = 32 MB
    n = 16384:  S is 16384² = 268M entries = 512 MB
    n = 131072: S is 131K²  = 17B entries  = 32 GB  ← just for ONE attention head

This is per head, per layer. For multi-head attention with 32 heads and 32 layers,
multiply accordingly.

Two problems from materializing S:
    1. Memory: O(n²) storage just for intermediate results
    2. Speed: O(n²) data moved through the HBM bandwidth bottleneck
```

### Why softmax makes this hard

```text
Why not just compute attention in tiles and never store the full S matrix?

The problem is softmax. Softmax over row i of S requires:

    softmax(Sᵢ) = exp(Sᵢⱼ) / Σⱼ exp(Sᵢⱼ)
                                 ^^^^^^^^^^^
                                 sum over ALL columns

To normalize row i, you need the sum over the ENTIRE row.
You can't compute softmax on a tile of S without seeing the rest of the row.

Or so it seems. This is exactly what Flash Attention solves.
```

---

## 2. Flash Attention — The Solution

### Core idea: tiling + online softmax

```text
Flash Attention computes EXACT attention (not an approximation)
without ever materializing the n×n matrix S or P in HBM.

Two key ideas:
    1. Tiling:          process Q, K, V in small blocks that fit in SRAM
    2. Online softmax:  compute softmax incrementally, one tile at a time,
                        without needing the full row first
```

### Online softmax — the key trick

Standard softmax needs the full row. But there's a way to compute it incrementally:

```text
Standard softmax of [a, b, c, d, e, f]:
    1. Find max:       m = max(a,b,c,d,e,f)
    2. Compute all:    exp(a-m), exp(b-m), ..., exp(f-m)
    3. Sum them:       L = Σ exp(xᵢ - m)
    4. Divide:         softmax(xᵢ) = exp(xᵢ - m) / L

    Requires seeing ALL elements before producing ANY output.

Online softmax — process in chunks, maintaining running statistics:

    Chunk 1: [a, b]
        m₁ = max(a, b)
        l₁ = exp(a - m₁) + exp(b - m₁)

    Chunk 2: [c, d]
        m₂ = max(m₁, c, d)                        ← update running max
        l₂ = l₁ × exp(m₁ - m₂) + exp(c - m₂) + exp(d - m₂)
             ^^^^^^^^^^^^^^^^^^^^
             rescale previous sum to new max

    Chunk 3: [e, f]
        m₃ = max(m₂, e, f)
        l₃ = l₂ × exp(m₂ - m₃) + exp(e - m₃) + exp(f - m₃)

    Final: softmax(xᵢ) = exp(xᵢ - m₃) / l₃

    Same exact result. But we never needed all 6 elements in memory at once.
```

The rescaling step `l₁ × exp(m₁ - m₂)` is what makes this work. When we see a new chunk with a larger max, we rescale our previous running sum to match. The output is mathematically identical to standard softmax.

### How tiling works

```text
Instead of computing the full n×n matrix:

    Standard:
        Load ALL of Q, K → compute FULL S (n×n) → store to HBM
        Load S → softmax → store P to HBM
        Load P, V → compute output → store to HBM

    Flash Attention:
        Divide Q into blocks of size Bᵣ rows (e.g., 64 rows)
        Divide K, V into blocks of size Bc rows (e.g., 64 rows)

        For each Q block:
            Initialize running output O = 0, running max m = -∞, running sum l = 0
            For each K, V block:
                Load Q block, K block, V block into SRAM     ← small, fits!
                Compute local scores: S_tile = Q_block × K_blockᵀ  (Bᵣ × Bc)
                Update running softmax statistics (m, l)      ← online softmax
                Rescale previous output and accumulate: O += softmax_tile × V_block
            Write final O block to HBM                        ← one write per Q block

Tile size example (n=4096, block=64):
    Full S matrix: 4096 × 4096 = 16M entries      ← never created
    One tile:      64 × 64 = 4K entries            ← fits in SRAM easily
    Number of tiles: (4096/64) × (4096/64) = 4096  ← processed sequentially
```

### What lives where

```text
Standard attention:
    HBM: Q, K, V, S (n×n), P (n×n), O           ← S and P are huge
    SRAM: whatever small piece is being computed

Flash Attention:
    HBM: Q, K, V, O                               ← no S or P stored at all
    SRAM: Q_block, K_block, V_block, S_tile, running stats (m, l, O_block)

    The n×n matrices S and P exist only as small tiles in SRAM,
    computed and consumed immediately, never written to HBM.
```

---

## 3. Analogy: Grading Exams

```text
Standard attention is like a teacher who:
    1. Collects ALL 100 exams
    2. Reads through ALL of them and writes every score on a giant 100×100 spreadsheet
    3. Puts the spreadsheet in a filing cabinet (HBM)
    4. Takes the spreadsheet back out to compute curved grades
    5. Puts the curved grades back in the filing cabinet
    6. Takes them out AGAIN to compute final results

    The filing cabinet is slow to access. Three round trips for a huge spreadsheet.

Flash Attention is like a teacher who:
    1. Takes a stack of 10 exams at a time to their desk (SRAM)
    2. Grades them, updates a running class average, stacks the graded exams
    3. Takes the next stack of 10, grades them, updates running average
    4. Never creates the giant spreadsheet at all
    5. Only goes to the filing cabinet to pick up and drop off small stacks

    Same final grades. Far fewer trips to the slow filing cabinet.
```

---

## 4. Concrete Walkthrough: n=4, block_size=2

```text
Q, K, V are each (4, d). We split into blocks of 2 rows.

Q blocks: Q₁ = Q[0:2], Q₂ = Q[2:4]
K blocks: K₁ = K[0:2], K₂ = K[2:4]
V blocks: V₁ = V[0:2], V₂ = V[2:4]

--- Processing Q₁ (rows 0-1) ---

    Load Q₁ into SRAM

    Tile (Q₁, K₁):
        Load K₁, V₁ into SRAM
        S_tile = Q₁ × K₁ᵀ                  → (2, 2) scores for positions 0-1 vs 0-1
        m₁ = row-wise max of S_tile
        l₁ = row-wise sum of exp(S_tile - m₁)
        O₁ = (1/l₁) × exp(S_tile - m₁) × V₁

    Tile (Q₁, K₂):
        Load K₂, V₂ into SRAM
        S_tile = Q₁ × K₂ᵀ                  → (2, 2) scores for positions 0-1 vs 2-3
        m₂ = max(m₁, row-wise max of S_tile)
        rescale = exp(m₁ - m₂)               ← correction factor
        l₂ = l₁ × rescale + row-wise sum of exp(S_tile - m₂)
        O₁ = (l₁ × rescale × O₁ + exp(S_tile - m₂) × V₂) / l₂
                   ^^^^^^^^^^^^^^^
                   rescale previous output to match new max

    Write O₁ (rows 0-1 of final output) to HBM. Done with Q₁.

--- Processing Q₂ (rows 2-3) ---

    Same process. Load Q₂, iterate over K₁ then K₂.
    Write O₂ (rows 2-3) to HBM.

Final output O = [O₁; O₂] — mathematically identical to standard attention.
The (4, 4) score matrix was never stored in HBM. Only (2, 2) tiles existed in SRAM.
```

---

## 5. What Flash Attention Changes (and What It Doesn't)

```text
                            Standard Attention    Flash Attention
HBM memory for attention    O(n²)                 O(n)
    (S and P matrices)      stored in HBM         never materialized

HBM reads/writes            O(n²)                 O(n² × d / SRAM_size)
    (bandwidth usage)       multiple full passes   fewer passes, smaller data

Total compute (FLOPs)       O(n² × d)             O(n² × d)
    (same math)             same                   same — NOT less work

Computation result          exact softmax          exact softmax
                            not approximate         identical output

Wall-clock speed            slower                 2-4× faster (memory-bound → compute-bound)
GPU memory usage            higher                 lower (no n×n intermediates)
```

Flash Attention does NOT reduce the amount of computation. It does the same math. The speedup comes entirely from better memory access patterns — fewer slow HBM round-trips by keeping intermediate results in fast SRAM.

### Practical impact

```text
Flash Attention is on by default in almost every modern framework:
    PyTorch: torch.nn.functional.scaled_dot_product_attention uses it automatically
    HuggingFace: enabled by default for supported models
    vLLM, TGI: use it for serving

Concrete numbers (A100 GPU, from Dao et al.):
    Standard attention (n=2048): 312 TFLOPs/s effective
    Flash Attention  (n=2048):   699 TFLOPs/s effective  ← 2.2× faster

    Standard attention (n=4096): OOM on some configs
    Flash Attention  (n=4096):   fits and runs

Enables longer sequences not by changing the model, but by removing
the O(n²) memory bottleneck that was the practical limit.
```

---

## 6. Flash Attention and Kernel Fusion

### What kernel fusion is

```text
A GPU "kernel" = one operation dispatched to the GPU.

Standard attention launches separate kernels:
    Kernel 1: S = Q × Kᵀ     → write S to HBM
    Kernel 2: P = softmax(S)  → read S from HBM, write P to HBM
    Kernel 3: O = P × V       → read P from HBM, write O to HBM

    Between kernels, data sits in HBM. Each kernel launch has overhead,
    and each read/write goes through the slow HBM bottleneck.

Flash Attention fuses all three into ONE kernel:
    Single kernel:
        For each tile:
            Compute S_tile, softmax, multiply by V — all in SRAM
        Write only the final O to HBM

    No intermediate HBM traffic. One kernel launch instead of three.
```

Kernel fusion is a general optimization principle: combine multiple sequential operations into one GPU kernel to avoid intermediate HBM round-trips. Flash Attention is the most impactful application of this idea in transformers.

---

## Summary

```text
Standard attention writes the full n×n score matrix to slow GPU memory (HBM),
reads it back for softmax, writes it again, reads it again for the output.
The GPU cores sit idle waiting for data. This is the bottleneck.

Flash Attention:
    1. Tiles Q, K, V into small blocks that fit in fast SRAM
    2. Uses online softmax to compute exact softmax incrementally per tile
    3. Never materializes the n×n matrix in HBM
    4. Fuses matmul + softmax + output into a single GPU kernel

Same math, same output. 2-4× faster, O(n) memory instead of O(n²).
Enabled by two insights: online softmax (incremental normalization)
and tiling (work in SRAM-sized chunks).
```
