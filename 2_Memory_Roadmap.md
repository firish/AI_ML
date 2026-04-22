# Memory & Hardware Roadmap for Understanding LLM Inference

The goal is **literacy, not expertise**: be able to read a paper that says "A100 80GB, 2TB/s HBM2e, 312 TFLOPS FP16" and know what every token means, reason about why decode is memory-bound, and recognize when an optimization is attacking bandwidth vs capacity vs compute — without dropping into embedded systems or chip design.

Depth guide: **Intuition** = understand what it does and why. **Working** = can do back-of-envelope math with it. **Deep** = derive/benchmark yourself (not needed for LLM literacy).

---

## Phase 1 — Numbers and Representation (the foundation)

> Goal: Know how many bytes a number takes and why that matters.

| # | Concept | Depth | Why |
|---|---------|-------|-----|
| 1 | Bits, bytes, words | Working | Memory footprint = params × bytes-per-param. Everything starts here. |
| 2 | Floating point: FP32, FP16, BF16, FP8 | Working | Exponent vs mantissa tradeoff. BF16 has FP32's range with FP16's size — why LLMs train in BF16 |
| 3 | Integer formats: INT8, INT4 | Working | Quantization targets. 4× / 8× memory reduction vs FP32 |
| 4 | Dynamic range vs precision | Intuition | Why FP8 training is hard, why quantization breaks some layers |

**Status:** Partially covered in `04_LLM_Inference_Optimization/05_Quantization.md` and quantization paper notes

---

## Phase 2 — The Memory Hierarchy (the single most important mental model)

> Goal: Internalize the pyramid. Once you see it, every optimization is "move the hot data closer."

| # | Concept | Depth | Why |
|---|---------|-------|-----|
| 1 | Jeff Dean's latency numbers | Working | L1 ~1ns, DRAM ~100ns, SSD ~100μs, network ~1ms. Memorize orders of magnitude |
| 2 | The pyramid: registers → L1/L2/L3 → DRAM → SSD → HDD → network | Working | Each level: ~10× slower, ~10× bigger, ~10× cheaper |
| 3 | Temporal and spatial locality | Working | Why caches work. Why FlashAttention tiles data. Why KV cache is laid out contiguously |
| 4 | Cache lines and prefetching | Intuition | Data moves in chunks (typically 64B), not single bytes. Memory access patterns matter |

**Status:** Not yet covered

---

## Phase 3 — Storage (the "cold" side)

> Goal: Read spec sheets. Understand load times and checkpoint I/O.

| # | Concept | Depth | Why |
|---|---------|-------|-----|
| 1 | HDD | Intuition | Spinning platters, ~100 MB/s, ~10ms seek. Cheap bulk storage |
| 2 | SSD (SATA) | Intuition | NAND flash, ~500 MB/s, no moving parts, ~0.1ms latency |
| 3 | NVMe | Working | SSD on PCIe instead of SATA. ~3–7 GB/s. "NVMe" is the *protocol*, not the medium |
| 4 | Filesystem and OS caching | Intuition | Why the second model load is faster than the first |
| 5 | Object storage (S3/GCS) | Intuition | Where checkpoints actually live in production |

**Status:** Not yet covered

---

## Phase 4 — RAM Types (the "hot" side — this is where LLMs live)

> Goal: Know the difference between memory *capacity* and memory *bandwidth*, and why HBM dominates LLM serving.

| # | Concept | Depth | Why |
|---|---------|-------|-----|
| 1 | DDR (DDR4/DDR5) | Working | System RAM on the CPU. ~50–100 GB/s. Large capacity, modest bandwidth |
| 2 | GDDR (GDDR6/6X) | Working | Consumer GPU memory. ~500–1000 GB/s. Used in 3090/4090 |
| 3 | HBM (HBM2/HBM2e/HBM3/HBM3e) | Working | Stacked memory next to GPU die. ~2–5 TB/s. The bottleneck in datacenter GPUs |
| 4 | Capacity vs bandwidth (they're different) | Working | H100: 80GB @ 3TB/s. CPU: 1TB @ 100GB/s. Different axes — optimize for the right one |
| 5 | Why HBM is supply-constrained | Intuition | Stacking is hard, TSMC + SK Hynix bottleneck. Explains GPU pricing/availability |

**Status:** Not yet covered

---

## Phase 5 — Compute Units (what actually does the math)

> Goal: Recognize the names in papers. Know why GPUs beat CPUs for matmul.

| # | Concept | Depth | Why |
|---|---------|-------|-----|
| 1 | CPU: few fat cores, branch prediction | Intuition | Optimized for latency and control flow. Bad at embarrassingly parallel math |
| 2 | GPU: thousands of thin cores (SMs, CUDA cores) | Working | Optimized for throughput. What every LLM runs on |
| 3 | TPU: systolic array | Intuition | Purpose-built matmul hardware. Google's answer to NVIDIA |
| 4 | FLOP, FLOPS, TFLOPS | Working | Multiply-adds per second. Always check FP16 vs FP32 vs sparse numbers |
| 5 | Tensor cores | Working | Specialized matmul units inside NVIDIA GPUs (Volta+). Where FP16/BF16 speed comes from |
| 6 | Warps, threads, SMs | Intuition | Enough vocabulary to read CUDA-flavored papers without getting lost |

**Status:** Compute units referenced in `04_Flash_Attention.md`, `08_Speculative_Decoding.md`, but not formally covered

---

## Phase 6 — Data Movement (the actual bottleneck)

> Goal: The roofline model. This phase is where "memory-bound" stops being a slogan and becomes a calculation.

| # | Concept | Depth | Why |
|---|---------|-------|-----|
| 1 | Bandwidth vs latency | Working | Different bottlenecks for different workloads. Streaming = bandwidth. Random access = latency |
| 2 | Arithmetic intensity (FLOPs per byte) | Working | The single most important number. Determines whether you're compute- or memory-bound |
| 3 | Roofline model | Working | Plot peak compute (ceiling) and peak bandwidth × intensity (slope). Where your workload sits tells you what to optimize |
| 4 | Memory-bound vs compute-bound | Working | Decode = low intensity = memory-bound. Prefill = high intensity = compute-bound |
| 5 | Why batching changes the regime | Working | Bigger batch → higher arithmetic intensity → eventually compute-bound |
| 6 | Kernel fusion and tiling | Intuition | Why FlashAttention works: keep data in SRAM, avoid HBM round-trips |

**Status:** Heavily referenced across `04_LLM_Inference_Optimization/` but not formally derived

---

## Phase 7 — Interconnects (multi-GPU)

> Goal: Know which link is the bottleneck when a model spans multiple GPUs or nodes.

| # | Concept | Depth | Why |
|---|---------|-------|-----|
| 1 | PCIe (Gen4/Gen5) | Working | CPU ↔ GPU bus. ~32–64 GB/s per direction x16. Orders of magnitude slower than HBM |
| 2 | NVLink | Working | GPU ↔ GPU within a node. ~600–900 GB/s (H100). Why 8-GPU nodes exist |
| 3 | NVSwitch | Intuition | Full all-to-all NVLink fabric. What DGX/HGX boxes use |
| 4 | InfiniBand / RoCE | Intuition | Node ↔ node across datacenter. ~200–400 Gb/s. Bottleneck for multi-node training |
| 5 | "PCIe wall" | Intuition | When model doesn't fit on one GPU, how you split determines whether PCIe or NVLink is the bottleneck |

**Status:** Referenced in `09_Model_Parallelism.md`

---

## Phase 8 — GPU Generations (the vocabulary)

> Goal: Recognize names and remember ballpark specs. No need to memorize tables.

| # | Concept | Depth | Why |
|---|---------|-------|-----|
| 1 | NVIDIA datacenter line: V100 → A100 → H100 → B100/B200 | Working | What every paper tests on. Know years, memory, bandwidth, tensor core gen |
| 2 | A100 specs | Working | 40/80GB HBM2e, ~2TB/s, 312 TFLOPS FP16. The reference GPU — most papers use this baseline |
| 3 | H100 specs | Working | 80GB HBM3, ~3TB/s, ~1000 TFLOPS FP16, FP8 support. Current workhorse |
| 4 | Consumer line (3090/4090/5090) | Intuition | GDDR, 24GB, cheaper. What hobbyist papers use |
| 5 | AMD MI250/MI300 | Intuition | Main non-NVIDIA alternative. MI300X: 192GB HBM — big capacity wins |
| 6 | Google TPU v4/v5 | Intuition | What Google's papers use. Different architecture, similar mental model |

**Status:** Various GPUs mentioned across inference optimization notes

---

## Phase 9 — Putting It Together (LLM-specific synthesis)

> Goal: Read a serving blog post or paper and reason about it end-to-end.

| # | Concept | Depth | Why |
|---|---------|-------|-----|
| 1 | Memory budget: weights + KV cache + activations | Working | "Does this model fit on this GPU?" is the first question in every deployment |
| 2 | Back-of-envelope math | Working | "70B params × 2 bytes (FP16) = 140GB → needs 2× A100 80GB" |
| 3 | Why decode is memory-bound | Working | Low arithmetic intensity per token. Explains every decode optimization you'll read about |
| 4 | Why prefill is compute-bound | Working | High arithmetic intensity. Different optimizations apply (FlashAttention, chunked prefill) |
| 5 | Reading a spec sheet critically | Working | FP16 vs FP8 vs sparse TFLOPS. HBM capacity vs bandwidth. PCIe gen |
| 6 | Connecting it back to your notes | Working | KV cache = capacity problem. Quantization = capacity + bandwidth. FlashAttention = bandwidth. PagedAttention = capacity fragmentation |

**Status:** This phase synthesizes everything in `04_LLM_Inference_Optimization/`

---

## Suggested pacing

- **Week 1** — Phases 1–2 (numbers + hierarchy). Memorize Jeff Dean's latency table.
- **Week 2** — Phases 3–4 (storage + RAM). Spec sheet literacy.
- **Week 3** — Phases 5–6 (compute + roofline). The roofline model is the payoff — it mathematically explains every optimization in your existing notes.
- **Week 4** — Phases 7–9 (interconnects + GPUs + synthesis). By the end, re-read your own inference optimization notes — they should feel deeper.

---

## What you DON'T need

- **Transistor-level / VLSI design** — how a gate is made is irrelevant for serving literacy
- **Assembly / microarchitecture** (out-of-order execution, branch prediction internals) — the "CPU has branch prediction" level is enough
- **OS internals** (page tables, MMU, scheduler) — PagedAttention borrows the *idea* of paging but you don't need the OS-kernel version
- **Detailed CUDA programming** — recognizing concepts (warps, shared memory, kernels) is plenty unless you're writing kernels
- **RDMA / networking protocol internals** — "InfiniBand is fast GPU-to-GPU networking" is enough
- **Power and cooling** — real but not something you need to reason about when reading papers
- **Chip fab / process nodes** (5nm, 3nm, etc.) — explains supply/pricing but not performance reasoning
