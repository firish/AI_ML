## Key Concepts: Model Parallelism

---

## 0. The Problem — Models Don't Fit on One GPU

### Why we need multiple GPUs

```text
A single GPU has finite memory. The largest widely available GPU:

    NVIDIA A100: 80 GB HBM
    NVIDIA H100: 80 GB HBM

Model weight sizes (FP16):
    LLaMA 7B:    14 GB   ← fits on one GPU
    LLaMA 70B:   140 GB  ← needs 2+ GPUs
    GPT-3 175B:  350 GB  ← needs 5+ GPUs
    LLaMA 405B:  810 GB  ← needs 11+ GPUs

And that's just weights. Add KV cache, activations, and framework overhead:
    LLaMA 70B serving: ~140 GB weights + 10-50 GB KV cache + overhead
                       = 200+ GB → 3× A100 80GB minimum

Two reasons to use multiple GPUs:
    1. Capacity: the model literally doesn't fit on one GPU
    2. Speed: split the work so each GPU does less → faster per token
```

### The three strategies

```text
There are three ways to spread a model across GPUs.
Each splits a different axis:

    Tensor Parallelism (TP):  split individual LAYERS across GPUs
    Pipeline Parallelism (PP): split LAYERS THEMSELVES across GPUs
    Data Parallelism (DP):     split REQUESTS across GPUs

    TP: one layer is sharded across 4 GPUs → each GPU does 1/4 of the layer
    PP: GPU 0 has layers 0-19, GPU 1 has layers 20-39, etc.
    DP: each GPU has a full model copy, handles different requests

These are not mutually exclusive. Production systems combine all three.
```

---

## 1. Tensor Parallelism — Split Each Layer

### How a single layer works (recap)

```text
One transformer layer, simplified:

    Input x: (batch, seq_len, d_model)    e.g., (1, 1, 8192) during decode

    Attention:
        Q = x × W_q    W_q is (8192, 8192)    ← large matrix multiply
        K = x × W_k    W_k is (8192, 8192)
        V = x × W_v    W_v is (8192, 8192)
        ... attention scores, softmax, output ...
        out = attn_out × W_o   W_o is (8192, 8192)

    FFN:
        h = x × W_up     W_up is (8192, 22016)   ← even larger
        h = SiLU(h)
        y = h × W_down   W_down is (22016, 8192)

Each weight matrix is huge. W_up for LLaMA 70B: 8192 × 28672 = 235M params = 470 MB.
```

### Splitting the weight matrices

```text
Tensor parallelism splits each weight matrix across GPUs.

Example: W_q (8192, 8192) across 4 GPUs, splitting columns:

    GPU 0: W_q[:, 0:2048]      → (8192, 2048)
    GPU 1: W_q[:, 2048:4096]   → (8192, 2048)
    GPU 2: W_q[:, 4096:6144]   → (8192, 2048)
    GPU 3: W_q[:, 6144:8192]   → (8192, 2048)

Each GPU computes a SLICE of Q:
    GPU 0: Q_0 = x × W_q_0  → (1, 1, 2048)   ← 1/4 of the Q vector
    GPU 1: Q_1 = x × W_q_1  → (1, 1, 2048)
    GPU 2: Q_2 = x × W_q_2  → (1, 1, 2048)
    GPU 3: Q_3 = x × W_q_3  → (1, 1, 2048)

For attention: this maps naturally to attention heads.
    LLaMA 70B has 64 heads. With TP=4: each GPU handles 16 heads.
    Each GPU computes attention for its 16 heads independently.
    No communication needed during the attention computation itself.
```

### Where communication happens

```text
After attention, the output projection W_o combines all heads:
    out = concat(head_0, ..., head_63) × W_o

With TP, each GPU has partial results. They need to combine:

    GPU 0: partial_out_0 = local_attn_0 × W_o_0   → (1, 1, 8192)
    GPU 1: partial_out_1 = local_attn_1 × W_o_1   → (1, 1, 8192)
    GPU 2: partial_out_2 = local_attn_2 × W_o_2   → (1, 1, 8192)
    GPU 3: partial_out_3 = local_attn_3 × W_o_3   → (1, 1, 8192)

    AllReduce: sum partial_out across all GPUs → each GPU gets the full result
    ↑ this is a COMMUNICATION step — GPUs exchange data

Same thing happens after the FFN's W_down projection.

Per layer: 2 AllReduce operations (one after attention, one after FFN).
Each AllReduce moves ~d_model × batch_size × 2 bytes across GPUs.

    LLaMA 70B, TP=4: 2 × 8192 × 2 bytes = 32 KB per AllReduce
    That's tiny — BUT it happens every layer (80 layers) and latency adds up.
```

### Why TP needs fast interconnect

```text
AllReduce latency happens on the critical path — the model can't proceed
until ALL GPUs have finished and exchanged results.

    NVLink (within a server):  ~600 GB/s, ~1-2 μs latency
    PCIe 4.0:                  ~32 GB/s, ~5-10 μs latency
    Network (across servers):  ~50-400 GB/s (InfiniBand), ~5-50 μs latency

    80 layers × 2 AllReduces × 2 μs = 320 μs overhead with NVLink    ← OK
    80 layers × 2 AllReduces × 50 μs = 8 ms overhead across network  ← significant

Rule of thumb:
    TP within a single server (NVLink): works great, up to TP=8
    TP across servers (network): too much latency, avoid if possible
```

---

## 2. Pipeline Parallelism — Split by Layers

### How it works

```text
Instead of splitting each layer, assign ENTIRE layers to different GPUs.

LLaMA 70B (80 layers), PP=4:
    GPU 0: layers 0-19    (embedding layer + first 20 transformer blocks)
    GPU 1: layers 20-39
    GPU 2: layers 40-59
    GPU 3: layers 60-79   (+ final LM head)

Forward pass:
    GPU 0 processes input through layers 0-19 → sends output to GPU 1
    GPU 1 processes through layers 20-39      → sends output to GPU 2
    GPU 2 processes through layers 40-59      → sends output to GPU 3
    GPU 3 processes through layers 60-79      → produces final output

Communication: only at the boundaries between stages.
    Send one activation tensor between GPUs: (batch, seq_len, d_model)
    LLaMA 70B: (1, 1, 8192) × 2 bytes = 16 KB per boundary, 3 boundaries.
    Much LESS communication than TP (which communicates every layer).
```

### The pipeline bubble problem

```text
With PP, GPUs are idle while waiting for the previous stage:

    Time →
    GPU 0: [compute]→ send → [idle........] [idle........] [idle........]
    GPU 1: [idle...] → recv → [compute]→ send → [idle........] [idle........]
    GPU 2: [idle.........] [idle...] → recv → [compute]→ send → [idle........]
    GPU 3: [idle.................] [idle.........] [idle...] → recv → [compute]

    GPU 0 finishes its layers and has nothing to do.
    GPU 3 waits for everyone else before it can start.
    This is the "pipeline bubble" — idle time at the start and end.

For a SINGLE request during decode (1 token), the bubble is terrible:
    Only one GPU is active at a time. 75% idle with PP=4.

The bubble shrinks with larger batches:
    While GPU 1 processes batch A, GPU 0 starts on batch B.
    With enough batches in flight, all GPUs stay busy.

    Time →
    GPU 0: [batch A][batch B][batch C][batch D] ...
    GPU 1:   [idle] [batch A][batch B][batch C] ...
    GPU 2:   [idle]   [idle] [batch A][batch B] ...
    GPU 3:   [idle]   [idle]   [idle] [batch A] ...

    After the initial fill, all GPUs are active. Called "microbatching".
```

### TP vs PP tradeoffs

```text
                    Tensor Parallelism          Pipeline Parallelism
Communication       every layer (2× AllReduce)  only at stage boundaries
Latency per token   low (all GPUs work together) higher (pipeline bubble)
Bandwidth needed    high (NVLink required)       low (one send per boundary)
Best for            within a server              across servers
Scales to           8 GPUs (one server)          many servers
Batch size needs    works at batch=1             needs large batches to fill pipeline
Memory per GPU      model_size / TP              model_size / PP (roughly)
```

---

## 3. Data Parallelism — Split by Requests

### How it works

```text
Each GPU (or group of GPUs) has a FULL copy of the model.
Different requests go to different replicas.

    GPU group 0 (full model copy): handles requests A, B, C
    GPU group 1 (full model copy): handles requests D, E, F
    GPU group 2 (full model copy): handles requests G, H, I

No communication between replicas during inference.
Each replica is independent. A load balancer routes requests.

This is the simplest form of scaling:
    Want 2× throughput? Add another replica.
    Want 10× throughput? Add 9 more replicas.

Limitation: each replica must fit the model.
    LLaMA 70B in FP16: each replica needs 2× A100 80GB (using TP=2).
    10 replicas = 20 A100 GPUs.
```

### When to use what

```text
Single user, minimum latency:
    TP within one server. All GPUs work on the same token together.
    More GPUs → faster per-token (each GPU does less work).

High throughput, many users:
    DP across servers. Each server handles different requests.
    More servers → more concurrent requests.

Model too large for one server:
    PP across servers + TP within each server.
    Minimize cross-server communication (PP has less than TP).
```

---

## 4. How Production Systems Combine Them

### The standard recipe

```text
LLaMA 70B serving on 8× A100 80GB (one server):

    Option A: TP=8
        Split every layer across all 8 GPUs.
        Each GPU: 140 GB / 8 = 17.5 GB weights + KV cache share.
        Lowest latency per token. Best for single-user or small batch.

    Option B: TP=4, DP=2
        Two replicas, each using 4 GPUs with tensor parallelism.
        Each GPU: 140 GB / 4 = 35 GB weights.
        Higher throughput (2 replicas), slightly higher latency than TP=8.

    Option C: TP=2, DP=4
        Four replicas, each using 2 GPUs.
        Even higher throughput, higher per-request latency.

The choice depends on SLA:
    Latency-sensitive (chatbot): maximize TP → fewest replicas, fastest per-token.
    Throughput-sensitive (batch processing): maximize DP → most replicas.
```

### Large-scale example

```text
LLaMA 405B serving across multiple servers:

    Each server: 8× H100 80GB (640 GB total per server)
    Model: 810 GB in FP16

    Setup: TP=8 within each server, PP=2 across 2 servers
        Server 0 (8 GPUs, TP=8): layers 0-63 sharded across 8 GPUs
        Server 1 (8 GPUs, TP=8): layers 64-126 sharded across 8 GPUs
        TP communication: NVLink within server (fast)
        PP communication: network between servers (one send per boundary)

    For throughput: add more pairs of servers as DP replicas.
        4 servers = 2 replicas (2 servers each)
        8 servers = 4 replicas
        Scale horizontally.

With INT4 quantization:
    Model: 810 GB → ~200 GB
    Fits on 4× H100 with TP=4. Single server. No PP needed.
    Quantization eliminated the need for pipeline parallelism entirely.
```

---

## 5. Expert Parallelism (for MoE Models)

### A note on Mixture-of-Experts models

```text
MoE models (Mixtral, GPT-4, DeepSeek) have a different structure:
    Instead of one FFN per layer, they have N expert FFNs.
    A router selects the top-K experts per token (typically K=2 of N=8).

This creates a natural parallelism dimension:

    Expert Parallelism (EP): place different experts on different GPUs.

    Mixtral 8x7B: 8 experts per layer
        EP=8: each GPU holds 1 expert per layer + shared attention weights
        When a token routes to expert 3 → GPU 3 processes it
        Tokens routed to expert 5 → GPU 5 processes it

    Communication: all-to-all shuffle
        After routing decisions, tokens must be sent to the GPU holding their expert.
        After expert computation, results sent back.
        This is an "all-to-all" communication pattern (different from AllReduce).

    EP is often combined with TP:
        TP for the attention layers (shared across all tokens)
        EP for the FFN experts (different tokens go to different GPUs)
```

---

## Summary

```text
Models that don't fit on one GPU need parallelism. Three strategies:

Tensor Parallelism (TP):
    Split each layer's weight matrices across GPUs.
    All GPUs work on the same token together.
    Communication: AllReduce every layer → needs fast interconnect (NVLink).
    Best within a single server. Minimizes per-token latency.

Pipeline Parallelism (PP):
    Assign entire layers to different GPUs.
    GPUs process in sequence: stage 0 → stage 1 → stage 2 → ...
    Communication: one activation send per stage boundary → low bandwidth.
    Pipeline bubble: GPUs idle waiting for previous stage. Needs microbatching.
    Best across servers (less communication than TP).

Data Parallelism (DP):
    Full model copy per replica. Different requests to different replicas.
    No communication during inference. Scales throughput linearly.
    Simplest but requires each replica to fit the model.

Production recipe:
    Within a server: TP (fast NVLink communication)
    Across servers: PP (low cross-server communication)
    For throughput: DP (independent replicas)
    With quantization: often eliminates the need for PP entirely.
```
