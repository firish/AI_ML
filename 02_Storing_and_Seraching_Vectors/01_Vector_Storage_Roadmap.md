## Memory for Vectors

### Storage Footprint
- Typically, “768” is the number of dimensions a vector has.
- Each dimension is stored as a single 32-bit (4-byte) floating-point value by default.

```text
memory_footprint = dims × bytes per value
                 = 768  × 4
                 = 3,072 bytes
```

- So a “768-d float32 vector” occupies 4 × 768 = 3 KB in raw form.
- If the model instead produced 1 024-d vectors, the raw footprint would be 1 024 × 4 = 4 KB, and so on.


### Datatype Choice
- A vector is a long list of decimal numbers—like [0.42, –1.17, …, 0.097 ] that capture subtle shades of meaning.
- If we forced those decimals into plain integers (… –1, 0, 1 …), most of the nuance would disappear, the same way rounding every price to the nearest dollar would ruin a financial spreadsheet.
- Floating-point storage keeps the decimals intact.
  
Why do we use 32-bit (4-byte) floats?
- They’re the sweet spot: enough precision that cosine distances hardly budge, yet small enough that millions of vectors fit in RAM or GPU memory.
- All modern CPUs/GPUs have hardware instructions tuned for 32-bit floats, so math stays fast.
- Going up to 64-bit (8 bytes) doubles memory and cuts speed in half, but rarely improves search accuracy.
- Going down to 16-bit floats or special quantized bytes is common for huge collections.


## Why ordinary databases fall short

- Relational or key-value stores excel at exact matching: “find the row whose key is 42” or “give me every order after June 1.” 
- Nearest-neighbour search is different—you need to ask “what’s close to this 768-number vector?”
- In high-dimensional space the usual tricks (B-trees, hash tables) degrade to nearly scanning the whole table.
- Modern vector stores layer an Approximate Nearest-Neighbour (ANN) index on top of raw storage so they can return “good-enough” neighbours in ≪ 1 ms instead of minutes.

## How a vector database is organised, at a glance
- Raw storage layer – keeps the original float32 vectors and any metadata (document ID, title).
- Index layer (ANN) – a data structure such as HNSW, IVF-PQ, or ScaNN that prunes 99 % of candidates before distance tests.
- Compression layer – optional; turns each 4 × 768-byte vector into as little as 32 bytes while preserving the geometry well enough for recall targets.
- Hybrid filters / re-ranker – combine ANN recall with metadata filters (e.g., tenant-ID) and optional exact re-ranking on the original floats for the final top-k.


## Learning Roadmap

| Step                                  | First questions to answer                                        | Core algorithms / ideas you should know                                                                                                                                                                                      |
| ------------------------------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1 · Vector geometry & similarity**  | *“How do we tell two vectors are close?”*                        | **Cosine similarity**, **Euclidean (L2)**, **Inner-product** (dot), Angular distance; why we L2-normalise for cosine; effect of dimensionality on each measure.                                                              |
| **2 · Exact search baselines**        | *“What happens if we don’t index at all?”*                       | Naïve brute-force (matrix multiply), BLAS/Faiss **FlatL2** scanner, SIMD/GPU batching, early-exit top-k heap. Establishes recall/latency ground truth for later steps.                                                       |
| **3 · Low-dim trees & hashing**       | *“How do classic structures fail as d grows?”*                   | **KD-Tree**, **Ball Tree**, **VP-Tree** (metric trees ≤30 d); **LSH** (Cosine / E2LSH), **SRP-Hash**; pros, curse-of-dimensionality symptoms.                                                                                |
| **4 · Modern ANN indexing**           | *“How do big vector DBs prune 99 % of candidates?”*              | **Graph-based:** HNSW (M, efConstruction, efSearch), NSG, SW-graph; **Quantiser-based:** IVF-Flat (coarse centroids), IVF-HNSW, **ScaNN partitioning**; parameter tuning for recall vs. latency.                             |
| **5 · Compression / quantisation**    | *“How do we fit billions of vectors in RAM/SSD?”*                | **Product Quantization (PQ)**, **Optimised PQ (OPQ)**, Residual / Additive quantisers, **Scalar int8 quant**, **PCA / Random Projection**, **Binary hashing** (SimHash, ITQ); asymmetric vs. symmetric distance computation. |
| **6 · Hybrid retrieval & re-ranking** | *“How do we combine ANN with metadata filters or exact scores?”* | Filter-then-ANN, ANN-then-exact-rerank, HNSW multi-layer filters, query-time L2 recheck, rescoring with language models.                                                                                                     |
| **7 · System-level operations**       | *“How do we keep the index live and reliable?”*                  | Delta-index + nightly rebuild, shard/replica layouts, streaming inserts, checkpointing, GPU vs. CPU memory hierarchies, monitoring recall\@k & tail-latency.                                                                 |
