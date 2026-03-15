# AI/ML Learning Roadmap

## How to read this file
- [x] = done
- [ ] = not yet started
- Phases are ordered by dependency — each one builds on the last.

---

## Phase 0 — Neural Network Basics (The foundation everything else builds on)
> Goal: Understand how a neural network learns at all. These concepts are assumed everywhere else.

- [x] 01 — Neurons, layers, and the forward pass (weighted sum, bias, how data flows)
- [x] 02 — Loss functions: MSE for regression, cross-entropy for classification
- [x] 03 — Backpropagation and gradient descent (how weights are updated, what a gradient is, Adam/AdamW)
- [x] 04 — Activation functions (ReLU, sigmoid, softmax, GELU — and why non-linearity matters)
- [x] 05 — BatchNorm vs LayerNorm (why transformers use LayerNorm, what normalisation does)
- [x] 06 — Overfitting, dropout, regularisation, early stopping

---

## Phase 1 — Making Vectors (How models turn things into numbers)
> Goal: Understand how text, documents, and images become fixed-length vectors.

- [x] 01 — What is a vector
- [x] 02 — How word vectors are created (word2vec: skip-gram, negative sampling)
- [x] 03 — Word vector walkthrough example
- [x] 04 — Sentence vectors: simple pooling & weighted pooling (TF-IDF, SIF)
- [x] 05 — Document vectors: doc2vec (PV-DM, PV-DBOW)
- [x] 06 — Book/long-text vectors: encoder pooling
- [x] 07 — How a custom text encoder works (Transformer internals: QKV, multi-head attention, residual connections, LayerNorm, positional encoding, [CLS] pooling)
- [x] 08 — Image vectors: off-the-shelf CNN, triplet fine-tuning, SimCLR/MoCo/BYOL
- [x] 09 — How a custom image encoder works (CNN internals: convolution, ReLU, pooling, GAP, ResNet skip connections)
- [x] 10 — Text-image pairs: CLIP / contrastive learning, joint embedding space

---

## Phase 2 — Storing & Searching Vectors (Vector DB internals)
> Goal: Understand how to index millions of vectors and retrieve nearest neighbours fast.
> Note: This is the infrastructure that makes RAG possible — will feel most motivated after Phase 3.

- [x] 01 — Vector storage roadmap overview
- [x] 02 — How to measure vector similarity (cosine, L2, dot product)
- [x] 03 — Flat / brute-force search
- [x] 04 — When flat indexing fails (the scale problem)
- [x] 05 — Graph-based ANN: concept
- [x] 06 — NSW (Navigable Small World graphs)
- [x] 07 — HNSW (Hierarchical NSW): layers, diversification, pruning, bootstrap
- [x] 08 — Space partitioning: concept, axis-aligned vs distance-based
- [x] 09 — KD-trees
- [x] 10 — Ball trees
- [x] 11 — VP-trees (metric trees)
- [x] 12 — IVF (Inverted File Index): k-means clustering + nprobe
- [x] 13 — Scalar Quantization (SQ): int8 compression
- [x] 14 — Product Quantization (PQ): subvector codebooks, ADC, OPQ, SDC vs ADC
- [x] 15 — IVF + PQ: residual encoding, full pipeline, FAISS string format
- [x] 16 — Filtered vector search: pre/post/in-filter, selectivity gap, bitmap indexes
- [x] 17 — Hybrid search: dense + sparse (BM25), RRF fusion, SPLADE
- [x] 18 — Re-ranking: exact distance, cross-encoder, bi-encoder vs cross-encoder, multi-stage pipelines

---

## Phase 3 — LLMs: How Models Generate Text (the decoder side)
> Goal: Understand GPT-style models. Phase 1 covered BERT (encoder = text → vector).
> This covers GPT (decoder = text → MORE text). Everything else in AI sits on top of this.

- [ ] 01 — Encoder vs Decoder vs Encoder-Decoder (BERT vs GPT vs T5)
          Key: causal masking, autoregressive generation, architecture taxonomy
- [ ] 02 — Tokenization deep dive (BPE, SentencePiece, why subwords)
- [ ] 03 — How LLMs generate text (next-token prediction, sampling, temperature, top-k, top-p, greedy)
- [ ] 04 — How LLMs are trained (pre-training on next-token → SFT → RLHF / DPO)
- [ ] 05 — LoRA and parameter-efficient fine-tuning (adapters, rank decomposition)

---

## Phase 4 — Inference Optimizations
> Goal: Understand how LLMs are made fast and cheap enough to serve in production.
> Prerequisite: Phase 3 (especially attention mechanics from Phase 1 file 07).

- [ ] 01 — KV Cache (why recomputing K and V for every token is wasteful, how caching fixes it)
- [ ] 02 — Flash Attention (memory-efficient attention via tiling)
- [ ] 03 — Paged Attention / vLLM (virtual memory for KV cache, high GPU utilisation)
- [ ] 04 — Speculative Decoding (small draft model + large verify model)
- [ ] 05 — Model Quantization (INT8/INT4, GPTQ, AWQ, GGUF — different from vector PQ)
- [ ] 06 — Mixture of Experts / MoE (sparse activation: only part of the network fires per token)
- [ ] 07 — Distillation & SLMs (training small models to mimic large ones)

---

## Phase 5 — RAG: Full Pipeline
> Goal: Combine Phase 2 (vector retrieval) + Phase 3 (LLM generation) into a real system.
> This is where Phase 2 finally feels fully motivated.

- [ ] 01 — Chunking strategies (fixed-size, sentence, semantic, hierarchical)
- [ ] 02 — Full RAG pipeline (chunk → embed → store → retrieve → augment prompt → generate)
- [ ] 03 — RAG evaluation (recall, precision, faithfulness, answer relevance — RAGAS framework)
- [ ] 04 — Guardrails and safety (input/output filtering, toxicity, hallucination detection)
- [ ] 05 — Fine-tuning vs RAG vs prompting (when to use which)

---

## Phase 6 — Agents & Tools
> Goal: Understand how LLMs go from answering questions to taking actions.

- [ ] 01 — ReAct pattern (Reason + Act: interleaving thinking and tool use)
- [ ] 02 — Tool calling / function calling (how LLMs call external APIs)
- [ ] 03 — LangChain and LangGraph (orchestration frameworks, DAGs vs cycles)
- [ ] 04 — Context engineering and memory (conversation history, working memory, long-term memory)
- [ ] 05 — MCP (Model Context Protocol: standardised tool/context interface)
- [ ] 06 — Multi-agent systems (orchestrator + specialist agents, message passing)

---

## Phase 7 — Reasoning
> Goal: Understand how models "think" beyond single-step responses.

- [ ] 01 — Chain of Thought (CoT): prompting models to reason step by step
- [ ] 02 — Tree of Thought (ToT): exploring multiple reasoning branches
- [ ] 03 — Reasoning models (o1/o3-style: internal scratchpad, RLHF on reasoning traces)

---

## Phase 8 — Advanced Topics & Projects
> Goal: Deep dives and applying everything to real projects.

- [ ] 01 — Diffusion models (how text-to-image / text-to-video generation works)
- [ ] 02 — Sparse attention patterns (Longformer, BigBird — handling very long contexts)
- [ ] 03 — Evals: how to measure model and system quality end-to-end

**Project ideas (once Phase 3-6 is done):**
- [ ] Build a RAG system end to end
- [ ] Remote SSH terminal to access Claude Code from phone
- [ ] Understand how Claude Code's skills files work and write a custom skill
- [ ] Understand how Claude Code picks which commands to run on a repo
- [ ] Build a Claude API-powered app using the Anthropic SDK

---

## Quick Reference: Architecture Taxonomy

| Architecture      | Example models        | Input → Output          | Covered in        |
|-------------------|-----------------------|-------------------------|-------------------|
| Encoder-only      | BERT, RoBERTa         | Text → vector           | Phase 1 file 07   |
| Decoder-only      | GPT, Llama, Claude    | Text → more text        | Phase 3 file 01   |
| Encoder-Decoder   | T5, BART              | Text → transformed text | Phase 3 file 01   |
| CNN               | ResNet, EfficientNet  | Image → vector          | Phase 1 files 08-09 |
| Vision Transformer| ViT, CLIP image tower | Image → vector          | Phase 1 file 08   |
| Multi-modal       | CLIP, ALIGN           | Image+Text → shared vec | Phase 1 file 10   |
